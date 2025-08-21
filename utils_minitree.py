import copy, numpy as np

def find_split_indeces(generate_response, num_splits, min_index=0):
    # find the tokens with the most uncertainty
    # min_index == 0 is on purpose; we never split on the first token; if that's desired, can set min_index to -1
    per_token_logprobs = generate_response["per_token_logprobs"]

    for i, token in enumerate(per_token_logprobs):
        token["index"] = i

    selected_per_token_logprobs = [token for token in per_token_logprobs if token["index"] > min_index]
    sorted_per_token_logprobs = sorted(selected_per_token_logprobs, key=lambda x: x["logprob"])
    selected_transitions = [token["index"] for token in sorted_per_token_logprobs[:num_splits]]

    for token in per_token_logprobs:
        del token["index"]

    return sorted(selected_transitions)

def reconstitute_conversation(conversation, todo_item, id2node, tokenizer):
    if todo_item["parent_sid"] == "root":
        return conversation

    parent = id2node[todo_item["parent_sid"]]

    reconstituted_conversation = copy.deepcopy(conversation)

    parent_prefix_text = tokenizer.decode(parent["response_tokens"][:todo_item["split_idx"]])
    if reconstituted_conversation[-1]["role"] == "assistant":
        reconstituted_conversation[-1]["content"] += parent_prefix_text
    else:
        reconstituted_conversation.append({"role": "assistant", "content": parent_prefix_text})

    return reconstituted_conversation

def merge_responses(response, parent, split_idx, tokenizer):
    # we need to: (1) merge the response_texts, (2) merge the response_tokens, (3) merge the per_token_logprobs, (4) merge the logprobs

    merged_response_tokens = parent["response_tokens"][:split_idx] + response["response_tokens"]
    merged_response_logprobs = parent["per_token_logprobs"][:split_idx] + response["per_token_logprobs"]

    response["response_text"] = tokenizer.decode(merged_response_tokens, skip_special_tokens=True)
    response["response_text_illustrated"] = tokenizer.decode(parent["response_tokens"][:split_idx]) + "|" + tokenizer.decode(response["response_tokens"], skip_special_tokens=True)
    response["response_tokens"] = merged_response_tokens
    response["per_token_logprobs"] = merged_response_logprobs

    response["node_start_idx"] = split_idx
    response["node_end_idx"] = len(merged_response_tokens)

    response["logprobs"] = sum([tok["logprob"] for tok in merged_response_logprobs])
    return response

def generate_backtrack_minitree(response_minitree):
    # wonderful code
    assert len(set(len(n['id']) for n in response_minitree)) == 1

    common_prefix = ""
    if "." in response_minitree[0]['id']:
        r_idx = response_minitree[0]['id'].rindex(".")
        common_prefix = response_minitree[0]['id'][:r_idx + 1]

    node_id2node = {common_prefix: "root"}
    depth = len(response_minitree[0]['id']) - 1 - len(common_prefix)

    backtrack_tree = []
    for this_depth in range(0, depth+1):
        parents = sorted(set(n["id"][:(len(common_prefix)+this_depth)] for n in response_minitree))
        for parent in parents:
            parent_b = node_id2node[parent]
            siblings = [n for n in response_minitree if n["id"].startswith(parent)]
            # for n in siblings:
            #     print(len(common_prefix), this_depth, len(n['id']), n["id"])
            deg_ids = sorted(set(n['id'][len(common_prefix)+this_depth] for n in siblings))
            left_idx = 0
            if parent_b != "root":
                left_idx = parent_b["right_idx"]

            for deg_id in deg_ids:
                node_id = parent+deg_id
                if parent_b != "root":
                    parent_b["children"].append(node_id)
                children = [n for n in response_minitree if n["id"].startswith(node_id)]
                min_length = min([len(n["response_tokens"]) for n in children])
                right_idx = 0
                while right_idx < min_length and len(set(n["response_tokens"][right_idx] for n in children)) == 1:
                    right_idx += 1 

                response_tokens = children[0]["response_tokens"][left_idx:right_idx]
                backtrack_node = {"id": node_id, "role": children[0]["role"], "parent_id": parent, "left_idx": left_idx, "right_idx": right_idx, "response_tokens": response_tokens, "children": [], "mini_depth": this_depth, "response_total_length": len(children[0]["response_tokens"])}
                if "per_token_logprobs" in children[0]:
                    backtrack_node["per_token_logprobs"] = children[0]["per_token_logprobs"][left_idx:right_idx]
                    backtrack_node["logprobs"] = sum([lp["logprob"] for lp in backtrack_node["per_token_logprobs"]])

                node_id2node[node_id] = backtrack_node
                backtrack_tree.append(backtrack_node)
    return backtrack_tree

def calculate_backtrack_scores_generalized(backtrack_tree, advantage_estimation="zero_mean"):
    # Calculate scores for the backtrack tree
    assert advantage_estimation in ["zero_mean", "zero_mean_no_neg"], f"Unknown advantage estimation method: {advantage_estimation}"
    id2node = {n["id"]: n for n in backtrack_tree}
    ordered_ids = sorted(id2node.keys(), key=lambda x: (len(x), x), reverse=True)
    for id in ordered_ids:
        node = id2node[id]
        if len(node["children"]) > 0:
            child_nodes = [id2node[child_id] for child_id in node["children"]]
            # mean = np.mean([id2node[child_id]["backtrack_score"] for child_id in node["children"]])
            mean = np.mean([child["backtrack_score"] for child in child_nodes])
            node["backtrack_score"] = mean
            # compute the children's advantage, while we're at it...
            for child in child_nodes:
                if advantage_estimation == "zero_mean":
                    child["advantage"] = child["backtrack_score"] - mean
                elif advantage_estimation == "zero_mean_no_neg":
                    child["advantage"] = max(child["backtrack_score"] - mean, 0.0)
        else:
            node["backtrack_score"] = node.get("score", 0.0)


def generate_backtrack_tree(response_tree):
    # This does it all across multiple turns (for MTCO)
    max_depth = max(n["depth"] for n in response_tree)

    backtrack_tree = [n for n in response_tree if n["depth"] == 0]
    for n in backtrack_tree:
        n["turn_depth"] = 0

    for depth in range(1, max_depth + 1):
        user_nodes = [n for n in response_tree if n["depth"] == depth and n["role"] == "user"]

        assistant_nodes = [n for n in response_tree if n["depth"] == depth and n["role"] == "assistant"]
        for user_node in user_nodes:
            user_node["turn_depth"] = depth
            user_node["children"] = [] # going to recompute what these are
            backtrack_tree.append(user_node)

            this_assistant_nodes = [an for an in assistant_nodes if an["parent"] == user_node["id"]]

            turn_children_mapping = {n["id"]: n["children"] for n in this_assistant_nodes}
            old_scores = {n["id"]: n["score"] for n in this_assistant_nodes if "score" in n}
            this_backtrack_subtree = generate_backtrack_minitree(this_assistant_nodes)
            for n in this_backtrack_subtree:
                n["turn_depth"] = depth
                if n["id"] in old_scores:
                    n["score"] = old_scores[n["id"]] # copy over the score, important for the leaves
                if n["id"] in turn_children_mapping: # this maps the last turn to the next turn's user turn
                    n["children"] = turn_children_mapping[n["id"]]

            user_node["children"] = [n["id"] for n in this_backtrack_subtree if n["mini_depth"] == 0] # the user's children is the root of the minitree, this connects the turns, hopefully

            backtrack_tree += this_backtrack_subtree
    return backtrack_tree
