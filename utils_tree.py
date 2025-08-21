from typing import List, Dict, Any
import re
import itertools
import tqdm
import numpy as np
from collections import Counter

def get_conversation_path(tree, node_id: str) -> List[Dict[str, Any]]:
    """Returns the conversation path from root to the given node, including all log nodes."""
    path = []
    current_id = node_id
    
    # First get all nodes up to root
    while current_id is not None:
        for node in tree:
            if node["id"] == current_id:
                path.append(node)
                current_id = node["parent"]
                break

    path = list(reversed(path))
    return path

def calculate_advantage(sibling_nodes, advantage_estimation="zero_mean"):
    assert advantage_estimation in ["zero_mean", "zero_mean_noneg"]
    mean_score = np.mean([node["backpropagated_score"] for node in sibling_nodes if node.get("strategy_name", "null") == "null"])
    for node in sibling_nodes:
        if advantage_estimation == "zero_mean":
            node["advantage"] = node["backpropagated_score"] - mean_score
        elif advantage_estimation == "zero_mean_noneg":
            node["advantage"] = max(node["backpropagated_score"] - mean_score, 0.0)
    return sibling_nodes

def calculate_backtrack_scores(tree, advantage_estimation="zero_mean", verbose=False):
    id2node = {node["id"]: node for node in tree}

    assistant_node_ids = sorted([id for id in id2node.keys() if re.match(r".*A\d+$", id)], key=len, reverse=True)
    for assistant_node_id in assistant_node_ids:
        user_node_child_ids = id2node[assistant_node_id]["children"]
        assistant_children_ids = []
        for user_node_child_id in user_node_child_ids:
            assistant_children_ids += id2node[user_node_child_id]["children"]
        backpropagated_score = None
        is_leaf = len(assistant_children_ids) == 0

        if is_leaf:
            backpropagated_score = id2node[assistant_node_id].get("score", 0.0)
        else:
            # only grab null strategy children
            children_scores = [id2node[child_id]["backpropagated_score"] for child_id in assistant_children_ids if id2node[child_id].get("strategy_name", "null") == "null"]
            backpropagated_score = sum(children_scores) / len(children_scores)

        id2node[assistant_node_id]["is_leaf"] = is_leaf
        id2node[assistant_node_id]["backpropagated_score"] = backpropagated_score
    
    # go over all the user nodes
    user_node_ids = sorted([id for id in id2node.keys() if re.match(r".*U\d+$", id)], key=len, reverse=True)
    for user_node_id in user_node_ids:
        user_node = id2node[user_node_id]
        children_node_ids = user_node["children"]
        children_nodes = [id2node[child_id] for child_id in children_node_ids]
        if len(children_nodes) <= 1:
            continue

        children_nodes = calculate_advantage(children_nodes, advantage_estimation)


    if verbose:
        for node_id in assistant_node_ids:
            print(f"{node_id} backpropagated_score: {id2node[node_id]['backpropagated_score']}")


def bin_score(x, num_bins=50):
    bin_width = 1.0 / num_bins
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    bin_index = min(int(x / bin_width), num_bins - 1)    
    return (bin_index + 0.5) * bin_width

def compute_backpropagated_score_distributions(tree, subdegree, num_bins=50):    
    id2node = {node["id"]: node for node in tree}
    assistant_nodes = [node for node in tree if node["role"] == "assistant"]
    assistant_nodes = sorted(assistant_nodes, key=lambda n: len(n["id"]), reverse=True)
    
    key_estimate = f"score_estimate_deg{subdegree}"
    
    def normalize_counter(counter, eps=1e-4):
        """Normalize a counter to probabilities"""
        total = sum(counter.values())
        if total == 0:
            return counter
        normalized = Counter({k: v / total for k, v in counter.items()})
        filtered = Counter({k: v for k, v in normalized.items() if v >= eps})
        filtered_total = sum(filtered.values())
        if filtered_total > 0:
            return Counter({k: v / filtered_total for k, v in filtered.items()})
        else:
            return counter

    for node in assistant_nodes:
        if node["is_leaf"]:
            binned_score = bin_score(node.get("score", 0.0), num_bins=num_bins)
            node[key_estimate] = Counter({binned_score: 1.0})
        else:
            user_node_child = id2node[node["children"][0]]
            assistant_descendant_ids = user_node_child["children"]
            assistant_descendants = [id2node[child_id] for child_id in assistant_descendant_ids]
            children_estimates = [n[key_estimate] for n in assistant_descendants]
            node[key_estimate] = Counter()
            
            for selected_ests in itertools.combinations(children_estimates, subdegree):
                unique_values = [list(est.keys()) for est in selected_ests]
                probabilities = [list(est.values()) for est in selected_ests]
                for value_combo in itertools.product(*unique_values):
                    avg_reward = bin_score(sum(value_combo) / len(value_combo), num_bins=num_bins)
                    
                    combo_prob = 1.0
                    for i, val in enumerate(value_combo):
                        idx = unique_values[i].index(val)
                        combo_prob *= probabilities[i][idx]
                    
                    node[key_estimate][avg_reward] += combo_prob
            
            node[key_estimate] = normalize_counter(node[key_estimate])

def compute_all_score_distributions(tree, num_bins=50):
    user_nodes = [n for n in tree if n["role"] == "user"]
    max_degree = max([len(n["children"]) for n in user_nodes])
    for subdegree in tqdm.trange(2, max_degree + 1):
        compute_backpropagated_score_distributions(tree, subdegree, num_bins=num_bins)

def compute_percentiles(distribution, percentiles=[0.5]):
    if not distribution:
        return [0.0] * len(percentiles)
    
    sorted_items = sorted(distribution.items())
    scores = [item[0] for item in sorted_items]
    probs = [item[1] for item in sorted_items]

    cumulative_probs = []
    cumsum = 0.0
    for prob in probs:
        cumsum += prob
        cumulative_probs.append(cumsum)
    
    results = []
    for percentile in percentiles:
        if percentile <= 0:
            results.append(scores[0])
            continue
        if percentile >= 1:
            results.append(scores[-1])
            continue
        
        found = False
        for i, cum_prob in enumerate(cumulative_probs):
            if cum_prob >= percentile:
                if i == 0:
                    results.append(scores[0])
                else:
                    prev_cum_prob = cumulative_probs[i-1]
                    if cum_prob == prev_cum_prob:
                        results.append(scores[i])
                    else:
                        weight = (percentile - prev_cum_prob) / (cum_prob - prev_cum_prob)
                        interpolated = scores[i-1] + weight * (scores[i] - scores[i-1])
                        results.append(interpolated)
                found = True
                break
        
        if not found:
            print(f"Warning: percentile {percentile} not found in distribution")
            results.append(scores[-1])
    
    return results

def compute_probability_flip(distrib1, distrib2, strict_flip=True):
    total_probability = 0.0
    for score_n1, prob_n1 in distrib1.items():
        for score_n2, prob_n2 in distrib2.items():
            if (strict_flip and score_n2 < score_n1) or (not strict_flip and score_n2 <= score_n1):
                total_probability += prob_n1 * prob_n2

    return total_probability


def compute_gap_distribution(distrib1, distrib2): # , rebin=False, num_bins=50
    # since it's a distribution, we might want to rebin to avoid explosion of size, not needed if not used recursively in backtracking
    gap_distribution = Counter()
    for score_n1, prob_n1 in distrib1.items():
        for score_n2, prob_n2 in distrib2.items():
            gap_distribution[score_n2 - score_n1] += prob_n1 * prob_n2
    # this doesn't work... because gaps can be in [-1, 1] ... would have to implement a new binning function for gaps
    # if rebin:
    #     gap_distribution = Counter({bin_score(k, num_bins=num_bins): v for k, v in gap_distribution.items()})
    return gap_distribution

def compute_binned_gap_distribution(gap_distribution, bins):
    binned_gap_distribution = Counter()
    for gap, prob in gap_distribution.items():
        for bin_idx, bin in enumerate(bins):
            if bin[0] <= gap <= bin[1]:
                binned_gap_distribution[bin] += prob
    return dict(binned_gap_distribution)

if __name__ == "__main__":
    import json

    # with open("data/mock_tree_data.json", "r") as f:
    #     global_tree = json.load(f)
    # calculate_backpropagated_scores(global_tree)

    # id2node = {node["id"]: node for node in global_tree}
    # assistant_node_ids = sorted([id for id in id2node.keys() if re.match(r".*A\d+$", id)], key=len, reverse=True)
    # for node_id in assistant_node_ids:
    #     print(node_id, id2node[node_id]["backpropagated_score"])

    example_distrib = Counter({0.0: 0.5, 1.5: 0.5})
    print(compute_percentiles(example_distrib, percentiles=[0.25, 0.5, 0.6, 0.75, 0.9]))


    distrib1 = {0.99: 0.7633928571428571, 0.75: 0.22321428571428573, 0.51: 0.013392857142857142}
    distrib2 = {0.99: 0.9375, 0.75: 0.0625}
    print(compute_probability_flip(distrib1, distrib2, strict_flip=True))
    print(compute_probability_flip(distrib1, distrib2, strict_flip=False))

    print(compute_gap_distribution(distrib1, distrib2))

    gap_bins = [(-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5), (-0.5, -0.4), (-0.4, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, -0.01), (-0.01, 0.01), (0.01, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    print(compute_binned_gap_distribution(compute_gap_distribution(distrib1, distrib2), gap_bins))
