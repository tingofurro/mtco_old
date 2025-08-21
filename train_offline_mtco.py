import torch, json, numpy as np, argparse, random, os, tqdm
from utils_tree import get_conversation_path
from model_generator_hf import GenerationModel
from utils import print_colored
parser = argparse.ArgumentParser(description="Train offline MTCO model")
parser.add_argument("--experiment_folder", type=str, default="experiments/exp_20250524_4/")
parser.add_argument("--model_name", type=str, default="microsoft/phi-4")
parser.add_argument("--learning_rate", type=float, default=2e-3)
args = parser.parse_args()

learning_rate = args.learning_rate

assistant_model = GenerationModel(model_name=args.model_name)

optimizer = torch.optim.SGD(assistant_model.model.parameters(), lr=learning_rate)
dataset_file = os.path.join(args.experiment_folder, "tree_logs.jsonl")

model_folder = os.path.join(args.experiment_folder, "offline_model")
os.makedirs(model_folder, exist_ok=True)

logs = []
with open(dataset_file, "r") as f:
    for line in f:
        logs.append(json.loads(line))

random.shuffle(logs)

for log in tqdm.tqdm(logs):
    # print("===========================================================")
    trace = log["trace"]
    user_node_ids = [node["id"] for node in trace if node["role"] == "user"]
    id2node = {node["id"]: node for node in trace}
    for user_node_id in user_node_ids[::-1]:
        user_node = id2node[user_node_id]
        current_conversation = get_conversation_path(trace, user_node_id)
        children_node_ids = user_node["children"]
        children_nodes = [id2node[child_id] for child_id in children_node_ids]
        back_prop_scores = [child_node["backpropagated_score"] for child_node in children_nodes]
        mean_backprop_score = np.mean(back_prop_scores)
        advantages = torch.tensor([back_prop_score - mean_backprop_score for back_prop_score in back_prop_scores]).to(assistant_model.device)

        if len(set(back_prop_scores)) == 1:
            pass
        else:
            print_colored(f"[log] Do backprop for {user_node_id} {children_node_ids}", "green")

            # with autocast(device_type=assistant_model.device):
            response_logprobs = assistant_model.get_logprobs(current_conversation, children_nodes) # relies on the response_tokens key

            # if any of the logprobs < -1000, then skip
            if any(logprob < -1000 for logprob in response_logprobs):
                print_colored(f"[log] Skipping {user_node_id} because of logprobs < -1000", "red")
                continue


            assert not torch.isnan(advantages).any(), "NaN in advantages"
            assert not torch.isnan(response_logprobs).any(), "NaN in response_logprobs"

            Loss = -torch.sum(advantages * response_logprobs)
            Loss.backward()

            assert not torch.isnan(Loss).any(), "NaN in Loss"
            torch.nn.utils.clip_grad_norm_(assistant_model.model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Convert to lists for correlation and printing
            advantages_num = advantages.tolist()
            logprobs_num = response_logprobs.tolist()
            response_lengths_num = [len(node["response_tokens"]) for node in children_nodes]

            # Print formatted information like in train_mtco.py
            print(f"Backprop scores:  {", ".join([(f'{x:.2f}').rjust(6) for x in back_prop_scores])}")
            print(f"Advantages:       {", ".join([(f'{x:.2f}').rjust(6) for x in advantages_num])}")
            print(f"Logprobs:         {", ".join([(f'{x:.2f}').rjust(6) for x in logprobs_num])}")
            print(f"Response lengths: {", ".join([str(x).rjust(6) for x in response_lengths_num])}")
            
            # Calculate and print correlations
            correlation_advantages_logprobs = np.corrcoef(advantages_num, logprobs_num)[0, 1]
            correlation_advantages_lengths = np.corrcoef(advantages_num, response_lengths_num)[0, 1]
            print(f"Corr(advantages, logprobs): {correlation_advantages_logprobs:.3f}")
            print(f"Corr(advantages, response_lengths): {correlation_advantages_lengths:.3f}")
            print(f"Loss: {Loss.item():.4f}")

            # recompute the logprobs, this time without gradients
            # logprobs2 = assistant_model.get_logprobs(current_conversation, children_nodes, use_grad=False)
            # print("logprobs (after backprop)", logprobs2.tolist())

            # loss2 = -torch.sum(advantages * logprobs2)
            # print("loss (after backprop)", loss2)

# save the model
assistant_model.model.save_pretrained(model_folder)
assistant_model.tokenizer.save_pretrained(model_folder)

# save the arguments
with open(os.path.join(model_folder, "args.json"), "w") as f:
    json.dump(vars(args), f)
