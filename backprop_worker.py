import multiprocessing, torch, numpy as np, setproctitle, os, time
from model_generator_hf import GenerationModel
from utils_tree import get_conversation_path
from utils import print_colored

def backprop_worker_process(model_path, save_path, trace, args_dict, result_queue, error_queue):
    setproctitle.setproctitle("backprop_worker")
    
    print(f"[Backprop Worker] Starting backprop worker (PID: {os.getpid()})")
    
    # Initialize timing tracking
    timings = {"model_load": 0, "backprop": 0, "model_save": 0}
    
    skip_leaf_update = args_dict.get("skip_leaf_update", False)
    reduction = args_dict.get("reduction", "sum")

    # Load model and optimizer
    T_model_load_start = time.time()
    print(f"[Backprop Worker] Loading model from {model_path}")
    assistant_model = GenerationModel(model_name=model_path, device=None)
    optimizer = torch.optim.SGD(assistant_model.model.parameters(), lr=args_dict["learning_rate"])
    T_model_load_end = time.time()
    timings["model_load"] = T_model_load_end - T_model_load_start
    
    print(f"[Backprop Worker] Model loaded successfully in {timings['model_load']:.2f}s")
    
    
    # Extract training parameters
    normalize_logprobs = args_dict.get("normalize_logprobs", False)
    advantage_estimation = args_dict.get("advantage_estimation", "zero_mean_noneg")
    
    # Process backprop
    user_node_ids = [node["id"] for node in trace if node["role"] == "user"]
    user_node_ids = sorted(user_node_ids, key=lambda x: len(x), reverse=True)  # leaf to root
    id2node = {node["id"]: node for node in trace}
    
    any_updates = False
    corrs_A_LP, corrs_A_RL, corrs_A_NLP = [], [], []
    num_skips_unstable = 0
    losses1 = []
    logprobs_map = {}
    
    print(f"[Backprop Worker] Processing {len(user_node_ids)} user nodes for backprop")
    
    T_backprop_start = time.time()
    for user_node_id in user_node_ids:
        user_node = id2node[user_node_id]
        current_conversation = get_conversation_path(trace, user_node_id)
        children_node_ids = user_node["children"]
        children_nodes = [id2node[child_id] for child_id in children_node_ids]
        # select only children with non-zero advantage
        selected_children_nodes = [child_node for child_node in children_nodes if "advantage" in child_node and child_node["advantage"] != 0.0]

        if skip_leaf_update:
            selected_children_nodes = [child_node for child_node in selected_children_nodes if not child_node["is_leaf"]] # filter out leaf nodes
        
        if len(selected_children_nodes) <= 1:
            continue # not enough nodes left for advantage to exist
        
        # if len(set(backprop_scores)) > 1:
        if any(child_node["advantage"] != 0.0 for child_node in selected_children_nodes):
            # backprop_scores = [child_node["backpropagated_score"] for child_node in selected_children_nodes]
            advantages = [child_node["advantage"] for child_node in selected_children_nodes]
            advantages = torch.tensor(advantages).to(assistant_model.device)

            print_colored(f"[Backprop Worker] Do backprop for {user_node_id}", "green")
            
            # Get logprobs
            response_logprobs = assistant_model.get_logprobs(current_conversation, selected_children_nodes, reduction=reduction)
            
            lengths = torch.tensor([len(node["response_tokens"]) if len(node["response_tokens"]) > 0 else 1 for node in selected_children_nodes]).to(assistant_model.device)
            normalized_logprobs = response_logprobs / lengths
            
            for child_node, logprob in zip(selected_children_nodes, response_logprobs.tolist()):
                logprobs_map[child_node["id"]] = logprob
            
            # Check for unstable logprobs
            if any(logprob < -1000 for logprob in response_logprobs):
                print_colored(f"[Backprop Worker] Skipping backprop for {user_node_id} because of NaN logprobs", "red")
                num_skips_unstable += 1
            else:
                if normalize_logprobs:
                    loss = -torch.sum(advantages * normalized_logprobs)
                else:
                    loss = -torch.sum(advantages * response_logprobs)
                
                losses1.append(loss.item())
                loss.backward()
                
                if not torch.isnan(loss).any():
                    torch.nn.utils.clip_grad_norm_(assistant_model.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    any_updates = True
                else:
                    print_colored(f"[Backprop Worker] NaN in loss for {user_node_id}; skipping update", "red")
            
            # Compute stats and print out based on all children nodes, not just selected ones
            response_lengths_num = [len(node["response_tokens"]) for node in children_nodes]
            all_backtrack_scores = [node["backpropagated_score"] for node in children_nodes]
            logprobs_num = [node["logprobs"] for node in children_nodes] # be careful, this is from VLLM, not from the HF model during backprop
            normalized_logprobs_num = [0.0 if resp_length == 0 else logprob / resp_length for logprob, resp_length in zip(logprobs_num, response_lengths_num)]
            advantages_num = [node["advantage"] for node in children_nodes]
            strategies = [child_node["strategy_name"] for child_node in children_nodes]
            
            # Print debug info
            print(f"Strategy:             {', '.join([(f'{x[:6].upper()}').rjust(7) for x in strategies])}")
            print(f"Backtrack scores:     {', '.join([(f'{x:.2f}').rjust(7) for x in all_backtrack_scores])}")
            print(f"Advantages:           {', '.join([(f'{x:.2f}').rjust(7) for x in advantages_num])}")
            
            correlation_advantages_logprobs = np.corrcoef(advantages_num, logprobs_num)[0, 1]
            correlation_advantages_lengths = np.corrcoef(advantages_num, response_lengths_num)[0, 1]
            correlation_advantages_normalized_logprobs = np.corrcoef(advantages_num, normalized_logprobs_num)[0, 1]
            
            corrs_A_LP.append(correlation_advantages_logprobs)
            corrs_A_RL.append(correlation_advantages_lengths)
            corrs_A_NLP.append(correlation_advantages_normalized_logprobs)
    
    T_backprop_end = time.time()
    timings["backprop"] = T_backprop_end - T_backprop_start
    
    # Save model if any updates were made
    if any_updates:
        print(f"[Backprop Worker] Saving updated model to {save_path}")
        assistant_model.save_model(save_path)
        print(f"[Backprop Worker] Model saved successfully")
        T_model_save_end = time.time()
        timings["model_save"] = T_model_save_end - T_backprop_end
        
    # Prepare results
    results = {
        "any_updates": any_updates,
        "corrs_A_LP": corrs_A_LP,
        "corrs_A_RL": corrs_A_RL,
        "corrs_A_NLP": corrs_A_NLP,
        "num_skips_unstable": num_skips_unstable,
        "losses1": losses1,
        "logprobs_map": logprobs_map,
        "num_backprop_updates": len(corrs_A_LP),
        "timings": timings
    }
    
    # Send results back
    result_queue.put(results)
    print(f"[Backprop Worker] Backprop completed successfully")

class BackpropWorker:
    """
    Manager class for backprop worker processes.
    """
    
    def __init__(self):
        self.process = None
        self.result_queue = None
        self.error_queue = None
    
    def run_backprop(self, model_path, save_path, trace, args_dict, timeout=300):
        self.result_queue = multiprocessing.Queue()
        self.error_queue = multiprocessing.Queue()
        
        self.process = multiprocessing.Process(
            target=backprop_worker_process,
            args=(model_path, save_path, trace, args_dict, self.result_queue, self.error_queue),
            daemon=False
        )
        
        print(f"[Backprop Manager] Starting backprop worker process")
        self.process.start()
        
        self.process.join(timeout=timeout)
        
        if self.process.is_alive():
            print(f"[Backprop Manager] Backprop worker timed out, terminating")
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                print(f"[Backprop Manager] Force killing backprop worker")
                self.process.kill()
                self.process.join()
            return None
        
        if not self.error_queue.empty():
            error_info = self.error_queue.get()
            print(f"[Backprop Manager] Error in backprop worker: {error_info['error']}")
            print(f"[Backprop Manager] Traceback: {error_info['traceback']}")
            return None
        
        if not self.result_queue.empty():
            results = self.result_queue.get()
            print(f"[Backprop Manager] Backprop completed successfully")
            return results
        else:
            print(f"[Backprop Manager] No results received from backprop worker")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

        if self.result_queue:
            self.result_queue.close()
        if self.error_queue:
            self.error_queue.close()
