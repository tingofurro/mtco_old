from utils_tree import get_conversation_path, calculate_backpropagated_scores
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from system_agent import SystemAgent
from typing import List, Dict, Any
from user_agent import UserAgent
from tasks import get_task
from llms import generate
import json, random, time

class SimulatorShardedTree:
    def __init__(self, task_name, sample, degree=2, assistant_model="gpt-4o-mini", user_model="gpt-4o-mini", assistant_temperature=1.0):
        assert degree >= 1, "Degree must be at least 1"

        self.task = get_task(task_name)
        self.dataset_fn = self.task.get_dataset_file()
        self.sample = sample
        self.degree = degree
        self.assistant_model = assistant_model
        self.user_model = user_model
        self.assistant_temperature = assistant_temperature
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(task_name, sample)

        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()

        # Initialize trace as a tree structure
        self.trace = [{
            "role": "system", 
            "content": self.system_message, 
            "timestamp": date_str(),
            "parent": None,
            "children": [],
            "id": "S",
            "depth": 0
        }]

    def get_num_turns(self, conversation_path: List[Dict[str, Any]], participant="assistant"):
        return sum(1 for msg in conversation_path if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):
        shards = self.sample["shards"]
        max_depth = len(shards) + 1
        print(f"Max depth will be: {max_depth}")

        timings = {"assistant_generation": 0, "user_generation": 0, "verification": 0, "answer_evaluation": 0, "total_path_time": 0}

        active_paths = [{"node_id": self.trace[0]["id"], "is_completed": False, "is_correct": False, "score": None, "parent": None}]

        while active_paths:
            path_start_time = time.time()
            node_is_completed = False
            current_path = active_paths.pop(0)
            current_node_id = current_path["node_id"]
            conversation_path = get_conversation_path(self.trace, current_node_id)

            num_assistant_responses = len([msg for msg in conversation_path if msg["role"] == "assistant"])
            
            # Check if all hints are revealed for this path
            shard_ids = set([msg["shard_id"] for msg in conversation_path 
                          if msg["role"] == "user" and "shard_id" in msg])
            
            shard_ids_list = [msg["shard_id"] for msg in conversation_path if msg["role"] == "user" and "shard_id" in msg]
            if verbose:
                print(f"Number of active paths: {len(active_paths)}; Current node: {current_node_id} (shard_ids: {shard_ids_list})")

            if len(shard_ids) == len(shards):
                if verbose:
                    print_colored(f"[log] all shards revealed for path {current_node_id}", "blue")
                continue

            if num_assistant_responses >= max_depth:
                if verbose:
                    print_colored(f"[log] max depth reached for path {current_node_id}", "blue")
                continue

            # 1. Generate user response
            user_start_time = time.time()
            user_response, shard_revealed_id = self.user_agent.generate_response(conversation_path, self.sample)
            user_end_time = time.time()
            timings["user_generation"] += user_end_time - user_start_time
            
            # Add user node
            user_node_id = f"{current_node_id}.U0"
            user_node = {"role": "user", "content": user_response, "timestamp": date_str(), "parent": current_node_id, "children": [], "id": user_node_id, "depth": len(conversation_path)}
            
            # Add shard information directly to user node if revealed
            if shard_revealed_id != -1:
                user_node["shard_id"] = shard_revealed_id
            
            self.trace.append(user_node)
            if verbose:
                print_colored(f"[Generating user      {user_node_id}]", "green")
                if shard_revealed_id != -1:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")
            
            # Update parent's children
            for node in self.trace:
                if node["id"] == current_node_id:
                    node["children"].append(user_node_id)
                    break

            # 2. Generate multiple assistant responses
            assistant_responses = []
            current_conversation = get_conversation_path(self.trace, user_node_id)
            conversation_so_far = extract_conversation(current_conversation, to_str=False)
            max_tokens = 4000 if "o1-" in self.assistant_model else 1000

            assistant_start_time = time.time()
            for _ in range(self.degree):
                assistant_response_obj = generate(conversation_so_far, model=self.assistant_model, temperature=self.assistant_temperature, step="sharded-tree-assistant-generation", return_metadata=True, max_tokens=max_tokens)
                assistant_responses.append(assistant_response_obj["message"])

            assistant_end_time = time.time()
            timings["assistant_generation"] += assistant_end_time - assistant_start_time
                
            for i, assistant_response in enumerate(assistant_responses):
                assistant_node_id = f"{user_node_id}.A{i}"
                assistant_node = {"role": "assistant", "content": assistant_response, "timestamp": date_str(), "parent": user_node_id, "children": [], "id": assistant_node_id, "depth": len(conversation_path) + 1}

                self.trace.append(assistant_node)
                user_node["children"].append(assistant_node_id)

                if verbose:
                    print_colored(f"[Generating assistant   {assistant_node_id}]", "red")

                # 3. Verify assistant response
                verification_start_time = time.time()
                current_conversation = get_conversation_path(self.trace, assistant_node_id)
                response_strategy = self.system_agent.classify_assistant_response(current_conversation)
                verification_end_time = time.time()
                timings["verification"] += verification_end_time - verification_start_time
                
                # Add verification information directly to assistant node
                assistant_node["response_strategy"] = response_strategy

                if verbose:
                    print_colored(f"[log] response strategy: {response_strategy}", "blue")

                if response_strategy == "answer_attempt":
                    evaluation_start_time = time.time()
                    extracted_answer = self.system_agent.extract_answer(current_conversation)
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    evaluation_end_time = time.time()
                    timings["answer_evaluation"] += evaluation_end_time - evaluation_start_time

                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)
                    if is_correct is not None and score is None:
                        score = 1 if is_correct else 0

                    if score == 1.0 and not is_correct:
                        is_correct = True

                    # Add evaluation information directly to assistant node
                    assistant_node["extracted_answer"] = extracted_answer
                    assistant_node["evaluation_return"] = evaluation_return
                    assistant_node["is_correct"] = is_correct
                    assistant_node["score"] = score

                    if verbose:
                        print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                    if is_correct:
                        node_is_completed = True
                        if verbose:
                            print_colored(f"[log] conversation path completed: {is_correct}; score: {score}", "blue")
                        continue

                if not node_is_completed:
                    active_paths.append({
                        "node_id": assistant_node_id,
                        "is_completed": False,
                        "is_correct": False,
                        "score": None
                    })
            
            path_end_time = time.time()
            timings["total_path_time"] += path_end_time - path_start_time
            
            # Print timing information at the end of each active_path iteration
            if verbose:
                print_colored(f"[Timings] User: {timings['user_generation']:.2f}s | Assistant: {timings['assistant_generation']:.2f}s | Verification: {timings['verification']:.2f}s | Evaluation: {timings['answer_evaluation']:.2f}s | Path: {timings['total_path_time']:.2f}s", "yellow")

        if save_log:
            # Calculate backpropagated scores before logging
            calculate_backpropagated_scores(self.trace, verbose=verbose)
            
            # Find the last answer evaluation from the assistant nodes
            final_assistant_node = None
            for node in reversed(self.trace):
                if node["role"] == "assistant" and "is_correct" in node:
                    final_assistant_node = node
                    break
            
            final_is_correct, final_score = False, 0
            if final_assistant_node:
                final_is_correct = final_assistant_node["is_correct"]
                final_score = final_assistant_node["score"]

            log_conversation("sharded-tree", self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.user_model, self.trace, final_is_correct, final_score)
        
        # Print final timings summary
        if verbose:
            print_colored("\n=== FINAL TIMING SUMMARY ===", "yellow")
            print_colored(f"User generation: {timings['user_generation']:.2f}s", "yellow")
            print_colored(f"Assistant generation: {timings['assistant_generation']:.2f}s", "yellow")
            print_colored(f"Verification: {timings['verification']:.2f}s", "yellow")
            print_colored(f"Answer evaluation: {timings['answer_evaluation']:.2f}s", "yellow")
            print_colored(f"Total path time: {timings['total_path_time']:.2f}s", "yellow")
            print_colored("==========================\n", "yellow")
            
        return final_is_correct, final_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="database")
    parser.add_argument("--assistant_model", type=str, default="l-phi4")
    parser.add_argument("--user_model", type=str, default="l-phi4")
    parser.add_argument("--assistant_temperature", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    dataset_fn = "data/sharded_instructions_600.json"
    with open(dataset_fn, "r") as f:
        data = json.load(f)

    data = [d for d in data if d["task"] == args.task]

    data = [d for d in data if len(d["shards"]) <= 4]

    sample = random.choice(data)

    simulator = SimulatorShardedTree(
        task_name=args.task,
        sample=sample,
        degree=args.degree,
        assistant_model=args.assistant_model,
        assistant_temperature=args.assistant_temperature,
        user_model=args.user_model
    )
    simulator.run(verbose=True, save_log=True) 
