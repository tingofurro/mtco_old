from utils import print_colored, extract_conversation, date_str, DoublePrint, subsample_responses
import json, random, time, numpy as np, torch, argparse, os, shutil, tqdm, copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_tree import get_conversation_path, calculate_backtrack_scores
from utils_guardrails import RepetitionGuardrail, MaxLengthGuardrail
from genserv_client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
from utils_experiments import make_exp_folder
from utils_log import log_conversation
from user_agent import UserAgent
from datetime import datetime
from tasks import get_task

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
parser.add_argument("--ignore_tasks", type=str, nargs="+", default=[])
parser.add_argument("--base_model", type=str, default="microsoft/phi-4")

# Tree Building Related
parser.add_argument("--max_tokens", type=int, default=1000)
parser.add_argument("--turn_degree", type=int, default=2)
parser.add_argument("--turn_depth", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument("--user_model", type=str, default="t-gpt-4o-mini")

# Misc
parser.add_argument("--genserv_port", type=int, default=5000)
parser.add_argument("--evalserv_port", type=int, default=5001)
parser.add_argument("--save_to_shm", action="store_true")
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--num_eval_workers", type=int, default=8)

args = parser.parse_args()


CURRENT_LATEST_MODEL_PATH = args.base_model

exp_folder = make_exp_folder(prefix="mtco")
print(f"Made new exp folder: {exp_folder}")

# guardrails
guardrails = [
    {"name": "repetition", "enforced": False, "guardrail": RepetitionGuardrail(name="repetition", description="Repetition", n_gram=5, repetition_threshold=5)},
    {"name": "max_length", "enforced": False, "guardrail": MaxLengthGuardrail(name="max_length", description="Max Length", max_length=1000)}
]

temperature = 1.0

MODEL_PATH = os.path.join(exp_folder, "model")

DoublePrint(os.path.join(exp_folder, "run_logs.ans"))
log_file = os.path.join(exp_folder, "tree_logs.jsonl")
run_stats_file = os.path.join(exp_folder, "run_stats.jsonl")
run_params_file = os.path.join(exp_folder, "run_params.json")

with open(run_params_file, "w") as f:
    json.dump(vars(args), f)

with open(args.dataset_fn, "r") as f:
    data = json.load(f)

samples = [d for d in data if d.get("split", "train") == "train" and d["task"] not in args.ignore_tasks and len(d["shards"]) == 4]

print(f"Training samples: {len(samples)}")


assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:{args.genserv_port}")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:{args.evalserv_port}")

if not eval_client.wait_for_service_ready(timeout=60):
    raise Exception("Evaluation service did not become ready within timeout")

status = eval_client.get_status()
loaded_tasks = status.get('service_status', {}).get('loaded_tasks', [])
print(f"Evaluation service ready with tasks: {loaded_tasks}")

# assistant_gen_client.load_model(CURRENT_LATEST_MODEL_PATH, num_gpus=args.num_gpus)

if not assistant_gen_client.wait_for_service_ready(timeout=180):
    raise Exception("Generation service did not become ready within timeout")


def run_tree_building_phase(sample, task_name, task, system_message, user_agent, assistant_gen_client, eval_client, args, temperature):
    """Run the tree building phase and return results"""
    tree_start_time = time.time()
    
    # Initialize timing tracking for tree building
    tree_timings = {"user_generation": 0, "system_functions": 0}
    
    shards = sample["shards"]
    max_depth = len(shards)

    active_jobs = {}
    active_eval_jobs = {}  # Track evaluation jobs separately
    trace = [{"role": "system", "content": system_message, "timestamp": date_str(), "parent": None, "children": [], "id": "S", "depth": 0}]

    def submit_path_job(path_info):
        current_node_id = path_info["node_id"]
        conversation_path = get_conversation_path(trace, current_node_id)

        num_assistant_responses = len([msg for msg in conversation_path if msg["role"] == "assistant"])
        shard_ids = set([msg["shard_id"] for msg in conversation_path if msg["role"] == "user" and "shard_id" in msg])

        if len(shard_ids) == len(shards) or num_assistant_responses >= max_depth:
            return None

        user_start_time = time.time()
        user_response, shard_revealed_id = user_agent.generate_response(conversation_path, sample)
        user_end_time = time.time()
        tree_timings["user_generation"] += user_end_time - user_start_time
        
        user_node_id = f"{current_node_id}.U0"
        user_node = {"role": "user", "content": user_response, "timestamp": date_str(), "parent": current_node_id, "children": [], "id": user_node_id, "depth": user_node_id.count("U")}

        if shard_revealed_id != -1:
            user_node["shard_id"] = shard_revealed_id

        trace.append(user_node)

        for node in trace:
            if node["id"] == current_node_id:
                node["children"].append(user_node_id)
                break

        current_conversation = get_conversation_path(trace, user_node_id)
        conversation_so_far = extract_conversation(current_conversation, to_str=False)

        job_infos = []

        tree_job = assistant_gen_client.build_tree(conversation_so_far, degree=args.turn_degree, depth=args.turn_depth, temperature=temperature, max_tokens=args.max_tokens)
        # job_result = assistant_gen_client.schedule_job(conversation_copy, n_responses=1, temperature=temperature, max_tokens=args.max_tokens)
        job_id = tree_job['job_id']
        job_infos.append({"job_id": job_id, "user_node_id": user_node_id, "path_info": path_info, "submit_time": time.time()}) # , "degree": args.degree, "response_index": i, "total_responses": n_responses
        
        return job_infos

    job_infos = submit_path_job({"node_id": trace[0]["id"], "is_completed": False, "is_correct": False, "score": None, "parent": None})
    for job_info in job_infos:
        active_jobs[job_info["job_id"]] = job_info

    old_progress_line = ""
    collision_rates = []


    while active_jobs or active_eval_jobs:
        assistant_turns = len([node for node in trace if node["role"] == "assistant"])
        user_turns = len([node for node in trace if node["role"] == "user"])
        total_nodes = len(trace)
        progress_line = f"Progress: {assistant_turns} assistant turns | {user_turns} user turns | {len(active_jobs)} gen jobs | {len(active_eval_jobs)} eval jobs | {total_nodes} total nodes"
        if progress_line != old_progress_line:
            print(f"\r{progress_line}", end="", flush=True)
            old_progress_line = progress_line

        completed_jobs = []
        completed_eval_jobs = []
        new_jobs_to_add = []
        
        for job_id, job_info in active_jobs.items():
            result = assistant_gen_client.check_job(job_id)

            if result["status"] == "completed":
                completed_jobs.append(job_id)
                
                # Extract the single response (since we now use n_responses=1)
                assistant_responses_flat = result["tree"]
                print(f"Assistant responses flat has : {len(assistant_responses_flat)} responses")

                user_node_id = job_info["user_node_id"]
                
                # Sort responses by response_index to maintain order
                already_response_texts = set()
                assistant_responses = [] # deduplicated responses
                for rd in assistant_responses_flat:
                    if rd["response_text"] not in already_response_texts:
                        already_response_texts.add(rd["response_text"])
                        assistant_responses.append(rd)

                expected_num_responses = args.turn_degree * args.turn_depth
                collision_rate = 100 * (1 - len(assistant_responses) / expected_num_responses)
                collision_rates.append(collision_rate)

                user_node = [n for n in trace if n["id"] == user_node_id][0]
                system_functions_start_time = time.time()

                # print(f"assistant_responses: {assistant_responses}")
                for i, assistant_response in enumerate(assistant_responses):
                    assistant_node_id = f"{user_node_id}.{assistant_response['subtree_id']}"
                    assistant_node = {"role": "assistant", "content": assistant_response["response_text"], "timestamp": date_str(), "parent": user_node_id, "children": [], "id": assistant_node_id, "depth": assistant_node_id.count("A"), "response_tokens": assistant_response["response_tokens"], "logprobs": assistant_response["logprobs"], "per_token_logprobs": assistant_response["per_token_logprobs"]}

                    assistant_node["any_enforced_guardrail_triggered"] = False
                    for guardrail in guardrails:
                        assistant_node[f"guardrail_{guardrail['name']}_triggered"] = guardrail["guardrail"].is_triggered(assistant_node["content"], assistant_node["response_tokens"])
                        if guardrail["enforced"] and assistant_node[f"guardrail_{guardrail['name']}_triggered"]:
                            assistant_node["any_enforced_guardrail_triggered"] = True

                    trace.append(assistant_node)
                    user_node["children"].append(assistant_node_id)

                    # Schedule evaluation job instead of doing it inline
                    current_conversation = get_conversation_path(trace, assistant_node_id)
                    
                    eval_job_result = eval_client.schedule_evaluation(conversation=current_conversation, task_name=task_name, sample=sample)
                    
                    eval_job_id = eval_job_result['job_id']
                    assistant_node["eval_job_id"] = eval_job_id  # Store for later reference
                    
                    # Track the evaluation job
                    active_eval_jobs[eval_job_id] = {"assistant_node_id": assistant_node_id, "job_info": job_info, "submit_time": time.time()}

                system_functions_end_time = time.time()
                tree_timings["system_functions"] += system_functions_end_time - system_functions_start_time

            elif result["status"] == "error":
                print(f"\nJob {job_id} failed with error: {result.get('error', 'Unknown error')}")
                completed_jobs.append(job_id)

        # Process evaluation jobs
        nodes_needing_new_jobs = []  # Collect nodes that need new generation jobs
        
        for eval_job_id, eval_job_info in active_eval_jobs.items():
            eval_result = eval_client.check_job(eval_job_id)
            
            if eval_result["status"] == "completed":
                completed_eval_jobs.append(eval_job_id)
                
                assistant_node_id = eval_job_info["assistant_node_id"]
                parent_job_info = eval_job_info["job_info"]
                
                # Find the assistant node in the trace
                assistant_node = None
                for node in trace:
                    if node["id"] == assistant_node_id:
                        assistant_node = node
                        break
                
                if assistant_node:
                    # Apply evaluation results to the assistant node
                    eval_data = eval_result["result"]
                    assistant_node["response_strategy"] = eval_data["response_strategy"]
                    if eval_data["response_strategy"] == "answer_attempt":
                        assistant_node["extracted_answer"] = eval_data.get("extracted_answer")
                        assistant_node["evaluation_return"] = eval_data.get("evaluation_return")
                        assistant_node["is_correct"] = eval_data.get("is_correct", False)
                        assistant_node["score"] = eval_data.get("score")
                    
                    # If not correct, mark for new job submission (don't submit immediately)
            
                    if assistant_node["any_enforced_guardrail_triggered"]:
                        assistant_node["score"] = 0.0 # we force it to have a score of 0.0, and prune the tree there
                    # if task is flipflop, we continue either way
                    elif task_name == "flipflop" or not assistant_node.get("is_correct", False): # only look for descendants if node is not correct and no enforced guardrail is triggered
                        nodes_needing_new_jobs.append({
                            "node_id": assistant_node_id,
                            "parent_job_info": parent_job_info
                        })
            
            elif eval_result["status"] == "error":
                print(f"\nEval job {eval_job_id} failed with error: {eval_result.get('error', 'Unknown error')}")
                completed_eval_jobs.append(eval_job_id)

        # Now batch process new job submissions (outside the eval checking loop)
        if nodes_needing_new_jobs:
            # Parallelize submit_path_job calls since user response generation can be concurrent
            with ThreadPoolExecutor(max_workers=min(args.num_eval_workers, len(nodes_needing_new_jobs))) as executor:
                future_to_node = {}
                
                for node_info in nodes_needing_new_jobs:
                    new_path_info = {
                        "node_id": node_info["node_id"], 
                        "is_completed": False, 
                        "is_correct": False, 
                        "score": None, 
                        "parent": node_info["parent_job_info"]["path_info"]["node_id"]
                    }
                    future = executor.submit(submit_path_job, new_path_info)
                    future_to_node[future] = node_info
                
                # Collect results as they complete
                for future in as_completed(future_to_node):
                    new_job_infos = future.result()
                    if new_job_infos:
                        for new_job_info in new_job_infos:
                            new_jobs_to_add.append(new_job_info)

        for job_id in completed_jobs:
            del active_jobs[job_id]
        
        for eval_job_id in completed_eval_jobs:
            del active_eval_jobs[eval_job_id]

        for new_job_info in new_jobs_to_add:
            active_jobs[new_job_info["job_id"]] = new_job_info

        if active_jobs or active_eval_jobs:
            time.sleep(2)

    print()

    tree_end_time = time.time()
    tree_total_timing = tree_end_time - tree_start_time
    
    return {'trace': trace, 'collision_rates': collision_rates, 'tree_timings': tree_timings, 'tree_total_timing': tree_total_timing}

num_iter = 0
while True:
    epoch = (num_iter // len(samples)) + 1
    if epoch > args.num_epochs:
        break

    is_new_epoch = num_iter % len(samples) == 0
    if is_new_epoch:
        random.shuffle(samples)

    sample = samples[num_iter % len(samples)]

    print_colored(f"[New sample] Task: {sample['task']} | Task ID: {sample['task_id']} | Shards: {len(sample['shards'])}", "green")

    num_iter += 1

    T_very_start = time.time()

    task_name = sample["task"]
    task = get_task(task_name)
    system_message = task.generate_system_prompt(sample)

    user_agent = UserAgent(task, args.user_model)
    tree_results = run_tree_building_phase(sample, task_name, task, system_message, user_agent, assistant_gen_client, eval_client, args, temperature)
    
    trace, collision_rates, tree_timings, tree_total_timing = tree_results['trace'], tree_results['collision_rates'], tree_results['tree_timings'], tree_results['tree_total_timing']
    
    print_colored("Tree building completed", "green")

    calculate_backtrack_scores(trace, advantage_estimation="zero_mean_noneg", verbose=False)

    id2node = {node["id"]: node for node in trace}


    log_dict = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "task": task_name, "task_id": sample["task_id"], "iteration": num_iter, "epoch": epoch, "avg_collision_rate": np.mean(collision_rates), "temperature": temperature}

    with open(run_stats_file, "a") as f:
        f.write(json.dumps(log_dict) + "\n")

    log_conversation(log_file, task_name, sample["task_id"], args.dataset_fn, CURRENT_LATEST_MODEL_PATH, args.user_model, trace, additional_info={"iteration": num_iter})

    print_colored(f"Iteration {num_iter} complete", "green")
