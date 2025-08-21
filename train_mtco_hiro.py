from utils import print_colored, extract_conversation, date_str, DoublePrint, subsample_responses, calculate_advantage
from utils_tree import get_conversation_path, calculate_backpropagated_scores
import json, random, time, numpy as np, torch, argparse, os, shutil, tqdm
from utils_guardrails import RepetitionGuardrail, MaxLengthGuardrail
from concurrent.futures import ThreadPoolExecutor, as_completed
from simulator_sharded import ConversationSimulatorSharded
from genserv_client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
from utils_experiments import make_exp_folder
from backprop_worker import BackpropWorker
from utils_log import log_conversation
from system_agent import SystemAgent
from user_agent import UserAgent
from datetime import datetime
from tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
parser.add_argument("--ignore_tasks", type=str, nargs="+", default=[])
parser.add_argument("--base_model", type=str, default="microsoft/phi-4")
parser.add_argument("--user_model", type=str, default="t-gpt-4o-mini")
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--num_eval_workers", type=int, default=8)
parser.add_argument("--workers_per_gpu", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=1000)
parser.add_argument("--degree", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-2)
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument("--normalize_logprobs", action="store_true")
parser.add_argument("--max_n_responses", type=int, default=None)
parser.add_argument("--genserv_port", type=int, default=5000)
parser.add_argument("--evalserv_port", type=int, default=5001)
parser.add_argument("--use_constant_temperature", action="store_true")
parser.add_argument("--advantage_estimation", type=str, default="zero_mean_noneg", choices=["zero_mean", "bounded", "bounded_symmetric", "zero_mean_noneg", "zero_mean_noneg_bounded"])
parser.add_argument("--skip_leaf_update", action="store_true")
parser.add_argument("--save_to_shm", action="store_true")

args = parser.parse_args()

CURRENT_LATEST_MODEL_PATH = args.base_model

# if args.base_model is an experiments folder's best_model, then no need to make a new exp folder
is_new_experiment = True
if args.base_model.startswith("experiments/") and "best_model" in args.base_model:
    folders = args.base_model.split("/")
    exp_folder = folders[0] + "/" + folders[1]
    print(f"Using existing exp folder: {exp_folder}")
    is_new_experiment = False
else:
    exp_folder = make_exp_folder(prefix="mtco")
    print(f"Made new exp folder: {exp_folder}")

# guardrails
guardrails = [
    {"name": "repetition", "enforced": False, "guardrail": RepetitionGuardrail(name="repetition", description="Repetition", n_gram=5, repetition_threshold=5)},
    {"name": "max_length", "enforced": False, "guardrail": MaxLengthGuardrail(name="max_length", description="Max Length", max_length=1000)}
]

temperature = 1.0
target_nlp = -0.25
temp_delta = 0.04

MODEL_PATH = os.path.join(exp_folder, "model")
if args.save_to_shm:
    # set a unique model name in /dev/shm in case we run multiple experiments at the same time
    import uuid
    uuid = str(uuid.uuid4())[:8]
    shm_path = "/dev/shm"
    if os.path.exists(shm_path):
        try:
            # Check if /dev/shm has enough space (at least 100GB free)
            free_space = os.statvfs(shm_path).f_bfree * os.statvfs(shm_path).f_bsize
            if free_space > 100 * 1024 * 1024 * 1024:  # 100GB in bytes
                MODEL_PATH = os.path.join(shm_path, f"model_{uuid}")
        except Exception as e:
            print(f"Error checking /dev/shm space: {e}. Using default model path: {MODEL_PATH}")
    else:
        print(f"Error with /dev/shm. Using default model path: {MODEL_PATH}")

DoublePrint(os.path.join(exp_folder, "run_logs.ans"))
log_file = os.path.join(exp_folder, "tree_logs.jsonl")
run_stats_file = os.path.join(exp_folder, "run_stats.jsonl")
run_params_file = os.path.join(exp_folder, "run_params.json")

if is_new_experiment:
    with open(run_params_file, "w") as f:
        json.dump(vars(args), f)

with open(args.dataset_fn, "r") as f:
    data = json.load(f)

samples = [d for d in data if d["split"] == "train" and d["task"] not in args.ignore_tasks]
validation_samples = [d for d in data if d["split"] == "validation" and d["task"] not in args.ignore_tasks]
num_eval_runs = min(10, 100 // len(validation_samples))
num_eval_workers = 40 # This shouldn't be hardcoded, it should be obtained from the genserv_app, it's really 4 GPUs x 10 concurrency on Philippe's MSR machines
max_validation_score = 0.0

all_validations_samples = [sample for sample in validation_samples for _ in range(num_eval_runs)]
random.shuffle(samples)

print(f"Training samples: {len(samples)}; Validation samples: {len(all_validations_samples)}")

if is_new_experiment:
    all_avg_leaf_node_scores = []
    best_avg_leaf_node_score = -1
    num_iter = 0
else:
    # recover from there
    all_log_lines = []
    with open(run_stats_file, "r") as f:
        for line in f:
            log_dict = json.loads(line)
            all_log_lines.append(log_dict)
    num_iter = len(all_log_lines)
    all_avg_leaf_node_scores = [log_dict["avg_leaf_node_scores"] for log_dict in all_log_lines]
    best_logs = [log_dict for log_dict in all_log_lines if log_dict["is_best_model"]]
    if len(best_logs) > 0:
        best_avg_leaf_node_score = best_logs[-1]["average_leaf_node_scores_over_last_n"]
    else:
        best_avg_leaf_node_score = -1
    print(f"Recovered from iteration {num_iter} (best avg leaf node score: {best_avg_leaf_node_score:.4f})")

assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:{args.genserv_port}")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:{args.evalserv_port}")
backprop_worker = BackpropWorker() # unfortunately needed, because of uncontrollable memory leaks in the backprop if kept in main process

if eval_client.get_status()["service_status"] == False:
    print("Loading evaluation service...")
    eval_load_result = eval_client.load_tasks(dataset_file=args.dataset_fn, num_workers=args.num_eval_workers)
    if not eval_client.wait_for_service_ready(timeout=60):
        raise Exception("Evaluation service did not become ready within timeout")
    print(f"Evaluation service ready with tasks: {eval_load_result.get('loaded_tasks', [])}")


def run_validation_sample(sample):
    simulator = ConversationSimulatorSharded(sample, assistant_model="gs-"+CURRENT_LATEST_MODEL_PATH, user_model=args.user_model, assistant_temperature=1.0, user_temperature=1.0, dataset_fn=args.dataset_fn, log_folder=exp_folder)
    sim_out = simulator.run(verbose=False, save_log=False)
    score = sim_out[1]
    score = 0.0 if score is None else score
    return score, sample["task"]

def run_validation_phase(all_validations_samples, num_eval_workers, max_validation_score, num_iter, exp_folder, CURRENT_LATEST_MODEL_PATH):
    """Run the validation phase and return results"""
    validation_start_time = time.time()
    
    validation_scores = []
    val_tasks = list(set([sample["task"] for sample in all_validations_samples]))
    validation_scores_per_task = {task: [] for task in val_tasks}

    with ThreadPoolExecutor(max_workers=num_eval_workers) as executor:
        futures = [executor.submit(run_validation_sample, sample) for sample in all_validations_samples]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running validation"):
            score, task = future.result()
            validation_scores.append(score)
            validation_scores_per_task[task].append(score)
    
    avg_validation_score = np.mean(validation_scores)
    validation_avg_scores_per_task = {task: np.mean(scores) for task, scores in validation_scores_per_task.items()}

    print_colored(f"[log] Validation score: {avg_validation_score:.4f} {validation_avg_scores_per_task}", "blue")
    
    is_best_model = False
    # don't save the base model
    if avg_validation_score > max_validation_score and CURRENT_LATEST_MODEL_PATH != args.base_model: 
        max_validation_score = avg_validation_score
        print_colored(f"[log] New max validation score: {max_validation_score:.4f}", "blue")
        # save the model
        best_model_folder = os.path.join(exp_folder, "best_model")
        # Remove existing best_model directory if it exists to avoid FileExistsError
        if os.path.exists(best_model_folder):
            shutil.rmtree(best_model_folder)
        shutil.copytree(CURRENT_LATEST_MODEL_PATH, best_model_folder)
        print_colored(f"[log] Saved best model to {best_model_folder}", "blue")
        is_best_model = True
    
    validation_end_time = time.time()
    validation_timing = validation_end_time - validation_start_time
    
    return {
        'avg_validation_score': avg_validation_score,
        'validation_avg_scores_per_task': validation_avg_scores_per_task,
        'is_best_model': is_best_model,
        'max_validation_score': max_validation_score,
        'validation_timing': validation_timing
    }

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

        # Check if all hints are revealed for this path
        shard_ids = set([msg["shard_id"] for msg in conversation_path if msg["role"] == "user" and "shard_id" in msg])

        # Skip if all shards revealed or max depth reached
        if len(shard_ids) == len(shards) or num_assistant_responses >= max_depth:
            return None

        # 1. Generate user response (this is still synchronous as it's fast, and we don't branch on it)
        user_start_time = time.time()
        user_response, shard_revealed_id = user_agent.generate_response(conversation_path, sample)
        user_end_time = time.time()
        tree_timings["user_generation"] += user_end_time - user_start_time
        
        # Add user node
        user_node_id = f"{current_node_id}.U0"
        user_node = {"role": "user", "content": user_response, "timestamp": date_str(), "parent": current_node_id, "children": [], "id": user_node_id, "depth": user_node_id.count("U")}

        # Add shard information directly to user node if revealed
        if shard_revealed_id != -1:
            user_node["shard_id"] = shard_revealed_id

        trace.append(user_node)

        # Update parent's children
        for node in trace:
            if node["id"] == current_node_id:
                node["children"].append(user_node_id)
                break

        # 2. Submit assistant generation job to generation service
        current_conversation = get_conversation_path(trace, user_node_id)
        conversation_so_far = extract_conversation(current_conversation, to_str=False)
        degree = args.degree
        n_responses = max(degree, args.max_n_responses) if args.max_n_responses is not None else degree
        job_result = assistant_gen_client.schedule_job(conversation_so_far, n_responses=n_responses, temperature=temperature, max_tokens=args.max_tokens)        
        job_id = job_result['job_id']

        return {"job_id": job_id, "user_node_id": user_node_id, "path_info": path_info, "submit_time": time.time(), "degree": degree}

    job_info = submit_path_job({"node_id": trace[0]["id"], "is_completed": False, "is_correct": False, "score": None, "parent": None})
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
        new_jobs_to_add = []  # Collect new jobs to add after iteration
        for job_id, job_info in active_jobs.items():
            result = assistant_gen_client.check_job(job_id)

            if result["status"] == "completed":
                completed_jobs.append(job_id)

                assistant_responses = result["responses"]
                expected_num_responses = job_info["degree"]

                collision_rate = 100 * (1 - len(assistant_responses) / expected_num_responses)
                collision_rates.append(collision_rate)

                user_node_id = job_info["user_node_id"]

                # subsample if max_n_responses is set and larger than the degree
                if args.max_n_responses is not None and job_info["degree"] < args.max_n_responses:
                    assistant_responses = subsample_responses(assistant_responses, n_responses=job_info["degree"])

                user_node = None
                for node in trace:
                    if node["id"] == user_node_id:
                        user_node = node
                        break

                system_functions_start_time = time.time()
                for i, assistant_response in enumerate(assistant_responses):
                    assistant_node_id = f"{user_node_id}.A{i}"
                    assistant_node = {"role": "assistant", "content": assistant_response["response_text"], "timestamp": date_str(), "parent": user_node_id, "children": [], "id": assistant_node_id, "depth": assistant_node_id.count("A"), "response_tokens": assistant_response["response_tokens"], "logprobs": assistant_response["logprobs"]}

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
                    new_job_info = future.result()
                    if new_job_info:
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
    
    return {
        'trace': trace,
        'collision_rates': collision_rates,
        'tree_timings': tree_timings,
        'tree_total_timing': tree_total_timing
    }

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
    timings = {"genserv_load": 0, "genserv_unload": 0, "assistant_generation": 0, "user_generation": 0, "system_functions": 0, "validation": 0, "best_model_copy": 0, "total_time": 0}

    T_very_start = time.time()

    task_name = sample["task"]
    task = get_task(task_name)
    system_message = task.generate_system_prompt(sample)

    user_agent = UserAgent(task, args.user_model)
    system_agent = SystemAgent(task_name, sample)

    # Part 0: Load model
    load_result = assistant_gen_client.load_model(CURRENT_LATEST_MODEL_PATH, num_gpus=args.num_gpus, workers_per_gpu=args.workers_per_gpu)

    if not assistant_gen_client.wait_for_service_ready(timeout=120):
        raise Exception("Generation service did not become ready within timeout")

    print(f"Loaded models on all GPUs.")
    T_genserv_loaded = time.time()
    timings["genserv_load"] += T_genserv_loaded - T_very_start

    # Part 1 & 2: Run Validation and Tree Building in Parallel
    print_colored("Starting validation and tree building in parallel...", "cyan")
    T_parallel_start = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        validation_future = executor.submit(run_validation_phase, all_validations_samples, num_eval_workers, max_validation_score, num_iter, exp_folder, CURRENT_LATEST_MODEL_PATH)
        
        tree_future = executor.submit(run_tree_building_phase, sample, task_name, task, system_message, user_agent, assistant_gen_client, eval_client, args, temperature)
        
        # Wait for both to complete and get results
        validation_results = validation_future.result()
        tree_results = tree_future.result()
    
    T_parallel_end = time.time()
    
    # Extract validation results
    avg_validation_score = validation_results['avg_validation_score']
    validation_avg_scores_per_task = validation_results['validation_avg_scores_per_task']
    is_best_model = validation_results['is_best_model']
    max_validation_score = validation_results['max_validation_score']
    validation_timing = validation_results['validation_timing']

    print_colored(f"Validation completed: {avg_validation_score:.4f} {validation_avg_scores_per_task}", "blue")

    # Extract tree building results
    trace = tree_results['trace']
    collision_rates = tree_results['collision_rates']
    tree_timings = tree_results['tree_timings']
    tree_total_timing = tree_results['tree_total_timing']
    
    print_colored("Tree building completed", "green")
    
    # Update timings from parallel execution
    timings["user_generation"] = tree_timings["user_generation"]
    timings["system_functions"] = tree_timings["system_functions"]
    timings["assistant_generation"] = tree_total_timing - tree_timings["user_generation"] - tree_timings["system_functions"]
    timings["validation"] = validation_timing
    
    actual_parallel_time = T_parallel_end - T_parallel_start
    max_individual_time = max(validation_timing, tree_total_timing)
    time_saved = max_individual_time - actual_parallel_time
    print_colored(f"Parallel execution completed in {actual_parallel_time:.2f}s (validation: {validation_timing:.2f}s, tree: {tree_total_timing:.2f}s, saved: {time_saved:.2f}s)", "cyan")

    # Part 3: Unload generation service and run backprop

    unload_result = assistant_gen_client.unload_model()
    print(f"Model unload result: {unload_result}")
    T_genserv_unloaded = time.time()
    timings["genserv_unload"] += T_genserv_unloaded - T_parallel_end

    calculate_backpropagated_scores(trace, verbose=False)
    
    backprop_args = {"learning_rate": args.learning_rate, "normalize_logprobs": args.normalize_logprobs, "advantage_estimation": args.advantage_estimation, "skip_leaf_update": args.skip_leaf_update}
    backprop_results = backprop_worker.run_backprop(model_path=CURRENT_LATEST_MODEL_PATH, save_path=MODEL_PATH, trace=trace, args_dict=backprop_args, timeout=600)
    timings.update(backprop_results["timings"])
    T_model_unload_start = time.time()
    backprop_worker.cleanup()
    T_model_unload_end = time.time()
    timings["model_unload"] = T_model_unload_end - T_model_unload_start

    if backprop_results:
        any_updates = backprop_results["any_updates"]
        corrs_A_LP, corrs_A_RL, corrs_A_NLP = backprop_results["corrs_A_LP"], backprop_results["corrs_A_RL"], backprop_results["corrs_A_NLP"]
        num_skips_unstable = backprop_results["num_skips_unstable"]
        losses1 = backprop_results["losses1"]
        all_logprobs = backprop_results["all_logprobs"]
        all_normalized_logprobs = backprop_results["all_normalized_logprobs"]
    else:
        print_colored("[log] Backprop worker failed, no updates applied; skipping this iteration", "red")
        # skip the rest of this iteration
        continue

    avg_leaf_node_scores = np.mean([node.get("score", 0.0) for node in trace if node.get("is_leaf", False)])
    print(f"Average leaf node score: {avg_leaf_node_scores}")

    if np.isnan(avg_leaf_node_scores):
        avg_leaf_node_scores = 0.0 # somehow it happens (rarely), needs investigation

    all_avg_leaf_node_scores.append(avg_leaf_node_scores)

    # decide on temperature movement
    if not args.use_constant_temperature and len(all_normalized_logprobs) > 0:
        avg_nlp = np.mean(all_normalized_logprobs)
        if avg_nlp > target_nlp: # nlp is too high (too close to zero), we're over-sharpening, need to increase temperature
            temperature += temp_delta
        elif temperature > 1.0: # don't drop below 1.0
            temperature -= temp_delta

        print_colored(f"[Temperature Dynamics] NLP: {avg_nlp:.2f}, Temperature: {temperature:.2f}", "green")

    depth_rls = {}
    total_assistant_responses = len([node for node in trace if node["role"] == "assistant"])
    guardrail_trigger_counts = {guardrail["name"]: 0 for guardrail in guardrails}

    for node in trace:
        if node["role"] == "assistant":
            depth = node["depth"]
            if depth not in depth_rls:
                depth_rls[depth] = []
            depth_rls[depth].append(len(node["response_tokens"]))

            for guardrail in guardrails:
                if node[f"guardrail_{guardrail['name']}_triggered"]:
                    guardrail_trigger_counts[guardrail["name"]] += 1

    perc_guardrail_triggered = {guardrail["name"]: (guardrail_trigger_counts[guardrail["name"]] / total_assistant_responses * 100) if total_assistant_responses > 0 else 0.0 for guardrail in guardrails}
    total_backprop_attempts = len(corrs_A_LP) + num_skips_unstable
    perc_skipped_backprop = (num_skips_unstable / total_backprop_attempts * 100) if total_backprop_attempts > 0 else 0.0

    avg_corr_A_LP = np.mean(corrs_A_LP) if corrs_A_LP else 0.0
    avg_corr_A_RL = np.mean(corrs_A_RL) if corrs_A_RL else 0.0
    avg_corr_A_NLP = np.mean(corrs_A_NLP) if corrs_A_NLP else 0.0
    
    avg_depth_rls = {depth: np.mean(rls) for depth, rls in depth_rls.items()}
    avg_response_length = np.mean([len(node["response_tokens"]) for node in trace if node["role"] == "assistant"])
    print(f"Average correlation between advantages and logprobs: {avg_corr_A_LP:.3f}")
    print(f"Average correlation between advantages and response lengths: {avg_corr_A_RL:.3f}")
    print(f"Average response length at each depth: {avg_depth_rls}")

    print(f"Skipped backprop: {num_skips_unstable}/{total_backprop_attempts} ({perc_skipped_backprop:.1f}%)")

    # , "average_leaf_node_scores_over_last_n": average_leaf_node_scores_over_last_n
    log_dict = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "task": task_name, "task_id": sample["task_id"], "iteration": num_iter, "epoch": epoch, "avg_leaf_node_scores": avg_leaf_node_scores, "is_best_model": is_best_model, "avg_response_length": avg_response_length, "total_nodes": len(trace), "num_backprop_updates": len(corrs_A_LP), "num_skips_unstable": num_skips_unstable, "num_skipped_backprop": num_skips_unstable, "perc_skipped_backprop": perc_skipped_backprop, "avg_collision_rate": np.mean(collision_rates), "temperature": temperature, "validation_score": avg_validation_score}

    log_dict.update({f"validation_score_{task}": score for task, score in validation_avg_scores_per_task.items()})

    for guardrail in guardrails:
        log_dict[f"guardrail_{guardrail['name']}_triggered"] = guardrail_trigger_counts[guardrail["name"]]
        log_dict[f"perc_guardrail_{guardrail['name']}_triggered"] = perc_guardrail_triggered[guardrail["name"]]
        print(f"[Guardrail] {guardrail['name']} triggered: {guardrail_trigger_counts[guardrail['name']]}/{total_assistant_responses} ({perc_guardrail_triggered[guardrail['name']]:.1f}%)")

    log_dict.update({"logprobs": all_logprobs, "normalized_logprobs": all_normalized_logprobs}) # tracking the normalized logprobs will be useful to understand over-sharpening if it occurs...

    log_dict.update({"corrs_A_LP": corrs_A_LP, "corrs_A_RL": corrs_A_RL, "corrs_A_NLP": corrs_A_NLP})
    log_dict.update({"losses1": losses1}) # yay, keep track of that as well

    log_dict.update({"avg_corr_A_LP": avg_corr_A_LP, "avg_corr_A_RL": avg_corr_A_RL, "avg_corr_A_NLP": avg_corr_A_NLP})

    log_dict.update({f"avg_response_length_depth_{depth}": avg_rl for depth, avg_rl in avg_depth_rls.items()})
    for timing_key, timing_value in timings.items():
        log_dict[f"timing/{timing_key}"] = timing_value

    # Store stats in run_stats.jsonl for the Streamlit dashboard
    with open(run_stats_file, "a") as f:
        f.write(json.dumps(log_dict) + "\n")

    T_model_saved = time.time()
    if any_updates:
        print("Model was saved by backprop worker")
        CURRENT_LATEST_MODEL_PATH = MODEL_PATH
        timings["model_save"] += 0  # Time is tracked in backprop worker

    # No model cleanup needed in main process since backprop worker handles it
    T_model_unloaded = time.time()
    timings["model_unload"] += 0  # No cleanup needed in main process

    T_best_model_copied = time.time()
    timings["best_model_copy"] += T_best_model_copied - T_model_unloaded

    T_very_end = time.time()
    timings["total_time"] += T_very_end - T_very_start

    log_conversation(log_file, task_name, sample["task_id"], args.dataset_fn, CURRENT_LATEST_MODEL_PATH, args.user_model, trace, additional_info={"iteration": num_iter, "avg_leaf_node_scores": avg_leaf_node_scores, "corrs_A_LP": corrs_A_LP, "corrs_A_RL": corrs_A_RL, "avg_response_length": avg_response_length, "avg_depth_rls": avg_depth_rls, "timings": timings, "avg_corr_A_LP": avg_corr_A_LP, "avg_corr_A_RL": avg_corr_A_RL, "avg_corr_A_NLP": avg_corr_A_NLP})

    print_colored("\n=== FINAL TIMING SUMMARY ===", "yellow")
    unaccounted_time = T_very_end - T_very_start - (sum(timings.values()) - timings["total_time"])
    for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print_colored(f"   {k.ljust(20)}: {v:.2f}s", "yellow")
    print_colored(f"   {'Unaccounted time'.ljust(20)}: {unaccounted_time:.2f}s", "yellow")
    print_colored("==========================\n", "yellow")

    print_colored(f"Iteration {num_iter} complete", "green")
