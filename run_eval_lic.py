import argparse, random, multiprocessing, json, tqdm
from simulator_sharded import ConversationSimulatorSharded
from simulator_full import ConversationSimulatorFull
from concurrent.futures import ThreadPoolExecutor
from utils_log_lic import get_run_counts
from collections import Counter

def run_simulation(todo):
    dataset_fn = todo["dataset_fn"]
    try:
        assistant_temp = todo.get("assistant_temperature", 1.0)
        user_temp = todo.get("user_temperature", 1.0)
        evalserv_port = todo.get("evalserv_port", 5001)
        if todo["conv_type"].startswith("full"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder, evalserv_port=evalserv_port)
        elif todo["conv_type"].startswith("concat"):
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], run_concat=True, temperature=assistant_temp, dataset_fn=dataset_fn, log_folder=args.log_folder, evalserv_port=evalserv_port)
        elif todo["conv_type"].startswith("sharded"):
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], user_model=todo["user_model"], assistant_temperature=assistant_temp, user_temperature=user_temp, dataset_fn=dataset_fn, log_folder=args.log_folder, evalserv_port=evalserv_port)
        conversation_simulator.run(verbose=args.verbose)

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        tqdm.tqdm.write(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_file", type=str, default="data/sharded_instructions_600.json", help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=10, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=10, help="Number of concat runs per model")
    parser.add_argument("--N_sharded_runs", type=int, default=10, help="Number of sharded runs per model")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"], # `, "gpt-4o"`
                        help="List of models to run experiments with")
    parser.add_argument("--tasks", nargs="+", default=["database", "actions", "math", "code"], help="Tasks to run experiments with") # "code", "database", "actions", "math", "data2text", "summary", "translation"
    parser.add_argument("--system_model", type=str, default="t-gpt-4o-mini", help="System model to use")
    parser.add_argument("--user_model", type=str, default="t-gpt-4o-mini", help="User model to use")
    parser.add_argument("--N_workers", type=int, default=8, help="Number of workers to run experiments with")
    parser.add_argument("--log_folder", type=str, default="logs", help="Log folder to use")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    parser.add_argument("--assistant_temperature", type=float, default=1.0, help="Temperature to use for assistant models")
    parser.add_argument("--user_temperature", type=float, default=1.0, help="Temperature to use for user models")
    parser.add_argument("--num_eval_workers", type=int, default=8, help="Number of workers to use for evaluation service")
    parser.add_argument("--evalserv_port", type=int, default=5001, help="Port for evaluation service")

    args = parser.parse_args()

    if any(model.startswith("gs-") for model in args.models):
        if  len(args.models) > 1:
            raise ValueError("Generation Service only supports one model at a time")
        # otherwise, let's load it
        from evalserv_client import EvaluationServiceClient
        client = EvaluationServiceClient(base_url=f"http://localhost:{args.evalserv_port}")
        eval_load_result = client.load_tasks(dataset_file=args.dataset_file, num_workers=args.num_eval_workers)
        if not client.wait_for_service_ready(timeout=60):
            raise Exception("Evaluation service did not become ready within timeout")
        print(f"Evaluation service ready with tasks: {eval_load_result.get('loaded_tasks', [])}")

        from genserv_client import GenerationServiceClient
        client = GenerationServiceClient()
        client.load_model(args.models[0].replace("gs-", ""))
        client.wait_for_service_ready()
        
        print("Generation Service is ready")

    dataset_fn = args.dataset_file

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    samples = [d for d in samples if d["split"] == "test"] # no need to evaluate on the training tasks
    samples = [d for d in samples if d["task"] in args.tasks]

    print(f"Loaded {len(samples)} samples")
    random.shuffle(samples)
    all_todos = []

    sharded_extra = f"-at{args.assistant_temperature}-ut{args.user_temperature}" if args.assistant_temperature != 1.0 or args.user_temperature != 1.0 else ""
    st_extra = f"-t{args.assistant_temperature}" if args.assistant_temperature != 1.0 else ""
    sharded_ct, full_ct, concat_ct = f"sharded{sharded_extra}", f"full{st_extra}", f"concat{st_extra}"

    all_tasks = list(set([sample["task"] for sample in samples]))
    for assistant_model in args.models:
        sharded_run_counts, full_run_counts, concat_run_counts = Counter(), Counter(), Counter()
        for task in all_tasks:
            sharded_run_counts.update(get_run_counts(sharded_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            full_run_counts.update(get_run_counts(full_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
            concat_run_counts.update(get_run_counts(concat_ct, task, assistant_model, dataset_fn, log_folder=args.log_folder))
        print(f"Sharded run counts: {sharded_run_counts}")
        print(f"Full run counts: {full_run_counts}")
        print(f"Concat run counts: {concat_run_counts}")

        for sample in samples:
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": full_ct, "dataset_fn": dataset_fn}] * (args.N_full_runs - full_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": concat_ct, "dataset_fn": dataset_fn}] * (args.N_concat_runs - concat_run_counts[sample["task_id"]])
            all_todos += [{"sample": sample, "assistant_model": assistant_model, "conv_type": sharded_ct, "user_model": args.user_model, "dataset_fn": dataset_fn}] * (args.N_sharded_runs - sharded_run_counts[sample["task_id"]])

    if args.assistant_temperature != 1.0 or args.user_temperature != 1.0:
        # update todos with temperature
        for todo in all_todos:
            todo["assistant_temperature"] = args.assistant_temperature
            todo["user_temperature"] = args.user_temperature
    
    # Add evalserv_port to all todos
    for todo in all_todos:
        todo["evalserv_port"] = args.evalserv_port

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print(Counter([todo["assistant_model"] for todo in all_todos]))
    print(Counter([todo["conv_type"] for todo in all_todos]))

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(tqdm.tqdm(executor.map(run_simulation, all_todos), total=len(all_todos)))
