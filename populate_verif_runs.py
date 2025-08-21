import argparse, random, multiprocessing, os, numpy as np, json, time
from simulator_full import ConversationSimulatorFull
from simulator_sharded import ConversationSimulatorSharded
from concurrent.futures import ThreadPoolExecutor
from tasks import get_task

def single_run(todo):
    try:
        if todo["conv_type"] == "full":
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"])
            return conversation_simulator.run(verbose=args.verbose, save_log=False)
        elif todo["conv_type"] == "concat":
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], run_concat=True)
            return conversation_simulator.run(verbose=args.verbose, save_log=False)
        elif todo["conv_type"] == "shuffle-concat":
            conversation_simulator = ConversationSimulatorFull(todo["sample"], assistant_model=todo["assistant_model"], run_shuffle_concat=True)
            return conversation_simulator.run(verbose=args.verbose, save_log=False)
        elif todo["conv_type"] == "sharded":
            conversation_simulator = ConversationSimulatorSharded(todo["sample"], assistant_model=todo["assistant_model"], user_model="t-gpt-4o-mini")
            return conversation_simulator.run(verbose=args.verbose, save_log=False)
        else:
            raise ValueError(f"Invalid conv_type: {todo['conv_type']}")
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m")
    return False, 0.0

def get_sample(sample_id, samples):
    for s in samples:
        if s["task_id"] == sample_id:
            return s
    raise ValueError(f"Sample {sample_id} not found")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_fn", type=str, default="sample_synthesis/data/code_synthetic_problems_0.1_verified.json", help="Dataset file to use")

    parser.add_argument("--N_full_runs", type=int, default=8, help="Number of full runs per model")
    parser.add_argument("--N_concat_runs", type=int, default=8, help="Number of concat runs per model")
    parser.add_argument("--N_shuffle_concat_runs", type=int, default=8, help="Number of shuffle-concat runs per model")
    parser.add_argument("--N_sharded_runs", type=int, default=8, help="Number of sharded runs per model")
    parser.add_argument("--verif_models", type=str, nargs="+", default=["t-gpt-4.1"], help="Assistant models to use")
    parser.add_argument("--N_workers", type=int, default=8, help="Number of workers to run experiments with")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--force_rerun", action="store_true")

    args = parser.parse_args()

    conv_types = ["full", "concat", "shuffle-concat", "sharded"]
    N_runs = {
        "full": args.N_full_runs,
        "concat": args.N_concat_runs,
        "shuffle-concat": args.N_shuffle_concat_runs,
        "sharded": args.N_sharded_runs
    }

    with open(args.dataset_fn, "r") as f:
        samples = json.load(f)

    random.shuffle(samples)

    todo_samples = {}
    for s in samples:
        todo_tasks = {}
        if "verifications" not in s or args.force_rerun:
            for model in args.verif_models:
                for conv_type in conv_types:
                    todo_tasks[(model, conv_type)] = N_runs[conv_type]
        else:
            for conv_type in conv_types:
                if f"{conv_type}-all" in s["verifications"]:
                    for model in args.verif_models:
                        existing_runs = len(s["verifications"][f"{conv_type}-all"][model])
                        if existing_runs < N_runs[conv_type]:
                            todo_tasks[(model, conv_type)] = N_runs[conv_type] - existing_runs
                else:
                    for model in args.verif_models:
                        todo_tasks[(model, conv_type)] = N_runs[conv_type]
        todo_samples[s["task_id"]] = todo_tasks

    total_runs = {conv_type: sum(todo_tasks[(model, conv_type)] for s, todo_tasks in todo_samples.items() for model in args.verif_models if (model, conv_type) in todo_tasks) for conv_type in conv_types}
    print(f"Samples to process: {len(todo_samples)}")
    print(f"Remaining runs by conversation type:")
    for conv_type, runs in total_runs.items():
        print(f"  {conv_type}: {runs} runs")

    # Initialize time tracking for periodic saves before processing samples
    SAVE_INTERVAL = 600  # save every 5 minutes
    last_save_time = time.time()

    for (sample_id, todo_tasks) in todo_samples.items():
        sample = get_sample(sample_id, samples)
        all_todos = []
        for (model, conv_type) in todo_tasks:
            all_todos += [{"sample": sample, "assistant_model": model, "conv_type": conv_type}] * todo_tasks[(model, conv_type)]

        with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
            results = list(executor.map(single_run, all_todos))

        # aggregate results by conv_type
        if args.force_rerun:
            results_by_conv_type = {}
        else:
            results_by_conv_type = sample.get("verifications", {})
            for conv_type in conv_types:
                results_by_conv_type[conv_type] = results_by_conv_type.get(f"{conv_type}-all", {model: [] for model in args.verif_models})

        for i, (is_correct, score) in enumerate(results):
            if score is None and is_correct is not None:
                score = 1 if is_correct else 0
            conv_type = all_todos[i]["conv_type"]
            assistant_model = all_todos[i]["assistant_model"]
            if conv_type not in results_by_conv_type:
                results_by_conv_type[conv_type] = {model: [] for model in args.verif_models}
            results_by_conv_type[conv_type][assistant_model].append(score)

        # save the results
        sample["verifications"] = {}
        for conv_type in conv_types:
            all_scores = [np.mean(results_by_conv_type[conv_type][model]).item() for model in args.verif_models]
            sample["verifications"][f"{conv_type}-avg"] = np.mean(all_scores).item()
            sample["verifications"][f"{conv_type}-all"] = results_by_conv_type[conv_type]
        avg_dict = {conv_type: sample["verifications"][f"{conv_type}-avg"] for conv_type in conv_types}
        print(f"[{sample['task_id']}] {avg_dict}")

        # Periodic save check
        current_time = time.time()
        if current_time - last_save_time > SAVE_INTERVAL:
            print(f"Performing periodic save at {current_time}")
            with open(args.dataset_fn, "w") as f:
                json.dump(samples, f, indent=4)
            last_save_time = current_time

    # Final save after all processing
    print("Performing final save")
    with open(args.dataset_fn, "w") as f:
        json.dump(samples, f, indent=4)
