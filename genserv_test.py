from concurrent.futures import ThreadPoolExecutor, as_completed
from simulator_sharded import ConversationSimulatorSharded
import json, time, multiprocessing, os, tqdm, numpy as np
from genserv_client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient

model_name = "microsoft/phi-4"
user_model_name = "t-gpt-4o-mini"

def run_validation(sample):
    simulator = ConversationSimulatorSharded(sample, assistant_model="gs-"+model_name, user_model=user_model_name, assistant_temperature=1.0, user_temperature=1.0, dataset_fn="data/sharded_instructions_600.json", log_folder="logs")
    sim_out = simulator.run(verbose=False, save_log=False)
    # print(sim_out)
    score = sim_out[1]
    score = 0.0 if score is None else score
    return score, sample["task"]

client = GenerationServiceClient(base_url="http://localhost:5000")
eval_client = EvaluationServiceClient(base_url="http://localhost:5001")
backend = client.what_backend()
print(backend)

T = time.time()
client.load_model(model_name)
client.wait_for_service_ready()
eval_client.load_tasks(dataset_file="data/sharded_instructions_600.json", num_workers=40)
eval_client.wait_for_service_ready()
print(f"Time taken to load model: {time.time() - T:.2f} seconds")

with open("data/sharded_instructions_600.json", "r") as f:
    data = json.load(f)

validation_samples = [d for d in data if d["split"] == "validation"]

num_eval_runs = 4
num_eval_workers = 40
max_validation_score = 0.0

all_validations_samples = [sample for sample in validation_samples for _ in range(num_eval_runs)]
val_tasks = list(set([sample["task"] for sample in all_validations_samples]))
validation_scores_per_task = {task: [] for task in val_tasks}

T = time.time()
with ThreadPoolExecutor(max_workers=num_eval_workers) as executor:
    futures = [executor.submit(run_validation, sample) for sample in all_validations_samples]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running validation"):
        score, task = future.result()
        validation_scores_per_task[task].append(score)
print(f"Time taken to run validation: {time.time() - T:.2f} seconds")

avg_validation_score = np.mean(list(validation_scores_per_task.values()))
print(f"Average validation score: {avg_validation_score:.4f}")
validation_avg_scores_per_task = {task: np.mean(scores) for task, scores in validation_scores_per_task.items()}
print(validation_avg_scores_per_task)
