from utils import print_colored, extract_conversation, date_str
from evalserv_client import EvaluationServiceClient
from utils_log_lic import log_conversation
from system_agent import SystemAgent
from user_agent import UserAgent
from tasks import get_task
from llms import generate
import json, random


class ConversationSimulatorSharded:
    def __init__(self, sample, assistant_model="gpt-4o-mini", user_model="gpt-4o-mini", assistant_temperature=1.0, user_temperature=1.0, dataset_fn=None, log_folder="logs", evalserv_port=5001):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.user_model = user_model
        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(self.task_name, self.sample)
        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()

        self.run_with_custom_temperature = assistant_temperature != 1.0 or user_temperature != 1.0
        self.assistant_temperature = assistant_temperature
        self.user_temperature = user_temperature

        # Initialize evaluation service client
        self.eval_client = EvaluationServiceClient(base_url=f"http://localhost:{evalserv_port}")
        
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):

        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set([msg["shard_id"] for msg in self.trace if msg["role"] == "user"])
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})", "blue")
                break # no need to keep going, nothing else to reveal

            # 1. get a user response
            user_response, shard_revealed_id = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "shard_id": shard_revealed_id})
            if verbose:
                print_colored(f"[user] {user_response}", "green")
                if shard_revealed_id != -1:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            # 2. get the assistant's response
            assistant_response_obj = generate(extract_conversation(self.trace, to_str=False), model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            assistant_log = {"role": "assistant", "content": assistant_response, "timestamp": date_str()}
            self.trace.append(assistant_log)
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            # 3. Use evaluation service for evaluation instead of direct calls
            sample_copy = self.sample.copy()
            # and then remove verifications
            if "verifications" in sample_copy:
                del sample_copy["verifications"]
            eval_job_result = self.eval_client.schedule_evaluation(conversation=self.trace, task_name=self.task_name, sample=sample_copy)
            eval_result = self.eval_client.wait_for_job_completion(eval_job_result['job_id'], timeout=120)
            
            assistant_log["response_strategy"] = 'none'
            assistant_log["is_correct"] = False
            assistant_log["score"] = 0.0
            
            if eval_result.get('status') == 'completed':
                eval_data = eval_result["result"]
                response_strategy = eval_data["response_strategy"]
                assistant_log["response_strategy"] = response_strategy
                
                if verbose:
                    print_colored(f"[log] response strategy: {response_strategy}", "blue")

                if response_strategy == "answer_attempt":
                    extracted_answer = eval_data.get("extracted_answer")
                    evaluation_return = eval_data.get("evaluation_return")
                    is_correct = eval_data.get("is_correct", None)
                    score = eval_data.get("score", None)
                    
                    assistant_log["extracted_answer"] = extracted_answer
                    assistant_log["evaluation_return"] = evaluation_return
                    assistant_log["is_correct"] = is_correct
                    assistant_log["score"] = score

                    if verbose:
                        print_colored(f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})", "blue")

                    if is_correct:
                        is_completed = True
                        if verbose:
                            print_colored(f"[log] conversation completed: {is_correct}; score: {score}", "blue")

        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
            log_conversation(conv_type=conv_type, task_name=self.task.get_task_name(), task_id=self.sample["task_id"], dataset_fn=self.dataset_fn, assistant_model=self.assistant_model, system_model="NA", user_model=self.user_model, trace=self.trace, is_correct=is_correct, score=score, log_folder=self.log_folder)
        return is_correct, score


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="code")
    parser.add_argument("--assistant_model", type=str, default="t-gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="t-gpt-4o-mini")
    parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
    # parser.add_argument("--dataset_fn", type=str, default="sample_synthesis/data/sharded_train_synthetic_0.1.json")
    parser.add_argument("--evalserv_port", type=int, default=5001)
    args = parser.parse_args()

    eval_client = EvaluationServiceClient(base_url=f"http://localhost:{args.evalserv_port}")
    # eval_client.load_tasks(dataset_file=args.dataset_fn, num_workers=40)
    eval_client.wait_for_service_ready()

    dataset_fn = args.dataset_fn
    with open(dataset_fn, "r") as f:
        data = json.load(f)

    data = [d for d in data if d["task"] == args.task]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorSharded(sample=sample, assistant_model=args.assistant_model, user_model=args.user_model, dataset_fn=dataset_fn, evalserv_port=args.evalserv_port)
    conversation_simulator.run(verbose=True, save_log=False)
