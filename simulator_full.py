import random

from utils_log_lic import log_conversation
from llms import generate
from tasks import get_task
from utils import date_str
from evalserv_client import EvaluationServiceClient


class ConversationSimulatorFull:
    def __init__(self, sample, assistant_model, run_concat=False, run_shuffle_concat=False, temperature=1.0, dataset_fn=None, log_folder=None, evalserv_port=5001):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.run_concat = run_concat
        self.run_shuffle_concat = run_shuffle_concat
        self.log_folder = log_folder
        self.run_custom_temperature = temperature != 1.0
        self.temperature = temperature

        self.system_message = self.task.generate_system_prompt(self.sample)
        
        # Initialize evaluation service client
        self.eval_client = EvaluationServiceClient(base_url=f"http://localhost:{evalserv_port}")

    def run(self, verbose=False, save_log=True):
        if self.run_shuffle_concat and self.run_concat:
            raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

        if self.run_shuffle_concat:
            conv_type = "shuffle-concat"

            random.shuffle(self.sample["shards"])

            input_prompt = self.task.populate_concat_prompt(self.sample)
        elif self.run_concat:
            conv_type = "concat"
            input_prompt = self.task.populate_concat_prompt(self.sample)
        else:
            conv_type = "full"
            input_prompt = self.task.populate_fully_specific_prompt(self.sample)

        # custom output dir for different temperatures to not mix up
        if self.run_custom_temperature:
            conv_type = f"{conv_type}-t{self.temperature}"

        is_reasoning_model = "o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model

        max_tokens = 16000 if is_reasoning_model else 1000
        trace = [{"role": "system", "content": self.system_message}, {"role": "user", "content": input_prompt}]

        if verbose:
            for msg in trace:
                print(f"\033[92m[{msg['role']}] {msg['content']}\033[0m")
                print("---")

        assistant_response_obj = generate(trace, model=self.assistant_model, return_metadata=True, temperature=self.temperature, max_tokens=max_tokens)


        assistant_response = assistant_response_obj["message"]
        if verbose:
            print(f"\033[91m[assistant] {assistant_response}\033[0m")

        trace.append({"role": "assistant", "content": assistant_response, "cost_usd": assistant_response_obj["total_usd"]})

        # eval_job_result = self.eval_client.schedule_evaluation(conversation=trace, task_name=self.task_name, sample=self.sample)
        # print(eval_job_result)
        # eval_result = self.eval_client.wait_for_job_completion(eval_job_result['job_id'], timeout=5)

        extracted_answer = self.task.extract_answer(assistant_response)
        evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

        is_correct = evaluation_return.get("is_correct", False)
        score = evaluation_return.get("score", 0.0)

        # trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
        # add it to the assistant response instead
        assistant_log = trace[-1]
        assistant_log["extracted_answer"] = extracted_answer
        assistant_log["is_correct"] = is_correct
        assistant_log["score"] = score

        if verbose:
            print('==================================================')
            icon = "\033[92m✔\033[0m" if is_correct else "\033[91m✘\033[0m"
            print(f"{icon} {extracted_answer} (score: {score})")

        if save_log:
            log_conversation(conv_type, self.task_name, self.sample["task_id"], self.dataset_fn, assistant_model=self.assistant_model, system_model="NA", user_model="NA", trace=trace, is_correct=is_correct, score=score, log_folder=self.log_folder)
        return is_correct, score



if __name__ == "__main__":
    from collections import Counter
    import json, argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
    parser.add_argument("--dataset_fn", type=str, default="sample_synthesis/data/code_synthetic_problems_0.1_verified.json")
    parser.add_argument("--assistant_model", type=str, default="t-gpt-4o-mini")
    parser.add_argument("--task", type=str, default="code")
    parser.add_argument("--run_concat", action="store_true")
    parser.add_argument("--run_shuffle_concat", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--evalserv_port", type=int, default=5001)
    args = parser.parse_args()

    # eval_client = EvaluationServiceClient(base_url=f"http://localhost:{args.evalserv_port}")
    # eval_client.load_tasks(dataset_file=args.dataset_fn, num_workers=40)
    # eval_client.wait_for_service_ready()

    if args.run_concat and args.run_shuffle_concat:
        raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

    with open(args.dataset_fn, "r") as f:
        data = json.load(f)

    data = [d for d in data if (d["task"] == args.task or args.task == "all")]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorFull(sample, args.assistant_model, run_concat=args.run_concat, run_shuffle_concat=args.run_shuffle_concat, evalserv_port=args.evalserv_port)
    conversation_simulator.run(verbose=args.verbose, save_log=False)
