from typing import List, Dict, Any
from task_base import Task
import json, random, re

class TaskMath(Task):
    def __init__(self):
        with open("prompts/math/math_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/math/math_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        self.answer_extraction_strategy = "task_specific"

    def get_task_name(self):
        return "math"

    def get_answer_description(self) -> str:
        return "The answer should be a single number (it could be decimal, or negative, or a fraction, etc.)."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def is_answer_attempt(self, text):
        return "```answer" in text

    def extract_answer(self, text):
        try:
            return re.findall(r"```answer\s*(.*?)\s*```", text, re.DOTALL)[-1]
        except:
            print(f"Error extracting answer: {text}")
            return ""

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:

        regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]

        # ground truth
        gold = sample["answer"].split("####")[1].strip().lower()

        try:
            # https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L42
            extracted_answer = extracted_answer.strip()
            # strict
            # extracted_answer = re.findall(r"(\-?[0-9\.\,]+)", extracted_answer)[0]
            # flexible
            extracted_answer = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", extracted_answer)[-1]
            extracted_answer = [m for m in extracted_answer if m][0]
        except:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(extracted_answer)}"}

        # custom formatting fix
        # if dollar mark is in the answer, check for the cents and trim if necessary
        if re.search(r'\$', extracted_answer) and extracted_answer.endswith(".00"):
            extracted_answer = extracted_answer.rstrip(".00")

        # ref: https://github.com/EleutherAI/lm-evaluation-harness/blob/52df63b7b30da53c481ed9090598d9189fab1d91/lm_eval/api/metrics.py#L198
        # further normalize $ and , for both extracted_answer and gold
        for regex in regexes_to_ignore:
            extracted_answer = re.sub(regex, "", extracted_answer)
            gold = re.sub(regex, "", gold)
        score = 1.0 if extracted_answer == gold else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[QUESTION]]", sample["question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = ""
        for shard in sample["shards"]:
            query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[QUESTION]]", query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        # FIXME(hiro): "completion" is not the best name for the field because we ask for a full function
        return response["answer"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process GSM8K sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "answer": sample["answer"],
        }
