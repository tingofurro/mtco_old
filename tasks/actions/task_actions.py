from tasks.actions.eval_bfcl import ast_checker, ast_parse, cleanup_input_str
from typing import Dict, Any
from task_base import Task
import json

class TaskActions(Task):
    def __init__(self):
        with open("prompts/actions/actions_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        with open("prompts/actions/actions_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()

        self.answer_extraction_strategy = "full_response"

    def get_task_name(self):
        return "actions"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions_600.json"

    def get_samples(self):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        data = [d for d in data if d["task"] == "actions"]
        return data

    def get_answer_description(self) -> str:
        # FIXME
        return (
            "The answer should be a series of valid function calls in the format of "
            "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. "
            "An answer may contain multiple function calls."
        )

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt.replace("[[FUNCTIONS]]", "{}".format(sample["function"]))

    def is_answer_attempt(self, text: str) -> bool:
        # check if ```python is present in the text
        clean_text = cleanup_input_str(text)
        return clean_text.startswith("[") and clean_text.endswith("]")

    def evaluator_function(self, predicted_answer: str, sample: Dict[str, Any]) -> Dict:
        """
        Evaluate if the predicted function call matches the expected format and functionality.
        """

        try:
            # attempt to decode ast out of the predicted answer
            decoded_output = ast_parse(predicted_answer.strip(), sample["language"])
        except Exception as e:
            # print(f"Error decoding AST: {e}")
            # print(f"\033[94mPredicted answer: |{predicted_answer}|\033[0m")
            return {"is_correct": False, "error": "Failing to parse the predicted answer as an AST"}

        result = ast_checker(
            sample["function"],
            decoded_output,
            sample["reference_answer"],
            sample["language"],
            sample["test_category"],
            "gpt-4o"
        )
        score = 1 if result["valid"] else 0
        return {"is_correct": result["valid"], "score": score, "error": result["error"]}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        # The official gorilla repo inserts functions as a un-indented one-liner
        return self.fully_specified_prompt.replace("[[QUESTION]]", sample["fully_specified_question"][0][0]["content"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = ""
        for shard in sample["shards"]:
            query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[QUESTION]]", query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process BFCL sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question": sample["fully_specified_question"][0][0]["content"],  # because their data supports multi-turn. Hardcoding for the single-turn version for now.
            "answer": sample["reference_answer"],
        }

if __name__ == "__main__":
    task = TaskActions()
    samples = task.get_samples()
    print(len(samples))
