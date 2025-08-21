from typing import Dict, Any, List
from task_base import Task
import json, sacrebleu

class TaskData2Text(Task):
    def __init__(self):
        with open(f"prompts/data2text/data2text_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open(f"prompts/data2text/data2text_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        self.answer_extraction_strategy = "full_response"

    def is_answer_attempt(self, text):
        return True # Always an answer attempt

    def get_task_name(self) -> str:
        return "data2text"

    def get_answer_description(self) -> str:
        return "A one-sentence description of the table."

    def generate_system_prompt(self, sample):
        return self.system_prompt

    def evaluator_function(self, extracted_answer, sample):
        # ToTTo has multiple references per example
        references = sample["references"]
        bleu = sacrebleu.corpus_bleu([extracted_answer.strip()], [[ref.strip()] for ref in references])
        return {"score": bleu.score / 100.0}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        # Replace placeholders with table HTML and metadata
        prompt = self.fully_specified_prompt
        prompt = prompt.replace("[[TABLE_HTML]]", sample["table_highlighted_html"])
        prompt = prompt.replace("[[FEWSHOT_DESCRIPTIONS]]", sample["fewshot_descriptions"])
        
        # Add metadata if available
        metadata_str = ""
        for key, value in sample["metadata"].items():
            metadata_str += f"{key}: {value}\n"
        prompt = prompt.replace("[[CONTEXT]]", metadata_str)
        
        return prompt

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        prompt = self.fully_specified_prompt

        # both of these are defined in the hints anyway
        prompt = prompt.replace("[[TABLE_HTML]]", "") # sample["table_html"]
        prompt = prompt.replace("[[FEWSHOT_DESCRIPTIONS]]", "") # sample["fewshot_descriptions"]
        context = ""

        for shard in sample["shards"]:
            context += f"Hint {shard['shard_id']}:\n{shard['shard']}\n\n"
        prompt = prompt.replace("[[CONTEXT]]", context)
        return prompt

    def populate_sharded_prompt(self, sample, turn_index):
        if turn_index < len(sample["shards"]):
            shard = sample["shards"][turn_index]
            return shard["shard"], shard["shard_id"]
        else:
            return None, -1

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process ToTTo sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "table_html": sample["table_html"],
            "table_highlighted_html": sample["table_highlighted_html"],
            "metadata": sample["metadata"],
            "references": sample["references"]
        }

if __name__ == "__main__":
    # Test code
    task = TaskData2Text()
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples")
    
    sample = samples[0]
    print(task.populate_concat_prompt(sample))
