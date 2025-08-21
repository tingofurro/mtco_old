from typing import Dict, Any, List, Union, Tuple
from task_base import Task
import json, sacrebleu

class TaskTranslation(Task):
    def __init__(self, version="0.1"):
        with open(f"prompts/translation/translation_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open(f"prompts/translation/translation_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        self.answer_extraction_strategy = "full_response"

    def get_dataset_file(self) -> str:
        return "data/sharded_translation.json"

    def get_samples(self) -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        data = [d for d in data if d["task"] == "translation"]
        return data

    def get_task_name(self) -> str:
        return "translation"

    def get_answer_description(self) -> str:
        return "A complete English translation of the German text."

    def generate_system_prompt(self, sample):
        return self.system_prompt

    def evaluator_function(self, extracted_answer, sample):
        bleu = sacrebleu.corpus_bleu([extracted_answer.strip()], [[sample["document_en"].strip()]])
        return {"score": bleu.score / 100.0}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[GERMAN_TEXT]]", sample["document_de"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        source_document = "Please translate the different chunks of the following German document into English. Translate the entire document without keeping chunking information.\n\n"
        for i, shard in enumerate(sample["shards"]):
            source_document += f"Document Chunk {i+1}:\n" + shard["shard"] + "\n\n"
        return self.fully_specified_prompt.replace("[[GERMAN_TEXT]]", source_document)

    def populate_sharded_prompt(self, sample, turn_index):
        if turn_index == 0:
            first_shard = sample["shards"][0]
            prompt = "You are translating a document from German to English that is being transcribed live. I will provide you with the document in chunks. At each turn, you should return the translation of the ENTIRE DOCUMENT (and not just the last chunk). You should consider all chunks together when translating, and not just the last chunk. You can optionally modify previously translated chunks if you think you had made mistakes.\n\nChunk 1:\n\n[[CHUNK_1]]"
            return prompt.replace("[[CHUNK_1]]", first_shard["shard"]), first_shard["shard_id"], 0.0
        elif turn_index <= len(sample["shards"]):
            prompt = f"I have received the latest chunk of the document. Please translate the entire document so far.\n\nChunk {turn_index+1}:\n\n[[CHUNK_{turn_index+1}]]"
            shard = sample["shards"][turn_index]
            return prompt.replace(f"[[CHUNK_{turn_index+1}]]", shard["shard"]), shard["shard_id"], 0.0
        else:
            return None, -1, 0.0

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process translation sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "german_text": sample["document_de"],
            "reference_translation": sample["document_en"]
        }

if __name__ == "__main__":
    # Test code
    task = TaskTranslation()
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples")
    
    # Test a sample
    sample = samples[0]
    print("\nGerman text:")
    print(sample["document_de"])
    print("\nReference translation:")
    print(sample["document_en"])
    
    # Test evaluation
    test_translation = sample["document_en"]  # Perfect translation should get high BLEU
    passed, feedback = task.evaluator_function(test_translation, sample)
    print(f"\nEvaluation feedback: {feedback}")
    print(f"Passed threshold: {passed}")
