from utils import extract_conversation
from tasks import get_task


class SystemAgent:
    def __init__(self, task_name, sample):
        self.task_name = task_name
        self.task = get_task(task_name)
        self.answer_extraction_strategy = self.task.answer_extraction_strategy
        self.sample = sample
        self.answer_description = self.task.get_answer_description()
        self.max_extraction_attempts = 3

        assert self.answer_extraction_strategy in ["full_response", "prefix_suffix", "gen", "task_specific"], f"Answer extraction strategy {self.answer_extraction_strategy} not supported"

        with open("prompts/system_turn_categorization.txt", "r", encoding="utf-8") as f:
            self.system_verification_prompt = f.read()
        with open("prompts/system_answer_extraction_gen.txt", "r", encoding="utf-8") as f:
            self.answer_extraction_prompt_gen = f.read()
        with open("prompts/system_answer_extraction_prefix_suffix.txt", "r", encoding="utf-8") as f:
            self.answer_extraction_prompt_prefix_suffix = f.read()


    def classify_assistant_response(self, conversation_so_far):
        if self.task_name in ["summary", "totto", "translation"]:
            # in these tasks, the assistant is explicitly instructed to provide an answer attempt at each turn
            return {"response_type": "answer_attempt"}, 0.0

        initial_query = self.sample["shards"][0]["shard"]
        shards = self.sample["shards"][1:]

        last_turn_text = extract_conversation(conversation_so_far, to_str=True, only_last_turn=True)

        # print("--------------------- TURN CLASSIFICATION ---------------------")
        # print(last_turn_text)

        response_strategy = ""

        if len(last_turn_text) < 5:
            response_strategy = "missing"
        elif self.task.is_answer_attempt(last_turn_text):
            response_strategy = "answer_attempt"
        elif len(last_turn_text) < 150:
            response_strategy = "discussion-short"
        else:
            response_strategy = "discussion-long"

        return response_strategy

    def extract_answer(self, conversation_so_far):
        assistant_response = [msg["content"] for msg in conversation_so_far if msg["role"] == "assistant"][-1]

        if self.answer_extraction_strategy == "full_response":
            return assistant_response # just return the full response
        elif self.answer_extraction_strategy == "task_specific":
            return self.task.extract_answer(assistant_response)
        else:
            raise ValueError(f"Answer extraction strategy {self.answer_extraction_strategy} not supported")

