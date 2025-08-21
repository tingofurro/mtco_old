import json, random

from llms import generate_json
from utils import extract_conversation, print_colored

class UserAgent:
    def __init__(self, task, model="gpt-4o"):
        self.model = model
        self.task = task
        self.task_name = task.get_task_name()

        if self.model.startswith("l-"):
            self.json_schema = {
                "type": "object",
                "properties": {"shard_id": {"type": "number"}, "response": {"type": "string"}},
                "required": ["shard_id", "response"]
            }
        else:
            self.json_schema = None

        with open("prompts/user_agent.txt", "r") as f:
            self.prompt_response = f.read()

    def generate_response(self, conversation, sample, temperature=1.0):
        num_user_msgs = sum(1 for msg in conversation if msg["role"] == "user")

        if self.task_name in ["translation", "summary", "data2text", "flipflop"]:
            return self.task.populate_sharded_prompt(sample, num_user_msgs)

        assistant_messages = [msg for msg in conversation if msg["role"] == "assistant"]

        if num_user_msgs == 0:
            first_shard = sample["shards"][0]
            shard_id = first_shard["shard_id"]
            initial_query = random.choice(first_shard["paraphrases"])

            # print_colored(f"[User Agent] Initial query: {initial_query}", "purple")

            return initial_query, shard_id

        shard_revealed_ids = [msg["shard_id"] for msg in conversation if "shard_id" in msg]
        shard_ids = [shard["shard_id"] for shard in sample["shards"]]
        shard_ids_revealed = [shard_id for shard_id in shard_ids if shard_id in shard_revealed_ids]
        shard_ids_not_revealed = [shard_id for shard_id in shard_ids if shard_id not in shard_revealed_ids]

        if len(assistant_messages) > 0 and assistant_messages[-1]["response_strategy"] in ["answer_attempt", "discussion-long"]:
            # we can select a random unrevealed shard, and select a random paraphrase of it
            # if the task is actions, then pick in order of the shards
            if self.task_name == "actions":
                shard_id = sorted(shard_ids_not_revealed)[0]
            else:
                shard_id = random.choice(shard_ids_not_revealed)
            shard = [shard for shard in sample["shards"] if shard["shard_id"] == shard_id][0]
            shard_text = random.choice(shard["paraphrases"])
            # print_colored(f"[User Agent] Rule-based response: {shard_text}", "purple")
            return shard_text, shard_id

        # then we have to use the an LLM to figure it out...

        shard_texts_revealed = [{"shard_id": s["shard_id"], "shard": s["shard"]} for s in sample["shards"] if s["shard_id"] in shard_ids_revealed] # remove any other irrelevant fields
        shard_texts_not_revealed = [{"shard_id": s["shard_id"], "shard": s["shard"]} for s in sample["shards"] if s["shard_id"] in shard_ids_not_revealed] # remove any other irrelevant fields

        shards_revealed_str = json.dumps(shard_texts_revealed)
        shards_not_revealed_str = json.dumps(shard_texts_not_revealed)


        user_agent_prompt_populated = self.prompt_response.replace("[[CONVERSATION_SO_FAR]]", extract_conversation(conversation, to_str=True, skip_system=True)).replace("[[SHARDS_NOT_REVEALED]]", shards_not_revealed_str) # .replace("[[SHARDS_REVEALED]]", shards_revealed_str)

        # print_colored(user_agent_prompt_populated, "purple")
        response_obj = None
        while response_obj is None:
            try:
                # print_colored(f"[{self.task_name}] Generating user responses with {self.model}; {assistant_messages[-1]['response_strategy']}; {num_user_msgs}; {shard_ids_revealed} {shard_ids_not_revealed}", "red")
                response_obj = generate_json([{"role": "user", "content": user_agent_prompt_populated}], model=self.model, timeout=100, return_metadata=True, temperature=temperature, is_json=True, json_schema=self.json_schema)
            except Exception as e:
                print(f"Error generating user response: {e}")

        response = response_obj["message"]

        if response is None or type(response) != dict:
            return "Please try again.", -1 # This should be extremely rare, when the user agent fails to generate proper JSON
        response_text = response.get("response", "")
        shard_id = response.get("shard_id", -1)

        # print_colored(f"[User Agent] LLM response: {response_text}", "purple")

        return response_text, shard_id
