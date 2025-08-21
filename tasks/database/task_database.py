import os
from tasks.database.eval_spider_exec import eval_exec_match
from typing import Dict, Any, List
from task_base import Task
import json, re

class TaskDatabase(Task):
    def __init__(self):
        with open(f"prompts/database/database_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open(f"prompts/database/database_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        self.answer_extraction_strategy = "task_specific"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions_600.json"


    def get_samples(self):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        data = [d for d in data if d["task"] == "database"]
        return data

    def get_task_name(self) -> str:
        return "database"

    def get_answer_description(self) -> str:
        return "If the response contains a complete SQL query (and not just partial or templated SQL used as an example), then it is an answer attempt. You must only extract the SQL query, nothing before or after, as it will be executed as is."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt.replace("[[SCHEMA]]", sample["schema_sql"])

    def is_answer_attempt(self, text: str) -> bool:
        return "```sql" in text
    
    def extract_answer(self, text: str) -> str:
        # Find the latest occurrence of ```sql and extract its contents
        sql_blocks = text.split("```sql")
        if len(sql_blocks) > 1:
            # Get the last SQL block
            last_sql_block = sql_blocks[-1]
            # Extract content up to the closing ```
            if "```" in last_sql_block:
                extracted_sql = last_sql_block.split("```")[0].strip()
                return extracted_sql
        
        # Fallback if no proper SQL block is found
        return text.strip()

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        # simpler and easier than using parsing, based on this paper
        # https://arxiv.org/pdf/2010.02840 (followup to spider)

        pred_sql = extracted_answer.replace("```sql", "").replace("```", "")
        pred_sql = re.sub(r"\s+", " ", pred_sql).strip()
        ref_sql = sample["reference_sql"]
        # if there's no data/spider/databases/ folder, then throw an error
        if not os.path.exists("data/spider/databases/"):
            raise FileNotFoundError("data/spider/databases/ folder not found; please see data/spider/README.md for instructions")


        try:
            is_correct = eval_exec_match(f"data/spider/databases/{sample['db_id']}/", pred_sql, ref_sql, plug_value=True, keep_distinct=False, progress_bar_for_each_datapoint=False) == 1
        except Exception as e:
            print(f"Error evaluating SQL: {e}")
            is_correct = False
        score = 1.0 if is_correct else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[DATABASE_SCHEMA]]", sample["schema_sql"]).replace("[[USER_QUERY]]", sample["fully_specified_question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        user_query = "Consider all the following:\n"

        for shard in sample["shards"]:
            user_query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[DATABASE_SCHEMA]]", sample["schema_sql"]).replace("[[USER_QUERY]]", user_query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response["sql"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process Spider sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question": sample["fully_specified_question"],
            "reference_sql": sample["reference_sql"],
            "db_id": sample["db_id"],
            "spider_difficulty": sample.get("spider_difficulty", "NA"),
            # "schema": sample["schema_sql"],
        }


if __name__ == "__main__":
    task = TaskDatabase()
    samples = task.get_samples()
    print(len(samples))
