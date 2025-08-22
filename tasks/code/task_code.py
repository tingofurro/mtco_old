from typing import List, Dict, Any, Tuple
import json, os, random, re, pickle, zlib, base64, ast


from task_base import Task
from tasks.code.eval_code import check_correctness

class TaskCode(Task):
    def __init__(self):
        # Get current file's directory, go up 2 levels, then into prompts
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        prompts_dir = os.path.join(project_root, "prompts")
        # print("================")

        with open(os.path.join(prompts_dir, "lcb/lcb_full_prompt.txt"), "r") as f:
            self.fully_specified_prompt_lcb = f.read()
        with open(os.path.join(prompts_dir, "lcb/lcb_system_prompt.txt"), "r") as f:
            self.system_prompt_lcb = f.read()
        with open(os.path.join(prompts_dir, "humaneval/humaneval_full_prompt.txt"), "r") as f:
            self.fully_specified_prompt_humaneval = f.read()
        with open(os.path.join(prompts_dir, "humaneval/humaneval_system_prompt.txt"), "r") as f:
            self.system_prompt_humaneval = f.read()

        self.seed = 42
        # random.seed(self.seed)

        self.answer_extraction_strategy = "task_specific"

    # def get_dataset_file(self) -> str:
    #     return "data/sharded_instructions_600.json"

    # def get_samples(self) -> List[Dict[str, Any]]:
    #     with open(self.get_dataset_file(), "r") as f:
    #         data = json.load(f)
    #     data = [d for d in data if d["task"] == "code"]
    #     return data

    def get_task_name(self):
        return "code"

    # merged answer description for LCB and HumanEval
    def get_answer_description(self) -> str:
        return ("A final answer must be a valid Python function that is defined (def ...) and returns "
                "something. The function should handle multiple test cases according to the problem specification. "
                "If the answer is just a natural language explanation, it is not a valid answer. "
                "Special symbols like newlines and quotes must be escaped.")

    # use LCB system prompt for all samples
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt_lcb

    def load_test_cases(self, sample):
        public_test_cases = json.loads(sample["public_test_cases"])  # type: ignore

        if "private_test_cases" in sample:
            try:
                private_test_cases = json.loads(sample["private_test_cases"])  # type: ignore
            except:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(sample["private_test_cases"].encode("utf-8"))  # type: ignore
                        )
                    )
                )  # type: ignore
        else:
            private_test_cases = []

        return json.dumps(
            {
                "inputs": [
                    t["input"]
                    for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t["output"]
                    for t in public_test_cases + private_test_cases
                ],
                "fn_name": sample["metadata"].get("func_name", None),
            }
        )

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        pred_python_code = extracted_answer.replace("```python", "").replace("```", "")

        if "def " not in pred_python_code:
            return {"is_correct": False, "pass@1": 0, "score": 0}

        # Adding imports for HE-derived samples
        if "prompt" in sample:
            # Extract imports from sample["prompt"] -- this affects full
            prompt_ast = ast.parse(sample["prompt"])
            imports = []
            for node in prompt_ast.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))

            # Prepend imports to pred_python_func
            if imports:
                pred_python_code = "\n".join(imports) + "\n\n" + pred_python_code

        # Force update the function name with the true function name
        old_func_name = pred_python_code.split("def ")[1].split("(")[0].strip()

        func_name = sample["name"] if "name" in sample else sample["metadata"]["func_name"] # tape, shouldn't be needed once unified
        func_name = sample["metadata"]["func_name"]

        pred_python_code = pred_python_code.replace(old_func_name, func_name)

        # let's trigger only if you can't find the transformed test cases
        # if sample.get("sample_type", "") == "code_synthetic" and "public_test_cases" not in sample:
        #     is_correct = evaluate_synthetic_problem(pred_python_code, sample["reference_tests"], sample["name"], printing=False)
        #     return {"is_correct": is_correct, "pass@1": 1 if is_correct else 0, "score": 1 if is_correct else 0}

        # load tests
        testcases = self.load_test_cases(sample)

        output, metadata = check_correctness(sample, pred_python_code, testcases, timeout=6)
        all_test_cases_passed = all(o is True for o in output)

        score = 1 if all_test_cases_passed else 0
        return {"is_correct": all_test_cases_passed, "pass@1": 1 if all_test_cases_passed else 0, "score": score}

    def get_formatting_preamble(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Outputs the formatting preamble and the starter code block"""
        # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py#L33
        call_preamble = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

        return call_preamble, f"```python\n{sample['starter_code'] if sample['starter_code'] else '# YOUR CODE HERE'}\n```"

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        if sample.get("sample_type") == "code_synthetic":
            return f"The user requires the help in implementing the Python function {sample['name']}. Here's the instruction provided by the user:\n\n{sample['description']}"
        
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return self._populate_fully_specific_prompt_lcb(sample)
        elif sample.get("source") == "humaneval":
            return self._populate_fully_specific_prompt_humaneval(sample)
        else:
            raise ValueError(f"Invalid source: {sample.get('source')}")

    def _populate_fully_specific_prompt_lcb(self, sample: Dict[str, Any]) -> str:
        query = sample["question_content"]
        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt_lcb.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def _populate_fully_specific_prompt_humaneval(self, sample: Dict[str, Any]) -> str:
        user_query = sample["prompt"]
        # wrap with python code block
        return self.fully_specified_prompt_humaneval.replace("[[INSTRUCTION]]", f"Complete the following incomplete function signature:\n```python\n{user_query}\n```")

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        if sample.get("sample_type") == "code_synthetic":
            concat_shards = "\n".join([f"- {shard['shard']}" for shard in sample["shards"]])
            return f"The user requires the help in implementing the Python function {sample['name']}. Here's the instruction provided by the user:\n\n{concat_shards}"
        
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return self._populate_concat_prompt_lcb(sample)
        elif sample.get("source") == "humaneval":
            return self._populate_concat_prompt_humaneval(sample)
        else:
            raise ValueError(f"Invalid source: {sample.get('source')}")

    def _populate_concat_prompt_lcb(self, sample: Dict[str, Any]) -> str:
        query = sample["shards"][0]["shard"] + "\n"
        for shard in sample["shards"][1:]:
            query += f"- {shard['shard']}\n"

        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt_lcb.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def _populate_concat_prompt_humaneval(self, sample: Dict[str, Any]) -> str:
        query = sample["shards"][0]["shard"] + "\n"
        for shard in sample["shards"][1:]:
            query += f"- {shard['shard']}\n"

        return self.fully_specified_prompt_humaneval.replace("[[INSTRUCTION]]", query)

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process python sample for annotation UI display"""
        if sample.get("source") in ["lcb_easy", "lcb_medium"]:
            return {
                "task_id": sample["task_id"],
                "prompt": sample["question_content"] + "\n\n" + sample["starter_code"],
            }
        elif sample.get("source") == "humaneval":
            return {
                "task_id": sample["task_id"],
                "prompt": sample["prompt"],
            }

    def is_answer_attempt(self, text: str) -> bool:
        # check if ```python is present in the text
        # print(text)
        return "```python" in text

    def extract_answer(self, text: str) -> str:
        # First try to extract Python code blocks from markdown
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)

        if "class Solution" in text:
            # the task_python_eval.py handles the function extraction.
            return [c for c in code_blocks if "class Solution" in c][-1]

        # If we found code blocks, try each one from last to first
        if code_blocks:
            for block in reversed(code_blocks):
                result = self._extract_function_from_code(block)
                if result:
                    return result
            return ""  # No function found in any code block

        # If no code blocks found, try to find the last function definition in the text
        text = text.strip()
        if text.startswith("```") or text.startswith("`"):
            text = text[text.find("\n"):].strip()
        import_idx = text.rfind("import")
        def_idx = text.rfind("def")
        start_idx = import_idx if import_idx >= 0 else def_idx
        if start_idx >= 0:
            text = text[start_idx:]
        return self._extract_function_from_code(text)

    def _add_parent_info(self, node, parent=None):
        """Add parent information to all nodes in the AST."""
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._add_parent_info(child, node)

    def _extract_function_from_code(self, code: str) -> str:
        """Helper method to extract function from pure Python code using AST."""
        try:
            # Parse the code
            tree = ast.parse(code)

            # Add parent information to all nodes
            self._add_parent_info(tree)

            # Find all import statements
            import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            imports = []
            for node in import_nodes:
                imports.append(ast.unparse(node))

            # Find all function definitions
            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_nodes:
                return ""  # No functions found

            # Get the last function definition that is at the top level
            last_function = None
            for node in reversed(function_nodes):
                # Check if the parent is the module (top level)
                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break

            if not last_function:
                return ""  # No top-level functions found

            # Get thec source lines from the original text
            source_lines = code.splitlines()

            # Function spans from its first line to the last line of its body
            # If there are decorators, start from the first decorator
            start_line = (last_function.decorator_list[0].lineno - 1
                         if last_function.decorator_list
                         else last_function.lineno - 1)  # ast line numbers are 1-based
            end_line = last_function.end_lineno

            # Extract the complete function text
            function_text = '\n'.join(source_lines[start_line:end_line])

            # Prepend imports if any were found
            if imports:
                return '\n'.join(imports) + '\n\n' + function_text
            return function_text
        except Exception:
            return ""

    def extract_function_body(self, code: str) -> str:
        """Extract the body of a function and convert it to top-level code.

        Args:
            code: String containing a Python function definition

        Returns:
            The function body as top-level code, or empty string if no function found
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Add parent information to all nodes
            self._add_parent_info(tree)

            # Find all function definitions
            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_nodes:
                print("No functions found")
                return ""  # No functions found

            # Get the last function definition that is at the top level
            last_function = None
            for node in reversed(function_nodes):
                # Check if the parent is the module (top level)
                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break

            if not last_function:
                print("No top-level functions found")
                return ""  # No top-level functions found

            # Get the source lines from the original text
            source_lines = code.splitlines()

            # Function body starts after the function definition line
            start_line = last_function.lineno  # ast line numbers are 1-based
            end_line = last_function.end_lineno

            # Extract the function body text
            body_lines = source_lines[start_line:end_line]

            # Find the minimum indentation level in the body
            min_indent = float('inf')
            for line in body_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            # Remove only the initial indentation level from each line
            body_lines = [line[min_indent:] if line.strip() else line for line in body_lines]

            return '\n'.join(body_lines)
        except Exception as e:
            print("Error extracting function body")
            print(e)
            return ""


if __name__ == "__main__":
    task = TaskCode()
    samples = task.get_samples()
    print(len(samples))
