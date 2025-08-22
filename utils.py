from datetime import datetime
import os, sys
from itertools import combinations
import numpy as np
import subprocess
import re

def extract_gpu_details():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        
        if not gpu_names:
            return {"gpu_name": "Unknown", "num_gpus": 0}
        
        # Use the first GPU name (assuming all GPUs are the same)
        gpu_name = gpu_names[0]
        num_gpus = len(gpu_names)
        
        return {"gpu_name": gpu_name, "num_gpus": num_gpus}
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try parsing nvidia-smi default output
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            lines = result.stdout.split('\n')
            
            gpu_name = None
            num_gpus = 0
            
            for line in lines:
                # Look for GPU lines that contain GPU name
                if '|' in line and 'NVIDIA' in line and 'PCIe' in line:
                    # Extract GPU name using regex
                    match = re.search(r'\|\s+\d+\s+([^|]+?)\s+(?:On|Off)', line)
                    if match:
                        if gpu_name is None:
                            gpu_name = match.group(1).strip()
                        num_gpus += 1
            
            return {"gpu_name": gpu_name or "Unknown", "num_gpus": num_gpus}
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"gpu_name": "Unknown", "num_gpus": 0}

def calculate_gpu_concurrency():
    gpu2concurrency = {"NVIDIA A100 80GB PCIe": 45, "NVIDIA A100 40GB": 10, "NVIDIA RTX A6000": 11, "NVIDIA H100 80GB HBM3": 45}
    gpu_details = extract_gpu_details()
    assert gpu_details["gpu_name"] in gpu2concurrency, f"GPU {gpu_details['gpu_name']} not found in gpu2concurrency"
    gpu_details["gpu_concurrency"] = gpu2concurrency[gpu_details["gpu_name"]]
    gpu_details["total_concurrency"] = gpu_details["num_gpus"] * gpu_details["gpu_concurrency"]
    return gpu_details

def load_env_vars(filepath='.env'):
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove 'export' if present and any leading/trailing whitespace
                if line.startswith('export '):
                    line = line[7:].strip()

                # Split on first '=' only
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Remove any quotes around the value
                    value = value.strip('\'"')
                    os.environ[key.strip()] = value


class DoublePrint:
    def __init__(self, name=None, mode="a"):
        if name is None:
            filename = sys.argv[0].split("/")[-1]
            name = "%s.log" % (filename.split(".")[0])
            name = os.path.join(os.environ["HOME"], "mtco/logs/", name)

        self.file = open(name, mode)
        if hasattr(sys.stdout, "isatty"):
            self.isatty = sys.stdout.isatty()
        else:
            self.isatty = False
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.prev_tqdm = False
        sys.stderr = self
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        data_str = str(data).strip()
        is_tqdm = data_str.count("[")==1 and data_str.count("]")==1 and data_str.count("|")==2 
        full_line = data_str+"\n"
        if not is_tqdm:
            full_line = "\033[94m "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\033[0m - "+full_line
        if is_tqdm and self.prev_tqdm:
            CURSOR_UP_ONE = '\x1b[1A'
            ERASE_LINE = '\x1b[2K'
            full_line = CURSOR_UP_ONE + ERASE_LINE +full_line

        if len(data_str) > 0:
            self.file.write(full_line)
            self.stdout.write(full_line)
            self.file.flush()
        self.prev_tqdm = is_tqdm

    def flush(self):
        self.file.flush()

def print_colored(text, color):
    if color == "red":
        print(f"\033[91m{text}\033[0m")
    elif color == "green":
        print(f"\033[92m{text}\033[0m")
    elif color == "blue":
        print(f"\033[94m{text}\033[0m")
    elif color == "purple":
        print(f"\033[95m{text}\033[0m")
    elif color == "yellow":
        print(f"\033[93m{text}\033[0m")
    elif color == "cyan":
        print(f"\033[96m{text}\033[0m")
    else:
        raise Exception(f"Unknown color: {color}")


def extract_conversation(simulation_trace, to_str=False, skip_system=False, only_last_turn=False):
    keep_roles = ["system", "assistant", "user"] if not skip_system else ["assistant", "user"]
    real_conversation = [msg for msg in simulation_trace if msg["role"] in keep_roles]
    if only_last_turn:
        # get all the user turns
        user_turn_idxs = [i for i, msg in enumerate(real_conversation) if msg["role"] == "user"]
        last_user_turn_idx = user_turn_idxs[-1]
        real_conversation = real_conversation[last_user_turn_idx + 1:]

    real_conversation = [{"role": msg["role"], "content": msg["content"]} for msg in real_conversation] # only keep role and content

    if to_str:
        return "\n\n".join([f"[{msg['role']}] {msg['content']}" for msg in real_conversation])
    else:
        return real_conversation


def date_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def jaccard_similarity(text1, text2):
    # Convert texts to sets of words
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 1.0


def subsample_responses(responses, n_responses=4, strategy="jaccard_exact"):
    """filter outputs to get a subset"""
    if len(responses) <= n_responses:
        return responses

    n = len(responses)

    if strategy == "jaccard_exact":
        sim_matrix = np.zeros((len(responses), len(responses)))
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sim_matrix[i, j] = jaccard_similarity(responses[i]["response_text"], responses[j]["response_text"])
                sim_matrix[j, i] = sim_matrix[i, j]
        
        best_score = float("inf")
        best_subset_idx = None
        for comb in combinations(range(len(responses)), n_responses):
            idx = np.array(comb)
            score = sim_matrix[np.ix_(idx, idx)].sum() / 2
            if score < best_score:
                best_score = score
                best_subset_idx = comb

        return [responses[i] for i in best_subset_idx]

    if strategy == "jaccard_greedy":
        sim_matrix = np.zeros((len(responses), len(responses)))
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                sim_matrix[i, j] = jaccard_similarity(responses[i]["response_text"], responses[j]["response_text"])
                sim_matrix[j, i] = sim_matrix[i, j]

        avg_sim = sim_matrix.mean(axis=1)
        subset = [int(np.argmin(avg_sim))]

        while len(subset) < n_responses:
            remaining = list(set(range(len(responses))) - set(subset))
            # similarity of each remaining item to the current subset
            scores = [sim_matrix[i, subset].sum() for i in remaining]
            next_idx = remaining[int(np.argmin(scores))]
            subset.append(next_idx)

        return [responses[i] for i in subset]

    elif strategy == "random":
        return np.random.choice(range(len(responses)), n_responses, replace=False)

    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def calculate_advantage(advantage_estimation, scores):
    if len(scores) == 0:
        return []

    if advantage_estimation == "zero_mean":
        return [score - np.mean(scores) for score in scores]
    elif advantage_estimation == "zero_mean_noneg":
        return [score - np.mean(scores) if score - np.mean(scores) > 0 else 0 for score in scores]
    elif advantage_estimation == "zero_mean_noneg_bounded":
        clamped_scores = [score - np.mean(scores) if score - np.mean(scores) > 0 else 0 for score in scores]
        max_value = max(clamped_scores)
        min_value = min(clamped_scores)
        if max_value == min_value:
            return [0 for _ in scores]
        else:
            return [(score - min_value) / (max_value - min_value) for score in clamped_scores]
    elif advantage_estimation == "zero_mean_positive_only_bounded_2":
        if len(set(scores)) == 1:
            if scores[0] > 0.5:
                return scores
            else:
                return [0 for _ in scores]
        else:
            clamped_scores = [score - np.mean(scores) if score - np.mean(scores) > 0 else 0 for score in scores]
            max_value = max(clamped_scores)
            min_value = min(clamped_scores)
            if max_value == min_value:
                return [0 for _ in scores]
            else:
                return [(score - min_value) / (max_value - min_value) for score in clamped_scores]
    elif advantage_estimation == "bounded":
        min_value = min(scores)
        max_value = max(scores)
        if max_value == min_value:
            return [0 for _ in scores]
        else:
            return [(score - min_value) / (max_value - min_value) for score in scores]
    elif advantage_estimation == "bounded_symmetric":
        # [-1, 1]
        min_value = min(scores)
        max_value = max(scores)
        if max_value == min_value:
            return [0 for _ in scores]
        else:
            return [2 * (score - min_value) / (max_value - min_value) - 1 for score in scores]
    else:
        raise ValueError(f"Invalid advantage estimation: {advantage_estimation}")


if __name__ == "__main__":
    print(calculate_gpu_concurrency())
