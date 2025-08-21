import json
import os
import git
import time
import pandas as pd
from bson.objectid import ObjectId
from collections import Counter
from datetime import datetime


def get_log_files(conv_type, task_name, assistant_model, force_create=False, log_folder="logs"):
    # Sanitize the assistant_model name for Windows compatibility
    # Replace characters that are invalid in Windows filenames: < > : " / \ | ? *
    sanitized_model = assistant_model
    for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        sanitized_model = sanitized_model.replace(char, '-')

    base_log_file = f"{log_folder}/{task_name}/{conv_type}/{conv_type}_{task_name}_{sanitized_model}.jsonl"

    # if the folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(base_log_file)):
        if not force_create:
            return []
        os.makedirs(os.path.dirname(base_log_file))

    # Get all matching log files including split files
    log_dir = os.path.dirname(base_log_file)
    base_name = os.path.basename(base_log_file).replace(".jsonl", "")
    log_files = []

    for file in os.listdir(log_dir):
        if file.startswith(base_name) and file.endswith(".jsonl"):
            log_files.append(os.path.join(log_dir, file))

    # if it doesn't exist, touch it
    if len(log_files) == 0:
        if force_create:
            with open(base_log_file, "w") as f:
                f.write("")
            log_files.append(base_log_file)
        else:
            return []

    return sorted(log_files)  # Sort to ensure consistent order


def get_run_counts(conv_type, task_name, assistant_model, dataset_fn, log_folder="logs"):
    dataset_fn = dataset_fn.split("/")[-1] # Remove folders
    log_files = get_log_files(conv_type, task_name, assistant_model, force_create=False, log_folder=log_folder)
    task_id_counts = Counter()
    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                d = json.loads(line)
                if d["dataset_fn"] == dataset_fn and d["assistant_model"] == assistant_model:
                    task_id_counts[d["task_id"]] += 1
    return task_id_counts


def log_conversation(conv_type, task_name, task_id, dataset_fn, assistant_model, system_model, user_model, trace, is_correct=None, score=None, additional_info={}, log_folder=None):
    log_files = get_log_files(conv_type, task_name, assistant_model, force_create=True, log_folder=log_folder)
    log_file = log_files[-1]

    dataset_fn = dataset_fn.split("/")[-1] # Remove folders

    # if the folders don't exist, create them
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    git_version = git.Repo(search_parent_directories=True).head.object.hexsha

    record = {"conv_id": str(ObjectId()), "conv_type": conv_type, "task": task_name, "task_id": task_id, "dataset_fn": dataset_fn, "assistant_model": assistant_model, "system_model": system_model, "user_model": user_model, "git_version": git_version, "trace": trace, "is_correct": is_correct, "score": score} # , "source_conv_id": source_conv_id
    record.update(additional_info) # sample-specific, for example for recap
    with open(log_file, "a") as f:
        f.write(json.dumps(record)+"\n")


def clean_model_name(model):
    if (model.startswith("t-") or model.startswith("l-") or model.startswith("b-")):
        model = model[2:]
    elif model.startswith("sfr-"):
        model = model[4:]

    bad_ends = ["-instruct", "-17b-16e"]
    for bad_end in bad_ends:
        if model.endswith(bad_end):
            model = model[:-len(bad_end)]
    return model

def load_results_from(folder, dataset_fn, merge_trapi=True):
    # TrAPI = internal MSR API to access OpenAI model. If merge_trapi=True, the trapi model acronyms are merged with regular openai models (t-gpt-4o -> gpt-4o)
    dataset_fn = dataset_fn.split("/")[-1]
    model_data = {}
    for fn in os.listdir(folder):
        if "__" in fn:
            model = fn.split("__")[0]
        else:
            model = fn.replace(".jsonl", "")

        model = model.split("_")[-1]

        with open(os.path.join(folder, fn), "r") as f:
            lines = f.read().split("\n")
        data = []
        num_fail = 0
        for line in lines:
            if len(line.strip()) == 0:
                continue
            try:
                data.append(json.loads(line))
            except:
                num_fail += 1
        if num_fail > 0:
            # let's update the file to remove the failed lines
            print(f"Removing {num_fail} failed lines from {fn}")
            with open(os.path.join(folder, fn), "w") as f:
                for d in data:
                    f.write(json.dumps(d)+"\n")
        data = [d for d in data if d["dataset_fn"] == dataset_fn]
        if len(data) > 0:
            for log in data:
                model = log["assistant_model"]
                if merge_trapi:
                    model = clean_model_name(model)
                if model not in model_data:
                    model_data[model] = []
                model_data[model].append(log)
    return model_data


def clean_up_logs(task_name, dataset_fn, ids=None, conv_types="all", models="all", is_mock=False, log_folder="logs"):
    assert models == "all" or (type(models) is list)

    # we're going to clean the files in the following manner:
    folder = f"{log_folder}/{task_name}"
    if conv_types == "all":
        conv_types = os.listdir(folder)

    N_filtered = Counter()
    for conv_type in conv_types:
        log_files = os.listdir(os.path.join(folder, conv_type))
        for log_file in log_files:
            kept_conversations = []
            with open(os.path.join(folder, conv_type, log_file), "r") as f:
                all_conversations = [json.loads(line) for line in f]
            for conversation in all_conversations:
                model = conversation["assistant_model"]
                if conversation["dataset_fn"] == dataset_fn and (models == "all" or model in models) and (ids is None or conversation["task_id"] in ids):
                    N_filtered[(conversation["dataset_fn"], conv_type, model)] += 1
                    continue
                kept_conversations.append(conversation)
            if not is_mock:
                with open(os.path.join(folder, conv_type, log_file), "w") as f:
                    for conversation in kept_conversations:
                        f.write(json.dumps(conversation)+"\n")

    print(f"Filtered {N_filtered.total()} conversations")
    for (dataset_fn, conv_type, model), count in N_filtered.items():
        print(f"{dataset_fn} {conv_type} {model}: {count}")


def check_latest_updates():
    """
    Checks all files in the logs folder and its subfolders for files modified in the last 3 minutes.
    Returns a list of tuples containing (file_path, last_modified_time).
    """
    logs_dir = "logs"
    recent_files = []
    current_time = time.time()
    three_minutes_ago = current_time - (3 * 60)  # 3 minutes in seconds

    for root, _, files in os.walk(logs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            last_modified = os.path.getmtime(file_path)

            if last_modified > three_minutes_ago:
                modified_time = datetime.fromtimestamp(last_modified)
                recent_files.append((file_path, modified_time))
    dataset = []
    for fn, mod_time in recent_files:
        fn = fn.split("\\")[-1].replace(".jsonl", "")
        conv_type, task_name, model = fn.split("_")
        dataset.append({"task_name": task_name, "conv_type": conv_type, "model": model, "mod_time": mod_time})
    return pd.DataFrame(dataset)


def split_large_file(file_path, max_size_mb=30):
    assert file_path.endswith(".jsonl")
    max_size_bytes = max_size_mb * 1024 * 1024
    fn = file_path.split("/")[-1]
    base_fn = fn.replace(".jsonl", "")

    if "__" in base_fn:
        base_fn, chunk_number = base_fn.split("__")
        chunk_number = int(chunk_number)
    else:
        chunk_number = 1

    folder = "/".join(file_path.split("/")[:-1])
    with open(file_path, "r") as f:
        lines = f.readlines()

    total_size = sum(len(line.encode("utf-8")) for line in lines)
    if total_size < 1.02*max_size_bytes: # the extra bit for the last entry that might have surpassed a tiny bit
        print(f"{file_path} skipped (under {max_size_mb}MB)")
        return

    current_size = 0
    current_chunk = []
    for line in lines:
        current_size += len(line.encode("utf-8"))
        if current_size > max_size_bytes:
            print(f"Creating {os.path.join(folder, f'{base_fn}__{chunk_number}.jsonl')}")
            with open(os.path.join(folder, f"{base_fn}__{chunk_number}.jsonl"), "w") as f:
                f.writelines(current_chunk)
            current_chunk = []
            current_size = 0
            chunk_number += 1
        current_chunk.append(line)
    if current_chunk:
        print(f"Creating {os.path.join(folder, f'{base_fn}__{chunk_number}.jsonl')}")
        with open(os.path.join(folder, f"{base_fn}__{chunk_number}.jsonl"), "w") as f:
            f.writelines(current_chunk)

    # remove the original file
    os.remove(file_path)


def split_files_in_folder(folder):
    for fn in os.listdir(folder):
        if fn.endswith(".jsonl"):
            split_large_file(os.path.join(folder, fn))
