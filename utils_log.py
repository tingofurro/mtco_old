from bson.objectid import ObjectId
import json, git


def log_conversation(log_file, task_name, task_id, dataset_fn, assistant_model, user_model, trace, additional_info={}):
    dataset_fn = dataset_fn.split("/")[-1]

    git_version = git.Repo(search_parent_directories=True).head.object.hexsha

    record = {"conv_id": str(ObjectId()), "task": task_name, "task_id": task_id, "dataset_fn": dataset_fn, "assistant_model": assistant_model, "user_model": user_model, "git_version": git_version, "trace": trace}
    record.update(additional_info) # sample-specific, for example for recap
    with open(log_file, "a") as f:
        f.write(json.dumps(record)+"\n")
