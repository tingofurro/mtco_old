from tasks.database import TaskDatabase
from tasks.math import TaskMath
from tasks.code import TaskCode
from tasks.summary import TaskSummary
from tasks.data2text import TaskData2Text
from tasks.actions import TaskActions
from tasks.translation import TaskTranslation
from tasks.flipflop import TaskFlipflop

def get_task(task_name, version=None):
    kwargs = {}
    if version is not None:
        kwargs["version"] = version

    if task_name.startswith("database"):
        return TaskDatabase(**kwargs)
    elif task_name == "code":
        return TaskCode(**kwargs)
    elif task_name == "translation":
        return TaskTranslation(**kwargs)
    elif task_name == "summary":
        return TaskSummary(**kwargs)
    elif task_name == "data2text":
        return TaskData2Text(**kwargs)
    elif task_name == "math":
        return TaskMath(**kwargs)
    elif task_name.startswith("actions"):
        return TaskActions(**kwargs)
    elif task_name == "flipflop":
        return TaskFlipflop(**kwargs)
    else:
        raise ValueError(f"Task {task_name} not supported")

def get_all_supported_task_names():
    """Return a list of all supported task names"""
    return [
        "database",
        "code",
        "translation",
        "summary",
        "data2text",
        "math",
        "actions",
        "flipflop"
    ]

if __name__ == "__main__":
    task = get_task("data2text")
    print(len(task.get_samples()))
