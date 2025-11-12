import os

import yaml

TASK_STEPS = {
    "flights": ["download_flightlist", "download_trajectories", "process", "all"],
    "metar": ["download", "parse", "process", "all"],
    "weather": ["download", "process", "all"],
    "all": ["all"],
}
ALL_STEPS = set(step for steps in TASK_STEPS.values() for step in steps)
ALL_TASKS = list(TASK_STEPS.keys())

def load_config(config_path, cli_args):
    task, step = validate_task_step(cli_args)

    config_path = os.path.join("configs", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLI Args take precedence over config file
    merged_config = {
        "airports": cli_args.airports.split(",") if cli_args.airports else config["defaults"]["airports"],
        "start": cli_args.start or config["defaults"]["start"],
        "end": cli_args.end or config["defaults"]["end"],
        "output_dir": config["defaults"]["output_dir"],
        "task": task,
        "step": step,
        "flights": config["flights"], 
    }
    return merged_config    

def validate_task_step(args):
    task = args.task
    step = args.step

    if task not in TASK_STEPS:
        print(f"Error: Unknown task '{task}'. Allowed tasks: {', '.join(TASK_STEPS.keys())}")
        exit(1)

    if step not in TASK_STEPS[task]:
        print(f"Error: Step '{step}' is not valid for task '{task}'. Allowed steps: {', '.join([s for s in TASK_STEPS[task] if s is not None])}")
        exit(1)

    return task, step
