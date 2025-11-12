import os
import copy
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
    merged_config = copy.deepcopy(config)

    merged_config = {**config["defaults"], **merged_config} # do not nest defaults under 'defaults', but make them top-level
    merged_config["airports"] = cli_args.airports.split(",") if cli_args.airports else merged_config["airports"]
    merged_config["start"] = cli_args.start or merged_config["start"] # allow CLI args to override config file
    merged_config["end"] = cli_args.end or merged_config["end"] # allow CLI args to override config file
    merged_config["task"] = task # allow CLI args to override config file
    merged_config["step"] = step # allow CLI args to override config file
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
