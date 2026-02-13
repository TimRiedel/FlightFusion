import argparse
from datetime import datetime
import os
import shutil

import yaml

from common.dataset_processor import ProcessingConfig
from trajectories import FlightInfoProcessor, TrajectoryProcessor
from utils.config_loader import ALL_STEPS, ALL_TASKS, TASK_STEPS, load_config
from utils.logger import logger
from weather import MetarProcessor, WeatherProcessor


def log_config(cfg):
    logger.info("==================== Configuration ====================")
    logger.info(f"Dataset name: {cfg['dataset_name']}")
    logger.info(f"Task: {cfg['task']}")
    logger.info(f"Step: {cfg['step']}")
    logger.info(f"Airports: {cfg['airports']}")
    logger.info(f"Start: {cfg['start']}")
    logger.info(f"End: {cfg['end']}")
    if 'days_of_month' in cfg:
        logger.info(f"Days of month: {cfg['days_of_month']}")
    logger.info(f"Radius: {cfg['radius_km']}")
    logger.info(f"Dataset dir: {cfg['dataset_dir']}")
    logger.info(f"Cache dir: {cfg['cache_dir']}\n")

def save_config(dataset_dir, dataset_name, config_name):
    output_path = os.path.join(dataset_dir, dataset_name, "config.yaml")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy(f"configs/{config_name}", output_path)

def main():
    parser = argparse.ArgumentParser(description="FlightFusion data pipeline")
    parser.add_argument("--airports", help="One or more airport ICAO codes (e.g. EDDM,EDDL)")
    parser.add_argument("--start", type=str, help="Start date/time (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--end", type=str, help="End date/time (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--task", choices=ALL_TASKS, default="all", help="Task to execute in the pipeline.")
    parser.add_argument("--step", choices=ALL_STEPS, default="all", help=f"Which step to execute from the specified task. Step 'all' is available for every task. For task 'flights': {', '.join(TASK_STEPS['trajectories'])}. For task 'metar': {', '.join(TASK_STEPS['metar'])}. For task 'weather': {', '.join(TASK_STEPS['weather'])}.")
    parser.add_argument("--config", default="debug.yaml", help="Path to config file")

    args = parser.parse_args()
    cfg = load_config(args.config, args)
    save_config(cfg["dataset_dir"], cfg["dataset_name"], args.config)

    airports = cfg["airports"]
    start_dt = datetime.fromisoformat(cfg["start"])
    end_dt = datetime.fromisoformat(cfg["end"])
    task = cfg["task"]
    step = cfg["step"]

    logger.info(f"🚀 Starting FlightFusion pipeline for airports: {airports}, from {start_dt} to {end_dt}\n")
    log_config(cfg)


    for icao in airports:
        processing_config = ProcessingConfig(
            dataset_name=cfg["dataset_name"],
            icao_code=icao,
            start_dt=start_dt,
            end_dt=end_dt,
            days_of_month=cfg.get("days_of_month", None),
            circle_radius_km=cfg["radius_km"],
            dataset_dir=cfg["dataset_dir"],
            cache_dir=cfg["cache_dir"]
        )

        if task in ["metar", "all"]:
            logger.info(f"==================== Running task 'metar' for {icao} ====================\n")

            metar_processor = MetarProcessor(processing_config, cfg)
            if step in ["all", "download"]:
                metar_processor.download()
            if step in ["all", "parse"]:
                metar_processor.parse()
            if step in ["all", "process"]:
                metar_processor.process()

        if task in ["weather", "all"]:
            logger.info(f"==================== Running task 'weather' for {icao} ====================\n")

            weather_processor = WeatherProcessor(processing_config, cfg["weather"])
            if step in ["all", "download"]:
                weather_processor.download()
            if step in ["all", "merge"]:
                weather_processor.process()

        if task in ["trajectories", "all"]:
            logger.info(f"==================== Running task 'trajectories' for {icao} ====================\n")

            trajectory_processor = TrajectoryProcessor(processing_config, cfg["trajectories"])
            if step in ["all", "download"]:
                trajectory_processor.download_trajectories()
            if step in ["all", "clean"]:
                trajectory_processor.clean_trajectories()
            if step in ["all", "process"]:
                trajectory_processor.process_trajectories()
            if step in ["all", "create_training_data"]:
                trajectory_processor.create_training_data()

        if task in ["flightinfo", "all"]:
            logger.info(f"==================== Running task 'flightinfo' for {icao} ====================\n")

            flightinfo_processor = FlightInfoProcessor(processing_config, cfg.get("flightinfo", {}))
            if step in ["all", "extract"]:
                flightinfo_processor.extract()

    logger.info("=======================================================================\n")
    logger.info("✅ All tasks completed successfully.\n")


if __name__ == "__main__":
    main()