import argparse
from datetime import datetime

from trajectories import TrajectoryProcessor
from utils.config_loader import ALL_STEPS, ALL_TASKS, TASK_STEPS, load_config
from utils.logger import logger
from weather import MetarProcessor, WeatherProcessor


def main():
    parser = argparse.ArgumentParser(description="FlightFusion data pipeline")
    parser.add_argument("--airports", help="One or more airport ICAO codes (e.g. EDDM,EDDL)")
    parser.add_argument("--start", type=str, help="Start date/time (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--end", type=str, help="End date/time (YYYY-MM-DDTHH:MM)")
    parser.add_argument("--task", choices=ALL_TASKS, default="all", help="Task to execute in the pipeline.")
    parser.add_argument("--step", choices=ALL_STEPS, default="all", help=f"Which step to execute from the specified task. Step 'all' is available for every task. For task 'flights': {', '.join(TASK_STEPS['flights'])}. For task 'metar': {', '.join(TASK_STEPS['metar'])}. For task 'weather': {', '.join(TASK_STEPS['weather'])}.")
    parser.add_argument("--config", default="debug_config.yaml", help="Path to config file")

    args = parser.parse_args()
    cfg = load_config(args.config, args)


    airports = cfg["airports"]
    start_dt = datetime.fromisoformat(cfg["start"])
    end_dt = datetime.fromisoformat(cfg["end"])
    task = cfg["task"]
    step = cfg["step"]

    logger.info(f"🚀 Starting FlightFusion pipeline for airports: {airports}, from {start_dt} to {end_dt}\n")

    if task in ["metar", "all"]:
        for icao in airports:
            logger.info(f"==================== Running task '{task}' for {icao} ====================\n")

            metar_processor = MetarProcessor(icao, start_dt, end_dt, cfg["radius_km"], cfg["output_dir"])
            if step in ["all", "download"]:
                metar_processor.download()
            if step in ["all", "parse"]:
                metar_processor.parse()
            if step in ["all", "process"]:
                metar_processor.process()

    if task in ["weather", "all"]:
        for icao in airports:
            logger.info(f"==================== Running task '{task}' for {icao} ====================\n")

            weather_processor = WeatherProcessor(icao, start_dt, end_dt, cfg["radius_km"], cfg["output_dir"], cfg["weather"])
            if step in ["all", "download"]:
                weather_processor.download()
            if step in ["all", "merge"]:
                weather_processor.merge()

    if task in ["flights", "all"]:
        for icao in airports:
            logger.info(f"==================== Running task '{task}' for {icao} ====================\n")

            trajectory_processor = TrajectoryProcessor(icao, start_dt, end_dt, cfg["radius_km"], cfg["output_dir"], cfg["flights"])
            if step in ["all", "download_flightlist"]:
                trajectory_processor.download_flightlist()
            if step in ["all", "download_trajectories"]:
                trajectory_processor.download_trajectories()
            if step in ["all", "process"]:
                trajectory_processor.process()


    logger.info("=======================================================================\n")
    logger.info("✅ All tasks completed successfully.\n")


if __name__ == "__main__":
    main()