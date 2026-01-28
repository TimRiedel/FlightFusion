import os
from datetime import datetime
import warnings

import pandas as pd
from common.dataset_processor import DatasetProcessor, ProcessingConfig
from traffic.core import Traffic, Flight
from traffic.data import airports

from common.dataset_processor import DatasetProcessor
from trajectories.processing import *
from utils.logger import logger
from utils.cache import Cache

# Suppress the RuntimeWarning about numexpr engine fallback
warnings.filterwarnings('ignore', message='.*numexpr does not support extension array dtypes.*', category=RuntimeWarning)


class TrajectoryProcessor(DatasetProcessor):
    def __init__(self, processing_config: ProcessingConfig, task_config: dict):
        super().__init__(processing_config, task_type="trajectories", task_config=task_config, create_temp_dir=True)
        self.cache = Cache(processing_config.cache_dir, "trajectories", self.icao)

        self.traffic_type = task_config["traffic_type"]
        self.crop_to_circle = task_config["crop_to_circle"]
        self.drop_attributes = task_config.get("drop_attributes", [])
        self.process_config = task_config["process"]
        self.create_training_data_config = task_config["create_training_data"]

    # ------------------------------------
    # Utility
    # ------------------------------------
    
    def _get_request_config(self, day_start_dt: datetime, day_end_dt: datetime):
        request_config = {
            "icao": self.icao,
            "start_dt": day_start_dt,
            "end_dt": day_end_dt,
            "radius_m": self.radius_m,
            "traffic_type": self.traffic_type
        }
        return request_config

    def _load_traffic(self, path: str) -> Traffic:
        traffic = Traffic(self._load_data(path))
        if traffic.data.empty:
            raise ValueError(f"Traffic data is empty at {path}.")
        return traffic

    # ------------------------------------
    # Step 1: Download trajectories
    # ------------------------------------

    def download_trajectories(self):
        logger.info(f"📡 Downloading trajectories for {self.icao}...")
        all_trajectories_path = self._get_temp_file_path_for(f"trajectories-raw")
        if os.path.exists(all_trajectories_path):
            logger.info(f"    ✓ Found existing trajectories file under {all_trajectories_path}, skipping download.")
            logger.info(f"✅ Finished downloading trajectories for {self.icao}.\n")
            return

        all_traffic_dfs = []
        for day in self.all_days:
            day_start_dt = day.strftime("%Y-%m-%d 00:00:00")
            day_end_dt = day.strftime("%Y-%m-%d 23:59:59")
            lat, lon = airports[self.icao].latlon

            logger.info(f"    - Downloading trajectories for day {day.strftime('%Y-%m-%d')}.")
            request_config = self._get_request_config(day_start_dt, day_end_dt)
            cache_path, exists_cached = self.cache.get_file_path(request_config)
            if exists_cached:
                logger.info(f"        ✓ Found cached trajectories for day {day.strftime('%Y-%m-%d')} under {cache_path}, skipping download.")
                traffic = Traffic(self._load_data(cache_path))
            else:
                traffic = download_traffic(self.icao, day_start_dt, day_end_dt, self.airport_circle, traffic_type=self.traffic_type)
                self._save_data(traffic.data, cache_path)

            traffic = drop_irrelevant_attributes(traffic, self.drop_attributes)
            traffic = assign_flight_id(traffic)
            traffic = assign_distance_to_target(traffic, lat, lon)
            if self.crop_to_circle:
                traffic = crop_traffic_to_circle(traffic, self.airport_circle)

            logger.info(f"        ✓ Saved trajectories for day {day.strftime('%Y-%m-%d')} to cache {cache_path}.")
            all_traffic_dfs.append(traffic.data)

        logger.info(f"    - Concatenating trajectories for all days into a single file...")
        all_traffic = Traffic(pd.concat(all_traffic_dfs, ignore_index=True))
        self._save_data(all_traffic.data, all_trajectories_path)
        logger.info(f"        ✓ Saved trajectories for all days to {all_trajectories_path}.")

        logger.info(f"✅ Finished downloading trajectories for {self.icao}. Saved to {all_trajectories_path}.\n")


    # ------------------------------------
    # Step 2: Clean trajectories
    # ------------------------------------

    def clean_trajectories(self):
        logger.info(f"🧹 Cleaning trajectories for {self.icao}...")

        cleaned_trajectories_path = self._get_temp_file_path_for("trajectories-cleaned")
        if self._check_current_step_file_exists(cleaned_trajectories_path, "cleaning"):
            return

        all_trajectories_path = self._get_temp_file_path_for("trajectories-raw")
        self._ensure_previous_step_file_exists(all_trajectories_path, "download")

        traffic = self._load_traffic(all_trajectories_path)
        traffic = filter_traffic_by_type(traffic, self.traffic_type)
        if traffic.data.empty:
            logger.warning(f"    ✗ Trajectories is empty for traffic type {self.traffic_type}.")
            return

        logger.info(f"    - Filtering trajectories by airlines...")
        traffic = self._filter_traffic_by_airlines(traffic)

        logger.info(f"    - Cleaning trajectories (removing outliers, duplicates, NaN values)...")
        traffic = self._clean_trajectories(traffic)

        logger.info(f"    - Saving cleaned trajectories...")
        self._save_data(traffic.data, cleaned_trajectories_path)
        logger.info(f"✅ Finished cleaning trajectories for {self.icao}. Saved to {cleaned_trajectories_path}.\n")
    
    def _filter_traffic_by_airlines(self, traffic: Traffic) -> Traffic:
        filter_traffic_by_airlines_config = self.process_config.get("filter_traffic_by_airlines", {})
        if filter_traffic_by_airlines_config.get("enabled", True):
            excluded_callsigns = filter_traffic_by_airlines_config.get("excluded_callsigns", [])
            top_n_airlines = filter_traffic_by_airlines_config.get("top_n_airlines", None)
            traffic = filter_traffic_by_most_common_airlines(traffic, excluded_callsigns=excluded_callsigns, top_n_airlines=top_n_airlines)
        return traffic

    def _clean_trajectories(self, traffic: Traffic) -> Traffic:
        clean_config = self.task_config.get("clean", {})
        processed_flight_dfs = []
        for flight in traffic:
            logger.info(f"        - Cleaning flight {flight.flight_id}...")
            processed_flight = remove_nan_values(flight, attribute="altitude")
            processed_flight = remove_nan_values(processed_flight, attribute="latitude")
            processed_flight = remove_nan_values(processed_flight, attribute="longitude")
            
            processed_flight = remove_duplicate_positions(processed_flight)
            
            # Remove altitude outliers
            alt_outlier_config = clean_config.get("remove_outliers_altitude", {})
            processed_flight = remove_outliers(
                processed_flight, 
                "altitude", 
                alt_outlier_config.get("threshold", 300),
                window_size=alt_outlier_config.get("window_size", 20),
                handle_outliers=alt_outlier_config.get("handle_outliers", "drop")
            )
            
            # Remove groundspeed outliers
            gs_outlier_config = clean_config.get("remove_outliers_groundspeed", {})
            processed_flight = remove_outliers(
                processed_flight, 
                "groundspeed", 
                gs_outlier_config.get("threshold", 20),
                window_size=gs_outlier_config.get("window_size", 10),
                handle_outliers=gs_outlier_config.get("handle_outliers", "drop")
            )
            
            # Remove lateral outliers
            lateral_config = clean_config.get("remove_lateral_outliers", {})
            processed_flight = remove_lateral_outliers(
                processed_flight,
                window_size=lateral_config.get("window_size", 100),
                deviation_factor=lateral_config.get("deviation_factor", 10)
            )
            
            processed_flight = recompute_track(processed_flight)
            processed_flight = remove_nan_values(processed_flight, attribute="altitude")
            processed_flight = remove_nan_values(processed_flight, attribute="distance")
            processed_flight = remove_nan_values(processed_flight, attribute="track")
            processed_flight = remove_nan_values(processed_flight, attribute="groundspeed")
            processed_flight_dfs.append(processed_flight.data)

        traffic_df = pd.concat(processed_flight_dfs, ignore_index=True)
        traffic_df = self._round_values(traffic_df)
        return Traffic(pd.concat(processed_flight_dfs, ignore_index=True))

    # ------------------------------------
    # Step 3: Process trajectories
    # ------------------------------------

    def process_trajectories(self):
        logger.info(f"🔍 Processing trajectories for {self.icao}...")

        processed_trajectories_path = self._get_output_file_path_for("trajectories-processed")
        if self._check_current_step_file_exists(processed_trajectories_path, "processing"):
            return

        cleaned_trajectories_path = self._get_temp_file_path_for("trajectories-cleaned")
        self._ensure_previous_step_file_exists(cleaned_trajectories_path, "clean")

        traffic = self._load_traffic(cleaned_trajectories_path)
        logger.info(f"    - Removing invalid flights...")
        traffic, invalid_traffic = self._remove_invalid_flights(traffic, self.icao)

        logger.info(f"    - Saving processed and removed trajectories...")
        self._save_data(traffic.data, processed_trajectories_path)
        removed_trajectories_path = self._get_temp_file_path_for("trajectories-processed-removed-flights")
        self._save_data(invalid_traffic.data, removed_trajectories_path)
        logger.info(f"✅ Finished processing trajectories for {self.icao}. Saved\n    - Valid trajectories to {processed_trajectories_path}.\n    - Removed trajectories to {removed_trajectories_path}.\n")


    def _remove_invalid_flights(self, processed_traffic: Traffic, icao: str) -> tuple[Traffic, Traffic]:
        remove_config = self.process_config.get("remove_flights", {})
        removed_traffic_dfs = []
        
        # Remove flights with small duration
        small_duration_config = remove_config.get("remove_small_duration", {})
        if small_duration_config.get("enabled", True):
            logger.info(f"        - Removing flights with small duration...")
            processed_traffic, small_duration_flights, reasons = remove_flights_with_small_duration(
                processed_traffic, 
                threshold_seconds=small_duration_config.get("threshold_seconds", 60)
            )
            removed_traffic_dfs.append(small_duration_flights.data)
            self._log_removal_reasons(reasons)
        # Remove non-continuous flights
        non_continuous_config = remove_config.get("remove_non_continuous", {})
        if non_continuous_config.get("enabled", True):
            logger.info(f"        - Removing non-continuous flights...")
            processed_traffic, non_continuous_flights, reasons = remove_non_continous_flights(
                processed_traffic, 
                continuity_threshold_seconds=non_continuous_config.get("continuity_threshold_seconds", 120)
            )
            removed_traffic_dfs.append(non_continuous_flights.data)
            self._log_removal_reasons(reasons)
        
        # Apply arrival-specific removals if needed
        if self.traffic_type in ["arrivals", "all"]:
            arrival_traffic = filter_traffic_by_type(processed_traffic, "arrivals")
            departure_traffic = filter_traffic_by_type(processed_traffic, "departures") if self.traffic_type == "all" else None
            
            if not arrival_traffic.data.empty:
                # Remove flights with go-around or holding
                go_around_config = remove_config.get("remove_go_around", {})
                if go_around_config.get("enabled", True):
                    logger.info(f"        - Removing flights with go-around or holding...")
                    arrival_traffic, go_around_holding_flights, reasons = remove_flights_with_go_around_holding(
                        arrival_traffic, 
                        icao, 
                        track_threshold=go_around_config.get("track_threshold", 330),
                        time_window_seconds=go_around_config.get("time_window_seconds", 500)
                    )
                    removed_traffic_dfs.append(go_around_holding_flights.data)
                    self._log_removal_reasons(reasons)

                # Remove flights without runway alignment
                runway_alignment_config = remove_config.get("remove_without_runway_alignment", {})
                if runway_alignment_config.get("enabled", True):
                    logger.info(f"        - Removing flights without runway alignment...")
                    arrival_traffic, no_runway_alignment_flights, reasons = remove_flights_without_runway_alignment(
                        arrival_traffic, 
                        icao, 
                        angle_tolerance=runway_alignment_config.get("angle_tolerance", 0.1),
                        min_alignment_duration_seconds=runway_alignment_config.get("min_alignment_duration_seconds", 40)
                    )
                    removed_traffic_dfs.append(no_runway_alignment_flights.data)
                    self._log_removal_reasons(reasons)

            # Merge back with departures if needed
            if self.traffic_type == "all" and departure_traffic is not None:
                processed_traffic = merge_traffic(arrival_traffic, departure_traffic)
            else:
                processed_traffic = arrival_traffic
        
        removed_traffic = Traffic(pd.concat(removed_traffic_dfs, ignore_index=True)) if removed_traffic_dfs else Traffic(pd.DataFrame())
        return processed_traffic, removed_traffic
    
    def _round_values(self, traffic_df: pd.DataFrame) -> pd.DataFrame:
        traffic_df['latitude'] = traffic_df['latitude'].round(6)
        traffic_df['longitude'] = traffic_df['longitude'].round(6)
        traffic_df['altitude'] = traffic_df['altitude'].round(0).astype('int16')
        traffic_df['groundspeed'] = traffic_df['groundspeed'].round(0).astype('int16')
        traffic_df['vertical_rate'] = traffic_df['vertical_rate'].round(0).astype('int16')
        traffic_df['track'] = traffic_df['track'].round(3)
        traffic_df['distance'] = traffic_df['distance'].round(3)
        return traffic_df

    def _log_removal_reasons(self, reasons: list[str]):
        for reason in reasons:
            logger.info(f"            - {reason}")


    # ------------------------------------
    # Step 4: Create training data
    # ------------------------------------

    def create_training_data(self):
        logger.info(f"📊 Creating training data for {self.icao}...")

        inputs_path = self._get_output_file_path_for("trajectories-inputs")
        horizons_path = self._get_output_file_path_for("trajectories-horizons")
        if os.path.exists(inputs_path) and os.path.exists(horizons_path):
            logger.info(f"    ✓ Found existing input and horizon segments under {inputs_path} and {horizons_path}, skipping creation. To rerun this step, delete the files and run the method again.")
            return

        processed_trajectories_path = self._get_output_file_path_for("trajectories-processed")
        self._ensure_previous_step_file_exists(processed_trajectories_path, "process")

        is_sampling_enabled = self.create_training_data_config.get("sampling", {}).get("enabled", True)
        is_clipping_enabled = self.create_training_data_config.get("clipping", {}).get("enabled", False)
        if is_sampling_enabled and is_clipping_enabled:
            raise ValueError("Sampling and clipping cannot be enabled at the same time.")

        traffic = self._load_traffic(processed_trajectories_path)
        selected_runways = self.create_training_data_config.get("selected_runways", None)
        if selected_runways is not None and len(selected_runways) > 0:
            logger.info(f"    - Filtering trajectories by selected runways {selected_runways}...")
            traffic, _ = filter_traffic_by_runway(traffic, selected_runways)


        logger.info(f"    - Converting to metric units...")
        traffic = traffic.query("is_arrival == True").drop(columns=["is_arrival"])
        traffic = self._convert_to_metric_units(traffic)

        logger.info(f"    - Computing local coordinates...")
        ref_lat, ref_lon = airports[self.icao].latlon
        traffic = assign_local_xy_coordinates(traffic, ref_lat=ref_lat, ref_lon=ref_lon)

        resampling_rate_seconds = self.create_training_data_config["resampling_rate_seconds"]
        logger.info(f"    - Resampling trajectories in {resampling_rate_seconds} seconds intervals...")
        traffic = traffic.resample(f"{resampling_rate_seconds}s").eval()

        logger.info(f"    - Computing velocity components...")
        traffic = assign_velocity_components(traffic, resampling_rate_seconds=resampling_rate_seconds)

        if is_sampling_enabled:
            data_period_minutes = self.create_training_data_config["sampling"]["data_period_minutes"]
            logger.info(f"    - Drawing samples from trajectories in {data_period_minutes} minutes intervals...")
            traffic = sample_trajectories(
                traffic,
                data_period_minutes=data_period_minutes,
                min_trajectory_length_minutes=self.create_training_data_config["sampling"]["min_trajectory_length_minutes"],
            )

        if is_clipping_enabled:
            trajectory_length_minutes = self.create_training_data_config["clipping"]["trajectory_length_minutes"]
            logger.info(f"    - Clipping trajectories to {trajectory_length_minutes} minutes before end...")
            traffic, removal_reasons = clip_trajectories(
                traffic,
                trajectory_length_minutes=trajectory_length_minutes,
            )
            self._log_removal_reasons(removal_reasons)

        logger.info(f"    - Segmenting input and horizon segments...")
        input_time_minutes = self.create_training_data_config["input_time_minutes"]
        horizon_time_minutes = self.create_training_data_config["horizon_time_minutes"]
        if is_clipping_enabled:
            horizon_time_minutes = min(horizon_time_minutes, trajectory_length_minutes - input_time_minutes)
        self._log_segmenting_info(input_time_minutes, horizon_time_minutes, resampling_rate_seconds)
        input_segments, horizon_segments = get_input_horizon_segments(traffic,
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            resampling_rate_seconds=resampling_rate_seconds
        )

        logger.info(f"    - Saving input segments to {inputs_path}...")
        input_segments.data.to_parquet(inputs_path)
        logger.info(f"    - Saving horizon segments to {horizons_path}.")
        horizon_segments.data.to_parquet(horizons_path)

        logger.info(f"✅ Finished creating training data for {self.icao}.\n")

    def _log_segmenting_info(self, input_time_minutes: int, horizon_time_minutes: int, resampling_rate_seconds: int):
        num_input_points = input_time_minutes * 60 // resampling_rate_seconds
        num_horizon_points = horizon_time_minutes * 60 // resampling_rate_seconds
        logger.info(f"        -> Input time: {input_time_minutes} minutes, Horizon time: {horizon_time_minutes} minutes, Resampling rate: {resampling_rate_seconds} seconds")
        logger.info(f"        -> Number of input points: {num_input_points}, Number of horizon points: {num_horizon_points}")

    def _convert_to_metric_units(self, traffic: Traffic) -> Traffic:
        traffic.data["altitude"] = (traffic.data["altitude"] * 0.3048).round(0).astype(int)                    # ft to m
        traffic.data["vertical_rate"] = (traffic.data["vertical_rate"] * (0.3048 / 60)).round(2)               # ft/min to m/s
        traffic.data["groundspeed"] = (traffic.data["groundspeed"] * 1.852 / 3.6).round(2)                     # kts to m/s
        return traffic

    def _convert_to_aviation_units(self, traffic: Traffic) -> Traffic:
        traffic.data["altitude"] = (traffic.data["altitude"] / 0.3048).round(0).astype(int)                    # m to ft
        traffic.data["vertical_rate"] = (traffic.data["vertical_rate"] / (0.3048 / 60)).round(0).astype(int)   # m/s to ft/min
        traffic.data["groundspeed"] = (traffic.data["groundspeed"] * 3.6 / 1.852).round(2)                     # m/s to kts
        return traffic