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
                all_traffic_dfs.append(traffic.data)
                continue

            traffic = download_traffic(self.icao, day_start_dt, day_end_dt, self.airport_circle, traffic_type=self.traffic_type)
            traffic = drop_irrelevant_attributes(traffic, self.drop_attributes)
            traffic = assign_flight_id(traffic)
            traffic = assign_distance_to_target(traffic, lat, lon)
            if self.crop_to_circle:
                traffic = crop_traffic_to_circle(traffic, self.airport_circle)

            self._save_data(traffic.data, cache_path)
            logger.info(f"        ✓ Saved trajectories for day {day.strftime('%Y-%m-%d')} to cache {cache_path}.")
            all_traffic_dfs.append(traffic.data)

        logger.info(f"    - Concatenating trajectories for all days into a single file...")
        all_traffic = Traffic(pd.concat(all_traffic_dfs, ignore_index=True))
        self._save_data(all_traffic.data, all_trajectories_path)
        logger.info(f"        ✓ Saved trajectories for all days to {all_trajectories_path}.")

        logger.info(f"✅ Finished downloading trajectories for {self.icao}. Saved to {all_trajectories_path}.\n")


    # ------------------------------------
    # Step 2: Process trajectories
    # ------------------------------------

    def process_trajectories(self):
        logger.info(f"🧹 Processing trajectories for {self.icao}...")

        all_trajectories_path = self._get_temp_file_path_for("trajectories-raw")
        if not os.path.exists(all_trajectories_path):
            raise FileNotFoundError(f"Cached trajectories not found at {all_trajectories_path}. Please run the download method first.")

        traffic = Traffic(self._load_data(all_trajectories_path))
        traffic = filter_traffic_by_type(traffic, self.traffic_type)
        if traffic.data.empty:
            logger.warning(f"    ✗ Trajectories is empty for traffic type {self.traffic_type}.")
            return

        logger.info(f"    - Filtering trajectories by airlines...")
        traffic = self._filter_traffic_by_airlines(traffic)

        logger.info(f"    - Cleaning trajectories (removing outliers, duplicates, NaN values)...")
        traffic = self._clean_trajectories(traffic)

        logger.info(f"    - Removing invalid flights...")
        traffic, invalid_traffic = self._remove_invalid_flights(traffic, self.icao)

        logger.info(f"    - Assigning speed components...")
        traffic = self._compute_features(traffic)

        logger.info(f"    - Saving processed trajectories...")
        processed_trajectories_path = self._get_output_file_path_for("trajectories-processed")
        self._save_data(traffic.data, processed_trajectories_path)
        removed_trajectories_path = self._get_temp_file_path_for("trajectories-processed-removed")
        self._save_data(invalid_traffic.data, removed_trajectories_path)
        logger.info(f"✅ Finished processing trajectories for {self.icao}. Saved\n    - Valid trajectories to {processed_trajectories_path}.\n    - Removed trajectories to {removed_trajectories_path}.\n")

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
                # Remove flights without runway alignment
                runway_alignment_config = remove_config.get("remove_without_runway_alignment", {})
                if runway_alignment_config.get("enabled", True):
                    logger.info(f"        - Removing flights without runway alignment...")
                    arrival_traffic, no_runway_alignment_flights, reasons = remove_flights_without_runway_alignment(
                        arrival_traffic, 
                        icao, 
                        final_approach_time_seconds=runway_alignment_config.get("final_approach_time_seconds", 180),
                        angle_tolerance=runway_alignment_config.get("angle_tolerance", 0.1),
                        min_duration_seconds=runway_alignment_config.get("min_duration_seconds", 40)
                    )
                    removed_traffic_dfs.append(no_runway_alignment_flights.data)
                    self._log_removal_reasons(reasons)
                
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

            # Merge back with departures if needed
            if self.traffic_type == "all" and departure_traffic is not None:
                processed_traffic = merge_traffic(arrival_traffic, departure_traffic)
            else:
                processed_traffic = arrival_traffic
        
        removed_traffic = Traffic(pd.concat(removed_traffic_dfs, ignore_index=True)) if removed_traffic_dfs else Traffic(pd.DataFrame())
        return processed_traffic, removed_traffic
    
    def _compute_features(self, traffic: Traffic) -> Traffic:
        latitude, longitude = airports[self.icao].latlon
        flights = []
        for flight in traffic:
            flight = assign_speed_components(flight, longitude, latitude)
            flight = assign_remaining_track_miles(flight)
            flight = flight.drop(columns=["distance_to_next"])
            flights.append(flight)
        return Traffic.from_flights(flights)

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
    # Step 3: Create training data
    # ------------------------------------

    def create_training_data(self):
        logger.info(f"📊 Creating training data for {self.icao}...")

        all_trajectories_path = self._get_output_file_path_for("trajectories-processed")
        if not os.path.exists(all_trajectories_path):
            raise FileNotFoundError(f"Cached trajectories not found at {all_trajectories_path}. Please run the process and download methods first.")

        traffic = Traffic(self._load_data(all_trajectories_path))
        traffic = traffic.query("is_arrival == True").drop(columns=["is_arrival"])
        traffic = self._convert_to_metric_units(traffic)

        logger.info(f"    - Sampling trajectories...")
        self._log_sampling_info()
        traffic = sample_trajectories(traffic, resampling_rate_seconds=self.create_training_data_config["resampling_rate_seconds"], data_period_minutes=self.create_training_data_config["data_period_minutes"], min_trajectory_length_minutes=self.create_training_data_config["min_trajectory_length_minutes"])
        logger.info(f"    - Segmenting input and horizon segments...")
        input_segments, horizon_segments = get_input_horizon_segments(traffic, input_time_minutes=self.create_training_data_config["input_time_minutes"], horizon_time_minutes=self.create_training_data_config["horizon_time_minutes"], resampling_rate_seconds=self.create_training_data_config["resampling_rate_seconds"])

        output_file_path_inputs = self._get_output_file_path_for("trajectories-inputs")
        output_file_path_horizons = self._get_output_file_path_for("trajectories-horizons")
        logger.info(f"    - Saving input segments to {output_file_path_inputs}...")
        input_segments.data.to_parquet(output_file_path_inputs)
        logger.info(f"    - Saving horizon segments to {output_file_path_horizons}.")
        horizon_segments.data.to_parquet(output_file_path_horizons)

        logger.info(f"✅ Finished creating training data for {self.icao}.\n")

    def _log_sampling_info(self):
        input_time_minutes = self.create_training_data_config["input_time_minutes"]
        horizon_time_minutes = self.create_training_data_config["horizon_time_minutes"]
        resampling_rate_seconds = self.create_training_data_config["resampling_rate_seconds"]
        num_input_samples = input_time_minutes * 60 // resampling_rate_seconds
        num_horizon_samples = horizon_time_minutes * 60 // resampling_rate_seconds
        logger.info(f"        -> Input time: {input_time_minutes} minutes, Horizon: {horizon_time_minutes} minutes, Resampling rate: {resampling_rate_seconds} seconds")
        logger.info(f"        -> Number of input samples: {num_input_samples}, Number of horizon samples: {num_horizon_samples}")

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