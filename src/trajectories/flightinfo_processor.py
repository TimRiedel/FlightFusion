import numpy as np
import pandas as pd
from traffic.data import aircraft, airports

from common.projections import get_transformer_wgs84_to_aeqd
from common.dataset_processor import DatasetProcessor, ProcessingConfig
from utils.logger import logger


class FlightInfoProcessor(DatasetProcessor):
    def __init__(self, processing_config: ProcessingConfig, task_config: dict):
        super().__init__(processing_config, task_type="flightinfo", task_config=task_config, create_temp_dir=False)

    # ------------------------------------
    # Step 1: Extract flight info
    # ------------------------------------

    def extract(self):
        """Extract flight info."""
        logger.info(f"📋 Extracting flight info for {self.icao}...")
        flightinfo__output_path = self._get_output_file_path_for("flightinfo")
        if self._check_current_step_file_exists(flightinfo__output_path, "extract"):
            return

        processed_trajectories_path = self._get_output_file_path_for("trajectories-processed")
        processed_trajectories_path = processed_trajectories_path.replace("flightinfo", "trajectories")
        self._ensure_previous_step_file_exists(processed_trajectories_path, "trajectories/process")

        trajectories_df = self._load_data(processed_trajectories_path)

        logger.info(f"    - Aggregating flight info...")
        flight_info_df = self._aggregate_flight_info(trajectories_df)
        flight_info_df["aircraft_type"] = flight_info_df["icao24"].map(self._get_aircraft_type)
        flight_info_df["airline"] = flight_info_df["callsign"].str[:3]

        logger.info(f"    - Adding runway data...")
        flight_info_df = self._add_runway_data(flight_info_df)

        flight_info_df = flight_info_df[["flight_id", "airport", "runway", "callsign", "airline", "icao24", "aircraft_type", "rwy_x", "rwy_y", "rwy_bearing", "rwy_bearing_sin", "rwy_bearing_cos"]]

        self._save_data(flight_info_df, flightinfo__output_path)
        logger.info(f"✅ Finished extracting flight info for {self.icao}. Saved to {flightinfo__output_path}\n")

    def _aggregate_flight_info(self, trajectories_df: pd.DataFrame) -> pd.DataFrame:
        flight_info_df = trajectories_df.groupby("flight_id").agg(
            airport=("airport", "first"),
            icao24=("icao24", "first"),
            callsign=("callsign", "first"),
            runway=("ILS", "last")
        ).reset_index()
        return flight_info_df

    def _get_aircraft_type(self, icao24: str) -> str:
        acft = aircraft.get(icao24)
        if acft:
            return acft.typecode
        return None

    def _add_runway_data(self, flight_info_df: pd.DataFrame) -> pd.DataFrame:
        rwy_x_list, rwy_y_list, rwy_bearing_list = [], [], []

        for index, row in flight_info_df.iterrows():
            icao = row["airport"]
            airport_lat, airport_lon = airports[icao].latlon

            ils = row["runway"]
            runways = airports[icao].runways[ils].tuple_runway
            runway = next(t for t in runways if t.name == ils)
            if runway is None:
                raise ValueError(f"Runway data not found for {icao} and {ils}")
            rwy_lat, rwy_lon, rwy_bearing = runway.latitude, runway.longitude, runway.bearing
            rwy_x, rwy_y = get_transformer_wgs84_to_aeqd(ref_lat=airport_lat, ref_lon=airport_lon).transform(rwy_lon, rwy_lat)

            rwy_x_list.append(rwy_x)
            rwy_y_list.append(rwy_y)
            rwy_bearing_list.append(rwy_bearing)


        flight_info_df["rwy_x"] = rwy_x_list
        flight_info_df["rwy_y"] = rwy_y_list
        flight_info_df["rwy_bearing"] = rwy_bearing_list

        flight_info_df["rwy_bearing_sin"] = np.sin(flight_info_df["rwy_bearing"] * 2 * np.pi / 360)
        flight_info_df["rwy_bearing_cos"] = np.cos(flight_info_df["rwy_bearing"] * 2 * np.pi / 360)

        return flight_info_df
