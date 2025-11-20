import contextlib
import io
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from common.dataset_processor import DatasetProcessor, ProcessingConfig
from traffic.core.flight import Flight
from traffic.data import opensky
from utils.logger import logger


class TrajectoryProcessor(DatasetProcessor):
    def __init__(self, processing_config: ProcessingConfig, task_config: dict):
        super().__init__(processing_config, task_type="flights", task_config=task_config, create_temp_dir=True)

        self.check_runway_alignment = task_config.get("check_runway_alignment", True)
        self.excluded_callsigns = task_config.get("excluded_callsigns", [])

    # ------------------------------------
    # Download Step 1: Flight List
    # ------------------------------------

    def download_flightlist(self):
        logger.info(f"📥 Downloading flight lists for {self.icao}...")

        flightlist_df = self._get_flightlist()

        logger.info(f"    → Processing flight list from {self.start_dt} to {self.end_dt}") 
        flightlist_df = self._remove_invalid(flightlist_df)
        flightlist_df = self._remove_departures(flightlist_df)
        flightlist_df = self._remove_same_departure_arrival(flightlist_df)

        path = self._get_temp_file_path_for("flightlist")
        self._save_data(flightlist_df, path, sortby="lastseen")
        logger.info(f"✅ Processed {len(flightlist_df)} flights, saved flightlist to {path}\n")

    def _get_flightlist(self) -> pd.DataFrame:
        flightlist_path = self._get_temp_file_path_for("flightlist")

        if os.path.exists(flightlist_path):
            logger.info(f"    ✓ Found existing raw flightlist data, skipping download.")
            flightlist_df = pd.read_parquet(flightlist_path)
            return flightlist_df
        else:
            logger.info(f"    → Fetching flight list from {self.start_dt} to {self.end_dt}") 
            flightlist_df = self._fetch_flight_list(self.icao, self.start_dt, self.end_dt)
            flightlist_df["flight_id"] = flightlist_df["callsign"].fillna("").str.strip() + "_" + flightlist_df["lastseen"].dt.strftime("%Y%m%d%H%M%S")

            self._save_data(flightlist_df, flightlist_path, sortby="lastseen")
            logger.info(f"        ✓ {len(flightlist_df)} flights saved to {flightlist_path}")
            return flightlist_df

    def _fetch_flight_list(self, icao: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        show_output = logger.isEnabledFor(logging.DEBUG)

        if show_output:
            return opensky.flightlist(
                airport=icao,
                start=start_dt.strftime("%Y-%m-%d 00:00:00"),
                stop=end_dt.strftime("%Y-%m-%d 23:59:59")
            )
        else:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                return opensky.flightlist(
                    airport=icao,
                    start=start_dt.strftime("%Y-%m-%d 00:00:00"),
                    stop=end_dt.strftime("%Y-%m-%d 23:59:59")
                )

    def _remove_invalid(self, df: pd.DataFrame):
        return df.dropna().drop_duplicates()

    def _remove_departures(self, df: pd.DataFrame):
        return df[df["arrival"] == self.icao]

    def _remove_same_departure_arrival(self, df: pd.DataFrame):
        return df[df["departure"] != df["arrival"]]

    def _remove_by_callsign(self, df: pd.DataFrame):
        removal_condition = (
            ~df["callsign"].str.match(r"^[A-Za-z]{3}.*\d") | # this query does not seem to work, as N81 is also detected as valid
            df["callsign"].str[:3].isin(self.excluded_callsigns)
        )
        return df[~removal_condition]


    # ------------------------------------
    # Download Step 2: Trajectories
    # ------------------------------------

    def download_trajectories(self):
        logger.info(f"📡 Downloading trajectories for {self.icao}...")

        flightlist_df = self._load_flightlist_data()
        daily_paths = []

        for day, group in flightlist_df.groupby("_date"):
            logger.info(f"    → Processing {len(group)} flights for {day}")
            day_trajs = []

            for _, row in group.iterrows():
                flight = self._fetch_flight_trajectory(row["callsign"], row["firstseen"], row["lastseen"])
                if not flight or flight.data is None or flight.data.empty:
                    logger.warning(f"        ⚠️ No trajectory data for flight {row['callsign']} from {row['firstseen']} to {row['lastseen']}. Skipping.")
                    continue
                logger.info(f"        ✓ Fetched trajectory for flight {flight.callsign}.")

                if self.check_runway_alignment: # TODO: this does not work for departures
                    rwy = self._check_runway_alignment(flight, row["firstseen"], row["lastseen"])
                    if rwy is None:
                        logger.warning(f"        ⚠️ No runway alignment found for flight {flight.callsign} on {row["lastseen"].date()}. Skipping.")
                    flight.data["rwy"] = rwy

                day_trajs.append(flight.data)

            day_merged = pd.concat(day_trajs, ignore_index=True)
            day_path = self._get_temp_file_path_for("trajectories", pd.to_datetime(day))
            self._save_data(day_merged, day_path, sortby="timestamp")
            logger.info(f"        ✓ Saved {len(day_merged)} trajectory points for {day} to {day_path}")
            daily_paths.append(day_path)

        merged = self._merge_files(daily_paths, sortby="timestamp")
        out_path = self._get_output_file_path_for("trajectories")
        self._save_data(merged, out_path)
        logger.info(f"✅ Saved merged trajectories to {out_path}")

    def _load_flightlist_data(self):
        flightlist_path = self._get_temp_file_path_for("flightlist")
        flightlist_df = self._load_data(flightlist_path)
        flightlist_df["_date"] = pd.to_datetime(flightlist_df["lastseen"]).dt.date
        return flightlist_df

    def _fetch_flight_trajectory(self, callsign, firstseen, lastseen):
        flight = opensky.history(
            start=firstseen,
            stop=lastseen,
            callsign=callsign,
            return_flight=True
        )
        return flight if flight is not None else None

    def _check_runway_alignment(self, flight: Flight, firstseen: datetime, lastseen: datetime):
        first_ts = firstseen.timestamp()
        last_ts = lastseen.timestamp()
        flight_last_3min = flight.skip(seconds=(last_ts - first_ts - 180))
        if flight_last_3min is not None:
            rwy = flight_last_3min.aligned_on_ils(self.icao, angle_tolerance=0.1, min_duration="40sec").final()
            if rwy is not None:
                return rwy.max("ILS")
        return None

    # ------------------------------------
    # Process Step
    # ------------------------------------

    def process(self):
        # TODO: remove all flights starting before start_dt

        # trajectories_df = self._remove_by_callsign(flightlist_df)
        # trajectories_df = self._remove_firstseen_before_startdate(trajectories_df)
        pass

    def _remove_firstseen_before_startdate(self, df: pd.DataFrame):
        return df[df["firstseen"] >= self.start_dt]