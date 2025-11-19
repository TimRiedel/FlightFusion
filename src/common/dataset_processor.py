import os
from datetime import datetime

import pandas as pd


class DatasetProcessor:
    def __init__(self, icao: str, start_dt: datetime, end_dt: datetime, radius_km: int, task_asset_dir: str, cfg: dict, create_raw_data_dir: bool = True):
        self.icao = icao.upper()
        self.start_dt = pd.to_datetime(start_dt).tz_localize("UTC") # enforce UTC timezone
        self.end_dt = pd.to_datetime(end_dt).tz_localize("UTC")
        self.radius_m = radius_km * 1000

        self.task_asset_dir = task_asset_dir
        self.output_dir = os.path.join(task_asset_dir, self.icao)
        os.makedirs(self.output_dir, exist_ok=True)

        if create_raw_data_dir:
            self.raw_data_dir = os.path.join(self.output_dir, "raw")
            os.makedirs(self.raw_data_dir, exist_ok=True)

        self.all_days = pd.date_range(start=self.start_dt, end=self.end_dt, freq="D")

    # --------------------
    # Utility
    # --------------------
    def _get_raw_file_path_for(self, data_type: str, day: datetime = None) -> str:
        if day is None:
            return os.path.join(
                self.raw_data_dir,
                f"{self.icao}_{data_type}_{self.start_dt.date()}_{self.end_dt.date()}.parquet"
            )
        return os.path.join(self.raw_data_dir, f"{self.icao}_{data_type}_{day.date()}.parquet")

    def _get_output_file_path_for(self, data_type: str) -> str:
        return os.path.join(self.output_dir, f"{self.icao}_{data_type}_{self.start_dt.date()}_{self.end_dt.date()}.parquet")

    def _save_data(self, df: pd.DataFrame, path: str, sortby: str = None):
        if sortby:
            df.sort_values(sortby, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(path)

    def _merge_files(self, daily_file_paths: list[str], sortby: str = None) -> pd.DataFrame:
        merged = pd.concat([pd.read_parquet(p) for p in daily_file_paths], ignore_index=True)
        if sortby:
            merged.sort_values(sortby, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _load_data(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")
        return pd.read_parquet(path)