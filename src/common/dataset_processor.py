import os
import logging
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from traffic.data import airports

from common.projections import get_circle_around_location

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration object for data processing tasks."""
    icao_code: str
    start_dt: datetime
    end_dt: datetime
    circle_radius_km: int
    dataset_dir: str
    cache_dir: str


class DatasetProcessor:
    def __init__(self, processing_config: ProcessingConfig, task_type: str, task_config: dict = {}, create_temp_dir: bool = True):
        self.icao = processing_config.icao_code.upper()
        self.start_dt = pd.to_datetime(processing_config.start_dt).tz_localize("UTC") # enforce UTC timezone
        self.end_dt = pd.to_datetime(processing_config.end_dt).tz_localize("UTC")
        self.radius_m = processing_config.circle_radius_km * 1000
        self.task_config = task_config

        lat, lon = airports[self.icao].latlon
        self.airport_circle = get_circle_around_location(lat, lon, self.radius_m)

        self.output_dir = os.path.join(processing_config.dataset_dir, self.icao, task_type)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create .temp directory for intermediate files
        if create_temp_dir:
            self.temp_dir = os.path.join(self.output_dir, ".temp")
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = None

        self.all_days = pd.date_range(start=self.start_dt, end=self.end_dt, freq="D")

    # --------------------
    # I/O Functions
    # --------------------
    def _get_temp_file_path_for(self, data_type: str, day: datetime = None, extension: str = "parquet") -> str:
        """Get path for temporary/intermediate files (stored in .temp directory)"""
        if day is None:
            return os.path.join(
                self.temp_dir,
                f"{self.icao}_{data_type}_{self.start_dt.date()}_{self.end_dt.date()}.{extension}"
            )
        return os.path.join(self.temp_dir, f"{self.icao}_{data_type}_{day.date()}.{extension}")

    def _get_output_file_path_for(self, data_type: str, extension: str = "parquet") -> str:
        """Get path for final output files (stored directly in airport directory)"""
        return os.path.join(self.output_dir, f"{self.icao}_{data_type}_{self.start_dt.date()}_{self.end_dt.date()}.{extension}")

    def _save_data(self, df: pd.DataFrame, path: str, sortby: str = None):
        """Save DataFrame to parquet file."""
        if sortby:
            df.sort_values(sortby, inplace=True)
        df.reset_index(drop=True, inplace=True)
        try:
            df.to_parquet(path)
        except (MemoryError, OSError) as e:
            # If save fails due to OOM or disk issues, remove incomplete file
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as remove_error:
                    logger.error(f"    ✗ Failed to save data to {path} and could not remove incomplete file: {remove_error}")
            else:
                logger.error(f"    ✗ Failed to save data to {path}: {e}")
            raise

    def _merge_files(self, daily_file_paths: list[str], sortby: str = None) -> pd.DataFrame:
        """Merge daily parquet files into a single DataFrame."""
        merged = pd.concat([pd.read_parquet(p) for p in daily_file_paths], ignore_index=True)
        if sortby:
            merged.sort_values(sortby, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load DataFrame from parquet file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")
        return pd.read_parquet(path)