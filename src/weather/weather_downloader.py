
from abc import ABC, abstractmethod

import cdsapi

from utils.logger import logger

class WeatherDownloader(ABC):
    def __init__(self, icao: str, dataset_name: str, output_dir: str):
        self.icao = icao.upper()
        self.dataset_name = dataset_name
        self.output_dir = output_dir

    @abstractmethod
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        pass

class CerraDownloader(WeatherDownloader):
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        dataset = "reanalysis-cerra-pressure-levels"
        request = {
            "variable": variables,
            "pressure_level": pressure_levels,
            "data_type": ["reanalysis"],
            "product_type": ["analysis", "forecast"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "time": [
                "00:00", "03:00", "06:00",
                "09:00", "12:00", "15:00",
                "18:00", "21:00"
            ],
            "leadtime_hour": ["1", "2"],
            "data_format": "grib"
        }
        
        try:
            cdsapi.Client().retrieve(dataset, request).download(output_path)
        except Exception as e:
            logger.error(f"✗ Error downloading {year}-{month:02d}: {e}")
            raise

class Era5Downloader(WeatherDownloader):
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "pressure_level": pressure_levels,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "time": [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
            ],
            "data_format": "grib",
            "download_format": "unarchived"
        }

        try:
            cdsapi.Client().retrieve(dataset, request).download(output_path)
        except Exception as e:
            logger.error(f"✗ Error downloading {year}-{month:02d}: {e}")
            raise