
from abc import ABC, abstractmethod

import cdsapi
import xarray as xr

from utils.logger import logger

class WeatherDownloader(ABC):
    def __init__(self, icao: str, dataset_name: str, output_dir: str):
        self.icao = icao.upper()
        self.dataset_name = dataset_name
        self.output_dir = output_dir

    @abstractmethod
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        pass

    @abstractmethod
    def retrieve_xr_dataset_from_grib(self, grib_path: str) -> xr.Dataset:
        pass

class CerraDownloader(WeatherDownloader):
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        dataset = "reanalysis-cerra-pressure-levels"
        request = {
            "variable": variables,
            "pressure_level": pressure_levels,
            "data_type": ["reanalysis"],
            "product_type": ["forecast"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "leadtime_hour": ["6", "9"],
            "data_format": "grib"
        }
        
        try:
            cdsapi.Client().retrieve(dataset, request).download(output_path)
        except Exception as e:
            logger.error(f"✗ Error downloading {year}-{month:02d}: {e}")
            raise

    def retrieve_xr_dataset_from_grib(self, grib_path: str) -> xr.Dataset:
        ds_forecast = xr.load_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        ds_forecast = ds_forecast.drop_vars(['valid_time'])

        merged_ds = None
        for time in ds_forecast["time"].values:
            for step in ds_forecast["step"].values:
                hourly_data = ds_forecast.sel(time=time, step=step)
                hourly_data = hourly_data.drop_vars(["time", "step"])
                hourly_data = hourly_data.expand_dims({"time": [time + step]})
                if merged_ds is None:
                    merged_ds = hourly_data
                else:
                    merged_ds = xr.concat([merged_ds, hourly_data], dim="time", coords="minimal")

        merged_ds = merged_ds.sortby("time")
        merged_ds = merged_ds.rename({'isobaricInhPa': 'level'})
        return merged_ds

class Era5Downloader(WeatherDownloader):
    def fetch_month(self, year: int, month: int, days: list[str], variables: list[str], pressure_levels: list[int], output_path: str):
        dataset = "reanalysis-era5-pressure-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "pressure_level": pressure_levels,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{day:02d}" for day in days],
            "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
            "area": [75, -35, 20, 45], # Europe (norht, west, south, east)
            "data_format": "grib",
            "download_format": "unarchived"
        }

        try:
            cdsapi.Client().retrieve(dataset, request).download(output_path)
        except Exception as e:
            logger.error(f"✗ Error downloading {year}-{month:02d}: {e}")
            raise

    def retrieve_xr_dataset_from_grib(self, grib_path: str) -> xr.Dataset:
        ds = xr.load_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        ds = ds.rename({'isobaricInhPa': 'level'})
        ds = ds.drop_vars(['number', 'step', 'valid_time'])
        return ds