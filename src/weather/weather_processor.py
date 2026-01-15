import hashlib
import json
import os
from datetime import datetime, timedelta
from calendar import monthrange

import shapely
import xarray as xr
import xesmf as xe
import numpy as np
from traffic.data import airports

from weather.weather_downloader import CerraDownloader, Era5Downloader
from common.dataset_processor import DatasetProcessor, ProcessingConfig
from common.projections import Bounds, get_circle_around_location, get_curvilinear_grid_around_location
from utils.logger import logger
from utils.cache import Cache
from utils.output_capture import console_output_prefix

class WeatherProcessor(DatasetProcessor):
    def __init__(self, processing_config: ProcessingConfig, task_config: dict = {},):
        super().__init__(processing_config, task_type="weather", task_config=task_config, create_temp_dir=False)

        self.cache = Cache(processing_config.cache_dir, "weather", self.icao)
        
        self.dataset_name = task_config.get("dataset_name")
        self.variables = task_config.get("variables")
        self.pressure_levels = task_config.get("pressure_levels")

        if self.dataset_name == "cerra":
            # We are using 6-hourly forecast data from CERRA, so we need to start at least 6 hours before the start date.
            self.start_dt = self.start_dt - timedelta(days=1)
            self.weather_downloader = CerraDownloader(self.icao, self.dataset_name, self.output_dir)
        elif self.dataset_name == "era5":
            self.weather_downloader = Era5Downloader(self.icao, self.dataset_name, self.output_dir)
        else:
            raise ValueError(f"Invalid dataset: {self.dataset_name}")

        self.grid_n_x = task_config.get("grid_num_x")
        self.grid_n_y = task_config.get("grid_num_y")

    # ------------------------------------
    # Utils
    # ------------------------------------

    def _get_request_config(self, year: int, month: int):
        request_config = {
            "dataset_name": self.dataset_name,
            "variables": self.variables,
            "pressure_levels": self.pressure_levels,
            "start_dt": str(self.start_dt),
            "end_dt": str(self.end_dt),
            "year": year,
            "month": month,
        }
        return request_config

    def _get_months_to_download(self) -> list[tuple[int, int]]:
        """Get all (year, month) tuples that overlap with the date range."""
        months = []
        current = datetime(self.start_dt.year, self.start_dt.month, 1)
        end = datetime(self.end_dt.year, self.end_dt.month, 1)
        
        while current <= end:
            months.append((current.year, current.month))
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return months

    def _get_days_to_download(self, year: int, month: int) -> list[str]:
        """Get all days in the given month that fall within the date range."""
        _, num_days_in_month = monthrange(year, month)
        
        start_day = self.start_dt.day if (year == self.start_dt.year and month == self.start_dt.month) else 1
        end_day = self.end_dt.day if (year == self.end_dt.year and month == self.end_dt.month) else num_days_in_month
        
        days = [f"{day:02d}" for day in range(start_day, end_day + 1)]
        return days


    # ------------------------------------
    # Step 1: Download and crop monthly files and save as ZARR
    # ------------------------------------

    def download(self):
        logger.info(f"📥 Downloading {self.dataset_name.upper()} reanalysis data for {self.icao}...")
        months_to_download = self._get_months_to_download()
        
        for year, month in months_to_download:
            days = self._get_days_to_download(year, month)
            request_config = self._get_request_config(year, month)
            file_path, exists_cached = self.cache.get_file_path(request_config)

            if exists_cached:
                logger.info(f"    - Month {month:02d}-{year} already downloaded with same configuration. Skipping download.")
            else:
                logger.info(f"    - Downloading {len(days)} days for {month:02d}-{year}...")
                with console_output_prefix("        | "):
                    self.weather_downloader.fetch_month(year, month, days, self.variables, self.pressure_levels, file_path)
                logger.info(f"        ✓ Finished downloading. Saved GRIB to {file_path}.")

        logger.info(f"✅ Finished downloading {self.dataset_name.upper()} reanalysis data for {self.icao}.\n")


    # ------------------------------------
    # Step 2: Process and merge monthly weather data
    # ------------------------------------

    def process(self):
        logger.info(f"📦 Merging {self.dataset_name.upper()} reanalysis data...")
        output_path = self._get_output_file_path_for("weather", extension="zarr")
        if self._check_current_step_file_exists(output_path, "merging"):
            return

        # Ensure all monthly downloads exist before starting processing
        months = self._get_months_to_download()
        for year, month in months:
            request_config = self._get_request_config(year, month)
            file_path, exists_cached = self.cache.get_file_path(request_config)
            if not exists_cached:
                raise FileNotFoundError(f"Cached {self.dataset_name.upper()} data not found at {file_path}. Please run the download method first.")

        # Merge monthly data into a single Zarr file
        merged_exists = False
        for year, month in months:
            request_config = self._get_request_config(year, month)
            file_path, exists_cached = self.cache.get_file_path(request_config)
            logger.info(f"    - Processing month {month:02d}-{year}...")

            logger.info(f"        - Bringing GRIB data into the correct format...")
            month_ds = self.weather_downloader.retrieve_xr_dataset_from_grib(file_path)

            logger.info(f"        - Regridding and cropping data to airport circle bounds...")
            month_ds = self._regrid_dataset_to_airport(month_ds, exact_curvilinear_grid=True)
            month_ds.attrs["dataset"] = self.dataset_name

            if not merged_exists:
                month_ds.to_zarr(output_path, mode="w", consolidated=False, zarr_format=3)
                merged_exists = True
            else:
                month_ds.to_zarr(output_path, mode="a", consolidated=False, append_dim="time", zarr_format=3)
            logger.info(f"        ✓ Finished regridding and cropping. Merged Zarr to {output_path}.")

        logger.info(f"✅ Finished merging {self.dataset_name.upper()} weather data.\n")

    def _regrid_dataset_to_airport(self, ds: xr.Dataset, exact_curvilinear_grid: bool = True):
        lat, lon = airports[self.icao].latlon

        if exact_curvilinear_grid:
            lat2d, lon2d = get_curvilinear_grid_around_location(lat, lon, self.radius_m, num_x=self.grid_n_x, num_y=self.grid_n_y)
            regridded_ds = self._regrid_dataset_curvilinear(ds, lat2d, lon2d)
            return regridded_ds
        else:
            bounds = self._get_enclosing_weather_bounds(ds, self.airport_circle) 
            regridded_ds = self._regrid_dataset_regular(ds, bounds=bounds, num_lat=self.grid_n_x, num_lon=self.grid_n_y)
            return regridded_ds

    def _regrid_dataset_curvilinear(self, ds: xr.Dataset, lat2d: np.ndarray, lon2d: np.ndarray):
        ds_out = xr.Dataset(
            coords={
                "latitude": (["y", "x"], lat2d),
                "longitude": (["y", "x"], lon2d),
            }
        )

        regridder = xe.Regridder(ds, ds_out, method="bilinear", periodic=False)
        regridded_ds = regridder(ds)
        return regridded_ds
            
    def _regrid_dataset_regular(self, ds: xr.Dataset, bounds: Bounds, num_lat: int, num_lon: int):
        min_lon, min_lat, max_lon, max_lat = bounds.min_lon, bounds.min_lat, bounds.max_lon, bounds.max_lat
        res_lon = (max_lon - min_lon) / num_lon
        res_lat = (max_lat - min_lat) / num_lat
        margin = 1e-6

        ds_out = xr.Dataset(
            {
                "latitude": (["latitude"], np.arange(min_lat, max_lat + margin, res_lat)),
                "longitude": (["longitude"], np.arange(min_lon, max_lon + margin, res_lon)),
            }
        )
        regridder = xe.Regridder(ds, ds_out, "bilinear")
        regridded_ds = regridder(ds)
        return regridded_ds

    def _get_enclosing_weather_bounds(self, ds: xr.Dataset, circle: shapely.geometry.Polygon):
        min_lon, min_lat, max_lon, max_lat = circle.bounds

        ds_lats = ds.latitude.values
        ds_lons = ds.longitude.values

        min_lon_candidates = ds_lons[ds_lons <= min_lon]
        min_lat_candidates = ds_lats[ds_lats <= min_lat]
        max_lon_candidates = ds_lons[ds_lons >= max_lon]
        max_lat_candidates = ds_lats[ds_lats >= max_lat]

        min_lon = min_lon_candidates.max() if min_lon_candidates.size > 0 else ds_lons.min()
        min_lat = min_lat_candidates.max() if min_lat_candidates.size > 0 else ds_lats.min()
        max_lon = max_lon_candidates.min() if max_lon_candidates.size > 0 else ds_lons.max()
        max_lat = max_lat_candidates.min() if max_lat_candidates.size > 0 else ds_lats.max()

        return Bounds(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)