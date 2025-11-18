import os
from datetime import datetime
from calendar import monthrange

import shapely
import xarray as xr
import xesmf as xe
import numpy as np
from traffic.data import airports

from weather.weather_downloader import CerraDownloader, Era5Downloader
from common.dataset_processor import DatasetProcessor
from common.projections import Bounds, get_circle_around_location, get_curvilinear_grid_around_location
from utils.logger import logger
from utils.output_capture import console_output_prefix

class WeatherProcessor(DatasetProcessor):
    def __init__(self, icao: str, start_dt: datetime, end_dt: datetime, radius_km: int, output_dir: str, cfg: dict = {},):
        output_dir = os.path.join(output_dir, "weather")
        super().__init__(icao, start_dt, end_dt, radius_km, output_dir, cfg)
        
        self.dataset_name = cfg.get("dataset_name")
        self.variables = cfg.get("variables")
        self.pressure_levels = cfg.get("pressure_levels")

        if self.dataset_name == "cerra":
            self.weather_downloader = CerraDownloader(icao, self.dataset_name, output_dir)
        elif self.dataset_name == "era5":
            self.weather_downloader = Era5Downloader(icao, self.dataset_name, output_dir)
        else:
            raise ValueError(f"Invalid dataset: {self.dataset_name}")

        self.grid_n_x = cfg.get("grid_num_x")
        self.grid_n_y = cfg.get("grid_num_y")

    # ------------------------------------
    # Utils
    # ------------------------------------

    def _get_raw_file_path_for(self, year: int, month: int, extension: str):
        return os.path.join(self.raw_data_dir, f"{self.icao}_{self.dataset_name}_{year}-{month:02d}.{extension}")

    def _get_output_file_path_for(self, data_type: str):
        return super()._get_output_file_path_for(data_type).replace("parquet", "zarr")

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
            grib_path = self._get_raw_file_path_for(year, month, "grib")
            zarr_path = self._get_raw_file_path_for(year, month, "zarr")
            exists_raw_grib = os.path.exists(grib_path)
            exists_cropped_zarr = os.path.exists(zarr_path)

            if exists_raw_grib or exists_cropped_zarr:
                logger.info(f"    - {month:02d}-{year} already downloaded. Skipping download...")
            else:
                logger.info(f"    - Downloading {len(days)} days for {month:02d}-{year}...")
                with console_output_prefix("        | "):
                    self.weather_downloader.fetch_month(year, month, days, self.variables, self.pressure_levels, grib_path)
                logger.info(f"        ✓ Finished downloading. Saved GRIB to {grib_path}.")

            if not exists_cropped_zarr:
                logger.info(f"    - Bringing GRIB data into the correct format. This might take a while...")
                ds = self.weather_downloader.retrieve_xr_dataset_from_grib(grib_path)
                logger.info(f"        ✓ Done.")

                logger.info(f"    - Regridding data to airport circle bounds...")
                ds = self._regrid_dataset_to_airport(ds, exact_curvilinear_grid=True)

                ds.to_zarr(zarr_path, mode="w", consolidated=False, zarr_format=3)
                os.remove(grib_path)
                logger.info(f"        ✓ Finished regridding. Removed grib file. Saved ZARR to {zarr_path}.")


        logger.info(f"✅ Finished downloading {self.dataset_name.upper()} reanalysis data for {self.icao}.\n")

    def _regrid_dataset_to_airport(self, ds: xr.Dataset, exact_curvilinear_grid: bool = True):
        lat, lon = airports[self.icao].latlon

        if exact_curvilinear_grid:
            lat2d, lon2d = get_curvilinear_grid_around_location(lat, lon, self.radius_m, num_x=self.grid_n_x, num_y=self.grid_n_y)
            regridded_ds = self._regrid_dataset_curvilinear(ds, lat2d, lon2d)
            return regridded_ds
        else:
            circle = get_circle_around_location(lat, lon, self.radius_m)
            bounds = self._get_enclosing_weather_bounds(ds, circle) 
            regridded_ds = self._regrid_dataset_regular(ds, bounds=bounds, num_lat=self.grid_n_x, num_lon=self.grid_n_y)

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

    # ------------------------------------
    # Step 2: Merge monthly ZARR files
    # ------------------------------------

    def merge(self):
        output_path = self._get_output_file_path_for("weather")
        logger.info(f"📦 Merging {self.dataset_name.upper()} reanalysis data...")

        months = self._get_months_to_download()
        merged_exists = os.path.exists(output_path)
        for year, month in months:
            month_file = self._get_raw_file_path_for(year, month, extension="zarr")
            logger.info(f"    - Merging month {month:02d}-{year} from {month_file}...")

            if not os.path.exists(month_file):
                logger.warning(f"        ✗ {month:02d}-{year} file not found under {month_file}. Skipping...")
                continue

            month_ds = xr.open_zarr(month_file, consolidated=False, zarr_format=3)
            if not merged_exists:
                month_ds.to_zarr(output_path, mode="w", consolidated=False, zarr_format=3)
                merged_exists = True
            else:
                month_ds.to_zarr(output_path, mode="a", consolidated=False, append_dim="time", zarr_format=3)

            logger.info(f"        ✓ Finished. Saved ZARR to {output_path}.")

        logger.info(f"✅ Finished merging {self.dataset_name.upper()} reanalysis data.\n")
