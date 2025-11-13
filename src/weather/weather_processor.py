import os
from datetime import datetime
from calendar import monthrange

import shapely
import xarray as xr
from traffic.data import airports

from weather.weather_downloader import CerraDownloader, Era5Downloader
from common.dataset_processor import DatasetProcessor
from common.projections import get_circle_around_location
from utils.logger import logger
from utils.output_capture import console_output_prefix

class WeatherProcessor(DatasetProcessor):
    def __init__(self, icao: str, start_dt: datetime, end_dt: datetime, radius_km: int, output_dir: str, cfg: dict = {},):
        output_dir = os.path.join(output_dir, "weather")
        super().__init__(icao, start_dt, end_dt, radius_km, output_dir, cfg)
        
        self.dataset_name = cfg.get("dataset_name")
        self.variables = cfg.get("variables")
        self.pressure_levels = cfg.get("pressure_levels")
        self.pressure_levels_str = [str(level) for level in self.pressure_levels]
        
        if self.dataset_name == "cerra":
            self.weather_downloader = CerraDownloader(icao, self.dataset_name, output_dir)
        elif self.dataset_name == "era5":
            self.weather_downloader = Era5Downloader(icao, self.dataset_name, output_dir)
        else:
            raise ValueError(f"Invalid dataset: {self.dataset_name}")

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
            logger.info(f"    - Downloading {len(days)} days for {month:02d}-{year}...")
            with console_output_prefix("        | "):
                self.weather_downloader.fetch_month(year, month, days, self.variables, self.pressure_levels, grib_path)
            logger.info(f"        ✓ Finished downloading. Saved GRIB to {grib_path}.")


            zarr_path = self._get_raw_file_path_for(year, month, "zarr")
            logger.info(f"    - Cropping data...")
            self._crop_monthly_grib_file(grib_path, zarr_path)
            logger.info(f"        ✓ Finished cropping {month:02d}-{year}. Removed grib file. Saved ZARR to {zarr_path}.")


        logger.info(f"✅ Finished downloading {self.dataset_name.upper()} reanalysis data for {self.icao}.\n")

    def _crop_monthly_grib_file(self, grib_path: str, zarr_path: str):
        # Indexpath is empty to avoid creating a .idx file
        ds = xr.load_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        ds = ds.rename({'isobaricInhPa': 'level'})
        ds = ds.drop_vars(['number', 'step', 'valid_time'])

        lat, lon = airports[self.icao].latlon
        circle = get_circle_around_location(lat, lon, self.radius_m)
        min_lon, min_lat, max_lon, max_lat = self._get_enclosing_weather_bounds(ds, circle) 

        # Latitude coordinates are descending (90.0 -> -90.0), so slice from upper to lower
        cropped_ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        cropped_ds.to_zarr(zarr_path, mode="w", consolidated=False, zarr_format=3)

        os.remove(grib_path)

    def _get_enclosing_weather_bounds(self, ds: xr.Dataset, circle: shapely.geometry.Polygon):
        min_lon, min_lat, max_lon, max_lat = circle.bounds

        ds_lats = ds.latitude.values
        ds_lons = ds.longitude.values

        lon_lower_candidates = ds_lons[ds_lons <= min_lon]
        lat_lower_candidates = ds_lats[ds_lats <= min_lat]
        lon_upper_candidates = ds_lons[ds_lons >= max_lon]
        lat_upper_candidates = ds_lats[ds_lats >= max_lat]

        lon_lower = lon_lower_candidates.max() if lon_lower_candidates.size > 0 else ds_lons.min()
        lat_lower = lat_lower_candidates.max() if lat_lower_candidates.size > 0 else ds_lats.min()
        lon_upper = lon_upper_candidates.min() if lon_upper_candidates.size > 0 else ds_lons.max()
        lat_upper = lat_upper_candidates.min() if lat_upper_candidates.size > 0 else ds_lats.max()

        return lon_lower, lat_lower, lon_upper, lat_upper

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
