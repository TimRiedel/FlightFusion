import os
from datetime import datetime
from calendar import monthrange

from traffic.data import airports
import xarray as xr

from weather.weather_downloader import CerraDownloader, Era5Downloader
from common.dataset_processor import DatasetProcessor
from common.projections import get_circle_around_location
from utils.logger import logger



class WeatherProcessor(DatasetProcessor):
    def __init__(self, icao: str, start_dt: datetime, end_dt: datetime, radius: int, output_dir: str, cfg: dict = {},):
        output_dir = os.path.join(output_dir, "weather")
        super().__init__(icao, start_dt, end_dt, radius, output_dir, cfg)
        
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

    def download(self):
        logger.info(f"📥 Downloading {self.dataset_name.upper()} reanalysis data for {self.icao}...")
        months_to_download = self._get_months_to_download()
        
        for year, month in months_to_download:
            days = self._get_days_to_download(year, month)
            grib_path = self._get_raw_file_path_for(year, month, "grib")
            logger.info(f"    - Downloading {len(days)} days for {month:02d}-{year}...")
            self.weather_downloader.fetch_month(year, month, days, self.variables, self.pressure_levels, grib_path)
            logger.info(f"        ✓ Finished downloading. Saved GRIB to {grib_path}.")


            zarr_path = self._get_raw_file_path_for(year, month, "zarr")
            logger.info(f"    - Cropping data...")
            self._crop_monthly_grib_file(grib_path, zarr_path)
            logger.info(f"        ✓ Finished cropping {month:02d}-{year}. Removed grib file. Saved ZARR to {zarr_path}.")


        logger.info(f"✅ Finished downloading {self.dataset_name.upper()} reanalysis data for {self.icao}.")

    def _get_raw_file_path_for(self, year: int, month: int, extension: str):
        return os.path.join(self.raw_data_dir, f"{self.icao}_{self.dataset_name}_{year}-{month:02d}.{extension}")

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

    def _crop_monthly_grib_file(self, grib_path: str, zarr_path: str):
        # Indexpath is empty to avoid creating a .idx file
        ds = xr.load_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}, decode_timedelta=True)
        ds = ds.rename({'isobaricInhPa': 'level'})
        ds = ds.drop_vars(['number', 'step', 'valid_time'])

        lat, lon = airports[self.icao].latlon
        circle = get_circle_around_location(lat, lon, self.radius_m)
        min_lon, min_lat, max_lon, max_lat = circle.bounds

        # Latitude coordinates are descending (90.0 -> -90.0), so we need to swap min/max for slice
        cropped_ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
        cropped_ds.to_zarr(zarr_path, mode="w", consolidated=False)

        os.remove(grib_path)

    def process(self):
        # TODO: Merge files from all months into a single zarr file
        pass