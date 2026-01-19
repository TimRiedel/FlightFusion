import argparse
import hashlib
import io
import json
import os
import time
from datetime import datetime
from enum import IntEnum

import numpy as np
import pandas as pd
import requests
from common.dataset_processor import DatasetProcessor, ProcessingConfig
from metar_taf_parser.model.enum import CloudQuantity, CloudType, Descriptive, Phenomenon
from metar_taf_parser.parser.parser import MetarParser
from utils.logger import logger
from utils.cache import Cache

OGIMET_BASE = "https://www.ogimet.com/cgi-bin/getmetar"

class MetarProcessor(DatasetProcessor):
    def __init__(self, processing_config: ProcessingConfig, task_config: dict = {}):
        super().__init__(processing_config, task_type="metar", task_config=task_config, create_temp_dir=True)
        
        self.cache = Cache(processing_config.cache_dir, "metar", self.icao)

    # --------------------
    # Utility
    # --------------------

    def _make_key(self, airport: str, date_iso: str):
        return hash((airport, date_iso))

    def _get_request_config(self):
        request_config = {
            "icao": self.icao,
            "start_dt": str(self.start_dt),
            "end_dt": str(self.end_dt)
        }
        return request_config

    
    # --------------------
    # Step 1: Download raw METAR reports
    # --------------------

    def download(self):
        logger.info(f"📥 Downloading METARs for {self.icao}...")

        request_config = self._get_request_config()
        file_path, exists_cached = self.cache.get_file_path(request_config)
        if exists_cached:
            logger.info(f"    ✓ Found cached METAR data, skipping download.\n")
            return

        raw_reports_df = self._fetch_raw_reports()
        self._save_data(raw_reports_df, file_path)
        logger.info(f"✅ Saved {len(raw_reports_df)} raw METAR reports to {file_path}\n")

    def _fetch_raw_reports(self):
        url = self._build_query_url()

        response = requests.get(url)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data, skip_blank_lines=True)

        # Normalize Spanish headers to English
        col_map = {
            "ESTACION": "airport",
            "ANO": "year",
            "MES": "month",
            "DIA": "day",
            "HORA": "hour",
            "MINUTO": "minutes",
            "PARTE": "message",
        }
        df = df.rename(columns=col_map)

        # Extract report type (e.g. METAR) and remove it from beginning of message
        for report in df.itertuples():
            report_words = report.message.split()
            report_type = report_words[0]
            df.at[report.Index, "message"] = report.message.replace(report_type, "").strip()  # Remove type prefix
            df.at[report.Index, "type"] = report_type

        return df

    def _build_query_url(self):
        """Build the OGIMET query URL for METAR/TAF reports selection."""
        params = {
            "begin": self.start_dt.strftime("%Y%m%d%H%M"),
            "end": self.end_dt.strftime("%Y%m%d%H%M"),
            "lang": "eng",
            "header": "yes",
            "icao": self.icao,
        }

        url = OGIMET_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        return url


    # --------------------
    # Step 2: Parse raw METAR reports
    # --------------------

    def parse(self):
        parsed_path = self._get_temp_file_path_for("parsed-metar")
        if self._check_current_step_file_exists(parsed_path, "parsing"):
            return

        request_config = self._get_request_config()
        file_path, exists_cached = self.cache.get_file_path(request_config)
        if not exists_cached:
            raise FileNotFoundError(f"Cached METAR data not found at {file_path}. Please run the download method first.")

        raw_reports_df = self._load_data(file_path)
        logger.info(f"⛅ Parsing {len(raw_reports_df)} METAR reports...")

        parsed_reports_map = {}
        for report in raw_reports_df.itertuples():
            try:
                date = datetime(report.year, report.month, report.day, report.hour, report.minutes)
                parsed_report = self._parse_report_message(report.message, date)
                parsed_reports_map[self._make_key(report.airport, date)] = parsed_report
            except Exception as e:
                print(f"Skipping report at {date}. Message: {report.message}. Error: {e}")
                continue

        parsed_reports_df = pd.DataFrame(parsed_reports_map.values())
        self._save_data(parsed_reports_df, parsed_path)
        logger.info(f"✅ Parsed {len(parsed_reports_df)} METAR reports, saved to {parsed_path}\n")

    def _parse_report_message(self, message: str, date: datetime):
        if message.startswith("COR"):
            message = message[3:].strip()  # Remove COR prefix for corrected reports and replace the original

        metar_parser = MetarParser()
        parsed = metar_parser.parse(message)
        ceiling = self._compute_ceiling(parsed.clouds)

        parsed_report = {
            "airport": self.icao,
            "datetime": date.isoformat(),
            "month": date.month,
            "temperature": self._safe_value(parsed.temperature),
            "dewpoint": self._safe_value(parsed.dew_point),
            "pressure": self._safe_value(parsed.altimeter),
            "wind_dir": self._safe_value(parsed.wind.degrees), 
            "wind_dir_variable": True if parsed.wind.degrees is None else False, # variable wind has no degrees
            "wind_speed": self._safe_value(parsed.wind.speed, 0),
            "wind_gust": parsed.wind.gust if parsed.wind.gust else parsed.wind.speed, # set wind gust to wind speed if not present
            "min_visibility": self._compute_min_of_visibility_and_rwy_range(parsed),
            "trend_visibility": self._compute_trend_visibility(parsed),
            "ceiling": self._safe_value(ceiling, 45000), # missing ceiling set to 45000 ft
            "ceiling_missing": True if ceiling is None else False,
            "clouds_TCU": self._exists_towering_cumulus(parsed),
            "clouds_CB": self._exists_cumulonimbus(parsed),
            "weather_TS": self._exists_thunderstorm(parsed),
            "weather_FG": self._exists_fog(parsed),
            "weather_SN": self._exists_precipitation_snow(parsed),
        }
        return parsed_report

    def _compute_ceiling(self, cloud_layers):
        min_ceiling = None
        for cloud in cloud_layers:
            if cloud.quantity in [CloudQuantity.BKN, CloudQuantity.OVC]:
                if cloud.height is None: # sometimes clouds are reported without height like BKN///CB, skip them
                    continue
                if min_ceiling is None or cloud.height < min_ceiling:
                    min_ceiling = cloud.height
        return min_ceiling

    def _get_visibility(self, parsed):
        min_distance = 10000
        if parsed.visibility and parsed.visibility.distance != "> 10km":
            min_distance = int(parsed.visibility.distance[:-1])
        return min_distance

    def _compute_min_of_visibility_and_rwy_range(self, parsed):
        min_rwy_range = 10000
        if parsed.runways_info:
            min_rwy_range = min(runway.min_range for runway in parsed.runways_info)
        return min(min_rwy_range, self._get_visibility(parsed))

    def _exists_towering_cumulus(self, parsed):
        return any(cloud.type == CloudType.TCU for cloud in parsed.clouds)

    def _exists_cumulonimbus(self, parsed):
        return any(cloud.type == CloudType.CB for cloud in parsed.clouds)

    def _exists_thunderstorm(self, parsed):
        return any(condition.descriptive == Descriptive.THUNDERSTORM for condition in parsed.weather_conditions) if parsed.weather_conditions else False

    def _exists_fog(self, parsed):
        for condition in parsed.weather_conditions:
            if any(phenomenon == Phenomenon.FOG for phenomenon in condition.phenomenons):
                return True
        return False

    def _exists_precipitation_snow(self, parsed):
        for condition in parsed.weather_conditions:
            if any(phenomenon == Phenomenon.SNOW for phenomenon in condition.phenomenons):
                return True
        return False

    def _compute_trend_visibility(self, parsed):
        min_visibility = self._get_visibility(parsed)
        if parsed.nosig:
            return min_visibility

        min_trend_visibility = 10000 # set to max visibility
        trend_visibility_present = False
        for trend in parsed.trends:
            if trend.visibility is not None and trend.visibility.distance != "> 10km":
                trend_visibility_present = True
                visibility = int(trend.visibility.distance[:-1])
                min_trend_visibility = min(min_trend_visibility, visibility)

        if trend_visibility_present:
            return min_trend_visibility
        else:
            return min_visibility


    def _safe_value(self, value, default=np.nan):
        return value if value is not None else default


    # --------------------
    # Step 3: Process parsed METAR reports for machine learning
    # --------------------

    def process(self):
        processed_path = self._get_output_file_path_for("metar")
        if self._check_current_step_file_exists(processed_path, "process"):
            return

        parsed_path = self._get_temp_file_path_for("parsed-metar")
        self._ensure_previous_step_file_exists(parsed_path, "parse")

        parsed_reports_df = self._load_data(parsed_path)
        logger.info(f"📈 Processing {len(parsed_reports_df)} METAR reports for machine learning...")

        processed_reports = parsed_reports_df.copy()
        processed_reports = self._add_cyclic_encodings(processed_reports)
        processed_reports = self._interpolate_missing_values(processed_reports)
        processed_reports = self._convert_units_to_metric(processed_reports)
        processed_reports = self._convert_booleans_to_int(processed_reports)
        processed_reports = self._round_numeric_columns(processed_reports)

        self._save_data(processed_reports, processed_path)
        logger.info(f"✅ Processed {len(processed_reports)} METAR reports, saved to {processed_path}\n")

    def _add_cyclic_encodings(self, processed_reports: pd.DataFrame):
        # Set wind dir to 0, if wind_dir is 360, because wind is always reported as 360 for direct northerly wind and never as 000
        processed_reports.loc[processed_reports["wind_dir"] == 360, "wind_dir"] = 0
        processed_reports["wind_dir_sin"] = np.sin(processed_reports["wind_dir"] * 2 * np.pi / 360)
        processed_reports["wind_dir_cos"] = np.cos(processed_reports["wind_dir"] * 2 * np.pi / 360)
        
        minutes_in_day = 24 * 60
        datetimes = pd.to_datetime(processed_reports["datetime"])
        day_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute
        processed_reports["time_of_day_sin"] = np.sin(day_minutes * 2 * np.pi / minutes_in_day)
        processed_reports["time_of_day_cos"] = np.cos(day_minutes * 2 * np.pi / minutes_in_day)

        processed_reports["month_sin"] = np.sin((processed_reports["month"] - 1) * 2 * np.pi / 12)
        processed_reports["month_cos"] = np.cos((processed_reports["month"] - 1) * 2 * np.pi / 12)

        processed_reports = processed_reports.drop(columns=["wind_dir", "month"])
        return processed_reports

    def _interpolate_missing_values(self, df):
        df = df.sort_values(by=["airport", "datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        # Interpolate temperature, dewpoint and wind directions per airport based on the surrounding time
        columns_to_interp = ["temperature", "dewpoint", "wind_dir_sin", "wind_dir_cos", "pressure"]
        for col in columns_to_interp:
            df[col] = df.groupby("airport")[col].transform(lambda g: g.interpolate(method="time"))

        return df.reset_index()

    def _convert_units_to_metric(self, processed_reports: pd.DataFrame):
        processed_reports["ceiling"] = processed_reports["ceiling"] * 0.3048  # Convert ceiling from feet to meters
        processed_reports["wind_speed"] = processed_reports["wind_speed"] * 1.852  # Convert wind speed from knots to km/h
        processed_reports["wind_gust"] = processed_reports["wind_gust"] * 1.852  # Convert wind gust from knots to km/h
        return processed_reports

    def _convert_booleans_to_int(self, processed_reports: pd.DataFrame):
        bool_columns = processed_reports.select_dtypes(include=["bool"]).columns
        processed_reports[bool_columns] = processed_reports[bool_columns].astype(int)
        return processed_reports

    def _round_numeric_columns(self, processed_reports: pd.DataFrame):
        to_int_columns = ["wind_speed", "wind_gust", "ceiling", "temperature", "dewpoint", "pressure"]

        to_float_columns = ["wind_dir_sin", "wind_dir_cos", "time_of_day_sin", "time_of_day_cos", "month_sin", "month_cos"]
        for col in to_int_columns:
            processed_reports[col] = processed_reports[col].round().astype(int)

        for col in to_float_columns:
            processed_reports[col] = processed_reports[col].round(6).astype(float)
        return processed_reports

