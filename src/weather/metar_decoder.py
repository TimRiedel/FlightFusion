import argparse
import io
import os
import time
from datetime import datetime
from enum import IntEnum

import numpy as np
import pandas as pd
import requests
from common.dataset_processor import DatasetProcessor
from metar_taf_parser.model.enum import CloudQuantity, CloudType, Descriptive
from metar_taf_parser.parser.parser import MetarParser
from utils.logger import logger

OGIMET_BASE = "https://www.ogimet.com/cgi-bin/getmetar"

class MetarDecoder(DatasetProcessor):
    def __init__(self, icao: str, start_dt: datetime, end_dt: datetime, output_dir: str, cfg: dict = {}):
        output_dir = os.path.join(output_dir, "metar")
        super().__init__(icao, start_dt, end_dt, output_dir, cfg)

    # --------------------
    # Utility
    # --------------------

    def _make_key(self, airport: str, date_iso: str):
        return hash((airport, date_iso))

    
    # --------------------
    # Step 1: Download raw METAR reports
    # --------------------

    def download(self):
        logger.info(f"📥 Downloading METARs for {self.icao}...")
        self._set_raw_data_dir("metar")

        raw_reports_df = self._fetch_raw_reports()
        path = self._get_raw_file_path_for("metar")
        self._save_data(raw_reports_df, path)
        logger.info(f"✅ Saved {len(raw_reports_df)} raw METAR reports to {path}\n")

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
        self._set_raw_data_dir("metar")
        raw_reports_df = self._load_data(self._get_raw_file_path_for("metar"))
        logger.info(f"⛅ Parsing {len(raw_reports_df)} METAR reports...")

        self._set_raw_data_dir("parsed-metar")
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
        path = self._get_raw_file_path_for("parsed-metar")
        self._save_data(parsed_reports_df, path)
        logger.info(f"✅ Parsed {len(parsed_reports_df)} METAR reports, saved to {path}\n")

    def _parse_report_message(self, message: str, date: datetime):
        if message.startswith("COR"):
            message = message[3:].strip()  # Remove COR prefix for corrected reports and replace the original

        metar_parser = MetarParser()
        parsed = metar_parser.parse(message)
        ceiling = self._compute_ceiling(parsed.clouds)

        parsed_report = {
            "airport": self.icao,
            "datetime": date.isoformat(),
            "hour": date.hour,
            "month": date.month,
            "wind_dir": self._safe_value(parsed.wind.degrees), 
            "wind_dir_variable": True if parsed.wind.degrees is None else False, # variable wind has no degrees
            "wind_speed": self._safe_value(parsed.wind.speed, 0),
            "wind_gust": parsed.wind.gust if parsed.wind.gust else parsed.wind.speed, # set wind gust to wind speed if not present
            "visibility": int(parsed.visibility.distance[:-1]) if (parsed.visibility and parsed.visibility.distance != "> 10km") else 10000, # missing values for visibility set to 10000
            "ceiling": self._safe_value(ceiling, 45000), # missing ceiling set to 45000 ft
            "ceiling_missing": True if ceiling is None else False,
            "clouds_TCU": any(cloud.type == CloudType.TCU for cloud in parsed.clouds),
            "clouds_CB": any(cloud.type == CloudType.CB for cloud in parsed.clouds),
            "weather_TS": any(condition.descriptive == Descriptive.THUNDERSTORM for condition in parsed.weather_conditions) if parsed.weather_conditions else False,
            "temperature": self._safe_value(parsed.temperature),
            "dewpoint": self._safe_value(parsed.dew_point),
            "pressure": self._safe_value(parsed.altimeter)
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

    def _safe_value(self, value, default=np.nan):
        return value if value is not None else default

    # --------------------
    # Step 3: Process parsed METAR reports for machine learning
    # --------------------

    def process(self):
        self._set_raw_data_dir("parsed-metar")
        parsed_reports_df = self._load_data(self._get_raw_file_path_for("parsed-metar"))
        logger.info(f"📈 Processing {len(parsed_reports_df)} METAR reports for machine learning...")

        processed_reports = parsed_reports_df.copy()
        processed_reports = self._add_cyclic_encodings(processed_reports)
        processed_reports = self._interpolate_missing_values(processed_reports)
        processed_reports = self._convert_units_to_metric(processed_reports)
        processed_reports = self._convert_booleans_to_int(processed_reports)

        path = self._get_output_file_path_for("processed-metar")
        self._save_data(processed_reports, path)
        logger.info(f"✅ Processed {len(processed_reports)} METAR reports, saved to {path}\n")

    def _add_cyclic_encodings(self, processed_reports: pd.DataFrame):
        # Cyclic encoding for wind direction, hour of day, and month of year
        processed_reports["wind_dir_sin"] = np.sin(processed_reports["wind_dir"] * (2 * np.pi / 360))
        processed_reports["wind_dir_cos"] = np.cos(processed_reports["wind_dir"] * (2 * np.pi / 360))
        processed_reports["hour_sin"] = np.sin(processed_reports["hour"] * (2 * np.pi / 24))
        processed_reports["hour_cos"] = np.cos(processed_reports["hour"] * (2 * np.pi / 24))
        processed_reports["month_sin"] = np.sin(processed_reports["month"] * (2 * np.pi / 12))
        processed_reports["month_cos"] = np.cos(processed_reports["month"] * (2 * np.pi / 12))

        # Drop original encoded columns
        processed_reports = processed_reports.drop(columns=["wind_dir", "hour", "month"])
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
        processed_reports["temperature"] = processed_reports["temperature"] + 273.15  # Convert temperature to Kelvin
        processed_reports["dewpoint"] = processed_reports["dewpoint"] + 273.15  # Convert dewpoint to Kelvin
        processed_reports["ceiling"] = processed_reports["ceiling"] * 0.3048  # Convert ceiling from feet to meters
        processed_reports["wind_speed"] = processed_reports["wind_speed"] * 1.852  # Convert wind speed from knots to km/h
        processed_reports["wind_gust"] = processed_reports["wind_gust"] * 1.852  # Convert wind gust from knots to km/h
        return processed_reports

    def _convert_booleans_to_int(self, processed_reports: pd.DataFrame):
        bool_columns = ["wind_dir_variable", "ceiling_missing", "clouds_TCU", "clouds_CB", "weather_TS"]
        processed_reports[bool_columns] = processed_reports[bool_columns].astype(int)
        return processed_reports
