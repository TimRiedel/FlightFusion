import argparse
import io
import os
import time
from datetime import datetime
from enum import IntEnum

import numpy as np
import pandas as pd
import requests
from metar_taf_parser.model.enum import CloudQuantity, CloudType, Descriptive
from metar_taf_parser.parser.parser import MetarParser

OGIMET_BASE = "https://www.ogimet.com/cgi-bin/getmetar"

def make_key(airport: str, date_iso: str):
    return hash((airport, date_iso))

def safe_value(value, default=np.nan):
    return value if value is not None else default

def compute_ceiling(cloud_layers):
    min_ceiling = None
    for cloud in cloud_layers:
        if cloud.quantity in [CloudQuantity.BKN, CloudQuantity.OVC]:
            if cloud.height is None: # sometimes clouds are reported without height like BKN///CB, skip them
                continue
            if min_ceiling is None or cloud.height < min_ceiling:
                min_ceiling = cloud.height
    return min_ceiling


def build_query_url(icao: str, start_dt: datetime, end_dt: datetime=None):
    """Build the OGIMET query URL for METAR/TAF reports selection."""
    params = {
        "begin": start_dt.strftime("%Y%m%d%H%M"),
        "end": end_dt.strftime("%Y%m%d%H%M"),
        "lang": "eng",
        "header": "yes",
        "icao": icao,
    }

    url = OGIMET_BASE + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    return url


def fetch_raw_reports(icao: str, start_dt: datetime, end_dt: datetime = None):
    url = build_query_url(icao, start_dt, end_dt)

    print("Fetching data from:", url)
    response = requests.get(url)
    response.raise_for_status()
    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data, skip_blank_lines=True)
    print("Data fetched.")

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

    # Extract report type (e.g. METAR) and remove it from message beginning of message
    for report in df.itertuples():
        report_words = report.message.split()
        report_type = report_words[0]
        df.at[report.Index, "message"] = report.message.replace(report_type, "").strip()  # Remove type prefix
        df.at[report.Index, "type"] = report_type

    return df

def parse_reports(raw_reports: pd.DataFrame):
    print("Parsing reports...")
    metar_parser = MetarParser()
    parsed_reports_map = {}

    for report in raw_reports.itertuples():
        message = report.message
        try:
            if message.startswith("COR"):
                message = message[3:].strip()  # Remove COR prefix for corrected reports and replace the original

            parsed = metar_parser.parse(message)
            date = datetime(report.year, report.month, report.day, report.hour, report.minutes)
            ceiling = compute_ceiling(parsed.clouds)

            parsed_report = {
                "airport": report.airport,
                "datetime": date.isoformat(),
                "hour": date.hour,
                "month": date.month,
                "wind_dir": safe_value(parsed.wind.degrees), 
                "wind_dir_variable": True if parsed.wind.degrees is None else False, # variable wind has no degrees
                "wind_speed": safe_value(parsed.wind.speed, 0),
                "wind_gust": parsed.wind.gust if parsed.wind.gust else parsed.wind.speed, # set wind gust to wind speed if not present
                "visibility": parsed.visibility.distance[:-1] if (parsed.visibility and parsed.visibility.distance != "> 10km") else 10000, # missing values for visibility set to 10000
                "ceiling": safe_value(ceiling, 45000), # missing ceiling set to 45000 ft
                "ceiling_missing": True if ceiling is None else False,
                "clouds_TCU": any(cloud.type == CloudType.TCU for cloud in parsed.clouds),
                "clouds_CB": any(cloud.type == CloudType.CB for cloud in parsed.clouds),
                "weather_TS": any(condition.descriptive == Descriptive.THUNDERSTORM for condition in parsed.weather_conditions) if parsed.weather_conditions else False,
                "temperature": safe_value(parsed.temperature),
                "dewpoint": safe_value(parsed.dew_point),
                "pressure": safe_value(parsed.altimeter)
            }
            parsed_reports_map[make_key(report.airport, date)] = parsed_report

        except Exception as e:
            print(f"Skipping report at {date}. Message: {message}. Error: {e}")
            continue

    parsed_df = pd.DataFrame(parsed_reports_map.values())
    return parsed_df

def interpolate_missing_values(df):
    df = df.sort_values(by=["airport", "datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    # Interpolate temperature, dewpoint and wind directions per airport based on the surrounding time
    columns_to_interp = ["temperature", "dewpoint", "wind_dir_sin", "wind_dir_cos", "pressure"]
    for col in columns_to_interp:
        df[col] = df.groupby("airport")[col].transform(lambda g: g.interpolate(method="time"))

    return df.reset_index()

def process_reports(parsed_reports: pd.DataFrame):
    print("Processing reports...")
    processed_reports = parsed_reports.copy()

    # Cyclic encoding for wind direction, hour of day, and month of year
    processed_reports["wind_dir_sin"] = np.sin(processed_reports["wind_dir"] * (2 * np.pi / 360))
    processed_reports["wind_dir_cos"] = np.cos(processed_reports["wind_dir"] * (2 * np.pi / 360))
    processed_reports["hour_sin"] = np.sin(processed_reports["hour"] * (2 * np.pi / 24))
    processed_reports["hour_cos"] = np.cos(processed_reports["hour"] * (2 * np.pi / 24))
    processed_reports["month_sin"] = np.sin(processed_reports["month"] * (2 * np.pi / 12))
    processed_reports["month_cos"] = np.cos(processed_reports["month"] * (2 * np.pi / 12))

    # Drop original encoded columns
    processed_reports = processed_reports.drop(columns=["wind_dir", "hour", "month"])

    # Interpolate missing values
    processed_reports = interpolate_missing_values(processed_reports)

    # Unit conversions
    processed_reports["temperature"] = processed_reports["temperature"] + 273.15  # Convert temperature to Kelvin
    processed_reports["dewpoint"] = processed_reports["dewpoint"] + 273.15  # Convert dewpoint to Kelvin
    processed_reports["ceiling"] = processed_reports["ceiling"] * 0.3048  # Convert ceiling from feet to meters
    processed_reports["wind_speed"] = processed_reports["wind_speed"] * 1.852  # Convert wind speed from knots to km/h
    processed_reports["wind_gust"] = processed_reports["wind_gust"] * 1.852  # Convert wind gust from knots to km/h

    # Convert booleans to 0/1 integers
    bool_columns = ["wind_dir_variable", "ceiling_missing", "clouds_TCU", "clouds_CB", "weather_TS"]
    processed_reports[bool_columns] = processed_reports[bool_columns].astype(int)

    return processed_reports


def main():
    parser = argparse.ArgumentParser(description="Download METAR/TAF from OGIMET and export to CSV.")
    parser.add_argument("--icao", type=str, help="ICAO code of the station, e.g. EDDB", default="EDDM")
    parser.add_argument("--start", type=str, help="Start UTC date/time YYYY-MM-DDTHH:MM or YYYY-MM-DD", default="2024-01-01T00:00")
    parser.add_argument("--end", type=str, help="Optional end UTC date/time YYYY-MM-DDTHH:MM", default="2025-01-01T00:00")
    parser.add_argument("--output", type=str, default="parsed_reports.csv", help="Output CSV filename")
    parser.add_argument("--use_downloaded", action="store_true", help="Use previously downloaded data instead of fetching from OGIMET")
    args = parser.parse_args()


    icao = args.icao.upper()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = start_dt
    if args.end:
        end_dt = datetime.fromisoformat(args.end)

    out_dir = os.path.join("assets", "datasets", "weather")
    raw_file = os.path.join(out_dir, f"{icao}_raw_metar_reports.csv")
    if args.use_downloaded and os.path.exists(raw_file):
        raw_reports_df = pd.read_csv(raw_file)
    else:
        raw_reports_df = fetch_raw_reports(icao, start_dt, end_dt)
    parsed_reports_df = parse_reports(raw_reports_df)
    processed_reports_df = process_reports(parsed_reports_df)

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, f"{icao}_raw_metar_reports.csv")
    parsed_path = os.path.join(out_dir, f"{icao}_parsed_metar_reports.csv")
    processed_path = os.path.join(out_dir, f"{icao}_processed_metar_reports.csv")

    raw_reports_df.to_csv(raw_path, index=False)
    parsed_reports_df.to_csv(parsed_path, index=False)
    processed_reports_df.to_csv(processed_path, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
