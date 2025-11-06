import argparse
import os
from datetime import datetime

import pandas as pd
from traffic.core.flight import Flight
from traffic.data import opensky


def filter_arrivals(flights_df, icao):
    filtered_df = flights_df[flights_df['arrival'] == icao]
    return filtered_df

def drop_invalid(flights_df):
    flights_df = flights_df.dropna()
    flights_df = flights_df.drop_duplicates()
    return flights_df

def drop_same_departure_arrival(flights_df):
    flights_df = flights_df[flights_df['departure'] != flights_df['arrival']]
    return flights_df

def remove_flights_by_callsign(flights_df):
    excluded_prefixes = ['AAO', 'AAX', 'ABF', 'ABP', 'AEA', 'AEL', 'AHY', 'AIA', 'AIB', 'AIZ', 'AIZ', 'AJO', 'AJT', 'ALE', 'ALX', 'AMB', 'ANE', 'AQS', 'ARG', 'ARL', 'ASJ', 'ASR', 'ATV', 'AUH', 'AUR', 'AUR', 'AWC', 'AWC', 'AXA', 'AXZ', 'BAF', 'BBA', 'BBB', 'BCI', 'BCO', 'BCS', 'BFY', 'BGF', 'BGH', 'BHZ', 'BPO', 'BRJ', 'BRO', 'BRS', 'BRX', 'BVR', 'BZE', 'CAO', 'CBM', 'CEF', 'CFC', 'CFG', 'CHX', 'CLF', 'CLY', 'CND', 'CRO', 'CRX', 'CWG', 'CYF', 'DAF', 'DBT', 'DCW', 'DFL', 'DNC', 'DNU', 'DOR', 'DSO', 'DUK', 'EAF', 'EAJ', 'EAT', 'EAU', 'ECC', 'EDC', 'EDG', 'EDW', 'EFF', 'EFW', 'EGT', 'EIS', 'EJM', 'ELF', 'ELJ', 'ETI', 'EUP', 'EVE', 'EXJ', 'EZE', 'FAC', 'FAL', 'FCK', 'FDS', 'FEG', 'FGB', 'FHM', 'FIX', 'FIZ', 'FLI', 'FMY', 'FOX', 'FPR', 'FRO', 'FRX', 'FSE', 'GAF', 'GAM', 'GAV', 'GCK', 'GDK', 'GEC', 'GER', 'GFA', 'GFM', 'GJE', 'GJI', 'GLJ', 'GMA', 'GNY', 'GSM', 'HAF', 'HAT', 'HCR', 'HCU', 'HHN', 'HKH', 'HKH', 'HLJ', 'HLR', 'HMJ', 'HMZ', 'HNA', 'HRT', 'HUA', 'HUE', 'HUM', 'HYP', 'HZS', 'HZX', 'IBI', 'INI', 'IRL', 'ITL', 'ITY', 'IVY', 'JAF', 'JAS', 'JCL', 'JEF', 'JEI', 'JET', 'JFL', 'JIV', 'JML', 'JNY', 'JOC', 'JPV', 'JTD', 'JTI', 'JTY', 'JVW', 'KAF', 'KAL', 'KAY', 'KFE', 'KIW', 'KJD', 'KOC', 'KRF', 'KRH', 'LAF', 'LBT', 'LEA', 'LEX', 'LFO', 'LHO', 'LHX', 'LIF', 'LIL', 'LJC', 'LLT', 'LMG', 'LMJ', 'LRQ', 'LSA', 'LSV', 'LTC', 'LUC', 'LVA', 'LXA', 'LXM', 'LYF', 'LYX', 'MAP', 'MBR', 'MBU', 'MED', 'MGR', 'MJE', 'MJF', 'MJN', 'MLH', 'MLM', 'MLT', 'MMO', 'MYJ', 'MYX', 'NAF', 'NAT', 'NGN', 'NGR', 'NIA', 'NJU', 'NOS', 'NOW', 'NPT', 'NRN', 'NUM', 'NVD', 'NYB', 'NYX', 'OBS', 'OCN', 'ONY', 'ORF', 'ORO', 'ORT', 'PAL', 'PBB', 'PCT', 'PEA', 'PEG', 'PEX', 'PGC', 'PJF', 'PJS', 'PJV', 'PJZ', 'PLF', 'PNC', 'PON', 'PRI', 'PVA', 'PVG', 'PYN', 'QAF', 'QAV', 'QFX', 'QQE', 'RCH', 'RDF', 'RDN', 'RES', 'RFF', 'RFR', 'RHH', 'ROF', 'ROJ', 'RRR', 'RSD', 'RYS', 'RZO', 'SAF', 'SAW', 'SAZ', 'SCO', 'SGP', 'SIO', 'SIS', 'SJI', 'SLJ', 'SMB', 'SON', 'SOW', 'SPA', 'SPL', 'SQF', 'SSG', 'SSR', 'STW', 'SUS', 'SVF', 'SWW', 'SXI', 'SXN', 'TAG', 'TAR', 'TBJ', 'TES', 'TEU', 'TFF', 'TFL', 'THA', 'TIH', 'TJD', 'TLJ', 'TLK', 'TOM', 'TRA', 'TRK', 'TTJ', 'TUI', 'TYJ', 'UAE', 'UAF', 'UEE', 'UKN', 'UPA', 'USY', 'VAJ', 'VAL', 'VCG', 'VCJ', 'VKG', 'VLB', 'VMP', 'VND', 'VPC', 'VSR', 'VVV', 'WDY', 'XED', 'XEN', 'XFL', 'XGO', 'XLS']
    removal_condition = (
        ~flights_df['callsign'].str.match(r'^[A-Za-z]{3}.*\d') |
        flights_df['callsign'].str[:3].isin(excluded_prefixes)
    ) 
    flights_df = flights_df[~removal_condition]
    return flights_df

def fetch_flight_list(icao: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    flightlist_df = opensky.flightlist(
            airport = icao,
            start = start_dt.strftime("%Y-%m-%d 00:00:00"),
            stop = end_dt.strftime("%Y-%m-%d 23:59:59")
    )
    print(f"Fetched {len(flightlist_df)} flights arriving at {icao}.")
    flightlist_df = filter_arrivals(flightlist_df, icao)
    print(f"\t- {len(flightlist_df)} flights after filtering for arrivals at {icao}.")
    flightlist_df = drop_invalid(flightlist_df)
    print(f"\t- {len(flightlist_df)} flights after dropping invalid entries.")
    flightlist_df = drop_same_departure_arrival(flightlist_df)
    print(f"\t- {len(flightlist_df)} flights after dropping same departure and arrival.")
    flightlist_df = remove_flights_by_callsign(flightlist_df)
    print(f"\t- {len(flightlist_df)} flights after removing unwanted callsigns.")
    return flightlist_df

def fetch_flight_trajectory(callsign: str, firstseen: datetime, lastseen: datetime) -> pd.DataFrame:
    flight = opensky.history(
        start = firstseen,
        stop = lastseen,
        callsign = callsign,
        return_flight = True
    )
    if flight is not None:
        trajectory = flight.data
        # columns_to_keep = ['timestamp', 'latitude', 'longitude', 'altitude', 'groundspeed', 'vertical_rate', 'icao24', 'callsign', 'track', 'hour']
        return flight
    return None

def check_runway_alignment(flight: Flight, firstseen: datetime, lastseen: datetime, dest_icao: str) -> str:
    firstseen = firstseen.timestamp()
    lastseen = lastseen.timestamp()
    flight_data_last_3_min = flight.skip(seconds=(lastseen - firstseen - 180))
    if flight_data_last_3_min is not None:
        rwy = flight_data_last_3_min.aligned_on_ils(dest_icao, angle_tolerance=0.1, min_duration='40sec').final()
        if rwy is not None:
            return rwy.max('ILS')
    return None

def fetch_trajectories(icao: str, flightlist_df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    all_trajectories = []

    for index, flight_info in flightlist_df.iterrows():
        flight= fetch_flight_trajectory(flight_info['callsign'], flight_info['firstseen'], flight_info['lastseen'])
        if flight.data is None or flight.data.empty:
            continue
        
        aligned_on_rwy = check_runway_alignment(flight, flight_info['firstseen'], flight_info['lastseen'], icao)
        if aligned_on_rwy is None:
            continue
        flight.data['rwy'] = aligned_on_rwy
        all_trajectories.append(flight.data)

    return pd.concat(all_trajectories, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description="Download a flightlist for a given airport and time range from OpenSky Network.")
    parser.add_argument("--icao", type=str, help="ICAO code of the station, e.g. EDDB", default="EDDB")
    parser.add_argument("--start", type=str, help="Start UTC date/time YYYY-MM-DD", default="2024-01-01T00:00")
    parser.add_argument("--end", type=str, help="Required end UTC date/time YYYY-MM-DDTHH:MM", default="2024-01-01T23:59")
    args = parser.parse_args()

    icao = args.icao.upper()
    out_dir = os.path.join("../assets", "datasets", "flights", icao)
    os.makedirs(out_dir, exist_ok=True)

    current_datetime = datetime.fromisoformat(args.start)
    end_datetime = datetime.fromisoformat(args.end)

    while current_datetime <= end_datetime:
        current_datetime_end = current_datetime + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        print(f"Fetching flight list for {icao} on {current_datetime.date()}...")
        daily_flights = fetch_flight_list(icao, current_datetime, current_datetime_end)
        daily_flights.to_parquet(os.path.join(out_dir, f"{icao}_flights_{current_datetime.date()}.parquet"))

        print(f"Fetching trajectories for {len(daily_flights)} flights...")
        daily_trajectories = fetch_trajectories(icao, daily_flights, current_datetime, current_datetime_end)

        daily_trajectories.to_parquet(os.path.join(out_dir, f"{icao}_trajectories_{current_datetime.date()}.parquet"))

        current_datetime += pd.Timedelta(days=1)

    print("Done.")

if __name__ == "__main__":
    main()
