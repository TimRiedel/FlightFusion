import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd
from traffic.core import FlightIterator, Traffic, Flight
from traffic.data import airports
from traffic.data.basic.runways import Threshold

from .great_circle_calculations import haversine_bearing, haversine_distance
from .compute_features import assign_rolling_cumulative_track_change
from .filter_traffic import filter_traffic_by_flight_ids

# Suppress the RuntimeWarning about numexpr engine fallback
warnings.filterwarnings('ignore', message='.*numexpr does not support extension array dtypes.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation with empty or all-NA entries.*')

def remove_invalid_flights(processed_traffic: Traffic, icao: str) -> tuple[Traffic, Traffic]:
    """
    Removes invalid flights from traffic using multiple validation criteria.
    
    Applies a series of filters to remove flights that are invalid based on:
    - Small duration (less than 60 seconds)
    - Non-continuous trajectories (gaps greater than 120 seconds)
    - Missing runway alignment
    - Go-around or holding patterns
    
    Parameters
    -------
    processed_traffic : Traffic
        Traffic object containing processed flight trajectories.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt). Used for runway
        alignment and go-around detection.
    
    Returns
    -------
    tuple[Traffic, Traffic]
        Tuple of (valid_traffic, removed_traffic) where:
        - valid_traffic: Traffic object containing only valid flights
        - removed_traffic: Traffic object containing all removed invalid flights
        - removal_reasons: List of reasons why each flight was removed
    """
    processed_traffic, small_duration_flights, small_duration_reasons = remove_flights_with_small_duration(processed_traffic, threshold_seconds=60)
    processed_traffic, non_continuous_flights, non_continuous_reasons = remove_non_continous_flights(processed_traffic, continuity_threshold_seconds=120)
    processed_traffic, no_runway_alignment_flights, no_runway_alignment_reasons = remove_flights_without_runway_alignment(processed_traffic, icao, final_approach_time_seconds=180, angle_tolerance=0.1, min_duration_seconds=40)
    processed_traffic, go_around_holding_flights, go_around_holding_reasons = remove_flights_with_go_around_holding(processed_traffic, icao, track_threshold=330, time_window_seconds=500)

    removed_traffic = Traffic(pd.concat([small_duration_flights.data, non_continuous_flights.data, go_around_holding_flights.data, no_runway_alignment_flights.data], ignore_index=True))
    removal_reasons = small_duration_reasons + non_continuous_reasons + no_runway_alignment_reasons + go_around_holding_reasons
    return processed_traffic, removed_traffic, removal_reasons

# --------------------------------
# Flight duration and continuity functions
# --------------------------------

def remove_flights_with_small_duration(traffic: Traffic, threshold_seconds: int = 60) -> tuple[Traffic, Traffic, list[str]]:
    """
    Removes flights with duration smaller than the specified threshold.
    
    Calculates the duration of each flight as the time difference between the
    first and last timestamp within each flight_id, and removes flights that
    are shorter than the threshold.

    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories. Must contain 'flight_id'
        and 'timestamp' columns.
    threshold_seconds : int, optional
        Minimum duration threshold in seconds. Flights with duration less than
        this value will be removed. Default is 60.

    Returns
    -------
    tuple[Traffic, Traffic, list[str]]
        Tuple containing:
        - valid_traffic: Traffic object where all flights with too short duration are removed
        - removed_traffic: Traffic object containing only flights that were removed
        - removal_reasons: List of reasons why each flight was removed, including
          flight_id and duration_seconds
    """
    traffic_df = traffic.data.copy()
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])

    flight_durations = (
        traffic_df.groupby('flight_id')['timestamp']
        .agg(['min', 'max'])
        .assign(duration_seconds=lambda x: (x['max'] - x['min']).dt.total_seconds())
        .reset_index()
    )
    short_flights = flight_durations[flight_durations['duration_seconds'] < threshold_seconds]
    short_flight_ids = short_flights['flight_id'].tolist()
    
    removal_reasons = []
    for _, row in short_flights.iterrows():
        flight_id = row['flight_id']
        duration = row['duration_seconds']
        removal_reasons.append(f"Flight {flight_id} has duration of {duration:.2f} s, which is less than threshold of {threshold_seconds} s.")

    valid_traffic, short_flights_traffic = filter_traffic_by_flight_ids(traffic, short_flight_ids)
    return valid_traffic, short_flights_traffic, removal_reasons


def remove_non_continous_flights(traffic: Traffic, continuity_threshold_seconds: int = 60) -> tuple[Traffic, Traffic, list[str]]:
    """
    Removes flights, that has gaps in the timestamp column, exceeding the continuity threshold.
    
    Checks each flight for gaps between consecutive timestamps. If the maximum
    gap exceeds the threshold, the flight is considered non-continuous and removed.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories. Must contain 'flight_id'
        and 'timestamp' columns.
    continuity_threshold_seconds : int, optional
        Maximum allowed gap between consecutive timestamps in seconds. Flights
        with gaps exceeding this threshold will be removed. Default is 60.
    
    Returns
    -------
    tuple[Traffic, Traffic, list[str]]
        Tuple containing:
        - continuous_traffic: Traffic object containing only continuous flights
        - non_continuous_traffic: Traffic object containing removed non-continuous flights
        - removal_reasons: List of reasons why each flight was removed, including
          flight_id and gap duration
    """
    traffic_df = traffic.data.copy()
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    non_continuous_flight_ids = []
    
    removal_reasons = []
    for flight_id, flight_df in traffic_df.groupby('flight_id'):
        flight_df = flight_df.sort_values(by='timestamp')
        flight_df['time_diff'] = flight_df['timestamp'].diff().dt.total_seconds()

        max_time_diff = flight_df['time_diff'].max()
        if max_time_diff > continuity_threshold_seconds:
            removal_reasons.append(f"Flight {flight_id} is not continuous. Gap ({max_time_diff} s) is greater than threshold ({continuity_threshold_seconds} s).")
            non_continuous_flight_ids.append(flight_id)
    
    continuous_traffic, non_continuous_traffic = filter_traffic_by_flight_ids(traffic, non_continuous_flight_ids)
    return continuous_traffic, non_continuous_traffic, removal_reasons


# --------------------------------
# Go-Around Functions
# --------------------------------

def remove_flights_with_go_around_holding(traffic: Traffic, icao: str, track_threshold: int = 330, time_window_seconds: int = 500) -> tuple[Traffic, Traffic, list[str]]:
    """
    Removes flights with go-around or holding patterns.
    
    Identifies and removes flights that exhibit go-around behavior or holding
    patterns. Uses two detection methods:
    1. Go-around climb segment detection (using traffic library)
    2. Track change detection (for go-arounds initiated before Final Approach Point)
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories. Must contain 'track' and
        'timestamp' columns.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt). Used for go-around
        detection.
    track_threshold : int, optional
        Threshold for cumulative track change in degrees. Flights exceeding this
        threshold are considered to have go-around or holding patterns. Default is 330.
    time_window_seconds : int, optional
        Time window in seconds for rolling cumulative track change calculation.
        Default is 500.
    
    Returns
    -------
    tuple[Traffic, Traffic, list[str]]
        Tuple containing:
        - valid_traffic: Traffic object containing flights without go-around/holding patterns
        - removed_traffic: Traffic object containing removed flights with go-around/holding
        - removal_reasons: List of reasons why each flight was removed
    """
    # Assign rolling cumulative track change to all flights before checking
    processed_flights = []
    for flight in traffic:
        processed_flight = assign_rolling_cumulative_track_change(flight, time_window_seconds=time_window_seconds)
        processed_flights.append(processed_flight.data)
    traffic = Traffic(pd.concat(processed_flights, ignore_index=True))
    
    removed_flight_ids = []
    removal_reasons = []
    for flight in traffic:
        if has_go_around_climb_segment(flight, icao):
            removed_flight_ids.append(flight.flight_id)
            removal_reasons.append(f"Flight {flight.flight_id} has a go around climb segment.")
            continue

        if has_track_change_for_go_around(flight, track_threshold):
            removed_flight_ids.append(flight.flight_id)
            removal_reasons.append(f"Flight {flight.flight_id} has a go-around track change or holding.")
            continue

    valid_traffic, removed_traffic = filter_traffic_by_flight_ids(traffic, removed_flight_ids)
    valid_traffic = valid_traffic.drop(columns=['rolling_cumulative_track_change', 'track_diff'])
    removed_traffic = removed_traffic.drop(columns=['rolling_cumulative_track_change', 'track_diff'])
    return valid_traffic, removed_traffic, removal_reasons


def has_go_around_climb_segment(flight: Flight, icao: str) -> bool:
    """
    Checks if the flight has a go-around climb segment, using the traffic library's has_go_around method.
    
    A go-around climb segment is a segment where the altitude is increasing
    after a descending segment, indicating the aircraft aborted its approach
    and initiated a go-around.
    
    Parameters
    -------
    flight : Flight
        Flight object to check for go-around climb segments.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    
    Returns
    -------
    bool
        True if the flight has a go-around climb segment, False otherwise.
        Returns False if an error occurs during detection (e.g., empty iterator raising StopIteration).
    """
    try:
        return flight.has(lambda f: f.go_around(icao))
    except RuntimeError as e:
        # Handle RuntimeError that occurs when the go_around generator raises StopIteration
        # (in Python 3.7+, StopIteration raised from a generator is converted to RuntimeError)
        # If we can't determine if there's a go-around, assume there isn't one
        print(f"Error detecting go-around for flight {flight.flight_id} at {icao}: {e}. Assuming no go-around.")
        return False


def has_track_change_for_go_around(flight: Flight, track_threshold: int = 300) -> bool:
    """
    Checks if the flight has significant track change indicating a go-around or holding pattern.
    
    There exist cases in the data where a go-around is initiated before the Final
    Approach Point (FAP). In these cases, the altitude was stable for the entire go-around
    and no go-around climb segment is present. The traffic library's has_go_around
    method does not account for this case.
    
    This function uses the pre-computed rolling_cumulative_track_change that should be
    assigned by calling assign_rolling_cumulative_track_change before this function.
    
    Parameters
    -------
    flight : Flight
        Flight object to check. Must have 'rolling_cumulative_track_change' column
        assigned (via assign_rolling_cumulative_track_change).
    track_threshold : int, optional
        Threshold for cumulative track change in degrees. Flights exceeding this
        threshold are considered to have go-around or holding patterns. Default is 300.
    
    Returns
    -------
    bool
        True if the flight has track change exceeding the threshold, False otherwise.
        Returns False if 'rolling_cumulative_track_change' column is not present.
    """
    flight_df = flight.data
    
    if 'rolling_cumulative_track_change' in flight_df.columns:
        max_track_change = flight_df['rolling_cumulative_track_change'].max()
        return max_track_change > track_threshold if not pd.isna(max_track_change) else False
    else:
        return False


# --------------------------------
# Runway alignment functions
# --------------------------------

def remove_flights_without_runway_alignment(traffic: Traffic, icao: str, final_approach_time_seconds: int = 180, angle_tolerance: float = 0.1, min_duration_seconds: int = 40) -> tuple[Traffic, Traffic]:
    """
    Removes flights without runway alignment and clips valid flights to runway threshold.
    
    Checks each flight for alignment with an ILS runway during the final approach
    segment. Flights with valid runway alignment are clipped to the runway threshold,
    while flights without alignment are removed.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories. Must contain 'timestamp',
        'latitude', and 'longitude' columns.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    final_approach_time_seconds : int, optional
        Duration of the final approach segment to analyze in seconds. Default is 180.
    angle_tolerance : float, optional
        Angular tolerance in radians for runway alignment detection. Default is 0.1.
    min_duration_seconds : int, optional
        Minimum duration of alignment required in seconds. Default is 40.
    
    Returns
    -------
    tuple[Traffic, Traffic, list[str]]
        Tuple containing:
        - valid_traffic: Traffic object containing flights with runway alignment,
          clipped to the runway threshold
        - removed_traffic: Traffic object containing flights without runway alignment
        - removal_reasons: List of reasons why each flight was removed
    """
    valid_flight_dfs = []
    removed_flight_dfs = []
    removal_reasons = []
    for flight in traffic:
        runway_alignments = _get_runway_alignments(flight, icao, final_approach_time_seconds, angle_tolerance, min_duration_seconds)
        if runway_alignments is not None:
            flight = _clip_flight_to_runway_threshold(flight, runway_alignments, icao)
            if flight.data.empty:
                removal_reasons.append(f"Flight {flight.flight_id} had empty data after clipping to runway alignment.")
            else:
                valid_flight_dfs.append(flight.data)
        else:
            removed_flight_dfs.append(flight.data)
            removal_reasons.append(f"Flight {flight.flight_id} does not have a runway alignment.")

    valid_df = pd.concat(valid_flight_dfs, ignore_index=True) if len(valid_flight_dfs) > 0 else pd.DataFrame(columns=traffic.data.columns)
    removed_df = pd.concat(removed_flight_dfs, ignore_index=True) if len(removed_flight_dfs) > 0 else pd.DataFrame(columns=traffic.data.columns)
    return Traffic(valid_df), Traffic(removed_df), removal_reasons

def _get_runway_alignments(flight: Flight, icao: str, final_approach_time_seconds: int = 180, angle_tolerance: float = 0.1, min_duration_seconds: int = 40) -> Optional[str]:
    """
    Gets runway alignments for a flight during the final approach segment.
    
    Analyzes the final approach segment of the flight to detect alignment with
    an ILS runway. Returns the runway alignment if found, None otherwise.
    
    Parameters
    -------
    flight : Flight
        Flight object to analyze. Must contain 'timestamp' column.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    final_approach_time_seconds : int, optional
        Duration of the final approach segment to analyze in seconds. Default is 180.
    angle_tolerance : float, optional
        Angular tolerance in radians for runway alignment detection. Default is 0.1.
    min_duration_seconds : int, optional
        Minimum duration of alignment required in seconds. Default is 40.
    
    Returns
    -------
    FlightIterator | None
        FlightIterator containing runway alignment segments if found, None otherwise.
    """
    flight_df = flight.data
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)

    firstseen = pd.to_datetime(flight_df['timestamp'].iloc[0])
    lastseen = pd.to_datetime(flight_df['timestamp'].iloc[-1])
    total_duration = (lastseen - firstseen).total_seconds()
    skip_seconds = int(max(0, total_duration - final_approach_time_seconds))
    final_segment = flight.skip(seconds=skip_seconds)

    if final_segment is not None:
        rwy_alignments = final_segment.aligned_on_ils(icao, angle_tolerance=angle_tolerance, min_duration=f"{min_duration_seconds}sec")
        if rwy_alignments is not None:
            return rwy_alignments
    return None


def _clip_flight_to_runway_threshold(flight: Flight, runway_alignments: FlightIterator, icao: str) -> Flight:
    """
    Clips a flight trajectory to the runway threshold.
    
    Clips the flight to the closest point before the runway threshold and adds
    a final point at the threshold location. Assigns airport and ILS information
    to aligned segments.
    
    Parameters
    -------
    flight : Flight
        Flight object to clip. Must contain 'latitude', 'longitude', 'altitude' (ft),
        'track', and 'timestamp' columns.
    runway_alignments : FlightIterator
        FlightIterator containing runway alignment segments from aligned_on_ils.
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    
    Returns
    -------
    Flight
        Flight object clipped to the runway threshold. The last point is set to
        the threshold location with altitude of the airport (ft) + 50ft (15m) and updated track bearing.
        Airport and ILS columns are added to aligned segments.
    """
    flight_df = flight.data.copy()
    last_alignment_idx = len(runway_alignments) - 1
    for i, alignment in enumerate(runway_alignments):

        idxs = alignment.data.index
        flight_df.loc[idxs, 'airport'] = alignment.data['airport']
        flight_df.loc[idxs, 'ILS'] = alignment.data['ILS']
        if i != last_alignment_idx:
            continue
        # Obtain the location of the threshold for the landing runway
        icao, ils = alignment.data.iloc[-1]['airport'], alignment.data.iloc[-1]['ILS']
        runway_thresholds = airports[icao].runways[ils].tuple_runway
        threshold = next(t for t in runway_thresholds if t.name == ils)

        # Calculate the closest point to the threshold, which is still before the threshold
        closest_point_idx, closest_distance = _calculate_closest_point_to_threshold(alignment.data, threshold)
        # if closest_distance > max_point_distance_from_threshold / 1000:
        #     print(f"Warning: Closest point is {closest_distance:.2f} km from threshold {threshold.name}, which is greater than the maximum distance of {max_point_distance_from_threshold / 1000:.2f} km.")
        
        # Clip the flight to the closest point and set the last point to the threshold
        flight_df = flight_df.loc[:closest_point_idx]
        last_point = flight_df.iloc[-1].copy()
        last_point['latitude'] = threshold.latitude
        last_point['longitude'] = threshold.longitude
        last_point['altitude'] = airports[icao].altitude + 15
        last_point['track'] = haversine_bearing(last_point['latitude'], last_point['longitude'], threshold.latitude, threshold.longitude)
        last_point['timestamp'] = pd.to_datetime(last_point['timestamp']) + pd.Timedelta(seconds=1)
        flight_df = pd.concat([flight_df, pd.DataFrame([last_point])], ignore_index=True)
    return Flight(flight_df)


def _calculate_closest_point_to_threshold(alignment_trajectory: pd.DataFrame, threshold: Threshold) -> tuple[int, float]:
    """
    Calculates the closest point to the runway threshold that is still before the threshold.
    
    Finds the point in the trajectory that is closest to the threshold while ensuring
    the bearing difference is within ±90 degrees (indicating the point is before the
    threshold, not past it).
    
    Parameters
    -------
    alignment_trajectory : pd.DataFrame
        DataFrame containing trajectory points with 'latitude' and 'longitude' columns.
    threshold : Threshold
        Threshold object representing the runway threshold location and bearing.
    
    Returns
    -------
    tuple[int, float]
        Tuple containing:
        - closest_point_idx: Index of the closest point in the trajectory
        - min_distance: Distance to the threshold in kilometers
    """
    closest_point_idx = 0
    min_distance = float('inf')
    for idx, row in alignment_trajectory.iterrows():
        distance = haversine_distance(row['latitude'], row['longitude'], threshold.latitude, threshold.longitude)
        bearing = haversine_bearing(row['latitude'], row['longitude'], threshold.latitude, threshold.longitude)
        bearing_diff = (threshold.bearing - bearing + 180) % 360 - 180 # range [-180, 180]
        
        if -90 <= bearing_diff <= 90 and distance < min_distance:
            min_distance = distance
            closest_point_idx = idx
    return closest_point_idx, min_distance