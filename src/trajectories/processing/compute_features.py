
import numpy as np
import pandas as pd
from traffic.core import Traffic, Flight

from .great_circle_calculations import haversine_distance


def assign_flight_id(traffic: Traffic, split_by_gap: bool = True, gap_threshold_minutes: int = 60) -> Traffic:
    """
    Assigns unique flight_id to each flight trajectory.
    
    If split_by_gap is True, handles cases where the same aircraft with the same
    callsign makes multiple approaches in a single day by detecting gaps in
    timestamps and assigning sequential numbers to each flight segment.
    
    Flight ID format: {callsign}_{icao24}_{date}_{sequence_number}
    Examples:
    - SFG55B_3cc108_20240915_1 (first flight of the day)
    - SFG55B_3cc108_20240915_2 (second flight of the day)
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight data with callsign, icao24, and timestamp columns.
    split_by_gap : bool, optional
        If True, splits flights by gap in timestamps. If False, assigns a single
        flight_id to all points of the same aircraft with the same callsign.
        Default is True.
    gap_threshold_minutes : int, optional
        Time gap threshold in minutes to detect separate flights. If time difference
        between consecutive points exceeds this threshold, they are considered
        separate flights. Default is 60.
    
    Returns
    -------
    Traffic
        Traffic object with assigned flight_id column. The flight_id column is
        placed first in the dataframe.
    """
    if not split_by_gap:
        traffic_df = traffic.data.copy()
        traffic_df = traffic_df.sort_values(by=['callsign', 'icao24', 'timestamp']).reset_index(drop=True)
        traffic_df['flight_id'] = traffic_df['callsign'] + "_" + traffic_df['icao24'] + "_" + traffic_df['timestamp'].dt.strftime("%Y%m%d")

        # Reorder columns to put flight_id first
        cols = ['flight_id'] + [col for col in traffic_df.columns if col != 'flight_id']
        traffic_df = traffic_df[cols]
        return Traffic(traffic_df)
    
    traffic_df = traffic.data.copy()
    traffic_df = traffic_df.sort_values(by=['callsign', 'icao24', 'timestamp']).reset_index(drop=True)
    
    traffic_df['timestamp'] = pd.to_datetime(traffic_df['timestamp'])
    traffic_df['date'] = traffic_df['timestamp'].dt.strftime("%Y%m%d")
    traffic_df['base_id'] = traffic_df['callsign'] + "_" + traffic_df['icao24'] + "_" + traffic_df['date']
    
    flight_segments = []
    for base_id, group in traffic_df.groupby('base_id'):
        group = group.sort_values(by='timestamp').reset_index(drop=True)
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds() / 60  # in minutes
        group['is_new_flight'] = (group['time_diff'] > gap_threshold_minutes) | (group['time_diff'].isna())
        # Convert boolean to int to avoid PyArrow dtype issues with cumsum
        group['flight_sequence'] = group['is_new_flight'].astype(int).cumsum()
        group['flight_id'] = group['base_id'] + "_" + group['flight_sequence'].astype(str)
        flight_segments.append(group)
    
    result_df = pd.concat(flight_segments, ignore_index=True)
    result_df = result_df.drop(columns=['date', 'base_id', 'time_diff', 'is_new_flight', 'flight_sequence'])

    # Reorder columns to put flight_id first
    cols = ['flight_id'] + [col for col in result_df.columns if col != 'flight_id']
    result_df = result_df[cols]
    return Traffic(result_df)


def assign_distance_to_target(traffic: Traffic, lat: float, lon: float) -> Traffic:
    """
    Assigns distance to a target location for each point in the trajectory.
    
    Calculates the haversine distance from each point in the trajectory to the
    specified target location (latitude, longitude).
    
    Credit: Ricardo Reinke
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing trajectory data with latitude and longitude columns.
    lat : float
        Latitude of the target location in degrees (WGS84).
    lon : float
        Longitude of the target location in degrees (WGS84).
    
    Returns
    -------
    Traffic
        Traffic object with 'distance' column assigned, containing the distance
        from each point to the target location in kilometers. The trajectory is
        sorted by timestamp.
    """
    traffic_df = traffic.data
    traffic_df = traffic_df.sort_values(by='timestamp').reset_index(drop=True)
    traffic_df['distance'] = traffic_df.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'], lat, lon),
        axis=1
    )
    return Traffic(traffic_df)


def assign_time_since_start_of_trajectory(flight: Flight) -> Flight:
    """
    Assigns time elapsed since the start of the trajectory for each point.
    
    Calculates the time difference in seconds from the first timestamp in the
    trajectory to each subsequent point.
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with a timestamp column.
    
    Returns
    -------
    Flight
        Flight object with 'time_since_start_of_trajectory' column assigned,
        containing the elapsed time in seconds from the trajectory start. The
        trajectory is sorted by timestamp.
    """
    flight_df = flight.data
    flight_df = flight_df.sort_values(by="timestamp")
    flight_df["time_since_start_of_trajectory"] = (flight_df["timestamp"] - flight_df["timestamp"].iloc[0]).dt.total_seconds()
    return Flight(flight_df)


def assign_distances_to_next_waypoint(flight: Flight) -> Flight:
    """
    Assigns distance to next waypoint for each point in the flight trajectory.
    
    Calculates the haversine distance between consecutive waypoints (latitude/longitude pairs).
    The last point in the trajectory has a distance of 0 (no next waypoint).
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with latitude and longitude columns.
    
    Returns
    -------
    Flight
        Flight object with 'distance_to_next' column assigned, containing the
        distance in kilometers to the next waypoint. The trajectory is sorted
        by timestamp before computation.
    """
    flight_df = flight.data.copy()
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)
    
    distances = []
    for i in range(1, len(flight_df)):
        prev_point = (flight_df.iloc[i - 1]['latitude'], flight_df.iloc[i - 1]['longitude'])
        curr_point = (flight_df.iloc[i]['latitude'], flight_df.iloc[i]['longitude'])
        distance = haversine_distance(prev_point[0], prev_point[1], curr_point[0], curr_point[1])
        distances.append(distance)
    distances.append(0)  # Last point has no next waypoint
    
    flight_df['distance_to_next'] = distances
    return Flight(flight_df)


def assign_rolling_cumulative_track_change(flight: Flight, time_window_seconds: int = 500) -> Flight:
    """
    Assigns rolling cumulative track change to a flight trajectory.
    
    Calculates the cumulative change in track (heading) over a rolling time window.
    Track differences are computed with wrap-around handling for angles (handles
    transitions across 0/360 degrees). The absolute value of the cumulative
    track change is stored.
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with track and timestamp columns.
    time_window_seconds : int, optional
        Time window in seconds for rolling cumulative calculation. Default is 500.
    
    Returns
    -------
    Flight
        Flight object with 'rolling_cumulative_track_change' column assigned,
        containing the absolute cumulative track change over the specified time
        window. The trajectory is sorted by timestamp before computation.
    """
    flight_df = flight.data.copy()
    flight_df['timestamp'] = pd.to_datetime(flight_df['timestamp'])
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Compute track difference with wrap-around handling
    flight_df['track_diff'] = flight_df['track'].diff()
    flight_df['track_diff'] = flight_df['track_diff'].apply(
        lambda x: x - 360 if x > 180 else x + 360 if x < -180 else x
    )
    
    # Set timestamp as index for time-based rolling window
    flight_df = flight_df.set_index('timestamp')
    
    # Compute rolling cumulative track change over time_window seconds
    flight_df['rolling_cumulative_track_change'] = flight_df['track_diff'].rolling(
        window=pd.Timedelta(seconds=time_window_seconds), min_periods=1, closed='right'
    ).sum().abs()
    
    flight_df = flight_df.reset_index() # Reset index to restore timestamp as a column
    flight_df['timestamp'] = pd.to_datetime(flight_df['timestamp'])
    return Flight(flight_df)
