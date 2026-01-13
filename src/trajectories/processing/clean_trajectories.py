
import numpy as np
import pandas as pd
from typing import Union
from shapely.geometry import Polygon
from traffic.core import Flight, Traffic

from .compute_features import assign_distance_to_next_waypoint


def crop_traffic_to_circle(traffic: Union[Traffic, Flight], circle_wgs84: Polygon) -> Union[Traffic, Flight]:
    """
    Crops/clips traffic to a geographic circle.
    
    Parameters
    -------
    traffic : Traffic | Flight
        Traffic or Flight object to crop.
    circle_wgs84 : Polygon
        Shapely Polygon representing the circle in WGS84 coordinates.
        
    Returns
    -------
    Traffic | Flight
        Cropped Traffic or Flight object containing only points within the circle.
        The return type matches the input type.
    """
    traffic = traffic.clip(circle_wgs84).eval()
    return traffic


def drop_irrelevant_attributes(
    traffic: Traffic,
    attributes: list[str] = ["alert", "spi", "geoaltitude", "last_position", "lastcontact", "serials", "hour"]
) -> Traffic:
    """
    Drops specified columns (attributes) from a Traffic object.

    Parameters
    ----------
    traffic : Traffic
        The Traffic object containing trajectory data.
    attributes : list[str], optional
        List of attribute (column) names to remove from the underlying DataFrame.
        Defaults to ["alert", "spi", "geoaltitude", "last_position", "lastcontact", "serials", "hour"].

    Returns
    -------
    Traffic
        New Traffic object with the specified columns removed.
    """
    return traffic.drop(columns=attributes)


def remove_nan_values(trajectory: Union[Flight, Traffic], attribute: str) -> Union[Flight, Traffic]:
    """
    Removes rows with NaN values for a specified attribute from a trajectory.
    
    Parameters
    -------
    trajectory : Flight | Traffic
        Flight or Traffic object containing trajectory data.
    attribute : str
        Name of the attribute/column to check for NaN values.
    
    Returns
    -------
    Flight | Traffic
        Flight or Traffic object with rows containing NaN values in the specified
        attribute removed. The return type matches the input type.
    """
    trajectory_df = trajectory.data
    trajectory_df = trajectory_df[trajectory_df[attribute].notna()]
    if isinstance(trajectory, Traffic):
        return Traffic(trajectory_df)
    else:
        return Flight(trajectory_df)


def remove_duplicate_positions(flight: Flight) -> Flight:
    """
    Removes consecutive duplicate positions from a flight trajectory.
    
    There are cases where the same position is recorded multiple times within
    a few seconds, leading to invalid data and a non-continuous trajectory.
    Only removes consecutive duplicates (same lat/lon appearing sequentially).
    If the aircraft flies through the same position twice with other positions
    in between, both occurrences are kept.
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with latitude and longitude columns.
    
    Returns
    -------
    Flight
        Flight object with consecutive duplicate positions removed. The trajectory
        is sorted by timestamp before duplicate removal.
    """
    flight_df = flight.data.copy()
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Identify consecutive duplicates: same lat/lon as previous row
    is_consecutive_duplicate = (
        (flight_df['latitude'] == flight_df['latitude'].shift(1)) & 
        (flight_df['longitude'] == flight_df['longitude'].shift(1))
    )
    
    flight_df["is_duplicate"] = is_consecutive_duplicate
    flight_df = flight_df[~is_consecutive_duplicate].reset_index(drop=True)
    
    # Remove temporary column used for duplicate detection
    flight_df = flight_df.drop(columns=['is_duplicate'], errors='ignore')

    return Flight(flight_df)


def remove_outliers(flight: Flight, attribute: str, threshold: float, window_size: int = 10, handle_outliers: str = "drop") -> Flight:
    """
    Removes or interpolates outliers in a specified attribute using a rolling window approach.
    
    Uses a progressive window approach:
    - For initial points: forward-looking window that gradually shifts to backward-looking
    - For points after window_size: standard backward-looking rolling window
    
    Common threshold values:
    - 30 for altitude
    - 20 for groundspeed
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data.
    attribute : str
        Name of the attribute/column to check for outliers (e.g., 'altitude', 'groundspeed').
    threshold : float
        Maximum allowed deviation from rolling minimum. Values exceeding this threshold
        are considered outliers. Use np.nan to disable outlier detection.
    window_size : int, optional
        Size of the rolling window for calculating the mean. Default is 10.
    handle_outliers : str, optional
        How to handle detected outliers. Options:
        - "drop": Remove outlier rows from the trajectory (default)
        - "interpolate": Replace outlier values with interpolated values
    
    Returns
    -------
    Flight
        Flight object with outliers removed or interpolated. The trajectory is sorted
        by timestamp, and additional columns 'rolling_mean_{attribute}' and
        '{attribute}_deviation' are added to the data.
    """
    if handle_outliers not in ["drop", "interpolate"]:
        raise ValueError(f"Invalid value for handle_outliers: {handle_outliers}. Must be 'drop' or 'interpolate'.")
    
    flight_df = flight.data.copy()
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Use pandas rolling for backward-looking means (excluding current point)
    # shift(1) moves the mean so it represents the mean of previous window_size points
    rm = flight_df[attribute].rolling(
        window_size, min_periods=window_size
    ).min().shift(1)

    # Convert rolling means and values to numpy arrays without read-only values
    rolling_means = np.zeros(len(rm))
    vals = np.zeros(len(flight_df))
    for i in range(len(flight_df)):
        rolling_means[i] = rm.iloc[i]
        value = flight_df[attribute].iloc[i]
        vals[i] = np.nan if pd.isna(value) else float(value)

    # Handle the first `window_size` points separately
    # Use a forward-backward looking window to calculate the rolling mean.
    for i in range(window_size):
        window = np.delete(vals[:window_size + 1], i)
        # Some flights have no groundspeed data at all,
        # if all values are Nan, set the rolling_means[i] to nan as well
        if all(np.isnan(window)):
            rolling_means[i] = np.nan
            continue
        m = np.nanmean(window)
        rolling_means[i] = m
        
        # Edge case: If the first point is an outlier, increase the rolling mean by the average differential slope
        if i == 0 and vals[i] >= m + threshold:
            # Calculate average differential slope in the window
            # Only consider non-NaN to avoid propagation of missing values
            diffs = np.diff(window)
            valid_diffs = diffs[~np.isnan(diffs)]
            avg_slope = np.nanmean(valid_diffs) if len(valid_diffs) > 0 else 0.0
            # If average slope is negative (descending aircraft), increase the rolling mean by the average slope and vice versa
            vals[i] = m - avg_slope
    flight_df[attribute] = vals

    # Assign rolling mean and deviation columns
    flight_df[f'rolling_mean_{attribute}'] = rolling_means
    flight_df[f'{attribute}_deviation'] = np.abs(flight_df[attribute] - flight_df[f'rolling_mean_{attribute}']).fillna(0)

    # Detect outliers and set them to NaN
    if not np.isnan(threshold):
        outliers = flight_df[f'{attribute}_deviation'] > threshold
    else:
        outliers = pd.Series([False] * len(flight_df))
    
    if handle_outliers == "drop":
        flight_df = flight_df[~outliers]
    elif handle_outliers == "interpolate":
        flight_df.loc[outliers, attribute] = np.nan
        flight_df[attribute] = flight_df[attribute].interpolate(method='linear')
    
    flight_df = flight_df.drop(columns=[f"rolling_mean_{attribute}", f"{attribute}_deviation"], errors='ignore')
    return Flight(flight_df)


def remove_lateral_outliers(flight: Flight, window_size: int = 100, deviation_factor: int = 10) -> Flight:
    """
    Removes lateral outliers from a flight trajectory based on distance to next waypoint.
    
    This method identifies outliers by computing the distance to the next waypoint
    for each point, then uses a rolling window to calculate deviations. Points with
    deviations exceeding a scaled threshold (based on median deviation) are removed.
    
    Credit: Ricardo Reinke
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with latitude and longitude columns.
    window_size : int, optional
        Size of the rolling window for calculating the mean distance. Default is 100.
    deviation_factor : int, optional
        Multiplier for the median deviation to determine the outlier threshold.
        Default is 10.
    
    Returns
    -------
    Flight
        Flight object with lateral outliers removed. The trajectory is filtered to
        keep only points within the calculated deviation threshold.
    """
    flight = assign_distance_to_next_waypoint(flight)
    flight_df = flight.data
    flight_df['distance_to_next'] = pd.to_numeric(flight_df['distance_to_next'], errors='coerce')
    flight_df = flight_df.dropna(subset=['distance_to_next'])

    flight_df['rolling_mean'] = flight_df['distance_to_next'].rolling(window_size, min_periods=1).mean()
    flight_df['deviation'] = np.abs(flight_df['distance_to_next'] - flight_df['rolling_mean']).fillna(0)

    median_deviation = flight_df['deviation'].median()
    scaled_threshold = deviation_factor * median_deviation if median_deviation > 0 else np.nan

    if not np.isnan(scaled_threshold): # keep only the points that are within the threshold
        df_clean = flight_df[flight_df['deviation'] <= scaled_threshold]
    else:
        df_clean = flight_df.copy()

    df_clean = df_clean.drop(columns=['rolling_mean', 'deviation', 'distance_to_next'])

    return Flight(df_clean)


def recompute_track(flight: Flight) -> Flight:
    """
    Computes track (heading/direction of movement) from latitude/longitude positions.
    
    Track is calculated as the bearing from each point to the next point in the
    trajectory using the great circle bearing formula. All track values are
    recomputed and overwrite any existing track values.
    
    Parameters
    -------
    flight : Flight
        Flight object containing trajectory data with latitude and longitude columns.
    
    Returns
    -------
    Flight
        Flight object with all track values recomputed from lat/lon positions.
        Track values are normalized to the range 0-360 degrees. The trajectory
        is sorted by timestamp before computation.
    """
    flight_df = flight.data.copy()
    flight_df = flight_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Compute track for all points (bearing from current point to next point)
    lat1 = np.radians(flight_df['latitude'].shift(1))
    lon1 = np.radians(flight_df['longitude'].shift(1))
    lat2 = np.radians(flight_df['latitude'])
    lon2 = np.radians(flight_df['longitude'])
    
    # Calculate bearing between consecutive points
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    
    computed_track = np.degrees(np.arctan2(x, y))
    computed_track = (computed_track + 360) % 360  # Normalize to 0-360
    
    # Update all track values (overwrite existing values)
    flight_df['track'] = computed_track
    return Flight(flight_df)


def clip_altitude(trajectory: Union[Flight, Traffic], lower_limit: int = 0, upper_limit: int = 40000) -> Union[Flight, Traffic]:
    """
    Clips altitude values to specified lower and upper limits.
    
    Values below the lower limit are set to the lower limit, and values above
    the upper limit are set to the upper limit.
    
    Parameters
    -------
    trajectory : Flight | Traffic
        Flight object containing trajectory data with an altitude column.
    lower_limit : int, optional
        Minimum allowed altitude value in feet. Default is 0.
    upper_limit : int, optional
        Maximum allowed altitude value in feet. Default is 40000.
    
    Returns
    -------
    Flight | Traffic
        Flight or Traffic object with altitude values clipped to the specified range. The return type matches the input type.
    """
    trajectory_df = trajectory.data
    trajectory_df['altitude'] = trajectory_df['altitude'].clip(lower=lower_limit, upper=upper_limit)
    if isinstance(trajectory, Traffic):
        return Traffic(trajectory_df)
    else:
        return Flight(trajectory_df)
