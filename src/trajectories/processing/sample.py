import pandas as pd
from traffic.core import Traffic, Flight

from .compute_features import assign_flight_sample_id

def clip_trajectories(traffic: Traffic, trajectory_length_minutes: int) -> Traffic:
    """
    Clips trajectories to a specified length.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories to clip.
    trajectory_length_minutes : int
        Length in minutes to clip the trajectories to.
    
    Returns
    -------
    Traffic
        Traffic object containing clipped flights.
    """
    clipped_flights = []
    removal_reasons = []
    for flight in traffic:
        if flight.duration < pd.Timedelta(minutes=trajectory_length_minutes):
            removal_reasons.append(f"Flight {flight.flight_id} is too short to clip. Skipping.")
        else:
            clipped_flight = flight.last(minutes=trajectory_length_minutes)
            clipped_flights.append(clipped_flight)
    return Traffic(pd.concat([flight.data for flight in clipped_flights], ignore_index=True)), removal_reasons


def sample_trajectories(traffic: Traffic, data_period_minutes: int, min_trajectory_length_minutes: int = 5) -> Traffic:
    """
    Samples trajectories by creating multiple prediction samples from each flight.
    
    For each flight creates multiple samples by taking segments at regular intervals.
    Each sample is assigned a unique sample ID (e.g., "flight_id_S1", "flight_id_S2").
    Flights shorter than min_trajectory_length_minutes are excluded from sampling.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories to sample.
    data_period_minutes : int
        Time interval in minutes between consecutive samples. Each sample starts
        data_period_minutes after the previous sample.
    min_trajectory_length_minutes : int, optional
        Minimum trajectory length in minutes required for a flight to be sampled.
        Flights shorter than this are excluded. Default is 5.
    
    Returns
    -------
    Traffic
        Traffic object containing sampled flights. Each sample is assigned a unique
        flight_id with a sample index suffix (e.g., "flight_id_S1").
    """
    traffic = traffic.drop(columns=["track_unwrapped"])
    sampled_flights = []
    for flight in traffic:
        flight_duration = flight.duration.components.minutes + 1
        num_prediction_samples = (flight_duration - min_trajectory_length_minutes) // data_period_minutes

        for sample_index in range(num_prediction_samples):
            sampled_flight = flight
            sampled_flight = assign_flight_sample_id(sampled_flight, sample_index + 1)
            sampled_flights.append(sampled_flight)

            flight = flight.skip(minutes=data_period_minutes)

    return Traffic(pd.concat([flight.data for flight in sampled_flights], ignore_index=True))


def get_input_horizon_segments(traffic: Traffic, input_time_minutes: int, horizon_time_minutes: int, resampling_rate_seconds: int) -> tuple[Traffic, Traffic]:
    """
    Splits each flight into input and horizon segments for prediction tasks.

    For each flight in the traffic, extracts two segments:
    - Input segment: First input_time_minutes of the flight.
    - Horizon segment: Next horizon_time_minutes after the input segment.

    This is typically used for trajectory prediction where the input segment
    represents historical data and the horizon segment represents the target
    prediction window (ground truth).
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories to split.
    input_time_minutes : int
        Duration in minutes for the input segment (historical data).
    horizon_time_minutes : int
        Duration in minutes for the horizon segment (ground truth).
    resampling_rate_seconds : int
        Resampling rate in seconds. Used for calculating the number of samples
        in each segment (informational only, does not resample the data).
    
    Returns
    -------
    tuple[Traffic, Traffic]
        Tuple of (input_segments, horizon_segments) where:
        - input_segments: Traffic object containing the first input_time_minutes
          of each flight
        - horizon_segments: Traffic object containing the next horizon_time_minutes
          of each flight after the input segment
    """
    input_time_seconds = input_time_minutes * 60
    horizon_seconds = horizon_time_minutes * 60
    num_input_samples = input_time_seconds // resampling_rate_seconds
    num_horizon_samples = horizon_seconds // resampling_rate_seconds
    # print(f"Input time: {input_time_minutes} minutes, Horizon: {horizon_time_minutes} minutes, Resampling rate: {resampling_rate_seconds} seconds")
    # print(f"    -> Number of input samples: {num_input_samples}, Number of horizon samples: {num_horizon_samples}")

    input_segments = []
    horizon_segments = []

    for flight in traffic:
        input_segment = flight.first(minutes=input_time_minutes)
        horizon_segment = flight.skip(minutes=input_time_minutes).first(minutes=horizon_time_minutes)
        input_segments.append(input_segment)
        horizon_segments.append(horizon_segment)

    input_segments_traffic = Traffic(pd.concat([flight.data for flight in input_segments], ignore_index=True))
    horizon_segments_traffic = Traffic(pd.concat([flight.data for flight in horizon_segments], ignore_index=True))
    return input_segments_traffic, horizon_segments_traffic