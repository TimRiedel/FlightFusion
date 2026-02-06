import pandas as pd
from traffic.core import Traffic, Flight


def filter_traffic_by_most_common_airlines(traffic: Traffic, excluded_callsigns: list[str] = ["CHX", "GAF", "BPO", "FCK", "AMB", "DCA", "PBW", "HUM", "DIL"], top_n_airlines: int | None = None) -> Traffic:
    """
    Filters traffic by the most common airlines.
    
    Filters traffic to include only flights from the top N airlines with the most flight counts in Germany,
    excluding specified callsigns. The airline ranking is based on approach counts from a predefined CSV file.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain a 'callsign' column.
    excluded_callsigns : list[str], optional
        List of 3-letter ICAO airline codes to exclude from filtering. To exclude
        no airlines, set to an empty list. Default is ["CHX", "GAF", "BPO", "FCK", "AMB", "DCA", "PBW", "HUM", "DIL"].
    top_n_airlines : int | None, optional
        Number of top airlines to filter by. If None, no filtering by top airlines
        is applied (only exclusion filtering). Default is None.
    
    Returns
    -------
    Traffic
        Traffic object containing the filtered flights. Only flights from the
        top N airlines (excluding specified callsigns) are retained.
    """
    traffic_df = traffic.data

    if excluded_callsigns:
        traffic_df = traffic_df[~traffic_df['callsign'].str[:3].isin(excluded_callsigns)]
    if top_n_airlines is not None and top_n_airlines > 0:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        airline_file = os.path.join(current_dir, "assets", "airline-approach-counts-germany.csv")

        airlines = pd.read_csv(airline_file)
        airlines = airlines.dropna(subset=['Name'])
        airlines = airlines[~airlines['ICAO'].isin(excluded_callsigns)]
        airlines = airlines.head(top_n_airlines)
        airlines = airlines['ICAO'].tolist()
        traffic_df = traffic_df[traffic_df['callsign'].str[:3].isin(airlines)] 
    return Traffic(traffic_df)


def filter_traffic_by_flight_ids(traffic: Traffic, flight_ids: list[str]) -> tuple[Traffic, Traffic]:
    """
    Splits traffic into two groups based on flight IDs.
    
    Separates the input traffic into kept and removed groups based on whether
    each flight's flight_id is in the provided list.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain a 'flight_id' column.
    flight_ids : list[str]
        List of flight IDs to use for filtering. Flights with these IDs will
        be placed in the removed_traffic group.
    
    Returns
    -------
    tuple[Traffic, Traffic]
        Tuple of (kept_traffic, removed_traffic) where:
        - kept_traffic: Traffic object containing flights NOT in flight_ids
        - removed_traffic: Traffic object containing flights IN flight_ids
    """
    traffic_df = traffic.data.copy()
    kept_df = traffic_df[~traffic_df['flight_id'].isin(flight_ids)]
    removed_df = traffic_df[traffic_df['flight_id'].isin(flight_ids)]
    return Traffic(kept_df), Traffic(removed_df)


def remove_flight_from_traffic(traffic: Traffic, flight: Flight) -> Traffic:
    """
    Removes a specific flight from traffic by flight_id.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain a 'flight_id' column.
    flight : Flight
        Flight object whose flight_id will be used to identify which flight
        to remove from the traffic.
    
    Returns
    -------
    Traffic
        Traffic object with the specified flight removed.
    """
    traffic_df = traffic.data
    traffic_df = traffic_df[traffic_df['flight_id'] != flight.flight_id]
    return Traffic(traffic_df)

def filter_traffic_by_type(traffic: Traffic, traffic_type: str) -> Traffic:
    """
    Filters traffic by arrival or departure type.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain an 'is_arrival' column.
    traffic_type : str
        Type of traffic to filter. Options:
        - "arrivals": Return only arrival flights
        - "departures": Return only departure flights
        - "all": Return all traffic without filtering
    
    Returns
    -------
    Traffic
        Traffic object filtered by the specified type. If traffic_type is "all",
        returns the original traffic object unchanged.
    """
    traffic_df = traffic.data
    if traffic_type in ["arrivals"]:
        return Traffic(traffic_df[traffic_df["is_arrival"] == True])
    elif traffic_type in ["departures"]:
        return Traffic(traffic_df[traffic_df["is_arrival"] == False])
    elif traffic_type == "all":
        return traffic
    else:
        raise ValueError(f"Invalid traffic type: {traffic_type}")

def filter_traffic_by_runway(traffic: Traffic, selected_runways: list[str] | None = None) -> Traffic:
    """
    Filters traffic by selected runways. Keeps only flights that approach one of the selected runways.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain an 'ILS' column.
    selected_runways : list[str] | None
        List of runways to filter by. If None, no filtering is applied.
    
    Returns
    -------
    Traffic
        Traffic object containing only flights that approach one of the selected runways (kept_traffic).
    Traffic
        Traffic object containing only flights that do not approach one of the selected runways (removed_traffic).
    """
    if selected_runways is None or len(selected_runways) == 0:
        return traffic
    traffic_df = traffic.data
    flight_ids = traffic_df[traffic_df["ILS"].isin(selected_runways)]["flight_id"].unique().tolist()
    # The method returns traffic not in the flight_ids list first
    removed_traffic, runway_traffic = filter_traffic_by_flight_ids(traffic, flight_ids)
    return runway_traffic, removed_traffic

def drop_flights_with_same_departure_arrival_airport(traffic: Traffic, flightlist: pd.DataFrame) -> Traffic:
    """
    Removes flights with the same arrival and departure airport.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object containing flight trajectories.
    flightlist : pd.DataFrame
        Flightlist DataFrame containing flight information.
    
    Returns
    -------
    Traffic
        Traffic object containing flights with different arrival and departure airport.
    Traffic
        Traffic object containing flights with the same arrival and departure airport.
    """
    removed_flight_ids = []
    for flight in list(traffic):

        matches = flightlist
        if flight.callsign is None or flight.icao24 is None or flight.start is None or flight.stop is None:
            raise ValueError(f"Flight {flight.flight_id} has missing callsign, icao24, start, or end.")

        matches = matches[
            (matches['callsign'] == flight.callsign) & 
            (matches['icao24'] == flight.icao24) &
            (pd.to_datetime(matches['firstseen']) <= flight.start) &
            (pd.to_datetime(matches['lastseen']) >= flight.stop)
        ]

        if len(matches) > 1:
            raise ValueError(f"Flight {flight.flight_id} has multiple matches in the flightlist.")
        if len(matches) == 0:
            raise ValueError(f"Flight {flight.flight_id} does not have a match in the flightlist.")
        match = matches.iloc[0]
        if pd.notna(match['departure']) and pd.notna(match['arrival']) and match['departure'] == match['arrival']:
            removed_flight_ids.append(flight.flight_id)

    traffic, removed_traffic = filter_traffic_by_flight_ids(traffic, removed_flight_ids)
    return traffic, removed_traffic

def drop_flights_with_recurring_callsigns_per_day(traffic: Traffic) -> tuple[Traffic, Traffic]:
    """
    Removes groups of flights that have multiple flight_id values for the same base_flight_id.
    Circumstances where this can occur:
        - Helicopter flights (e.g. rescue helicopters)
        - General aviation flights doing multiple flights in a day
        - Flights that leave the specified airport radius and return to the same airport (split into departure and arrival flights)
    Removing them prevents us from training on undesired flights.
    
    Adds a base_flight_id column by taking the first three groups of the string (for example: RYR2732_4d234a_20241020_1 -> RYR2732_4d234a_20241020).
    Groups by base_flight_id and checks if any group has multiple distinct flight_id values.
    If a group has multiple flight_id values, the entire group is removed.
    
    Parameters
    -------
    traffic : Traffic
        Traffic object to filter. Must contain a 'flight_id' column.
    
    Returns
    -------
    tuple[Traffic, Traffic]
        Tuple of (kept_traffic, removed_traffic) where:
        - kept_traffic: Traffic object containing flights where each base_flight_id has only one flight_id
        - removed_traffic: Traffic object containing flights where base_flight_id groups have multiple flight_id values
    """
    traffic_df = traffic.data.copy()
    
    # Add base_flight_id column by removing sampling suffix pattern "_S<digits>"
    def extract_base_flight_id(flight_id: str) -> str:
        """Extract base flight ID by taking the first three groups of the string (for example: RYR2732_4d234a_20241020_1 -> RYR2732_4d234a_20241020)."""
        parts = str(flight_id).split('_')
        return '_'.join(parts[:3]) if len(parts) >= 3 else str(flight_id)
    
    traffic_df['base_flight_id'] = traffic_df['flight_id'].apply(extract_base_flight_id)
    
    # Group by base_flight_id and check for multiple flight_id values
    groups_with_multiple_flight_ids = []
    for base_flight_id, group in traffic_df.groupby('base_flight_id'):
        unique_flight_ids = group['flight_id'].nunique()
        if unique_flight_ids > 1:
            groups_with_multiple_flight_ids.append(base_flight_id)
    
    # Filter out groups with multiple flight_id values
    kept_df = traffic_df[~traffic_df['base_flight_id'].isin(groups_with_multiple_flight_ids)]
    removed_df = traffic_df[traffic_df['base_flight_id'].isin(groups_with_multiple_flight_ids)]
    
    # Drop the temporary base_flight_id column
    kept_df = kept_df.drop(columns=['base_flight_id'])
    removed_df = removed_df.drop(columns=['base_flight_id'])
    
    return Traffic(kept_df), Traffic(removed_df)


def merge_traffic(traffic1: Traffic, traffic2: Traffic) -> Traffic:
    """
    Merges two Traffic objects into a single Traffic object.
    
    Concatenates the data from both Traffic objects, combining all flights
    into a single Traffic object.
    
    Parameters
    -------
    traffic1 : Traffic
        First Traffic object to merge.
    traffic2 : Traffic
        Second Traffic object to merge.
    
    Returns
    -------
    Traffic
        Traffic object containing all flights from both input Traffic objects.
        The dataframes are concatenated with index reset.
    """
    return Traffic(pd.concat([traffic1.data, traffic2.data], ignore_index=True))