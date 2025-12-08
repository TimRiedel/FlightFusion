import datetime
import pandas as pd
import shapely
from traffic.data import opensky
from traffic.core import Traffic

def download_flightlist(icao: str, start: str, end: str):
    """
    Downloads a flight list from OpenSky Network for a specified airport and time range.
    
    Parameters
    -------
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    start : str
        Start time for the query in format YYYY-MM-DDTHH:MM.
    end : str
        End time for the query in format YYYY-MM-DDTHH:MM.
    
    Returns
    -------
    DataFrame
        DataFrame containing flight list information from OpenSky Network.
    """
    flightlist = opensky.flightlist(
        airport=icao,
        start=start,
        stop=end
    )
    return flightlist

def download_flight(callsign: str, start: str, end: str):
    """
    Downloads a single flight trajectory from OpenSky Network by callsign.
    
    Parameters
    -------
    callsign : str
        Aircraft callsign (e.g., 'DLH123').
    start : str
        Start time for the query in format YYYY-MM-DDTHH:MM.
    end : str
        End time for the query in format YYYY-MM-DDTHH:MM.
    
    Returns
    -------
    Flight
        Flight object containing the trajectory data for the specified callsign
        and time range.
    """
    flight = opensky.history(
        callsign=callsign,
        start=start,
        stop=end,
        return_flight=True
    )
    return flight

def download_traffic(icao: str, start: str, end: str, bounds: tuple | shapely.geometry.Polygon, traffic_type: str = "all"):
    """
    Downloads traffic data from OpenSky Network for arrivals and/or departures.
    
    Downloads flight trajectories for a specified airport and time range, optionally
    filtered by arrivals or departures. Each point in the returned traffic is marked
    with an 'is_arrival' flag indicating whether it belongs to an arrival or departure
    at the given airport.
    
    Parameters
    -------
    icao : str
        ICAO code of the airport (e.g., 'EDDF' for Frankfurt).
    start : str
        Start time for the query in format YYYY-MM-DDTHH:MM.
    end : str
        End time for the query in format YYYY-MM-DDTHH:MM.
    bounds : tuple | shapely.geometry.Polygon
        Geographic bounds to filter traffic. Can be a tuple (min_lon, min_lat, max_lon, max_lat)
        or a shapely.geometry.Polygon object like a circle, which has the bounds attribute.
    traffic_type : str, optional
        Type of traffic to download. Options:
        - "all": Download both arrivals and departures (default)
        - "arrivals": Download only arrivals
        - "departures": Download only departures
    
    Returns
    -------
    Traffic
        Traffic object containing flight trajectories. Each point has an 'is_arrival'
        column indicating whether it belongs to an arrival (True) or departure (False).
    """
    if traffic_type not in ["all", "arrivals", "departures"]:
        raise ValueError(f"Invalid traffic type: {traffic_type}. Must be 'all', 'arrivals', or 'departures'.")

    if traffic_type in ["all", "departures"]:
        departure_traffic = opensky.history(
            departure_airport=icao,
            start=start,
            stop=end,
            bounds=bounds
        )
        departure_df = departure_traffic.data
        departure_df["is_arrival"] = False
        departure_traffic = Traffic(departure_df)

    if traffic_type in ["all", "arrivals"]:
        arrival_traffic = opensky.history(
            arrival_airport=icao,
            start=start,
            stop=end,
            bounds=bounds
        )
        arrival_df = arrival_traffic.data
        arrival_df["is_arrival"] = True
        arrival_traffic = Traffic(arrival_df)

    if traffic_type == "all":
        traffic = Traffic(pd.concat([departure_traffic.data, arrival_traffic.data]))
    elif traffic_type == "departures":
        traffic = departure_traffic
    elif traffic_type == "arrivals":
        traffic = arrival_traffic

    return traffic