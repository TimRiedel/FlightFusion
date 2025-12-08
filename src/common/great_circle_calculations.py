import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius: int = 6371) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    The Haversine formula determines the shortest distance over the Earth's surface
    between two points, accounting for the Earth's spherical shape.
    
    Parameters
    -------
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.
        earth_radius: Radius of the Earth in kilometers. Default is 6371 km.

    Returns
    -------
        float: Distance between the two points in kilometers.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


def haversine_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing (direction) from one point to another on a great circle.
    
    The bearing is the initial compass direction you would need to travel from the first point
    to reach the second point, measured clockwise from north (0° = North, 90° = East,
    180° = South, 270° = West).
    
    Parameters
    -------
        lat1: Latitude of the starting point in degrees.
        lon1: Longitude of the starting point in degrees.
        lat2: Latitude of the destination point in degrees.
        lon2: Longitude of the destination point in degrees.
    
    Returns
    -------
        float: Initial bearing in degrees, normalized to the range [0, 360).
        
    Note
    -------
        This is the initial bearing at the starting point. The bearing changes along
        the great circle path.
    """
    lambda1, phi1, lambda2, phi2 = map(np.radians, [lon1, lat1, lon2, lat2])

    y = np.sin(lambda2 - lambda1) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lambda2 - lambda1)
    theta = np.arctan2(y, x)

    return np.degrees(theta) % 360
