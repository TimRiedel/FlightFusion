import numpy as np
import pyproj
from dataclasses import dataclass
from shapely.geometry import Point
from shapely.ops import transform

@dataclass
class Bounds:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

def get_proj_wgs84():
    return pyproj.CRS("EPSG:4326")

def get_proj_aeqd(lat, lon):
    return pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")

def get_circle_around_location(lat, lon, radius_m):
    """
    Creates a circle around a location in meters using the AEQD projection and projects it back to WGS84.
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        radius_m: Radius of the circle in meters

    Returns:
        circle_wgs84: Circle around the location in WGS84
    """
    proj_wgs84 = get_proj_wgs84()
    proj_aeqd = get_proj_aeqd(lat, lon)
    project_to_aeqd = pyproj.Transformer.from_crs(proj_wgs84, proj_aeqd, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(proj_aeqd, proj_wgs84, always_xy=True).transform

    # Build the circle
    point = Point(lon, lat)
    circle_aeqd = transform(project_to_aeqd, point).buffer(radius_m)   # circle in meters
    circle_wgs84 = transform(project_to_wgs84, circle_aeqd)            # circle in lat/lon

    return circle_wgs84

def get_curvilinear_grid_around_location(lat0, lon0, radius_m, num_x, num_y):
    """
    Build a curvilinear lat-lon grid centered on (lat0, lon0),
    spanning `radius_m * 2` x `radius_m * 2` in local AEQD space,
    with resolution num_x x num_y.
    """
    proj_wgs84 = get_proj_wgs84()
    proj_aeqd = get_proj_aeqd(lat0, lon0)
    to_aeqd = pyproj.Transformer.from_crs(proj_wgs84, proj_aeqd, always_xy=True).transform
    to_wgs  = pyproj.Transformer.from_crs(proj_aeqd, proj_wgs84, always_xy=True).transform

    x = np.linspace(-radius_m, radius_m, num_x)
    y = np.linspace(-radius_m, radius_m, num_y)
    X, Y = np.meshgrid(x, y)

    lon, lat = to_wgs(X, Y)

    return lat, lon