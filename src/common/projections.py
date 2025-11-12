import pyproj
from shapely.geometry import Point
from shapely.ops import transform

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
    # Create a local AEQD projection for building accurate circle in meters
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    proj_aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    project_to_aeqd = pyproj.Transformer.from_crs(proj_wgs84, proj_aeqd, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(proj_aeqd, proj_wgs84, always_xy=True).transform

    # Build the circle
    point = Point(lon, lat)
    circle_aeqd = transform(project_to_aeqd, point).buffer(radius_m)   # circle in meters
    circle_wgs84 = transform(project_to_wgs84, circle_aeqd)            # back to lat/lon

    return circle_wgs84