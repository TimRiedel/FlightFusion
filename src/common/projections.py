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

def get_projection_wgs84_to_aeqd(lat, lon):
    return pyproj.Transformer.from_crs(get_proj_wgs84(), get_proj_aeqd(lat, lon), always_xy=True).transform

def get_projection_aeqd_to_wgs84(lat, lon):
    return pyproj.Transformer.from_crs(get_proj_aeqd(lat, lon), get_proj_wgs84(), always_xy=True).transform

def get_circle_around_location(lat, lon, radius_m):
    """Create a circel around a geographic location with a specified radius in meters.
    It uses the Azimuthal Equidistant (AEQD) projection to ensure accurate
    distance measurements, then projects the resulting circle back to WGS84 coordinates.
    
    Parameters
    -------
        lat: Latitude of the center point in degrees (WGS84).
        lon: Longitude of the center point in degrees (WGS84).
        radius_m: Radius of the circle in meters.
    
    Returns
    -------
        shapely.geometry.Polygon: A polygon representing the circular buffer in WGS84
            coordinates. The circle appears as an ellipse in lat/lon space due to
            the projection transformation.
    """
    point = Point(lon, lat)
    circle_aeqd = transform(get_projection_wgs84_to_aeqd(lat, lon), point).buffer(radius_m)   # circle in meters
    circle_wgs84 = transform(get_projection_aeqd_to_wgs84(lat, lon), circle_aeqd)            # back to lat/lon

    return circle_wgs84

def get_curvilinear_grid_around_location(lat0, lon0, radius_m, num_x, num_y):
    """Generate a curvilinear latitude-longitude grid centered on a location.
    
    Creates a regular grid in local Azimuthal Equidistant (AEQD) space centered on
    the specified location, then projects it back to WGS84 coordinates. The grid
    spans a square region of `radius_m * 2` by `radius_m * 2` meters in the local
    projection, which appears as a curvilinear grid in lat/lon space.
    
    Parameters
    -------
        lat0: Latitude of the grid center point in degrees (WGS84).
        lon0: Longitude of the grid center point in degrees (WGS84).
        radius_m: Half-width of the grid in meters. The grid spans from -radius_m
            to +radius_m in both x and y directions in AEQD space.
        num_x: Number of grid points in the x-direction (longitude).
        num_y: Number of grid points in the y-direction (latitude).
    
    Returns
    -------
        tuple: A tuple (lat, lon) containing:
            - lat: 2D numpy array of latitude values in degrees, shape (num_y, num_x).
            - lon: 2D numpy array of longitude values in degrees, shape (num_y, num_x).
    """
    x = np.linspace(-radius_m, radius_m, num_x)
    y = np.linspace(-radius_m, radius_m, num_y)
    X, Y = np.meshgrid(x, y)

    transformer = get_projection_aeqd_to_wgs84(lat0, lon0)
    lon, lat = transformer(X, Y)

    return lat, lon