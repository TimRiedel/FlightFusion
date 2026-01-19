from itertools import islice
import os
from cartopy.io.img_tiles import OSM
from cartopy.mpl.feature_artist import cfeature
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from traffic.core import Traffic
from traffic.data import airports

from .projections import get_circle_around_location
from .great_circle_calculations import haversine_distance


# --------------------------------
# Plotting Functions
# --------------------------------

def make_altitude_histogram(traffic):
    plt.hist(traffic['altitude'], bins=100)
    plt.xlabel('Altitude (in ft)')
    plt.ylabel('Frequency')
    plt.title(f"Altitude Histogram")
    plt.show()

def plot_flight_altitude_profile(flight, title=None):
    """
    Plot altitude and groundspeed profile for a flight.
    
    Parameters:
    -----------
    flight : Flight
        Flight object to plot
    title : str, optional
        Optional title for the plot. If not provided, defaults to 
        "Altitude and Groundspeed Profile for {callsign} on {date}"
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    flight.plot_time(ax=ax, y=["altitude", "groundspeed"], secondary_y=["groundspeed"])
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=0)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    if title is None:
        date_str = pd.to_datetime(flight.start).strftime("%Y-%m-%d")
        title = f"Altitude and Groundspeed Profile for {flight.callsign} on {date_str}"
    ax.set_title(title)
    return fig, ax

def plot_traffic(traffic, icao, attribute='altitude', plot_osm=True, plot_individual_flights=False, title=None, airport_circle=None,
                cmap='viridis', y_axis_label='Altitude (in feet)'):
    """
    Configurable plotting function for traffic data.
    
    Parameters:
    -----------
    traffic : Traffic
        Traffic object containing flight data
    icao : str
        ICAO airport code
    plot_osm : bool, default=True
        Whether to add OSM map tiles
    attribute : str, default='altitude'
        Attribute to use for colorization
    plot_individual_flights : bool, default=False
        If True, plot each flight in separate subplots
    airport_circle : shapely.geometry.Polygon, optional
        Circle geometry in WGS84 around the airport. If provided, the circle will be plotted
        and the plot extent will be set to the circle bounds. Otherwise, bounds are computed
        from the trajectories.
    cmap : str, default='viridis'
        Colormap to use for the attribute visualization. Examples: 'viridis', 'plasma', 'inferno', 
        'magma', 'Reds', 'YlOrRd', 'hot', 'cool', etc.
    y_axis_label : str, default='Altitude (in feet)'
        Label for the y-axis (colorbar).
    title : str, optional
        Title for the plot(s). If plot_individual_flights=True, each subplot title will be 'title - callsign'
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax or axes : matplotlib.axes or array of axes
        The axes object(s)
    """
    all_trajectories = traffic.data
    lat, lon = airports[icao].latlon
    proj_lcc = get_map_projection(lat, lon)
    norm = create_normalization(all_trajectories, attribute)
    
    if plot_individual_flights:
        num_flights = len(list(traffic))
        num_cols = 3
        num_rows = (num_flights + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(24, 7 * num_rows), dpi=120)

        for idx, flight in enumerate(traffic):
            ax = fig.add_subplot(num_rows, num_cols, idx + 1, projection=proj_lcc)
            ax = populate_plot(ax, flight.data, icao, plot_osm, norm, attribute, 
                                airport_circle, cmap, y_axis_label)
            ax = set_subplot_title(ax, title, icao, flight.callsign)
        plt.subplots_adjust(wspace=0.3, hspace=0.3) # Adjust spacing between subplots to prevent overlap
    else:
        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax = plt.axes(projection=proj_lcc)
        ax = populate_plot(ax, traffic.data, icao, plot_osm, norm, attribute, 
                          airport_circle, cmap, y_axis_label)
        ax = set_subplot_title(ax, title, icao)
    return fig, ax


# --------------------------------
# Helper Functions
# --------------------------------

def populate_plot(ax, trajectories, icao, plot_osm, norm, attribute, airport_circle,
                  cmap='viridis', y_axis_label=None):
    """Populate the plot with traffic data and optional circle overlay."""
    ax = setup_axes_for_traffic(ax, trajectories, icao, plot_osm, airport_circle)
    if airport_circle is not None:
        ax = add_circle_to_plot(ax, icao, airport_circle)
    ax = add_colorized_traffic_scatter(ax, trajectories, norm, cmap=cmap, attribute=attribute)
    ax = add_colorbar(ax, norm, cmap=cmap, y_axis_label=y_axis_label)
    return ax

# ---------- Traffic Scatter ----------

def add_colorized_traffic_scatter(ax, trajectories, norm, cmap='viridis', attribute='altitude'):
    ax.scatter(
        trajectories['longitude'], 
        trajectories['latitude'],
        c=trajectories[attribute],
        cmap=cmap,
        norm=norm,
        alpha=0.5, 
        s=1.0,
        transform=ccrs.PlateCarree(),
        zorder=3 
    )
    return ax

# ---------- Axes Setup ----------

def setup_axes_for_traffic(ax, trajectories, icao, plot_osm, airport_circle=None):
    """Set up axes with bounds, extent, and map features.
    
    If airport_circle is provided: use circle bounds for extent.
    Otherwise: compute bounds from trajectories.
    """
    if airport_circle is not None:
        bounds = get_bounds_from_circle(airport_circle)
    else:
        bounds = compute_bounds(trajectories)
    ax = set_extent(ax, bounds)
    ax = add_map_to_plot(ax, icao, plot_osm)
    return ax

def set_extent(ax, bounds):
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    # ax.set_extent([min_lon - 0.1, max_lon + 0.1, min_lat - 0.1, max_lat + 0.1], crs=ccrs.PlateCarree())
    return ax

def get_map_projection(lat, lon):
    return ccrs.LambertConformal(
        central_longitude=lon,
        central_latitude=lat,
        standard_parallels=(lat-1.5, lat+1.5)
    )

# ---------- Circle Plotting ----------

def add_circle_to_plot(ax, icao, circle_wgs84):
    """Add a circle overlay to the plot. Radius is calculated from the circle geometry."""
    lat, lon = airports[icao].latlon

    boundary_point = circle_wgs84.boundary.coords[0]
    radius_km = haversine_distance(lat, lon, boundary_point[1], boundary_point[0])
    x, y = circle_wgs84.exterior.xy
    ax.plot(x, y, color='black', linewidth=1, transform=ccrs.PlateCarree())

    # Add radius label in top right corner
    ax.text(0.98, 0.98, f"{radius_km:.0f} km radius", 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return ax

# ---------- Map Plotting ----------

def add_map_to_plot(ax, icao, plot_osm=True):
    tiler = OSM() if plot_osm else None
    if tiler is not None:
        ax.add_image(tiler, 10) 
        ax.add_feature(cfeature.OCEAN, facecolor="#a6cee3", alpha=0.6, zorder=1)
        ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgrey", alpha=0.6, zorder=1)
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5, alpha=0.8, zorder=2)
    else:
        ax.coastlines(resolution='10m')

    ax.gridlines(
        draw_labels={"bottom": "x", "left": "y", "right": "y"},
        linestyle='--',
        alpha=0.5,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
        xpadding=10,
        ypadding=10
    )
    
    airports[icao].plot(ax, footprint=False, runways=dict(linewidth=1))
    return ax

def get_bounds_from_circle(circle_wgs84, buffer_percent=0.06):
    """Get bounds from a circle geometry in the format required by set_extent.
    
    Adds a buffer around the circle bounds to improve visual appearance.
    
    Parameters:
    -----------
    circle_wgs84 : shapely.geometry.Polygon
        Circle geometry in WGS84
    buffer_percent : float, default=0.08
        Buffer percentage to add around the circle (8% by default)
    
    Returns:
    --------
    tuple : (min_lon, max_lon, min_lat, max_lat)
    """
    bounds = circle_wgs84.bounds  # (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Calculate buffer based on the size of the circle
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    lon_buffer = lon_range * buffer_percent
    lat_buffer = lat_range * buffer_percent
    
    # Apply buffer
    min_lon -= lon_buffer
    max_lon += lon_buffer
    min_lat -= lat_buffer
    max_lat += lat_buffer
    
    return min_lon, max_lon, min_lat, max_lat

def compute_bounds(trajectories, center_lat=None, center_lon=None, max_radius_km=None):
    """Compute bounds for the plot.
    
    If center_lat and center_lon are provided, the plot will be centered on those coordinates.
    Otherwise, it will be centered on the trajectories.

    Computes the maximum distance from the center to points directly north, south, east, and west
    of the center, using the trajectory bounding box extents.

    Parameters:
    -----------
    max_radius_km : float, optional
        Maximum radius in kilometers when centering on a specific point (e.g., airport).
        This prevents zooming out too far when trajectories are spread out.
    """
    min_lon, max_lon = trajectories['longitude'].min(), trajectories['longitude'].max()
    min_lat, max_lat = trajectories['latitude'].min(), trajectories['latitude'].max()

    # Use provided center or compute from trajectories
    if center_lat is None or center_lon is None:
        center_lat = (max_lat + min_lat) / 2
        center_lon = (max_lon + min_lon) / 2

    # Use bounding box approach for both cases
    buffer = 0.1
    cardinal_points = [
        (center_lon, max_lat + buffer),  # North
        (center_lon, min_lat - buffer),  # South
        (max_lon + buffer, center_lat),  # East
        (min_lon - buffer, center_lat),  # West
    ]
    
    max_distance = 0
    for point_lon, point_lat in cardinal_points:
        distance = haversine_distance(point_lat, point_lon, center_lat, center_lon)
        if distance > max_distance:
            max_distance = distance

    # Cap the maximum distance if specified (useful when centering on airport)
    if max_radius_km is not None and max_distance > max_radius_km:
        max_distance = max_radius_km

    max_distance *= 1000 # convert to meters
    circle_wgs84 = get_circle_around_location(center_lat, center_lon, max_distance)
    bounds = circle_wgs84.bounds # (min_lon, min_lat, max_lon, max_lat)
    return bounds[0], bounds[2], bounds[1], bounds[3] # min_lon, max_lon, min_lat, max_lat (as required by set_extent)

# ---------- Colorization ----------

def add_colorbar(ax, norm, cmap='viridis', y_axis_label='Altitude (in feet)'):
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.675, pad=0.15)
    cbar.set_label(y_axis_label, labelpad=10, fontsize=10)
    return ax

def create_normalization(trajectories, attribute):
    """Create a color normalization based on trajectory attribute values."""
    return mcolors.Normalize(vmin=trajectories[attribute].min(), vmax=trajectories[attribute].max())

# ---------- Subplot Titles ----------

def set_subplot_title(ax, title, icao=None, callsign=None):
    """Set title for a subplot. If callsign provided, format as 'title - callsign'."""
    fontsize = 12
    pad = 10
    if callsign:
        if title:
            ax.set_title(f"{title} - {callsign}", fontsize=fontsize, pad=pad)
        else:
            ax.set_title(f"{callsign}", fontsize=fontsize, pad=pad)
    else:
        if title:
            plt.title(title, fontsize=fontsize, pad=pad)
        else:
            plt.title(f"{icao}", fontsize=fontsize, pad=pad)
    return ax


# ---------- Saving ----------

def save_plot(fig, save_path):
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot at {save_path}")
    plt.close(fig)

def save_day_plot(fig, plot_type, icao, timestamp):
    print("Saving plot")
    output_dir = f"/sc/home/tim.riedel/masterthesis/assets/output/traffic_plots"
    save_path = os.path.join(output_dir, f"{icao}_{plot_type}_{pd.to_datetime(timestamp).strftime('%Y%m%d')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot at {save_path}")
    plt.close(fig)


# --------------------------------
# Batch Plotting
# --------------------------------

def batch_plot_individual_flights(traffic, icao, date, plot_osm=True, batch_size=20, attribute="altitude", cmap="viridis"):
    def batched(iterable, n):
        """Batch data into tuples of length n."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            # Concatenate the dataframes of the flights in the batch
            batch_df = pd.concat([flight.data for flight in batch])
            yield Traffic(batch_df)

    folder = f"/sc/home/tim.riedel/masterthesis/assets/output/traffic_plots/{icao}_individual_flights_{pd.to_datetime(date).strftime('%Y-%m-%d')}"
    os.makedirs(folder, exist_ok=True)

    for i, batch in enumerate(batched(traffic, batch_size)):
        fig = plot_traffic(batch, icao, plot_osm=plot_osm, plot_individual_flights=True, title=f"{icao} - {date}", attribute=attribute, cmap=cmap)
        path = os.path.join(folder, f"batch_{i}.png")
        save_plot(fig, path)
