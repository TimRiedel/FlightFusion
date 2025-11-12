# FlightFusion

FlightFusion is a data processing pipeline designed to combine and pre-process diverse aviation and weather datasets for machine learning applications in trajectory prediction. It focuses on approach and arrival flight trajectories obtained from historical [OpenSky Network](https://opensky-network.org/) ADS-B data, enriched with surrounding traffic context and weather information from multiple sources — including METARs from [Ogimet](https://www.ogimet.com/home.phtml.en) and atmospheric variables from the [Copernicus Climate Data Service (CDS)](https://cds.climate.copernicus.eu/).

The pipeline integrates, cleans, and synchronizes these heterogeneous data streams to produce a unified, ML-ready representation suitable for tasks such as flight path prediction, anomaly detection, and operational analytics. Using this pipeline, it is possible to generate consistent and scalable datasets for various timeframes and airports across Germany.

# Pre-Requisites
### OpenSky Access
To download flight trajectory data, you must create an account and request historic data access for the [OpenSky Network](https://opensky-network.org/) Trino database.
1. Create an account.
2. Apply for data access at [https://opensky-network.org/my-opensky/request-data](https://opensky-network.org/my-opensky/request-data)
3. On the "Account" page, create a new API client and retrieve the `client_id` and `client_secret`.
4. Locate [traffic](https://traffic-viz.github.io/) library's configuration file on your machine, usually located under `$HOME/.config/traffic/traffic.conf`, and edit the following lines by pasting your credentials:
   ```
   [opensky]
   username =
   password =
   ```

### Access to Climate Data Store (CDS) by ECMWF
In order to download atmospheric weather data, you need to have API access to the Climate Data Store (CDS) by the European Centre for Medium-Range Weather Forecasts (ECMWF). Please follow the instructions here, to obtain a Personal Access Token for the API and place it in the `$HOME/.cdsapirc` file: [https://cds.climate.copernicus.eu/how-to-api](https://cds.climate.copernicus.eu/how-to-api)



