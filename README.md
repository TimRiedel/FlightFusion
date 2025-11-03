# FlightFusion

FlightFusion is a data processing pipeline designed to combine and pre-process diverse aviation and weather datasets for machine learning applications in trajectory prediction. It focuses on approach and arrival flight trajectories obtained from historical [OpenSky Network](https://opensky-network.org/) ADS-B data, enriched with surrounding traffic context and weather information from multiple sources — including METARs from [Ogimet](https://www.ogimet.com/home.phtml.en) and atmospheric variables from the [Copernicus Climate Data Service (CDS)](https://cds.climate.copernicus.eu/).

The pipeline integrates, cleans, and synchronizes these heterogeneous data streams to produce a unified, ML-ready representation suitable for tasks such as flight path prediction, anomaly detection, and operational analytics. Using this pipeline, it is possible to generate consistent and scalable datasets for various timeframes and airports across Germany.
