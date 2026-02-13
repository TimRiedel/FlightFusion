import hashlib
import json
import os


class Cache:
    def __init__(self, cache_dir: str, data_type: str, icao: str = None):
        """
        Initialize a Cache instance.
        
        Args:
            cache_dir: Base cache directory path
            data_type: Type of data being cached ("weather", "metar", or "trajectories")
            icao: ICAO code (required for "metar" and "trajectories" data types)
        """
        if data_type not in ["weather", "metar", "trajectories", "flightlist"]:
            raise ValueError(f"Invalid data type for cache: {data_type}")
        if data_type in ["metar", "trajectories", "flightlist"] and icao is None:
            raise ValueError(f"ICAO code is required for {data_type} cache")

        self.cache_dir = os.path.join(cache_dir, data_type)
        self.data_type = data_type
        self.icao = icao
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def make_key(self, request_config: dict):
        """Generate a hash key from a request configuration."""
        hash_str = json.dumps(request_config, sort_keys=True)
        hash_val = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:10]
        return hash_val

    def get_file_path(self, request_config: dict):
        """
        Get the cache file path for a given request configuration.
        
        Args:
            request_config: Dictionary containing the configuration to hash
        
        Returns:
            file_path: Path to the cached file
            exists: True if the file exists, False otherwise
        """
        hash_val = self.make_key(request_config)
        file_path, exists = None, False

        if self.data_type == "weather":
            file_path = self._get_weather_file_path(request_config, hash_val)
            exists = os.path.exists(file_path)
        elif self.data_type == "metar":
            file_path = self._get_metar_file_path(request_config, hash_val)
            exists = os.path.exists(file_path)
        elif self.data_type == "trajectories":
            file_path = self._get_trajectories_file_path(request_config, hash_val)
            exists = os.path.exists(file_path)
        elif self.data_type == "flightlist":
            file_path = self._get_flightlist_file_path(request_config, hash_val)
            exists = os.path.exists(file_path)
        return file_path, exists

    def _get_weather_file_path(self, request_config: dict, hash_val: str):
        year = request_config["year"]
        month = request_config["month"]
        dataset_name = request_config["dataset_name"]
        return os.path.join(self.cache_dir, f"{dataset_name}_{year}-{month:02d}-{hash_val}.grib")

    def _get_metar_file_path(self, request_config: dict, hash_val: str):
        return os.path.join(self.cache_dir, f"{self.icao}_metar_{hash_val}.parquet")

    def _get_trajectories_file_path(self, request_config: dict, hash_val: str):
        return os.path.join(self.cache_dir, f"{self.icao}_trajectories_{hash_val}.parquet")

    def _get_flightlist_file_path(self, request_config: dict, hash_val: str):
        return os.path.join(self.cache_dir, f"{self.icao}_flightlist_{hash_val}.parquet")