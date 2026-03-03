from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from traffic.core import Traffic


@dataclass
class Scene:
    """
    Lightweight representation of a multi-flight scene for trajectory prediction.

    A scene captures all flights present at a given input start time and records
    the time windows needed to slice each flight into input and horizon segments.
    """
    scene_id: int
    flight_ids: list[str]
    input_start_time: pd.Timestamp
    prediction_start_time: pd.Timestamp
    prediction_end_time: pd.Timestamp


class SceneCreationStrategy(ABC):
    """
    Abstract base class for scene creation strategies.

    A strategy iterates over traffic, selects anchor times, and for each anchor
    collects all qualifying co-present flights into a Scene object.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
    ):
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.min_trajectory_length_minutes = min_trajectory_length_minutes
        self.min_traffic_duration_minutes = input_time_minutes + 1

    @abstractmethod
    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        """
        Creates a list of Scene objects from a resampled Traffic object.

        Args:
            traffic: Resampled Traffic object containing all flights.

        Returns:
            List of Scene objects in anchor-time order across all qualifying flights.
        """
        pass

    def collect_flight_ids(
        self,
        traffic: Traffic,
        input_start_time: pd.Timestamp,
        prediction_end_time: pd.Timestamp,
    ) -> list[str]:
        """
        Collects flight IDs of all flights present at input_start_time with sufficient duration.

        A flight qualifies if:
        - It has waypoints in [input_start_time, prediction_end_time].
        - Its first waypoint is at or before input_start_time (i.e. it has already started).
        - Its duration within the window is at least min_traffic_duration_minutes.
        """
        window_traffic = traffic.query(
            f"timestamp >= '{input_start_time}' and timestamp <= '{prediction_end_time}'"
        )
        if window_traffic is None:
            return []

        flight_ids = []
        for flight in window_traffic:
            if flight.start > input_start_time:
                continue
            if flight.duration.components.minutes < self.min_traffic_duration_minutes:
                continue
            flight_ids.append(flight.flight_id)
        return flight_ids

    def build_scene(self, traffic: Traffic, scene_id: int, input_start_time: pd.Timestamp) -> Scene | None:
        """
        Builds a single Scene for a given anchor time, or returns None if no
        qualifying flights are present.
        """
        prediction_start_time = input_start_time + timedelta(minutes=self.input_time_minutes)
        prediction_end_time = prediction_start_time + timedelta(minutes=self.horizon_time_minutes)
        flight_ids = self.collect_flight_ids(
            traffic, input_start_time, prediction_end_time
        )
        if not flight_ids:
            return None
        return Scene(
            scene_id=scene_id,
            flight_ids=flight_ids,
            input_start_time=input_start_time,
            prediction_start_time=prediction_start_time,
            prediction_end_time=prediction_end_time,
        )


class FlightAppearsSceneCreationStrategy(SceneCreationStrategy):
    """
    Creates one scene per qualifying flight, anchored at that flight's start time.
    """

    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        scenes = []
        scene_id = 0
        for flight in traffic:
            if flight.duration.components.minutes < self.min_trajectory_length_minutes:
                continue
            scene = self.build_scene(traffic, scene_id, flight.start)
            if scene is not None:
                scenes.append(scene)
                scene_id += 1
        return scenes


class SamplingSceneCreationStrategy(SceneCreationStrategy):
    """
    Creates multiple scenes per qualifying flight by sliding the anchor forward
    by data_period_minutes after each scene.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
        data_period_minutes: int,
    ):
        super().__init__(input_time_minutes, horizon_time_minutes, min_trajectory_length_minutes)
        self.data_period_minutes = data_period_minutes

    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        scenes = []
        scene_id = 0
        for flight in traffic:
            if flight.duration.components.minutes < self.min_trajectory_length_minutes:
                continue

            flight_duration = flight.duration.components.minutes + 1
            num_samples = (flight_duration - self.min_trajectory_length_minutes) // self.data_period_minutes

            current_flight = flight
            for _ in range(num_samples):
                scene = self.build_scene(traffic, scene_id, current_flight.start)
                if scene is not None:
                    scenes.append(scene)
                    scene_id += 1
                current_flight = current_flight.skip(minutes=self.data_period_minutes)

        return scenes


def scenes_to_dataframe(scenes: list[Scene]) -> pd.DataFrame:
    """
    Converts a list of Scene objects to a flat manifest DataFrame.

    Each row corresponds to one flight within one scene.

    Args:
        scenes: List of Scene objects.

    Returns:
        DataFrame with columns: scene_id, flight_id, input_start_time,
        prediction_start_time, prediction_end_time.
    """
    rows = []
    for scene in scenes:
        for flight_id in scene.flight_ids:
            rows.append({
                "scene_id": scene.scene_id,
                "flight_id": flight_id,
                "input_start_time": scene.input_start_time,
                "prediction_start_time": scene.prediction_start_time,
                "prediction_end_time": scene.prediction_end_time,
            })
    return pd.DataFrame(rows)


def build_scene_creation_strategy(config: dict) -> SceneCreationStrategy:
    """
    Instantiates a SceneCreationStrategy from a config dict.

    Expected config keys:
        type (str): "flight_appears" or "sampling"
        input_time_minutes (int)
        horizon_time_minutes (int)
        min_trajectory_length_minutes (int)
        data_period_minutes (int): only required for "sampling"

    Args:
        config: Scene creation config dict (from the training YAML).

    Returns:
        An instantiated SceneCreationStrategy.
    """
    strategy_type = config["type"]
    input_time_minutes = config["input_time_minutes"]
    horizon_time_minutes = config["horizon_time_minutes"]
    min_trajectory_length_minutes = config["min_trajectory_length_minutes"]

    if strategy_type == "flight_appears":
        return FlightAppearsSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
        )
    elif strategy_type == "sampling":
        return SamplingSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            data_period_minutes=config["data_period_minutes"],
        )
    else:
        raise ValueError(
            f"Unknown scene creation strategy type: '{strategy_type}'. "
            "Expected 'flight_appears' or 'sampling'."
        )

