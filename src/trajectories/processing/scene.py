import math
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

    def __len__(self) -> int:
        return len(self.flight_ids)


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
        min_horizon_length_minutes: int = 3,
    ):
        """
        Args:
            input_time_minutes: Length of the input window used to condition the model.
            horizon_time_minutes: Length of the output prediction horizon.
            min_trajectory_length_minutes: Minimum duration a flight must have to ever
                be used as an *anchor* (i.e. prediction target). This is only applied
                in flight‑centric strategies such as FlightAppearsSceneCreationStrategy
                and SamplingSceneCreationStrategy.
            min_scene_presence_minutes: Minimum remaining time a flight must have from
                the scene's prediction_start_time in order to be considered in that scene.

                Operationally, a flight becomes uninteresting for RTD prediction once
                it has passed the final approach fix, which is typically about
                3 minutes before touchdown. At that point, separation to following
                traffic is already resolved and the remaining trajectory is almost
                deterministic. We therefore require context flights to extend at least
                min_horizon_length_minutes into the prediction horizon and ignore
                aircraft that are closer than this threshold to landing.
        """
        self.input_time_minutes = input_time_minutes
        self.horizon_time_minutes = horizon_time_minutes
        self.min_trajectory_length_minutes = min_trajectory_length_minutes

        self.min_horizon_length_minutes = min_horizon_length_minutes
        self.min_scene_presence_minutes = input_time_minutes + min_horizon_length_minutes
        

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
        - Its duration within the window is at least min_scene_presence_minutes,
          meaning it is present for the full input window and still a few minutes
          into the prediction horizon, i.e. not already past the final approach fix.
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

            window_end = min(flight.stop, prediction_end_time)
            presence_minutes = (window_end - input_start_time).total_seconds() / 60
            if presence_minutes < self.min_scene_presence_minutes:
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

    def build_scene_from_flight_ids(self, scene_id: int, input_start_time: pd.Timestamp, flight_ids: list[str]) -> Scene | None:
        prediction_start_time = input_start_time + timedelta(minutes=self.input_time_minutes)
        prediction_end_time = prediction_start_time + timedelta(minutes=self.horizon_time_minutes)
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

    This guarantees that the furthest-from-airport moment — when the flight first
    enters the sector — is always captured, which is the operationally most interesting
    prediction point.

    The drawback is that a flight contributes exactly one training sample, regardless
    of how long its approach takes. The model never sees mid-approach queries for any
    flight, so it cannot learn to predict RTD at later stages of the approach. In
    low-density traffic periods this leaves large temporal gaps in coverage; in
    high-density periods the gaps are partially filled by scenes anchored by other
    flights, but only incidentally.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
        min_horizon_length_minutes: int = 3,
    ):
        super().__init__(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_horizon_length_minutes=min_horizon_length_minutes,
        )

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

    Because the anchor slides along each flight independently, this strategy provides
    good temporal coverage: every flight contributes samples from its first appearance
    through to its final qualifying segment. The furthest-from-airport moment is also
    always captured, since the anchor starts at flight.start.

    The critical drawback is that traffic density and sample count are coupled. When
    many flights are in the air simultaneously, each of those flights independently
    generates scenes at overlapping anchor times, producing near-duplicate scenes that
    differ only in which flight triggered the anchor. A single flight can appear dozens
    of times across all scenes, which biases the model toward high-traffic constellations
    and wastes training budget on redundant data.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
        data_period_minutes: int,
        min_horizon_length_minutes: int = 3,
    ):
        super().__init__(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
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


class ContinuousSceneCreationStrategy(SceneCreationStrategy):
    """
    Creates scenes by sliding a single time window globally at a fixed interval,
    independent of individual flights.

    Because the iterator is time-based rather than flight-based, a busy period with
    ten simultaneous flights generates the same number of scenes as a quiet period
    with one flight. This keeps the training distribution proportional to real traffic
    patterns and avoids the near-duplicate oversampling of SamplingSceneCreationStrategy.

    The drawback is that the global window is anchored to global_start and advances in
    fixed steps, so it will generally not land on the exact moment a flight first
    appears. A flight's earliest segment — where the approach is longest and RTD most
    uncertain — can be missed or under-represented by up to data_period_minutes.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
        data_period_minutes: int,
        min_horizon_length_minutes: int = 3,
    ):
        super().__init__(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
        self.data_period_minutes = data_period_minutes

    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        global_start = traffic.data["timestamp"].min()
        global_end = traffic.data["timestamp"].max()
        data_period_minutes = timedelta(minutes=self.data_period_minutes)

        current_time = global_start
        scenes = []
        scene_id = 0
        while current_time <= global_end:
            prediction_start_time = current_time + timedelta(minutes=self.input_time_minutes)
            prediction_end_time = prediction_start_time + timedelta(minutes=self.horizon_time_minutes)
            flight_ids = self.collect_flight_ids(
                traffic,
                input_start_time=current_time,
                prediction_end_time=prediction_end_time,
            )
            if flight_ids:
                num_agents = len(flight_ids)

                scene = self.build_scene_from_flight_ids(scene_id, current_time, flight_ids)
                scenes.append(scene)
                scene_id += 1
            current_time += data_period_minutes

        return scenes


class TrafficAdaptiveSceneCreationStrategy(SceneCreationStrategy):
    """
    A continuous sliding-window strategy whose step size adapts to traffic density.

    In high-traffic periods the window advances in small steps (down to
    min_data_period_minutes, default 1 min) to capture rapidly changing
    constellations. In low-traffic periods it advances in larger steps (up to
    max_data_period_minutes, default 4 min) to avoid redundant, near-identical
    scenes. Between the two thresholds the step size follows a power curve that
    keeps steps near the maximum for most traffic levels and only drops 
    to the minimum at high traffic counts.

    Traffic density is measured as the peak number of flights within airport_radius_m
    of the airport (x_coord=0, y_coord=0) across the full scene window
    [input_start, prediction_end]. Taking the peak over the window rather than
    a snapshot at input_start means that aircraft currently far out but converging
    toward the terminal area will drive a high density signal — and thus a small
    step size — even before they arrive.
    """

    def __init__(
        self,
        input_time_minutes: int,
        horizon_time_minutes: int,
        min_trajectory_length_minutes: int,
        min_horizon_length_minutes: int = 3,
        min_data_period_minutes: int = 1,
        max_data_period_minutes: int = 5,
        airport_radius_m: float = 50_000,
        low_traffic_threshold: int | None = None,
        high_traffic_threshold: int | None = None,
    ):
        """
        Args:
            min_data_period_minutes: Step size when traffic >= high_traffic_threshold.
            max_data_period_minutes: Step size when traffic <= low_traffic_threshold.
            airport_radius_m: Radius in metres around the airport origin within
                which flights are counted for the density signal (default 50 000 m).
            low_traffic_threshold: Flight count at or below which the maximum
                (slowest) step size is used. If None, derived from the minimum
                of the density series at runtime.
            high_traffic_threshold: Flight count at or above which the minimum
                (fastest) step size is used. If None, derived from the maximum
                of the density series at runtime.
        """
        super().__init__(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
        self.min_data_period_minutes = min_data_period_minutes
        self.max_data_period_minutes = max_data_period_minutes
        self.airport_radius_m = airport_radius_m
        self.low_traffic_threshold = low_traffic_threshold
        self.high_traffic_threshold = high_traffic_threshold

    def _build_traffic_count_series(
        self,
        traffic: Traffic,
        global_start: pd.Timestamp,
        global_end: pd.Timestamp,
    ) -> pd.Series:
        """
        Returns a Series indexed by 1-minute timestamps with the number of
        flights within airport_radius_m of the airport at each minute.
        """
        minute_index = pd.date_range(global_start, global_end, freq="1min")
        counts = pd.Series(0, index=minute_index, dtype=int)

        df = traffic.data
        radius_sq = self.airport_radius_m ** 2
        within = df.loc[
            df["x_coord"] ** 2 + df["y_coord"] ** 2 <= radius_sq,
            ["timestamp", "flight_id"],
        ]
        within = within.copy()
        within["timestamp"] = within["timestamp"].dt.floor("1min")
        within = within.drop_duplicates()

        per_minute = within.groupby("timestamp")["flight_id"].nunique()
        per_minute = per_minute.reindex(minute_index, fill_value=0)
        counts = per_minute.astype(int)
        return counts

    def _peak_traffic_in_window(
        self,
        counts: pd.Series,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> int:
        """Max airport-vicinity density across the full scene window."""
        return int(counts.loc[window_start.floor("1min"):window_end.floor("1min")].max())

    def _data_period_for_traffic(
        self, peak_count: int, low_threshold: int, high_threshold: int
    ) -> int:
        """Interpolate the step size between the two thresholds.

        Uses a power curve (exponent > 1) so that the step size stays close to
        max_data_period_minutes across most of the traffic range and only drops
        sharply to min_data_period_minutes at very high counts.
        """
        CURVE_EXPONENT = 3

        if high_threshold <= low_threshold:
            return self.max_data_period_minutes
        if peak_count >= high_threshold:
            return self.min_data_period_minutes
        if peak_count <= low_threshold:
            return self.max_data_period_minutes

        ratio = (peak_count - low_threshold) / (high_threshold - low_threshold)
        data_period = self.max_data_period_minutes - ratio ** CURVE_EXPONENT * (
            self.max_data_period_minutes - self.min_data_period_minutes
        )
        return round(data_period)

    def create_scenes(self, traffic: Traffic) -> list[Scene]:
        global_start = traffic.data["timestamp"].min()
        global_end = traffic.data["timestamp"].max()
        scene_window_minutes = self.input_time_minutes + self.horizon_time_minutes

        counts = self._build_traffic_count_series(traffic, global_start, global_end)

        low_threshold = self.low_traffic_threshold if self.low_traffic_threshold is not None else int(counts.min())
        high_threshold = self.high_traffic_threshold if self.high_traffic_threshold is not None else int(counts.max())

        current_time = global_start
        scenes = []
        scene_id = 0
        while current_time <= global_end:
            prediction_end_time = current_time + timedelta(minutes=scene_window_minutes)
            flight_ids = self.collect_flight_ids(
                traffic,
                input_start_time=current_time,
                prediction_end_time=prediction_end_time,
            )
            if flight_ids:
                scene = self.build_scene_from_flight_ids(scene_id, current_time, flight_ids)
                scenes.append(scene)
                scene_id += 1

            density = self._peak_traffic_in_window(counts, current_time, prediction_end_time)
            step = self._data_period_for_traffic(density, low_threshold, high_threshold)
            current_time += timedelta(minutes=step)

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
        type (str): "flight_appears", "sampling", "continuous", "anchored_continuous",
            "traffic_adaptive"
        input_time_minutes (int)
        horizon_time_minutes (int)
        min_trajectory_length_minutes (int)
        min_horizon_length_minutes (int, optional; default 3)
        data_period_minutes (int): required for "sampling", "continuous",
            and "anchored_continuous"

    Args:
        config: Scene creation config dict (from the training YAML).

    Returns:
        An instantiated SceneCreationStrategy.
    """
    strategy_type = config["type"]
    input_time_minutes = config["input_time_minutes"]
    horizon_time_minutes = config["horizon_time_minutes"]
    min_trajectory_length_minutes = config["min_trajectory_length_minutes"]
    min_horizon_length_minutes = config.get("min_horizon_length_minutes", 3)

    if strategy_type == "flight_appears":
        return FlightAppearsSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
    elif strategy_type == "sampling":
        return SamplingSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            data_period_minutes=config["data_period_minutes"],
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
    elif strategy_type == "continuous":
        return ContinuousSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            data_period_minutes=config["data_period_minutes"],
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
    elif strategy_type == "anchored_continuous":
        return AnchoredContinuousSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            data_period_minutes=config["data_period_minutes"],
            min_anchor_gap_minutes=config.get("min_anchor_gap_minutes", 1),
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
    elif strategy_type == "traffic_adaptive":
        return TrafficAdaptiveSceneCreationStrategy(
            input_time_minutes=input_time_minutes,
            horizon_time_minutes=horizon_time_minutes,
            min_trajectory_length_minutes=min_trajectory_length_minutes,
            min_data_period_minutes=config.get("min_data_period_minutes", 1),
            max_data_period_minutes=config.get("data_period_minutes", 5),
            airport_radius_m=config.get("traffic_count_airport_radius_m", 50_000),
            min_horizon_length_minutes=min_horizon_length_minutes,
        )
    else:
        raise ValueError(
            f"Unknown scene creation strategy type: '{strategy_type}'. "
            "Expected 'flight_appears', 'sampling', 'continuous', "
            "'anchored_continuous', or 'traffic_adaptive'."
        )