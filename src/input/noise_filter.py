"""Firmware-proven gravity, bias, stationary, EMA, and deadband filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.calibration.sensor_to_body import SENSOR_ORDER, map_vector_to_body
from src.calibration.tpose import TposeCalibration
from src.input.bno085_adapter import SensorPacket
from src.input.orientation_adapter import normalize_quaternion, quaternion_wxyz_to_matrix


@dataclass(frozen=True)
class NoiseFilterConfig:
    gravity_seconds: float = 2.0
    bias_seconds: float = 5.0
    calibration_gyro_limit_degrees: float = 3.0
    gyro_stationary_degrees: float = 1.0
    stationary_seconds: float = 0.35
    bias_adaptation: float = 0.02
    filter_alpha: float = 0.25
    minimum_deadband: float = 0.05
    noise_multiplier: float = 3.0

    def __post_init__(self):
        durations = (self.gravity_seconds, self.bias_seconds, self.stationary_seconds)
        if min(durations) <= 0:
            raise ValueError("Calibration and stationary durations must be positive")
        if min(self.calibration_gyro_limit_degrees, self.gyro_stationary_degrees) <= 0:
            raise ValueError("Gyroscope limits must be positive")
        if not 0 < self.bias_adaptation <= 1 or not 0 < self.filter_alpha <= 1:
            raise ValueError("Adaptation and filter alpha must be in (0, 1]")


@dataclass(frozen=True)
class FilteredSample:
    sensor_name: str
    node_id: int
    timestamp_ms: int
    sequence: int
    acceleration_world: np.ndarray
    angular_velocity_world: np.ndarray
    orientation: np.ndarray
    stationary: bool
    received_time_ms: float


@dataclass
class _SensorState:
    stage: str = "gravity"
    stage_started_ms: float | None = None
    gravity_samples: list[np.ndarray] = field(default_factory=list)
    gravity_world: np.ndarray | None = None
    bias_samples: list[np.ndarray] = field(default_factory=list)
    bias_body: np.ndarray | None = None
    noise_std: np.ndarray | None = None
    deadband: np.ndarray | None = None
    filtered: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    stationary_since_ms: float | None = None
    stationary: bool = False
    previous_quaternion: np.ndarray | None = None
    calibration_restarts: int = 0

    def restart_calibration(self, now_ms: float) -> None:
        previous_quaternion = self.previous_quaternion
        restarts = self.calibration_restarts + 1
        self.__dict__.update(_SensorState().__dict__)
        self.previous_quaternion = previous_quaternion
        self.calibration_restarts = restarts
        self.stage_started_ms = now_ms


class FiveSensorNoiseFilter:
    """Apply the documented filter independently to chronological native samples."""

    def __init__(
        self,
        calibration: TposeCalibration,
        config: NoiseFilterConfig | None = None,
    ):
        self.calibration = calibration
        self.config = config or NoiseFilterConfig()
        self.states = {name: _SensorState() for name in SENSOR_ORDER}

    @property
    def ready(self) -> bool:
        return all(state.stage == "ready" for state in self.states.values())

    def stages(self) -> Mapping[str, str]:
        return {name: state.stage for name, state in self.states.items()}

    def diagnostics(self) -> dict[str, dict[str, object]]:
        result = {}
        for name, state in self.states.items():
            result[name] = {
                "stage": state.stage,
                "stationary": state.stationary,
                "gravity_world": None if state.gravity_world is None else state.gravity_world.tolist(),
                "bias_body": None if state.bias_body is None else state.bias_body.tolist(),
                "noise_std": None if state.noise_std is None else state.noise_std.tolist(),
                "deadband": None if state.deadband is None else state.deadband.tolist(),
                "calibration_restarts": state.calibration_restarts,
            }
        return result

    def process(
        self, packet: SensorPacket, received_time_ms: float
    ) -> FilteredSample | None:
        name = packet.sensor_name
        state = self.states[name]
        config = self.config
        quaternion = normalize_quaternion(packet.quaternion_wxyz)
        if state.previous_quaternion is not None and np.dot(quaternion, state.previous_quaternion) < 0:
            quaternion = -quaternion
        state.previous_quaternion = quaternion

        raw_rotation = quaternion_wxyz_to_matrix(quaternion)
        orientation = self.calibration.orientation(name, raw_rotation)
        acceleration_body = map_vector_to_body(packet.acceleration, name)
        gyro_body_degrees = map_vector_to_body(packet.angular_velocity_degrees, name)
        gyro_magnitude = float(np.linalg.norm(gyro_body_degrees))
        if not np.isfinite(acceleration_body).all() or not np.isfinite(gyro_magnitude):
            raise ValueError(f"Non-finite sensor values for {name}")

        if state.stage_started_ms is None:
            state.stage_started_ms = received_time_ms

        if state.stage != "ready" and gyro_magnitude >= config.calibration_gyro_limit_degrees:
            state.restart_calibration(received_time_ms)
            return None

        acceleration_world = orientation @ acceleration_body
        if state.stage == "gravity":
            state.gravity_samples.append(acceleration_world)
            if received_time_ms - state.stage_started_ms >= config.gravity_seconds * 1000:
                state.gravity_world = np.mean(np.stack(state.gravity_samples), axis=0)
                if not np.isfinite(state.gravity_world).all() or not 5.0 < np.linalg.norm(state.gravity_world) < 15.0:
                    state.restart_calibration(received_time_ms)
                    return None
                state.stage = "bias"
                state.stage_started_ms = received_time_ms
            return None

        gravity_body = orientation.T @ state.gravity_world
        residual_body = acceleration_body - gravity_body
        if state.stage == "bias":
            state.bias_samples.append(residual_body)
            if received_time_ms - state.stage_started_ms >= config.bias_seconds * 1000:
                samples = np.stack(state.bias_samples)
                state.bias_body = samples.mean(axis=0)
                state.noise_std = samples.std(axis=0)
                state.deadband = np.maximum(
                    config.minimum_deadband,
                    config.noise_multiplier * state.noise_std,
                )
                if not all(
                    np.isfinite(value).all()
                    for value in (state.bias_body, state.noise_std, state.deadband)
                ):
                    state.restart_calibration(received_time_ms)
                    return None
                state.filtered.fill(0)
                state.stage = "ready"
                state.stage_started_ms = received_time_ms
            return None

        gyro_still = gyro_magnitude < config.gyro_stationary_degrees
        if gyro_still:
            if state.stationary_since_ms is None:
                state.stationary_since_ms = received_time_ms
        else:
            state.stationary_since_ms = None
        state.stationary = (
            state.stationary_since_ms is not None
            and received_time_ms - state.stationary_since_ms
            >= config.stationary_seconds * 1000
        )

        if state.stationary:
            alpha = config.bias_adaptation
            state.bias_body = (1 - alpha) * state.bias_body + alpha * residual_body
        bias_corrected = residual_body - state.bias_body
        beta = config.filter_alpha
        state.filtered = (1 - beta) * state.filtered + beta * bias_corrected
        deadbanded = np.where(np.abs(state.filtered) < state.deadband, 0.0, state.filtered)
        output_body = np.zeros(3) if state.stationary else deadbanded
        output_world = orientation @ output_body
        gyro_world = np.deg2rad(orientation @ gyro_body_degrees)
        return FilteredSample(
            sensor_name=name,
            node_id=packet.node_id,
            timestamp_ms=packet.timestamp_ms,
            sequence=packet.sequence,
            acceleration_world=output_world.astype(np.float32),
            angular_velocity_world=gyro_world.astype(np.float32),
            orientation=orientation.astype(np.float32),
            stationary=state.stationary,
            received_time_ms=received_time_ms,
        )
