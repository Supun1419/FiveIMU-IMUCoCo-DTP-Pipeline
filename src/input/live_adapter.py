"""T-pose collection and post-filter synchronization for live IMUCoCo input."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch

from src.calibration.sensor_to_body import SENSOR_ORDER, map_vector_to_body
from src.calibration.tpose import TposeCalibration
from src.input.bno085_adapter import SensorPacket
from src.input.noise_filter import FilteredSample
from src.input.orientation_adapter import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    rotation_matrix_to_r6d,
)
from src.input.synchronizer import DAY_MS, slerp


class TposeCollector:
    def __init__(self, duration_seconds: float = 2.0, gyro_limit_degrees: float = 3.0):
        self.duration_ms = duration_seconds * 1000
        self.gyro_limit_degrees = gyro_limit_degrees
        self.latest_seen: dict[str, float] = {}
        self.samples = defaultdict(list)
        self.started_ms: float | None = None
        self.calibration: TposeCalibration | None = None
        self.restarts = 0

    def reset(self) -> None:
        self.samples.clear()
        self.started_ms = None
        self.calibration = None
        self.restarts += 1

    def add(self, packet: SensorPacket, received_time_ms: float) -> TposeCalibration | None:
        name = packet.sensor_name
        self.latest_seen[name] = received_time_ms
        gyro = map_vector_to_body(packet.angular_velocity_degrees, name)
        if np.linalg.norm(gyro) >= self.gyro_limit_degrees:
            self.reset()
            return None
        if self.started_ms is None:
            if set(self.latest_seen) != set(SENSOR_ORDER):
                return None
            self.samples.clear()
            self.started_ms = received_time_ms
        self.samples[name].append(quaternion_wxyz_to_matrix(packet.quaternion_wxyz))
        if received_time_ms - self.started_ms >= self.duration_ms:
            if all(self.samples[name] for name in SENSOR_ORDER):
                self.calibration = TposeCalibration.from_sensor_rotations(self.samples)
                return self.calibration
        return None


@dataclass(frozen=True)
class LiveFrame:
    timestamp_ms: float
    imu: torch.Tensor
    angular_velocity: torch.Tensor
    stationary: tuple[bool, ...]
    sequences: tuple[int, ...]


class FilteredSynchronizer:
    """Interpolate already filtered native-rate samples onto a common timeline."""

    def __init__(self, buffer_size: int = 64):
        self.buffers = {name: deque(maxlen=buffer_size) for name in SENSOR_ORDER}
        self.offsets: dict[str, float] = {}
        self.day_offsets: dict[str, int] = defaultdict(int)
        self.previous_raw: dict[str, int] = {}
        self.latest_received: dict[str, float] = {}

    def add(self, sample: FilteredSample) -> None:
        name = sample.sensor_name
        previous = self.previous_raw.get(name)
        if previous is not None and sample.timestamp_ms < previous - DAY_MS / 2:
            self.day_offsets[name] += DAY_MS
        self.previous_raw[name] = sample.timestamp_ms
        unwrapped = sample.timestamp_ms + self.day_offsets[name]
        self.offsets.setdefault(name, sample.received_time_ms - unwrapped)
        synchronized_time = unwrapped + self.offsets[name]
        self.latest_received[name] = sample.received_time_ms
        self.buffers[name].append((synchronized_time, sample))

    def status(self, now_ms: float, stale_ms: float, max_skew_ms: float) -> tuple[list[str], list[str], float]:
        missing = [name for name in SENSOR_ORDER if not self.buffers[name]]
        stale = [
            name for name in SENSOR_ORDER
            if name in self.latest_received and now_ms - self.latest_received[name] > stale_ms
        ]
        newest = [self.buffers[name][-1][0] for name in SENSOR_ORDER if self.buffers[name]]
        skew = max(newest) - min(newest) if len(newest) == len(SENSOR_ORDER) else float("inf")
        if skew > max_skew_ms:
            return missing, stale, skew
        return missing, stale, skew

    def latest_common_time(self) -> float | None:
        if any(not self.buffers[name] for name in SENSOR_ORDER):
            return None
        return min(self.buffers[name][-1][0] for name in SENSOR_ORDER)

    def earliest_common_time(self) -> float | None:
        """Return the oldest timestamp still bracketable by every sensor."""
        if any(not self.buffers[name] for name in SENSOR_ORDER):
            return None
        return max(self.buffers[name][0][0] for name in SENSOR_ORDER)

    def frame(self, target_time_ms: float) -> LiveFrame | None:
        interpolated = []
        for name in SENSOR_ORDER:
            items = list(self.buffers[name])
            bracket = next(
                ((left, right) for left, right in zip(items, items[1:]) if left[0] <= target_time_ms <= right[0]),
                None,
            )
            if bracket is None:
                return None
            (left_time, left), (right_time, right) = bracket
            duration = right_time - left_time
            amount = 0.0 if duration <= 0 else (target_time_ms - left_time) / duration
            acceleration = (1 - amount) * left.acceleration_world + amount * right.acceleration_world
            angular_velocity = (1 - amount) * left.angular_velocity_world + amount * right.angular_velocity_world
            q = slerp(
                matrix_to_quaternion_wxyz(left.orientation),
                matrix_to_quaternion_wxyz(right.orientation),
                amount,
            )
            orientation = quaternion_wxyz_to_matrix(q)
            channels = np.concatenate((rotation_matrix_to_r6d(orientation), acceleration))
            interpolated.append((channels, angular_velocity, right))
        imu = torch.from_numpy(np.stack([item[0] for item in interpolated]).astype(np.float32))
        angular_velocity = torch.from_numpy(
            np.stack([item[1] for item in interpolated]).astype(np.float32)
        )
        if not torch.isfinite(imu).all() or not torch.isfinite(angular_velocity).all():
            raise ValueError("Synchronized frame contains non-finite values")
        return LiveFrame(
            timestamp_ms=target_time_ms,
            imu=imu,
            angular_velocity=angular_velocity,
            stationary=tuple(item[2].stationary for item in interpolated),
            sequences=tuple(item[2].sequence for item in interpolated),
        )


def recover_live_target(
    target_time_ms: float | None,
    earliest_time_ms: float,
    common_time_ms: float,
    step_ms: float,
) -> tuple[float, int]:
    """Keep a live cursor inside rolling buffers without accumulating latency."""
    if target_time_ms is None:
        return earliest_time_ms, 0
    backlog_ms = common_time_ms - target_time_ms
    if target_time_ms < earliest_time_ms or backlog_ms > 2.0 * step_ms:
        return common_time_ms, max(0, int(backlog_ms / step_ms))
    return target_time_ms, 0
