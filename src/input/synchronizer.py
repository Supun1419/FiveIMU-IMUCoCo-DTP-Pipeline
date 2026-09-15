"""Timestamp alignment and interpolation for the five independent IMU clocks."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from src.calibration.sensor_to_body import NODE_TO_SENSOR
from src.input.bno085_adapter import SensorPacket
from src.input.orientation_adapter import normalize_quaternion


DAY_MS = 86_400_000


@dataclass(frozen=True)
class SyncedPacket:
    packet: SensorPacket
    synchronized_time_ms: float


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quaternion(q0 + amount * (q1 - q0))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return (
        np.sin((1 - amount) * angle) * q0 + np.sin(amount * angle) * q1
    ) / np.sin(angle)


class PacketSynchronizer:
    """Map each node's first timestamp to one host timeline and interpolate."""

    def __init__(self, buffer_size: int = 64):
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        self.offsets: dict[int, float] = {}
        self.day_offsets: dict[int, int] = defaultdict(int)
        self.previous_raw: dict[int, int] = {}

    def add(self, packet: SensorPacket, received_time_ms: float) -> None:
        previous = self.previous_raw.get(packet.node_id)
        if previous is not None and packet.timestamp_ms < previous - DAY_MS / 2:
            self.day_offsets[packet.node_id] += DAY_MS
        self.previous_raw[packet.node_id] = packet.timestamp_ms
        unwrapped = packet.timestamp_ms + self.day_offsets[packet.node_id]
        self.offsets.setdefault(packet.node_id, received_time_ms - unwrapped)
        synchronized = unwrapped + self.offsets[packet.node_id]
        self.buffers[packet.node_id].append(SyncedPacket(packet, synchronized))

    def latest_common_time(self) -> float | None:
        if any(node not in self.buffers or not self.buffers[node] for node in NODE_TO_SENSOR):
            return None
        return min(self.buffers[node][-1].synchronized_time_ms for node in NODE_TO_SENSOR)

    def frame(self, target_time_ms: float | None = None) -> dict[str, SensorPacket] | None:
        target = self.latest_common_time() if target_time_ms is None else target_time_ms
        if target is None:
            return None
        result = {}
        for node_id, sensor_name in NODE_TO_SENSOR.items():
            items = list(self.buffers[node_id])
            bracket = next(
                (
                    (left, right)
                    for left, right in zip(items, items[1:])
                    if left.synchronized_time_ms <= target <= right.synchronized_time_ms
                ),
                None,
            )
            if bracket is None:
                return None
            left, right = bracket
            duration = right.synchronized_time_ms - left.synchronized_time_ms
            amount = 0.0 if duration <= 0 else (target - left.synchronized_time_ms) / duration
            result[sensor_name] = SensorPacket(
                node_id=node_id,
                timestamp_ms=int(round(target % DAY_MS)),
                sequence=right.packet.sequence,
                acceleration=(
                    (1 - amount) * left.packet.acceleration
                    + amount * right.packet.acceleration
                ),
                quaternion_wxyz=slerp(
                    left.packet.quaternion_wxyz,
                    right.packet.quaternion_wxyz,
                    amount,
                ),
                angular_velocity_degrees=(
                    (1 - amount) * left.packet.angular_velocity_degrees
                    + amount * right.packet.angular_velocity_degrees
                ),
            )
        return result
