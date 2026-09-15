"""Convert synchronized BNO085 packets into IMUCoCo five-sensor frames."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import field
from typing import Mapping

import numpy as np
import torch

from src.calibration.sensor_to_body import NODE_TO_SENSOR, SENSOR_ORDER
from src.calibration.tpose import TposeCalibration
from src.input.orientation_adapter import quaternion_wxyz_to_matrix, rotation_matrix_to_r6d


@dataclass(frozen=True)
class SensorPacket:
    node_id: int
    timestamp_ms: int
    sequence: int
    acceleration: np.ndarray
    quaternion_wxyz: np.ndarray
    angular_velocity_degrees: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    @property
    def sensor_name(self) -> str:
        return NODE_TO_SENSOR[self.node_id]


@dataclass(frozen=True)
class PacketParseDiagnostic:
    packet: SensorPacket | None
    reason: str
    node_id: int | None = None
    field_count: int | None = None


def diagnose_globalpose_packet(line: str) -> PacketParseDiagnostic:
    """Parse a packet and retain why a visible DATA record was rejected."""
    position = line.find("DATA,")
    if position < 0:
        return PacketParseDiagnostic(None, "not_data")
    try:
        fields = next(csv.reader([line[position:]]))
    except csv.Error:
        return PacketParseDiagnostic(None, "csv_error")
    field_count = len(fields)
    node_id = None
    try:
        node_id = int(fields[1])
    except (IndexError, ValueError):
        return PacketParseDiagnostic(None, "invalid_node_id", field_count=field_count)
    if field_count not in (20, 21):
        return PacketParseDiagnostic(None, "field_count", node_id, field_count)
    try:
        if node_id not in NODE_TO_SENSOR:
            return PacketParseDiagnostic(None, "unknown_node_id", node_id, field_count)
        packet = SensorPacket(
            node_id=node_id,
            timestamp_ms=int(fields[2]),
            sequence=int(fields[6]),
            acceleration=np.asarray(fields[7:10], dtype=np.float64),
            quaternion_wxyz=np.asarray(fields[16:20], dtype=np.float64),
            angular_velocity_degrees=np.asarray(fields[10:13], dtype=np.float64),
        )
        if not all(
            np.isfinite(value).all()
            for value in (
                packet.acceleration,
                packet.angular_velocity_degrees,
                packet.quaternion_wxyz,
            )
        ):
            return PacketParseDiagnostic(None, "nonfinite", node_id, field_count)
        if np.linalg.norm(packet.quaternion_wxyz) < 1e-8:
            return PacketParseDiagnostic(None, "zero_quaternion", node_id, field_count)
        return PacketParseDiagnostic(packet, "ok", node_id, field_count)
    except (IndexError, ValueError):
        return PacketParseDiagnostic(None, "invalid_numeric_field", node_id, field_count)


def parse_globalpose_packet(line: str) -> SensorPacket | None:
    """Parse the current 21-field or legacy 20-field five-IMU DATA packet."""
    return diagnose_globalpose_packet(line).packet


def build_imucoco_frame(
    packets: Mapping[str, SensorPacket],
    calibration: TposeCalibration,
    *,
    acceleration_includes_gravity: bool = True,
) -> torch.Tensor:
    """Build one `[5, 9]` frame in the canonical project sensor order."""
    missing = set(SENSOR_ORDER) - set(packets)
    if missing:
        raise ValueError(f"Missing synchronized sensors: {sorted(missing)}")
    channels = []
    for sensor_name in SENSOR_ORDER:
        packet = packets[sensor_name]
        raw_rotation = quaternion_wxyz_to_matrix(packet.quaternion_wxyz)
        orientation = calibration.orientation(sensor_name, raw_rotation)
        acceleration = calibration.linear_acceleration(
            sensor_name,
            packet.acceleration,
            raw_rotation,
            includes_gravity=acceleration_includes_gravity,
        )
        channels.append(np.concatenate((rotation_matrix_to_r6d(orientation), acceleration)))
    return torch.from_numpy(np.stack(channels).astype(np.float32))
