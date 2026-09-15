"""Per-mount T-pose calibration into the synthetic IMUCoCo world frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.calibration.sensor_to_body import (
    SENSOR_ORDER,
    map_orientation_to_body,
    map_relative_orientation_to_model,
    map_vector_to_body,
)
from src.input.orientation_adapter import project_to_rotation_matrix


GRAVITY_MODEL = np.array((0.0, -9.80665, 0.0), dtype=np.float64)


def mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("At least one rotation is required")
    return project_to_rotation_matrix(np.mean(np.stack(rotations), axis=0))


@dataclass(frozen=True)
class TposeCalibration:
    """Mapped sensor orientations observed while the subject holds a T-pose."""

    references: Mapping[str, np.ndarray]

    @classmethod
    def from_sensor_rotations(
        cls, samples: Mapping[str, Sequence[np.ndarray]]
    ) -> "TposeCalibration":
        missing = set(SENSOR_ORDER) - set(samples)
        if missing:
            raise ValueError(f"Missing T-pose sensors: {sorted(missing)}")
        references = {
            name: mean_rotation(
                [map_orientation_to_body(rotation, name) for rotation in samples[name]]
            )
            for name in SENSOR_ORDER
        }
        return cls(references=references)

    def orientation(
        self, sensor_name: str, sensor_to_world: np.ndarray
    ) -> np.ndarray:
        mapped = map_orientation_to_body(sensor_to_world, sensor_name)
        relative = project_to_rotation_matrix(self.references[sensor_name].T @ mapped)
        return project_to_rotation_matrix(
            map_relative_orientation_to_model(relative, sensor_name)
        )

    def linear_acceleration(
        self,
        sensor_name: str,
        acceleration_sensor: np.ndarray,
        sensor_to_world: np.ndarray,
        *,
        includes_gravity: bool = True,
    ) -> np.ndarray:
        acceleration_body = map_vector_to_body(acceleration_sensor, sensor_name)
        acceleration_model = self.orientation(sensor_name, sensor_to_world) @ acceleration_body
        if includes_gravity:
            acceleration_model = acceleration_model + GRAVITY_MODEL
        return acceleration_model.astype(np.float32)
