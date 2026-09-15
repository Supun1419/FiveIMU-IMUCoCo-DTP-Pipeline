"""Fixed hardware mounting transforms shared with the GlobalPose sensors."""

from __future__ import annotations

import numpy as np


SENSOR_ORDER = (
    "left_wrist",
    "right_wrist",
    "torso",
    "left_ankle",
    "right_ankle",
)

NODE_TO_SENSOR = {
    1: "left_wrist",
    2: "right_wrist",
    0: "torso",
    3: "left_ankle",
    4: "right_ankle",
}

SENSOR_TO_NODE = {name: node for node, name in NODE_TO_SENSOR.items()}

# Columns are sensor +X, +Y and +Z expressed in the anatomical model basis.
SENSOR_TO_BODY = {
    "torso": np.array(((-1, 0, 0), (0, -1, 0), (0, 0, 1)), dtype=np.float64),
    "left_wrist": np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=np.float64),
    "right_wrist": np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0)), dtype=np.float64),
    "left_ankle": np.eye(3, dtype=np.float64),
    "right_ankle": np.eye(3, dtype=np.float64),
}

# The ankle firmware X channel is opposite to the pretrained model signal frame.
# Conjugating rotations by this basis keeps them proper while changing only X.
SENSOR_TO_MODEL_BASIS = {
    name: (
        np.diag((-1, 1, 1)).astype(np.float64)
        if name in ("left_ankle", "right_ankle")
        else np.eye(3, dtype=np.float64)
    )
    for name in SENSOR_ORDER
}


def map_orientation_to_body(
    sensor_to_world: np.ndarray, sensor_name: str
) -> np.ndarray:
    """Return the mounted anatomical body orientation in the sensor world."""
    return np.asarray(sensor_to_world, dtype=np.float64) @ SENSOR_TO_BODY[sensor_name].T


def map_vector_to_body(vector_sensor: np.ndarray, sensor_name: str) -> np.ndarray:
    vector = np.asarray(vector_sensor, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {vector.shape}")
    return SENSOR_TO_MODEL_BASIS[sensor_name] @ SENSOR_TO_BODY[sensor_name] @ vector


def map_relative_orientation_to_model(
    body_orientation: np.ndarray, sensor_name: str
) -> np.ndarray:
    """Express a calibrated relative rotation in the model signal basis."""
    basis = SENSOR_TO_MODEL_BASIS[sensor_name]
    orientation = np.asarray(body_orientation, dtype=np.float64)
    return basis @ orientation @ basis
