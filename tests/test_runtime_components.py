import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from firmware.mqtt_firmware_subscriber import parse_batch
from src.calibration.sensor_to_body import SENSOR_ORDER, map_vector_to_body
from src.input.noise_filter import NoiseFilterConfig
from src.input.orientation_adapter import normalize_quaternion


def test_mqtt_batch_parser_preserves_sensor_values():
    payload = {
        "b": 7,
        "t0": 1000,
        "d": [[9, 1050, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
    }
    rows = parse_batch(
        "FYP_IMU_Session/Node3_Batch",
        json.dumps(payload).encode("utf-8"),
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "Node3"
    assert rows[0]["timestamp_ms"] == 1050
    assert rows[0]["quat_w"] == 1.0


def test_runtime_sensor_contract_and_ankle_x_conversion():
    assert SENSOR_ORDER == (
        "left_wrist",
        "right_wrist",
        "torso",
        "left_ankle",
        "right_ankle",
    )
    np.testing.assert_allclose(
        map_vector_to_body(np.array((1.0, 2.0, 3.0)), "left_ankle"),
        (-1.0, 2.0, 3.0),
    )


def test_quaternion_and_calibration_defaults():
    np.testing.assert_allclose(
        normalize_quaternion(np.array((2.0, 0.0, 0.0, 0.0))),
        (1, 0, 0, 0),
    )
    config = NoiseFilterConfig()
    assert config.calibration_gyro_limit_degrees == 3.0
    assert config.gyro_stationary_degrees == 1.0
