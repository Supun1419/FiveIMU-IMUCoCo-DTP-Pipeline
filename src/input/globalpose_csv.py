"""Read calibrated GlobalPose recordings in the IMUCoCo sensor order."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from src.calibration.sensor_to_body import SENSOR_ORDER
from src.input.orientation_adapter import rotation_matrix_to_r6d


def load_calibrated_recording(path: Path, max_frames: int | None = None) -> torch.Tensor:
    frames = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sensors = []
            for name in SENSOR_ORDER:
                acceleration = np.array(
                    [float(row[f"{name}_a{axis}"]) for axis in "xyz"], dtype=np.float32
                )
                rotation = np.array(
                    [float(row[f"{name}_R{r}{c}"]) for r in range(3) for c in range(3)],
                    dtype=np.float64,
                ).reshape(3, 3)
                sensors.append(np.concatenate((rotation_matrix_to_r6d(rotation), acceleration)))
            frames.append(np.stack(sensors))
            if max_frames is not None and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return torch.from_numpy(np.stack(frames).astype(np.float32))
