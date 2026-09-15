"""Quaternion and rotation conversion for the released IMUCoCo contract."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Quaternion must be finite and nonzero")
    return quaternion / norm


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a sensor-to-world WXYZ quaternion to a rotation matrix."""
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized WXYZ quaternion."""
    m = project_to_rotation_matrix(matrix)
    trace = np.trace(m)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = (0.25 * scale, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        quaternion = ((m[2, 1] - m[1, 2]) / scale, 0.25 * scale, (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale)
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        quaternion = ((m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale, 0.25 * scale, (m[1, 2] + m[2, 1]) / scale)
    else:
        scale = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        quaternion = ((m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale, (m[1, 2] + m[2, 1]) / scale, 0.25 * scale)
    return normalize_quaternion(np.asarray(quaternion))


def project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation input must be a finite 3x3 matrix")
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_matrix_to_r6d(matrix: np.ndarray) -> np.ndarray:
    """Flatten the first two matrix columns in upstream IMUCoCo order."""
    rotation = project_to_rotation_matrix(matrix)
    return rotation[:, :2].T.reshape(6).astype(np.float32)
