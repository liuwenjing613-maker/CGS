"""Coordinate conversions shared by the local Habitat bridge."""
from __future__ import annotations

import numpy as np
import quaternion


HABITAT_SENSOR_FROM_OPENCV_CAMERA = np.diag(
    [1.0, -1.0, -1.0, 1.0]
).astype(np.float64)


def state_to_matrix(position, rotation) -> np.ndarray:
    """Return a world-from-state homogeneous transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion.as_rotation_matrix(rotation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def habitat_sensor_pose_to_opencv_c2w(position, rotation) -> np.ndarray:
    """Convert a Habitat sensor state to an OpenCV camera-to-world pose."""
    return state_to_matrix(position, rotation) @ HABITAT_SENSOR_FROM_OPENCV_CAMERA


def transform_point(transform: np.ndarray, point) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = np.asarray(point, dtype=np.float64)
    return (np.asarray(transform, dtype=np.float64) @ homogeneous)[:3]
