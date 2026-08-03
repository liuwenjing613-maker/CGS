from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import quaternion


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local"))
from pose_utils import (  # noqa: E402
    habitat_sensor_pose_to_opencv_c2w,
    state_to_matrix,
    transform_point,
)


def test_rotation_is_valid():
    rotation = quaternion.from_rotation_vector(np.asarray([0.0, 0.73, 0.0]))
    transform = habitat_sensor_pose_to_opencv_c2w([1.0, 2.0, 3.0], rotation)
    matrix = transform[:3, :3]
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5)


def test_center_ray_points_forward():
    transform = habitat_sensor_pose_to_opencv_c2w(
        [0.0, 0.0, 0.0], quaternion.one
    )
    assert np.allclose(transform_point(transform, [0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])


def test_translation_is_preserved():
    transform = state_to_matrix([1.0, 2.0, 3.0], quaternion.one)
    assert np.allclose(transform[:3, 3], [1.0, 2.0, 3.0])
