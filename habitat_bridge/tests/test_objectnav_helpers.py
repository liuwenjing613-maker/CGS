import math
import sys
from pathlib import Path

import numpy as np


LOCAL = Path(__file__).resolve().parents[1] / "local"
sys.path.insert(0, str(LOCAL))

from run_objectnav import look_at_yaw, transform_points, verify_expected_class  # noqa: E402


def test_transform_points_homogeneous_translation_and_axis_flip():
    transform = np.asarray(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, -1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = transform_points(transform, [[1.0, 2.0, 3.0]])
    assert np.allclose(result, [[3.0, 1.0, 1.0]])


def test_look_at_yaw_matches_habitat_forward_axis():
    source = [0.0, 0.0, 0.0]
    assert math.isclose(look_at_yaw(source, [0.0, 0.0, -1.0]), 0.0)
    assert math.isclose(look_at_yaw(source, [-1.0, 0.0, 0.0]), math.pi / 2.0)
    assert math.isclose(look_at_yaw(source, [1.0, 0.0, 0.0]), -math.pi / 2.0)


def test_expected_class_is_fail_closed():
    assert verify_expected_class("sofa chair", "sofa")
    assert verify_expected_class("folded dining chair", "dining chair")
    assert not verify_expected_class("folded chair", "sofa")

