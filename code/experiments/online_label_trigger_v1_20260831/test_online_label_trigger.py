#!/usr/bin/env python3
"""Small adversarial tests for the online label trigger evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_online_label_trigger import (  # noqa: E402
    evaluate_rule_scene,
    mask_assignment,
    metric_core,
    production_features,
    target_features,
)


def observations(labels: list[int]) -> tuple[list[str], dict[str, dict]]:
    uids = [f"o{index}" for index in range(len(labels))]
    rows = {
        uid: {"class_id": label, "frame_idx": index}
        for index, (uid, label) in enumerate(zip(uids, labels))
    }
    return uids, rows


def test_production_features() -> None:
    uids, rows = observations([1, 1, 1, 2, 2])
    result = production_features({"1": 3, "2": 2}, uids, rows)
    assert result["num_observations"] == 5
    assert result["persistence_gate"] is True
    assert result["repeated_alt_fraction"] == 0.4
    assert 0.0 < result["entropy_n5"] <= 1.0
    try:
        production_features({"1": 4, "2": 1}, uids, rows)
    except RuntimeError:
        pass
    else:
        raise AssertionError("histogram/member mismatch must fail closed")


def test_mask_targets() -> None:
    semantic = np.ones((10, 10), dtype=np.uint16)
    semantic[:, 6:] = 2
    labels = {1: "chair", 2: "table"}
    mixed = mask_assignment(np.ones((10, 10), dtype=bool), semantic, labels)
    assert mixed["mask_mixed"] is True
    assert mixed["mask_two_foreground"] is True
    pure_mask = np.zeros((10, 10), dtype=bool)
    pure_mask[:, :5] = True
    pure = mask_assignment(pure_mask, semantic, labels)
    assert pure["mask_mixed"] is False

    observation_gt = {
        "a": {"gt_assignment_eligible": True, "mask_mixed": True, "mask_two_foreground": True, "frame_idx": 1},
        "b": {"gt_assignment_eligible": True, "mask_mixed": True, "mask_two_foreground": True, "frame_idx": 2},
        "c": {"gt_assignment_eligible": True, "mask_mixed": False, "mask_two_foreground": False, "frame_idx": 3},
    }
    target = target_features(["a", "b", "c"], observation_gt)
    assert target["repeated_mixed"] is True
    assert target["repeated_two_foreground"] is True


def test_metrics_and_causal_timing() -> None:
    perfect = metric_core([False, False, True, True], [0.1, 0.2, 0.8, 0.9])
    assert perfect["auroc"] == 1.0
    assert perfect["average_precision"] == 1.0

    positive_trace = [
        {
            "object_uid": "p",
            "frame_idx": 4,
            "raw_frame": 20,
            "event_sequence": 4,
            "num_observations": 5,
            "entropy_n5": 0.7,
            "repeated_mixed": False,
        },
        {
            "object_uid": "p",
            "frame_idx": 7,
            "raw_frame": 35,
            "event_sequence": 7,
            "num_observations": 8,
            "entropy_n5": 0.8,
            "repeated_mixed": True,
        },
    ]
    negative_trace = [
        {
            "object_uid": "n",
            "frame_idx": 5,
            "raw_frame": 25,
            "event_sequence": 5,
            "num_observations": 5,
            "entropy_n5": 0.1,
            "repeated_mixed": False,
        }
    ]
    scene = {
        "summary": {"scene": "synthetic"},
        "finals": [
            {
                "object_uid": "p",
                "is_background": False,
                "num_observations": 8,
                "gt_eligible_count": 8,
                "repeated_mixed": True,
            },
            {
                "object_uid": "n",
                "is_background": False,
                "num_observations": 5,
                "gt_eligible_count": 5,
                "repeated_mixed": False,
            },
        ],
        "traces": {"p": positive_trace, "n": negative_trace},
    }
    result = evaluate_rule_scene(scene, "entropy_n5", 0.5, "repeated_mixed")
    assert result["endpoint_precision"] == 1.0
    assert result["endpoint_recall"] == 1.0
    assert result["current_precision_at_first_fire"] == 0.0
    assert result["early_positive_trigger_count"] == 1
    assert result["post_delay_median_processed"] == 0.0


def main() -> int:
    test_production_features()
    test_mask_targets()
    test_metrics_and_causal_timing()
    print("all online label trigger tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
