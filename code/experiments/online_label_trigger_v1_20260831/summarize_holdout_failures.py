#!/usr/bin/env python3
"""Join frozen-rule outcomes to final object traces for auditable failure analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, action="append", required=True)
    parser.add_argument("--classes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = args.evaluation_dir.resolve()
    holdout = json.loads((evaluation / "holdout_summary.json").read_text(encoding="utf-8"))
    threshold = float(holdout["frozen_rule"]["threshold"])
    score_name = str(holdout["frozen_rule"]["score"])
    classes = [line.strip() for line in args.classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def class_names(class_ids: list[str]) -> list[str]:
        names = []
        for value in class_ids:
            index = int(value)
            names.append(classes[index] if 0 <= index < len(classes) else f"<unknown:{index}>")
        return names

    final_by_scene: dict[str, dict[str, dict]] = {}
    traces_by_scene: dict[str, dict[str, list[dict]]] = {}
    for directory in args.scene_dir:
        rows = read_jsonl(directory.resolve() / "final_objects.jsonl")
        scene = directory.name
        final_by_scene[scene] = {row["object_uid"]: row for row in rows}
        traces: dict[str, list[dict]] = {}
        for row in read_jsonl(directory.resolve() / "causal_trace.jsonl"):
            traces.setdefault(row["object_uid"], []).append(row)
        for values in traces.values():
            values.sort(key=lambda row: int(row["event_sequence"]))
        traces_by_scene[scene] = traces

    outcomes = read_jsonl(evaluation / "holdout_object_records.jsonl")
    enriched: list[dict] = []
    for outcome in outcomes:
        scene = outcome["scene"]
        final = final_by_scene[scene][outcome["object_uid"]]
        if bool(outcome["final_positive"]):
            category = "TP" if bool(outcome["fired"]) else "FN"
        else:
            category = "FP" if bool(outcome["fired"]) else "TN"
        crossing = None
        if bool(outcome["fired"]):
            crossing = next(
                row
                for row in traces_by_scene[scene][outcome["object_uid"]]
                if int(row["num_observations"]) >= int(holdout["frozen_rule"]["minimum_observations"])
                and float(row[score_name]) >= threshold
            )
            if int(crossing["frame_idx"]) != int(outcome["first_frame"]):
                raise RuntimeError(
                    f"first-fire frame mismatch for {scene}/{outcome['object_uid']}"
                )
        final_score = float(final[score_name])
        enriched.append(
            {
                **outcome,
                "category": category,
                "class_name": final["class_name"],
                "final_score": final_score,
                "first_fire_score": float(crossing[score_name]) if crossing else None,
                "first_fire_num_observations": int(crossing["num_observations"]) if crossing else None,
                "first_fire_class_name": crossing["class_name"] if crossing else None,
                "first_fire_persistent_alt_class_names": class_names(crossing["persistent_alt_labels"]) if crossing else [],
                "final_score_below_frozen_threshold": final_score < threshold,
                "num_observations": int(final["num_observations"]),
                "label_count": int(final["label_count"]),
                "dominant_ratio": float(final["dominant_ratio"]),
                "repeated_alt_count": int(final["repeated_alt_count"]),
                "persistent_alt_labels": list(final["persistent_alt_labels"]),
                "persistent_alt_class_names": class_names(final["persistent_alt_labels"]),
                "mixed_count": int(final["mixed_count"]),
                "mixed_fraction": float(final["mixed_fraction"]),
                "two_foreground_count": int(final["two_foreground_count"]),
                "two_foreground_fraction": float(final["two_foreground_fraction"]),
                "strict_target": bool(final["repeated_two_foreground"]),
                "raw_entropy": float(final["raw_entropy"]),
            }
        )

    categories = ("TP", "FP", "FN", "TN")
    per_scene = {}
    for scene in sorted(final_by_scene):
        selected = [row for row in enriched if row["scene"] == scene]
        per_scene[scene] = {
            "counts": {category: sum(row["category"] == category for row in selected) for category in categories},
            "score_range": {
                category: quantiles([row["final_score"] for row in selected if row["category"] == category])
                for category in categories
            },
            "class_counts": {
                category: dict(Counter(row["class_name"] for row in selected if row["category"] == category).most_common())
                for category in categories
            },
        }

    payload = {
        "schema_version": "online-label-trigger-failure-analysis/1.0",
        "decision": holdout["decision"],
        "score": score_name,
        "threshold": threshold,
        "post_hoc_threshold_tuning_performed": False,
        "object_count": len(enriched),
        "pooled_counts": {category: sum(row["category"] == category for row in enriched) for category in categories},
        "per_scene": per_scene,
        "false_positives": sorted(
            (row for row in enriched if row["category"] == "FP"),
            key=lambda row: (-row["first_fire_score"], row["scene"], row["object_uid"]),
        ),
        "false_negatives": sorted(
            (row for row in enriched if row["category"] == "FN"),
            key=lambda row: (-row["final_score"], row["scene"], row["object_uid"]),
        ),
        "true_positives": sorted(
            (row for row in enriched if row["category"] == "TP"),
            key=lambda row: (-row["first_fire_score"], row["scene"], row["object_uid"]),
        ),
        "true_negatives": sorted(
            (row for row in enriched if row["category"] == "TN"),
            key=lambda row: (-row["final_score"], row["scene"], row["object_uid"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "counts": payload["pooled_counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
