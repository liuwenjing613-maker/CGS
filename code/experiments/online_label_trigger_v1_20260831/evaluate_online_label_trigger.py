#!/usr/bin/env python3
"""Causal evaluation for the online observation-label consistency trigger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA = "online-label-trigger-eval/1.0"
VALID_MAPPER_STATUSES = {"completed", "MAP_COMPLETED_EVIDENCE_VALID"}
BG_LABELS = {"wall", "floor", "ceiling"}
STRICT_EXCLUDED = BG_LABELS | {"unknown", "undefined"}
SCORE_NAMES = (
    "entropy_n5",
    "minority_n5",
    "entropy_persistent",
    "minority_persistent",
    "repeated_alt_fraction",
)
THRESHOLDS = {
    "entropy_n5": (0.20, 0.30, 0.40, 0.50, 0.60),
    "entropy_persistent": (0.20, 0.30, 0.40, 0.50, 0.60),
    "minority_n5": (0.10, 0.15, 0.20, 0.25, 0.30),
    "minority_persistent": (0.10, 0.15, 0.20, 0.25, 0.30),
    "repeated_alt_fraction": (0.10, 0.15, 0.20, 0.25, 0.30),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze-scene")
    analyze.add_argument("--scene", required=True)
    analyze.add_argument("--source-scene", required=True)
    analyze.add_argument("--exp-root", type=Path, required=True)
    analyze.add_argument("--gt-root", type=Path, required=True)
    analyze.add_argument("--objects-json", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--overwrite", action="store_true")

    select = sub.add_parser("select-dev")
    select.add_argument("--scene-dir", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)

    holdout = sub.add_parser("evaluate-holdout")
    holdout.add_argument("--frozen-rule", type=Path, required=True)
    holdout.add_argument("--scene-dir", type=Path, action="append", required=True)
    holdout.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSONL {path}:{line_number}") from error
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def load_instance_labels(path: Path, source_scene: str) -> dict[int, str]:
    payload = read_json(path)
    scans = [item for item in payload["scans"] if item["scan"] == source_scene]
    if len(scans) != 1:
        raise RuntimeError(f"expected exactly one objects record for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def source_frame_number(source_frame_id: str) -> int:
    match = re.search(r"(\d+)$", str(source_frame_id))
    if not match:
        raise ValueError(f"cannot parse source frame: {source_frame_id}")
    return int(match.group(1))


def mask_assignment(mask: np.ndarray, semantic: np.ndarray, labels: dict[int, str]) -> dict:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return {
            "gt_assignment_eligible": False,
            "gt_top_id": None,
            "gt_top_label": None,
            "gt_top_pixels": 0,
            "gt_second_id": None,
            "gt_second_label": None,
            "gt_second_pixels": 0,
            "gt_purity": 0.0,
            "gt_second_fraction": 0.0,
            "gt_supported_fraction": 0.0,
            "mask_mixed": False,
            "mask_two_foreground": False,
        }
    ids, counts = np.unique(semantic[mask], return_counts=True)
    candidates = sorted(
        (
            (int(count), int(instance_id))
            for instance_id, count in zip(ids.tolist(), counts.tolist())
            if int(instance_id) in labels
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not candidates:
        return {
            "gt_assignment_eligible": False,
            "gt_top_id": None,
            "gt_top_label": None,
            "gt_top_pixels": 0,
            "gt_second_id": None,
            "gt_second_label": None,
            "gt_second_pixels": 0,
            "gt_purity": 0.0,
            "gt_second_fraction": 0.0,
            "gt_supported_fraction": 0.0,
            "mask_mixed": False,
            "mask_two_foreground": False,
        }
    top_count, top_id = candidates[0]
    second_count, second_id = candidates[1] if len(candidates) > 1 else (0, None)
    top_label = labels[top_id]
    second_label = labels[second_id] if second_id is not None else None
    purity = top_count / area
    second_fraction = second_count / area
    eligible = top_count >= 25
    return {
        "gt_assignment_eligible": bool(eligible),
        "gt_top_id": int(top_id),
        "gt_top_label": str(top_label),
        "gt_top_pixels": int(top_count),
        "gt_second_id": int(second_id) if second_id is not None else None,
        "gt_second_label": str(second_label) if second_label is not None else None,
        "gt_second_pixels": int(second_count),
        "gt_purity": float(purity),
        "gt_second_fraction": float(second_fraction),
        "gt_supported_fraction": float(sum(count for count, _ in candidates) / area),
        "mask_mixed": bool(eligible and (purity < 0.8 or second_fraction >= 0.1)),
        "mask_two_foreground": bool(
            eligible
            and second_id is not None
            and second_id != top_id
            and second_count >= 25
            and second_fraction >= 0.1
            and top_label not in STRICT_EXCLUDED
            and second_label not in STRICT_EXCLUDED
        ),
    }


def resolve_ref(exp_root: Path, ref: dict) -> Path:
    path = Path(str(ref["path"]))
    return path if path.is_absolute() else (exp_root / path).resolve()


def audit_observations(
    *,
    exp_root: Path,
    evidence_root: Path,
    gt_root: Path,
    labels: dict[int, str],
    output: Path,
    overwrite: bool,
) -> tuple[dict[str, dict], dict]:
    cache = output / "observation_gt.jsonl"
    if cache.is_file() and not overwrite:
        rows = read_jsonl(cache)
        return {row["obs_uid"]: row for row in rows}, {"cache_reused": True, "rows": len(rows)}

    frames = read_jsonl(evidence_root / "frames.jsonl")
    frame_by_uid = {row["frame_uid"]: row for row in frames}
    observations = read_jsonl(evidence_root / "observations.jsonl")
    kept = [row for row in observations if row.get("status") == "kept"]
    semantic_cache: dict[int, np.ndarray] = {}
    rows = []
    hash_failures = []
    for index, observation in enumerate(kept):
        frame = frame_by_uid[observation["frame_uid"]]
        raw_frame = source_frame_number(frame["source_frame_id"])
        if raw_frame not in semantic_cache:
            gt_path = gt_root / f"frame{raw_frame:06d}.npz"
            with np.load(gt_path) as handle:
                semantic_cache[raw_frame] = np.asarray(handle["semantic"])
        ref = observation.get("processed_mask_ref") or observation.get("mask_ref")
        mask_path = resolve_ref(exp_root, ref)
        if ref.get("sha256") and sha256_file(mask_path) != ref["sha256"]:
            hash_failures.append(str(mask_path))
        with np.load(mask_path) as handle:
            key = str(ref.get("key") or "mask")
            mask = np.asarray(handle[key])
            if ref.get("index") is not None:
                mask = mask[int(ref["index"])]
        assignment = mask_assignment(mask, semantic_cache[raw_frame], labels)
        rows.append(
            {
                "obs_uid": observation["obs_uid"],
                "frame_uid": observation["frame_uid"],
                "frame_idx": int(frame["frame_idx"]),
                "raw_frame": int(raw_frame),
                "class_id": int(observation["class_id"]),
                "class_name": str(observation["class_name"]),
                "mask_path": str(mask_path),
                "mask_sha256": ref.get("sha256"),
                "mask_area": int(np.asarray(mask, dtype=bool).sum()),
                **assignment,
            }
        )
        if (index + 1) % 1000 == 0:
            print(f"GT audit: {index + 1}/{len(kept)} observations", flush=True)
    if hash_failures:
        raise RuntimeError(f"processed mask hash failures: {hash_failures[:3]}")
    write_jsonl(cache, rows)
    return {row["obs_uid"]: row for row in rows}, {
        "cache_reused": False,
        "rows": len(rows),
        "processed_mask_hash_failures": 0,
    }


def normalized_entropy(counts: Iterable[int]) -> float:
    values = np.asarray([int(value) for value in counts if int(value) > 0], dtype=float)
    if len(values) <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def production_features(
    class_histogram: dict[str, int], member_uids: list[str], observations: dict[str, dict]
) -> dict:
    """This function deliberately receives no mask or GT fields."""
    counts = Counter({str(key): int(value) for key, value in class_histogram.items()})
    n = int(sum(counts.values()))
    members = list(dict.fromkeys(str(uid) for uid in member_uids))
    observed_counts = Counter(str(observations[uid]["class_id"]) for uid in members)
    if counts != observed_counts:
        raise RuntimeError(f"class histogram/member mismatch: {counts} != {observed_counts}")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant = ordered[0][0] if ordered else None
    dominant_count = ordered[0][1] if ordered else 0
    entropy = normalized_entropy(counts.values())
    minority = float(1.0 - dominant_count / max(n, 1))
    frames_by_label: dict[str, set[int]] = defaultdict(set)
    for uid in members:
        item = observations[uid]
        frames_by_label[str(item["class_id"])].add(int(item["frame_idx"]))
    persistent_alt_labels = {
        label
        for label, count in counts.items()
        if label != dominant and count >= 2 and len(frames_by_label[label]) >= 2
    }
    repeated_alt_count = int(sum(counts[label] for label in persistent_alt_labels))
    persistent = bool(persistent_alt_labels)
    eligible = n >= 5
    return {
        "num_observations": n,
        "label_count": int(len(counts)),
        "dominant_class_id": dominant,
        "dominant_count": int(dominant_count),
        "dominant_ratio": float(dominant_count / max(n, 1)),
        "raw_entropy": float(entropy),
        "raw_minority_ratio": minority,
        "persistent_alt_labels": sorted(persistent_alt_labels),
        "repeated_alt_count": repeated_alt_count,
        "persistence_gate": persistent,
        "entropy_n5": float(entropy) if eligible else 0.0,
        "minority_n5": minority if eligible else 0.0,
        "entropy_persistent": float(entropy) if eligible and persistent else 0.0,
        "minority_persistent": minority if eligible and persistent else 0.0,
        "repeated_alt_fraction": float(repeated_alt_count / max(n, 1)) if eligible else 0.0,
    }


def target_features(member_uids: list[str], observation_gt: dict[str, dict]) -> dict:
    members = [observation_gt[uid] for uid in dict.fromkeys(member_uids) if uid in observation_gt]
    eligible = [item for item in members if item["gt_assignment_eligible"]]
    mixed = [item for item in eligible if item["mask_mixed"]]
    strict = [item for item in eligible if item["mask_two_foreground"]]
    mixed_fraction = float(len(mixed) / max(len(eligible), 1))
    strict_fraction = float(len(strict) / max(len(eligible), 1))
    return {
        "gt_member_count": int(len(members)),
        "gt_eligible_count": int(len(eligible)),
        "mixed_count": int(len(mixed)),
        "mixed_fraction": mixed_fraction,
        "repeated_mixed": bool(len(mixed) >= 2 and mixed_fraction >= 0.05),
        "two_foreground_count": int(len(strict)),
        "two_foreground_fraction": strict_fraction,
        "repeated_two_foreground": bool(len(strict) >= 2 and strict_fraction >= 0.05),
        "mixed_member_frames": sorted({int(item["frame_idx"]) for item in mixed}),
        "strict_member_frames": sorted({int(item["frame_idx"]) for item in strict}),
    }


def event_sequence_map(path: Path) -> dict[str, int]:
    return {
        row["event_uid"]: int(row.get("event_sequence", index))
        for index, row in enumerate(read_jsonl(path))
    }


def build_traces(
    evidence_root: Path, observation_gt: dict[str, dict]
) -> tuple[list[dict], list[dict], dict]:
    frames = read_jsonl(evidence_root / "frames.jsonl")
    frame_by_uid = {row["frame_uid"]: row for row in frames}
    raw_observations = read_jsonl(evidence_root / "observations.jsonl")
    observations = {
        row["obs_uid"]: {
            "class_id": int(row["class_id"]),
            "frame_idx": int(frame_by_uid[row["frame_uid"]]["frame_idx"]),
        }
        for row in raw_observations
        if row.get("status") == "kept"
    }
    final_membership = [row for row in read_json(evidence_root / "final_membership.json") if row["status"] == "active"]
    final_by_uid = {str(row["object_uid"]): row for row in final_membership}
    sequences = event_sequence_map(evidence_root / "mapping_events.jsonl")
    versions = read_jsonl(evidence_root / "object_versions.jsonl")
    latest_per_frame: dict[tuple[str, int], tuple[int, dict]] = {}
    missing_members = set()
    for line_index, version in enumerate(versions):
        uid = str(version["object_uid"])
        if uid not in final_by_uid or version.get("status") != "active":
            continue
        frame = frame_by_uid[version["frame_uid"]]
        frame_idx = int(frame["frame_idx"])
        order = int(sequences.get(version.get("trigger_event_uid"), line_index))
        members = list(dict.fromkeys(str(item) for item in version["member_observation_uids"]))
        missing_members.update(uid for uid in members if uid not in observations)
        key = (uid, frame_idx)
        if key not in latest_per_frame or order > latest_per_frame[key][0]:
            latest_per_frame[key] = (order, {**version, "_frame": frame, "_members": members})
    if missing_members:
        raise RuntimeError(f"object versions reference missing observations: {list(missing_members)[:3]}")

    trace_rows = []
    for (uid, frame_idx), (order, version) in sorted(
        latest_per_frame.items(), key=lambda item: (item[0][0], item[0][1], item[1][0])
    ):
        final = final_by_uid[uid]
        members = version["_members"]
        trace_rows.append(
            {
                "object_uid": uid,
                "frame_idx": frame_idx,
                "raw_frame": source_frame_number(version["_frame"]["source_frame_id"]),
                "event_sequence": order,
                "operation": str(version.get("operation")),
                "class_name": str(version.get("class_name") or final.get("class_name") or "unknown"),
                "is_background": str(final.get("class_name") or "unknown") in BG_LABELS,
                "member_observation_uids": members,
                **production_features(version["class_histogram"], members, observations),
                **target_features(members, observation_gt),
            }
        )

    max_frame = max(int(row["frame_idx"]) for row in frames)
    trace_by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in trace_rows:
        trace_by_uid[row["object_uid"]].append(row)
    for uid, final in final_by_uid.items():
        members = list(dict.fromkeys(str(item) for item in final["member_observation_uids"]))
        final_features = production_features(final["class_histogram"], members, observations)
        final_target = target_features(members, observation_gt)
        synthetic = {
            "object_uid": uid,
            "frame_idx": max_frame,
            "raw_frame": source_frame_number(frames[-1]["source_frame_id"]),
            "event_sequence": 10**18,
            "operation": "FINAL_MEMBERSHIP",
            "class_name": str(final.get("class_name") or "unknown"),
            "is_background": str(final.get("class_name") or "unknown") in BG_LABELS,
            "member_observation_uids": members,
            **final_features,
            **final_target,
        }
        existing = trace_by_uid.get(uid, [])
        if not existing or stable_json_hash(
            {
                "members": existing[-1]["member_observation_uids"],
                "hist": [existing[-1][name] for name in SCORE_NAMES],
            }
        ) != stable_json_hash(
            {"members": members, "hist": [synthetic[name] for name in SCORE_NAMES]}
        ):
            trace_rows.append(synthetic)
            trace_by_uid[uid].append(synthetic)

    trace_rows.sort(key=lambda row: (row["frame_idx"], row["event_sequence"], row["object_uid"]))
    final_rows = []
    for uid, final in final_by_uid.items():
        rows = sorted(trace_by_uid[uid], key=lambda row: (row["frame_idx"], row["event_sequence"]))
        endpoint = dict(rows[-1])
        endpoint.pop("member_observation_uids", None)
        endpoint["trace_updates"] = len(rows)
        final_rows.append(endpoint)
    final_rows.sort(key=lambda row: row["object_uid"])
    integrity = {
        "frame_count": len(frames),
        "processed_frame_count": int(sum(bool(row["processed"]) for row in frames)),
        "source_frames_exact_0_1995_stride5": [source_frame_number(row["source_frame_id"]) for row in frames]
        == list(range(0, 2000, 5)),
        "kept_observations": len(observations),
        "final_active_objects": len(final_rows),
        "object_version_rows": len(versions),
        "causal_trace_rows": len(trace_rows),
        "missing_final_member_observations": 0,
    }
    return trace_rows, final_rows, integrity


def metric_core(labels: list[bool], scores: list[float]) -> dict:
    y = np.asarray(labels, dtype=np.uint8)
    score = np.asarray(scores, dtype=float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    output = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "auroc": None,
        "average_precision": None,
        "ap_lift": None,
    }
    if len(y) and len(np.unique(y)) == 2:
        output["auroc"] = float(roc_auc_score(y, score))
        output["average_precision"] = float(average_precision_score(y, score))
        output["ap_lift"] = float(output["average_precision"] - output["prevalence"])
    if len(y):
        order = np.argsort(-score, kind="stable")
        count = max(1, int(math.ceil(0.2 * len(y))))
        top = float(y[order[:count]].mean())
        bottom = float(y[order[-count:]].mean())
        output.update(
            {
                "top20_k": count,
                "top20_error_rate": top,
                "bottom20_error_rate": bottom,
                "top_bottom_ratio": float(top / bottom) if bottom else None,
            }
        )
    return output


def bootstrap_metrics(
    labels: list[bool], scores: list[float], *, seed: int, repetitions: int = 2000
) -> dict:
    base = metric_core(labels, scores)
    y = np.asarray(labels, dtype=np.uint8)
    score = np.asarray(scores, dtype=float)
    if len(y) < 2 or len(np.unique(y)) != 2:
        return base
    rng = np.random.default_rng(seed)
    auc, ap, lift = [], [], []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), len(y))
        sample_y = y[indices]
        if len(np.unique(sample_y)) != 2:
            continue
        sample_score = score[indices]
        sample_ap = average_precision_score(sample_y, sample_score)
        auc.append(roc_auc_score(sample_y, sample_score))
        ap.append(sample_ap)
        lift.append(sample_ap - sample_y.mean())
    for name, values in (("auroc", auc), ("average_precision", ap), ("ap_lift", lift)):
        if values:
            base[f"{name}_ci95"] = [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
    return base


def endpoint_metrics(rows: list[dict], target: str, seed: int) -> dict:
    selected = [
        row
        for row in rows
        if not row["is_background"]
        and row["num_observations"] >= 5
        and row["gt_eligible_count"] >= 5
    ]
    return {
        score: bootstrap_metrics(
            [bool(row[target]) for row in selected],
            [float(row[score]) for row in selected],
            seed=seed + index * 1009,
        )
        for index, score in enumerate(SCORE_NAMES)
    }


def analyze_scene(args: argparse.Namespace) -> int:
    exp_root = args.exp_root.resolve()
    evidence_root = exp_root / "evidence"
    gt_root = args.gt_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(evidence_root / "manifest.json")
    gt_manifest = read_json(gt_root / "manifest.json")
    if manifest.get("status") not in VALID_MAPPER_STATUSES:
        raise RuntimeError(f"mapper did not complete: {manifest.get('status')}")
    if int(gt_manifest["frame_count"]) != 400 or gt_manifest["frames"] != list(range(0, 2000, 5)):
        raise RuntimeError("GT sidecar protocol mismatch")
    labels = load_instance_labels(args.objects_json.resolve(), args.source_scene)
    observation_gt, gt_audit = audit_observations(
        exp_root=exp_root,
        evidence_root=evidence_root,
        gt_root=gt_root,
        labels=labels,
        output=output,
        overwrite=args.overwrite,
    )
    traces, final_rows, trace_integrity = build_traces(evidence_root, observation_gt)
    manifest_keys = set(manifest)
    leakage_keys = sorted(key for key in manifest_keys if str(key).lower().startswith("gt"))
    gt_generation_method = str(gt_manifest.get("generation_method") or "habitat_egl")
    cpu_gt_proof_valid = (
        gt_generation_method != "cpu_sparse_raycast"
        or (
            int(gt_manifest.get("parity_proof_count", 0)) >= 2
            and bool(gt_manifest.get("parity_all_passed", False))
            and len(
                {
                    str(item.get("scene"))
                    for item in gt_manifest.get("parity_proofs", [])
                }
            )
            >= 2
        )
    )
    integrity = {
        **trace_integrity,
        "mapper_manifest_status": manifest.get("status"),
        "mapper_manifest_gt_keys": leakage_keys,
        "gt_frame_count": int(gt_manifest["frame_count"]),
        "gt_min_within_5cm": float(gt_manifest["alignment_summary"]["min_within_5cm"]),
        "gt_max_p90_depth_error_m": float(gt_manifest["alignment_summary"]["max_p90_abs_depth_m"]),
        "gt_generation_method": gt_generation_method,
        "gt_cpu_parity_proof_count": int(gt_manifest.get("parity_proof_count", 0)),
        "gt_cpu_parity_all_passed": bool(gt_manifest.get("parity_all_passed", False)),
        "observation_gt_audit": gt_audit,
    }
    gates = {
        "mapper_completed": integrity["mapper_manifest_status"] in VALID_MAPPER_STATUSES,
        "frames_exact": integrity["source_frames_exact_0_1995_stride5"],
        "all_frames_processed": integrity["processed_frame_count"] == 400,
        "no_missing_members": integrity["missing_final_member_observations"] == 0,
        "no_gt_in_mapper_manifest": not leakage_keys,
        "gt_alignment_within_5cm_ge_0p99": integrity["gt_min_within_5cm"] >= 0.99,
        "cpu_gt_has_two_passing_dev_parity_proofs": cpu_gt_proof_valid,
    }
    if not all(gates.values()):
        raise RuntimeError(f"integrity gate failed: {gates}")
    seed = int(hashlib.sha256(args.scene.encode()).hexdigest()[:8], 16)
    summary = {
        "schema_version": SCHEMA,
        "scene": args.scene,
        "source_scene": args.source_scene,
        "exp_root": str(exp_root),
        "evidence_root": str(evidence_root),
        "gt_root": str(gt_root),
        "gt_manifest_sha256": sha256_file(gt_root / "manifest.json"),
        "mapper_manifest_sha256": sha256_file(evidence_root / "manifest.json"),
        "integrity": integrity,
        "integrity_gates": gates,
        "endpoint_eligible_objects": int(
            sum(
                not row["is_background"]
                and row["num_observations"] >= 5
                and row["gt_eligible_count"] >= 5
                for row in final_rows
            )
        ),
        "endpoint_targets": {
            target: int(
                sum(
                    bool(row[target])
                    for row in final_rows
                    if not row["is_background"]
                    and row["num_observations"] >= 5
                    and row["gt_eligible_count"] >= 5
                )
            )
            for target in ("repeated_mixed", "repeated_two_foreground")
        },
        "endpoint_metrics": {
            target: endpoint_metrics(final_rows, target, seed + target_index * 100000)
            for target_index, target in enumerate(("repeated_mixed", "repeated_two_foreground"))
        },
    }
    write_jsonl(output / "causal_trace.jsonl", traces)
    write_jsonl(output / "final_objects.jsonl", final_rows)
    write_csv(output / "final_objects.csv", final_rows)
    write_json(output / "scene_summary.json", summary)
    write_json(output / "integrity.json", {"integrity": integrity, "gates": gates})
    print(json.dumps({"scene": args.scene, "targets": summary["endpoint_targets"], "gates": gates}, indent=2))
    return 0


def load_scene_dir(path: Path) -> dict:
    summary = read_json(path / "scene_summary.json")
    traces = read_jsonl(path / "causal_trace.jsonl")
    finals = read_jsonl(path / "final_objects.jsonl")
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in traces:
        by_uid[row["object_uid"]].append(row)
    for rows in by_uid.values():
        rows.sort(key=lambda row: (row["frame_idx"], row["event_sequence"]))
    return {"path": path, "summary": summary, "traces": by_uid, "finals": finals}


def safe_div(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def fbeta(precision: float | None, recall: float | None, beta: float = 0.5) -> float:
    if precision is None or recall is None or precision == 0 or recall == 0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def evaluate_rule_scene(scene: dict, score_name: str, threshold: float, target: str) -> dict:
    eligible_finals = [
        row
        for row in scene["finals"]
        if not row["is_background"]
        and row["num_observations"] >= 5
        and row["gt_eligible_count"] >= 5
    ]
    records = []
    for final in eligible_finals:
        uid = final["object_uid"]
        trace = scene["traces"][uid]
        eligible_trace = [row for row in trace if row["num_observations"] >= 5]
        first = next((row for row in eligible_trace if float(row[score_name]) >= threshold), None)
        onset = next((row for row in eligible_trace if bool(row[target])), None)
        post = next(
            (
                row
                for row in eligible_trace
                if onset is not None
                and int(row["frame_idx"]) >= int(onset["frame_idx"])
                and float(row[score_name]) >= threshold
            ),
            None,
        )
        records.append(
            {
                "object_uid": uid,
                "final_positive": bool(final[target]),
                "fired": first is not None,
                "current_positive_at_fire": bool(first[target]) if first is not None else False,
                "first_frame": int(first["frame_idx"]) if first is not None else None,
                "first_raw_frame": int(first["raw_frame"]) if first is not None else None,
                "onset_frame": int(onset["frame_idx"]) if onset is not None else None,
                "onset_raw_frame": int(onset["raw_frame"]) if onset is not None else None,
                "post_frame": int(post["frame_idx"]) if post is not None else None,
                "post_delay_processed": int(post["frame_idx"] - onset["frame_idx"])
                if post is not None and onset is not None
                else None,
                "early_fire": bool(
                    first is not None and onset is not None and first["frame_idx"] < onset["frame_idx"]
                ),
            }
        )
    positives = sum(row["final_positive"] for row in records)
    negatives = len(records) - positives
    fired = [row for row in records if row["fired"]]
    tp = sum(row["final_positive"] for row in fired)
    fp = len(fired) - tp
    precision = safe_div(tp, len(fired))
    recall = safe_div(tp, positives)
    current_tp = sum(row["current_positive_at_fire"] for row in fired)
    current_precision = safe_div(current_tp, len(fired))
    post_detected = [row for row in records if row["final_positive"] and row["post_frame"] is not None]
    delays = [row["post_delay_processed"] for row in post_detected]
    output = {
        "scene": scene["summary"]["scene"],
        "score": score_name,
        "threshold": float(threshold),
        "target": target,
        "n": len(records),
        "positives": int(positives),
        "negatives": int(negatives),
        "fired": len(fired),
        "endpoint_tp": int(tp),
        "endpoint_fp": int(fp),
        "endpoint_precision": precision,
        "endpoint_recall": recall,
        "endpoint_f0p5": fbeta(precision, recall),
        "current_positive_at_first_fire": int(current_tp),
        "current_precision_at_first_fire": current_precision,
        "negative_trigger_rate": safe_div(fp, negatives),
        "early_positive_trigger_count": int(sum(row["early_fire"] and row["final_positive"] for row in records)),
        "post_onset_detected": len(post_detected),
        "post_onset_recall": safe_div(len(post_detected), positives),
        "post_delay_median_processed": float(np.median(delays)) if delays else None,
        "post_delay_p90_processed": float(np.quantile(delays, 0.9)) if delays else None,
        "records": records,
    }
    return output


def pool_rule_metrics(metrics: list[dict]) -> dict:
    records = [row for metric in metrics for row in metric["records"]]
    positives = sum(row["final_positive"] for row in records)
    negatives = len(records) - positives
    fired = [row for row in records if row["fired"]]
    tp = sum(row["final_positive"] for row in fired)
    fp = len(fired) - tp
    precision = safe_div(tp, len(fired))
    recall = safe_div(tp, positives)
    current_precision = safe_div(sum(row["current_positive_at_fire"] for row in fired), len(fired))
    delays = [
        row["post_delay_processed"]
        for row in records
        if row["final_positive"] and row["post_delay_processed"] is not None
    ]
    return {
        "n": len(records),
        "positives": int(positives),
        "negatives": int(negatives),
        "fired": len(fired),
        "endpoint_tp": int(tp),
        "endpoint_fp": int(fp),
        "endpoint_precision": precision,
        "endpoint_recall": recall,
        "endpoint_f0p5": fbeta(precision, recall),
        "current_precision_at_first_fire": current_precision,
        "negative_trigger_rate": safe_div(fp, negatives),
        "post_onset_recall": safe_div(len(delays), positives),
        "post_delay_median_processed": float(np.median(delays)) if delays else None,
        "post_delay_p90_processed": float(np.quantile(delays, 0.9)) if delays else None,
    }


def candidate_rank_metrics(scenes: list[dict], target: str) -> dict:
    return {
        score: {
            scene["summary"]["scene"]: scene["summary"]["endpoint_metrics"][target][score]
            for scene in scenes
        }
        for score in SCORE_NAMES
    }


def select_dev(args: argparse.Namespace) -> int:
    scenes = [load_scene_dir(path.resolve()) for path in args.scene_dir]
    scene_names = [scene["summary"]["scene"] for scene in scenes]
    if sorted(scene_names) != ["office0", "room0"]:
        raise RuntimeError(f"expected DEV room0/office0, got {scene_names}")
    rank = candidate_rank_metrics(scenes, "repeated_mixed")
    rank_qualified = {
        score: all(
            metrics.get("auroc") is not None
            and metrics["auroc"] > 0.5
            and metrics.get("ap_lift") is not None
            and metrics["ap_lift"] > 0
            for metrics in by_scene.values()
        )
        for score, by_scene in rank.items()
    }
    rules = []
    for score in SCORE_NAMES:
        for threshold in THRESHOLDS[score]:
            per_scene = [
                evaluate_rule_scene(scene, score, threshold, "repeated_mixed") for scene in scenes
            ]
            pooled = pool_rule_metrics(per_scene)
            gate = bool(
                rank_qualified[score]
                and all(
                    metric["endpoint_precision"] is not None
                    and metric["endpoint_precision"] >= 0.75
                    and metric["negative_trigger_rate"] is not None
                    and metric["negative_trigger_rate"] <= 0.25
                    for metric in per_scene
                )
                and pooled["endpoint_recall"] is not None
                and pooled["endpoint_recall"] >= 0.40
            )
            rules.append(
                {
                    "score": score,
                    "threshold": threshold,
                    "rank_qualified": rank_qualified[score],
                    "dev_gate_passed": gate,
                    "per_scene": [{k: v for k, v in item.items() if k != "records"} for item in per_scene],
                    "pooled": pooled,
                    "min_scene_f0p5": min(item["endpoint_f0p5"] for item in per_scene),
                    "min_scene_recall": min(item["endpoint_recall"] or 0.0 for item in per_scene),
                }
            )

    qualified = [rule for rule in rules if rule["dev_gate_passed"]]
    candidate_pool = qualified or rules
    simplicity = {
        "entropy_n5": 2,
        "minority_n5": 2,
        "entropy_persistent": 1,
        "minority_persistent": 1,
        "repeated_alt_fraction": 1,
    }
    selected = max(
        candidate_pool,
        key=lambda rule: (
            rule["min_scene_f0p5"],
            rule["min_scene_recall"],
            simplicity[rule["score"]],
            rule["threshold"],
        ),
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        scene["summary"]["scene"]: sha256_file(scene["path"] / "scene_summary.json")
        for scene in scenes
    }
    frozen = {
        "schema_version": "online-label-trigger-rule/1.0",
        "dev_scenes": scene_names,
        "holdout_scenes": ["room1", "office1"],
        "target": "repeated_mixed",
        "minimum_observations": 5,
        "score": selected["score"],
        "threshold": float(selected["threshold"]),
        "dev_gate_passed": bool(selected["dev_gate_passed"]),
        "selection_was_fallback": not bool(qualified),
        "source_scene_summary_sha256": source_hashes,
        "rank_metrics": rank,
        "selected_dev_metrics": selected,
        "threshold_grid": {key: list(value) for key, value in THRESHOLDS.items()},
        "rule_sha256": None,
    }
    frozen["rule_sha256"] = stable_json_hash({k: v for k, v in frozen.items() if k != "rule_sha256"})
    write_json(output / "frozen_rule.json", frozen)
    write_json(output / "dev_selection_summary.json", {"rank": rank, "rank_qualified": rank_qualified, "rules": rules, "selected": selected})
    print(json.dumps({"selected": selected, "rule_sha256": frozen["rule_sha256"]}, indent=2))
    return 0


def evaluate_holdout(args: argparse.Namespace) -> int:
    frozen = read_json(args.frozen_rule.resolve())
    expected_hash = frozen["rule_sha256"]
    actual_hash = stable_json_hash({k: v for k, v in frozen.items() if k != "rule_sha256"})
    if actual_hash != expected_hash:
        raise RuntimeError("frozen rule hash mismatch")
    scenes = [load_scene_dir(path.resolve()) for path in args.scene_dir]
    names = [scene["summary"]["scene"] for scene in scenes]
    if sorted(names) != ["office1", "room1"]:
        raise RuntimeError(f"expected HOLDOUT room1/office1, got {names}")
    score = frozen["score"]
    threshold = float(frozen["threshold"])
    primary_rank = candidate_rank_metrics(scenes, "repeated_mixed")[score]
    strict_rank = candidate_rank_metrics(scenes, "repeated_two_foreground")[score]
    per_scene_full = [
        evaluate_rule_scene(scene, score, threshold, "repeated_mixed") for scene in scenes
    ]
    pooled = pool_rule_metrics(per_scene_full)
    per_scene = [{k: v for k, v in row.items() if k != "records"} for row in per_scene_full]
    gates = {
        "primary_rank_both": all(
            metric.get("auroc") is not None
            and metric["auroc"] >= 0.65
            and metric.get("ap_lift") is not None
            and metric["ap_lift"] >= 0.10
            for metric in primary_rank.values()
        ),
        "pooled_endpoint_precision": pooled["endpoint_precision"] is not None
        and pooled["endpoint_precision"] >= 0.75,
        "pooled_endpoint_recall": pooled["endpoint_recall"] is not None
        and pooled["endpoint_recall"] >= 0.40,
        "pooled_current_precision": pooled["current_precision_at_first_fire"] is not None
        and pooled["current_precision_at_first_fire"] >= 0.60,
        "negative_trigger_rate_each": all(
            row["negative_trigger_rate"] is not None and row["negative_trigger_rate"] <= 0.25
            for row in per_scene
        ),
        "median_post_delay": pooled["post_delay_median_processed"] is not None
        and pooled["post_delay_median_processed"] <= 10,
        "strict_direction_both": all(
            metric.get("auroc") is not None and metric["auroc"] > 0.5
            for metric in strict_rank.values()
        ),
    }
    stop = any(
        metric.get("auroc") is None
        or metric["auroc"] <= 0.5
        or metric.get("ap_lift") is None
        or metric["ap_lift"] <= 0
        for metric in primary_rank.values()
    ) or any(
        metric.get("auroc") is None or metric["auroc"] <= 0.5
        for metric in strict_rank.values()
    )
    decision = "GO" if all(gates.values()) else ("STOP" if stop else "MODIFY")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_records = []
    for row in per_scene_full:
        for record in row["records"]:
            all_records.append({"scene": row["scene"], **record})
    summary = {
        "schema_version": SCHEMA,
        "decision": decision,
        "frozen_rule": frozen,
        "frozen_rule_path": str(args.frozen_rule.resolve()),
        "frozen_rule_sha256": sha256_file(args.frozen_rule.resolve()),
        "holdout_scenes": names,
        "primary_rank": primary_rank,
        "strict_rank": strict_rank,
        "per_scene_trigger": per_scene,
        "pooled_trigger": pooled,
        "gates": gates,
    }
    write_json(output / "holdout_summary.json", summary)
    write_jsonl(output / "holdout_object_records.jsonl", all_records)
    write_csv(output / "holdout_object_records.csv", all_records)
    print(json.dumps({"decision": decision, "gates": gates, "pooled": pooled}, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "analyze-scene":
        return analyze_scene(args)
    if args.command == "select-dev":
        return select_dev(args)
    if args.command == "evaluate-holdout":
        return evaluate_holdout(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
