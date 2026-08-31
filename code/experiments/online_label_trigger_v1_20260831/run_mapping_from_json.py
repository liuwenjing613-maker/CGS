#!/usr/bin/env python3
"""Run the unmodified online mapper from an audited frozen JSON config."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper-root", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument(
        "--detections-exp-suffix",
        help=(
            "Audited cache location in cached mode, or output cache location in "
            "on_the_fly mode; omission retains the source config value."
        ),
    )
    parser.add_argument(
        "--detection-mode",
        choices=("cached", "on_the_fly"),
        default="cached",
        help="Use an existing cache or run the frozen detector causally per frame.",
    )
    parser.add_argument(
        "--save-generated-detections",
        action="store_true",
        help="Persist detections generated in on_the_fly mode for audit/replay.",
    )
    parser.add_argument("--run-record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapper_root = args.mapper_root.resolve()
    config_path = args.config_json.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    source_protocol = {
        "start": int(payload["start"]),
        "end": int(payload["end"]),
        "stride": int(payload["stride"]),
    }
    if source_protocol["start"] != 0 or source_protocol["stride"] != 5:
        raise RuntimeError(f"source config is not a 0::5 replay: {source_protocol}")
    if source_protocol["end"] not in {-1, 2000}:
        raise RuntimeError(f"unexpected source end frame: {source_protocol}")
    source_force_detection = bool(payload.get("force_detection", False))
    if args.detection_mode == "on_the_fly" and not source_force_detection:
        raise RuntimeError(
            "on_the_fly mode is allowed only when the frozen source config also "
            "declares force_detection=true"
        )
    if args.save_generated_detections and args.detection_mode != "on_the_fly":
        raise RuntimeError("--save-generated-detections requires on_the_fly mode")

    old_paths = {
        "dataset_config": Path(str(payload["dataset_config"])).resolve(),
        "classes_file": Path(str(payload["classes_file"])).resolve(),
    }
    new_paths = {
        "dataset_config": mapper_root / "conceptgraph/dataset/dataconfigs/replica/replica.yaml",
        "classes_file": mapper_root / "conceptgraph/scannet200_classes.txt",
    }
    path_parity = {}
    for name in old_paths:
        old_hash = sha256_file(old_paths[name])
        new_hash = sha256_file(new_paths[name])
        if old_hash != new_hash:
            raise RuntimeError(f"{name} content mismatch: {old_hash} != {new_hash}")
        path_parity[name] = {
            "old": str(old_paths[name]),
            "new": str(new_paths[name]),
            "sha256": old_hash,
        }

    payload.update(
        {
            "repo_root": str(mapper_root),
            "dataset_config": str(new_paths["dataset_config"]),
            "classes_file": str(new_paths["classes_file"]),
            "exit_early_file": str(mapper_root / "conceptgraph/hydra_configs/early_exit.json"),
            "latest_pcd_filepath": None,
            "exp_suffix": args.exp_suffix,
            # Some audited baseline configs encode the full 400-frame sequence
            # as end=-1.  Pin the effective causal replay to the preregistered
            # explicit interval so every run has identical frame semantics.
            "start": 0,
            "end": 2000,
            "stride": 5,
            "force_detection": args.detection_mode == "on_the_fly",
            "make_edges": False,
            "save_evidence": True,
            "evidence_mode": "strict",
            # Strict evidence audit requires the processed mask and the full
            # observation point cloud.  The trigger evaluator never reads PCD;
            # this artifact exists only for replay/audit integrity.
            "evidence_save_observation_pcd": True,
            "evidence_observation_pcd_max_points": 0,
            "save_parity_trace": True,
            "save_pcd": True,
            "save_json": True,
            "save_objects_all_frames": False,
            "save_detections": bool(args.save_generated_detections),
            "save_video": False,
            "periodically_save_pcd": False,
            "use_rerun": False,
            "save_rerun": False,
            "vis_render": False,
            "use_wandb": False,
        }
    )
    if args.detections_exp_suffix:
        payload["detections_exp_suffix"] = args.detections_exp_suffix
    if int(payload["start"]) != 0 or int(payload["end"]) != 2000 or int(payload["stride"]) != 5:
        raise RuntimeError("expected frozen 0:2000:5 protocol")

    record = {
        "schema_version": "online-label-trigger-mapping/1.0",
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "source_protocol": source_protocol,
        "mapper_root": str(mapper_root),
        "mapper_entry_sha256": sha256_file(
            mapper_root / "conceptgraph/slam/rerun_realtime_mapping.py"
        ),
        "detection_protocol": {
            "mode": args.detection_mode,
            "source_config_force_detection": source_force_detection,
            "save_generated_detections": bool(args.save_generated_detections),
            "detections_exp_suffix": payload.get("detections_exp_suffix"),
            "causal_per_frame": True,
        },
        "path_content_parity": path_parity,
        "effective_config": payload,
    }
    args.run_record.parent.mkdir(parents=True, exist_ok=True)
    args.run_record.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(mapper_root))
    os.chdir(mapper_root)
    module = importlib.import_module("conceptgraph.slam.rerun_realtime_mapping")
    wrapped = getattr(module.main, "__wrapped__", None)
    if wrapped is None:
        raise RuntimeError("Hydra wrapped mapper entry is unavailable")
    wrapped(OmegaConf.create(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
