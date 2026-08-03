#!/usr/bin/env python3
"""Package a completed server mapping run into a portable map_bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np


def finite_stats(objects):
    centers = []
    classes = {}
    nan_centers = 0
    for obj in objects:
        tag = str(obj.get("class_name", "unknown"))
        classes[tag] = classes.get(tag, 0) + 1
        bbox_np = obj.get("bbox_np")
        if bbox_np is None:
            nan_centers += 1
            continue
        bbox_np = np.asarray(bbox_np, dtype=np.float64)
        if bbox_np.ndim != 2 or bbox_np.shape[1] != 3:
            nan_centers += 1
            continue
        values = [float(v) for v in bbox_np.mean(axis=0)]
        if not all(math.isfinite(v) for v in values):
            nan_centers += 1
            continue
        centers.append(values)
    if centers:
        mins = [min(row[i] for row in centers) for i in range(3)]
        maxs = [max(row[i] for row in centers) for i in range(3)]
        radius = max(math.sqrt(sum(v * v for v in row)) for row in centers)
    else:
        mins = maxs = [float("nan")] * 3
        radius = float("nan")
    return {
        "num_objects": len(objects),
        "num_nan_centers": nan_centers,
        "bbox_center_min": mins,
        "bbox_center_max": maxs,
        "max_radius_m": radius,
        "class_histogram": dict(sorted(classes.items())),
    }


def checksum_tree(root: Path) -> None:
    files = sorted(path for path in root.iterdir() if path.is_file())
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mapping-suffix", required=True)
    parser.add_argument("--detection-suffix", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    sequence_dir = Path(args.sequence_dir).resolve()
    mapping_dir = sequence_dir / "exps" / args.mapping_suffix
    detection_dir = sequence_dir / "exps" / args.detection_suffix
    if not mapping_dir.is_dir():
        raise SystemExit(f"mapping output missing: {mapping_dir}")
    pcd = mapping_dir / f"pcd_{args.mapping_suffix}.pkl.gz"
    obj_json = mapping_dir / f"obj_json_{args.mapping_suffix}.json"
    edge_json = mapping_dir / f"edge_json_{args.mapping_suffix}.json"
    for path in (pcd, obj_json, edge_json, sequence_dir / "metadata.json"):
        if not path.is_file():
            raise SystemExit(f"required output missing: {path}")

    bundle = Path(args.result_root).resolve() / args.run_id / "map_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    shutil.copy2(pcd, bundle / "object_map.pkl.gz")
    shutil.copy2(obj_json, bundle / "objects.json")
    shutil.copy2(edge_json, bundle / "edges.json")
    shutil.copy2(sequence_dir / "metadata.json", bundle / "sequence_metadata.json")
    config = mapping_dir / f"config_params_{args.mapping_suffix}.json"
    if not config.is_file():
        config = mapping_dir / "config_params.json"
    if config.is_file():
        shutil.copy2(config, bundle / "mapping_config.json")

    with gzip.open(pcd, "rb") as handle:
        result = pickle.load(handle)
    objects = result.get("objects", []) if isinstance(result, dict) else []
    metadata = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = {
        "format_version": "cgs-map-bundle-v1",
        "sequence_id": metadata["sequence_id"],
        "run_id": args.run_id,
        "map_frame": "first_opencv_camera",
        "habitat_world_frame": "habitat",
        "T_habitat_world_from_cg_map": metadata["T_habitat_world_from_cg_map"],
        "files": {
            "object_map": "object_map.pkl.gz",
            "objects": "objects.json",
            "edges": "edges.json",
            "sequence_metadata": "sequence_metadata.json",
        },
    }
    stats = finite_stats(objects)
    if stats["num_objects"] <= 0 or stats["num_nan_centers"] > 0:
        raise SystemExit(f"invalid map statistics: {stats}")
    (bundle / "map_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (bundle / "map_statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    tags = json.loads((bundle / "objects.json").read_text(encoding="utf-8"))
    if isinstance(tags, dict):
        tags = [item.get("object_tag", "unknown") for item in tags.values()]
    (bundle / "query_classes.txt").write_text("\n".join(sorted(set(map(str, tags)))) + "\n", encoding="utf-8")
    checksum_tree(bundle)
    (bundle / "COMPLETE").touch()
    print(f"bundle={bundle}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
