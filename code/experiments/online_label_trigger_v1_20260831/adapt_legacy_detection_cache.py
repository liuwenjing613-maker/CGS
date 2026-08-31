#!/usr/bin/env python3
"""Losslessly adapt legacy one-file detection caches to the current mapper schema.

Only four bookkeeping fields that the current mapper expects are added.  Every
pre-existing field is fingerprinted before and after serialization so that the
manifest can prove that masks, boxes, class IDs, and features did not change.
No image, map output, or ground-truth input is read by this adapter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "xyxy",
    "confidence",
    "class_id",
    "mask",
    "classes",
    "image_crops",
    "image_feats",
    "text_feats",
}
ADDED_KEYS = ("detection_class_labels", "labels", "edges", "captions")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray) and value.dtype != object:
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": digest,
        }
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "kind": type(value).__name__,
        "length": len(value) if hasattr(value, "__len__") else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_legacy(frame: int, data: dict[str, Any]) -> int:
    missing = sorted(REQUIRED_KEYS - set(data))
    if missing:
        raise ValueError(f"frame {frame}: missing legacy keys {missing}")
    unexpected_current = sorted(set(ADDED_KEYS) & set(data))
    if unexpected_current:
        raise ValueError(
            f"frame {frame}: source is not legacy; already has {unexpected_current}"
        )

    n = len(data["class_id"])
    for key in ("xyxy", "confidence", "mask", "image_crops", "image_feats", "text_feats"):
        if len(data[key]) != n:
            raise ValueError(f"frame {frame}: len({key})={len(data[key])}, expected {n}")

    classes = list(data["classes"])
    class_ids = np.asarray(data["class_id"])
    if class_ids.ndim != 1:
        raise ValueError(f"frame {frame}: class_id must be one-dimensional")
    if n and (int(class_ids.min()) < 0 or int(class_ids.max()) >= len(classes)):
        raise ValueError(
            f"frame {frame}: class_id outside [0, {len(classes) - 1}]"
        )
    return n


def write_gzip_pickle(path: Path, value: Any) -> None:
    # Fixed gzip mtime makes the adapted artifact reproducible.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = list(range(args.start, args.end, args.stride))
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    existing = list(args.dst_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"destination must be empty to preserve auditability: {args.dst_dir}"
        )

    started = time.time()
    records: list[dict[str, Any]] = []
    total_detections = 0
    source_key_sets: set[tuple[str, ...]] = set()

    for ordinal, frame in enumerate(frames, start=1):
        source = args.src_dir / f"frame{frame:06d}.pkl.gz"
        if not source.is_file():
            raise FileNotFoundError(source)
        with gzip.open(source, "rb") as handle:
            legacy = pickle.load(handle)
        if not isinstance(legacy, dict):
            raise TypeError(f"frame {frame}: expected dict, got {type(legacy).__name__}")

        n = validate_legacy(frame, legacy)
        source_key_sets.add(tuple(sorted(legacy)))
        before = {key: fingerprint(value) for key, value in legacy.items()}
        classes = list(legacy["classes"])
        class_ids = np.asarray(legacy["class_id"])
        detection_labels = [
            f"{classes[int(class_id)]} {index}"
            for index, class_id in enumerate(class_ids)
        ]

        adapted = dict(legacy)
        adapted["detection_class_labels"] = detection_labels
        # make_edges=false in the frozen mapping configuration.  These three
        # fields are therefore bookkeeping only; labels/edges are not consumed,
        # and empty captions are expanded deterministically by filter_captions.
        adapted["labels"] = list(detection_labels)
        adapted["edges"] = []
        adapted["captions"] = []

        output = args.dst_dir / f"frame{frame:06d}.pkl.gz"
        write_gzip_pickle(output, adapted)
        # The current mapper first checks the extensionless path, then its loader
        # opens the adjacent .pkl.gz.  A zero-byte sentinel avoids changing code.
        (args.dst_dir / f"frame{frame:06d}").touch(exist_ok=False)

        with gzip.open(output, "rb") as handle:
            roundtrip = pickle.load(handle)
        after = {key: fingerprint(roundtrip[key]) for key in legacy}
        if before != after:
            raise RuntimeError(f"frame {frame}: legacy field changed during adaptation")
        if roundtrip["detection_class_labels"] != detection_labels:
            raise RuntimeError(f"frame {frame}: derived labels failed round-trip")

        records.append(
            {
                "frame": frame,
                "detections": n,
                "source_sha256": sha256_file(source),
                "output_sha256": sha256_file(output),
                "legacy_payload_fingerprint": before,
            }
        )
        total_detections += n
        if ordinal % 50 == 0 or ordinal == len(frames):
            print(f"[{args.scene}] adapted {ordinal}/{len(frames)} frames", flush=True)

    manifest = {
        "schema_version": "legacy-detection-adapter/1.0",
        "scene": args.scene,
        "protocol": {"start": args.start, "end_exclusive": args.end, "stride": args.stride},
        "source_dir": str(args.src_dir.resolve()),
        "destination_dir": str(args.dst_dir.resolve()),
        "source_key_sets": [list(keys) for keys in sorted(source_key_sets)],
        "added_fields": {
            "detection_class_labels": "classes[class_id] + stable raw detection index",
            "labels": "same deterministic labels; inert because make_edges=false",
            "edges": "empty; make_edges=false",
            "captions": "empty; mapper deterministically expands to None captions",
        },
        "ground_truth_or_map_inputs_read": False,
        "frame_count": len(records),
        "total_detections": total_detections,
        "all_legacy_fields_roundtrip_identical": True,
        "duration_seconds": time.time() - started,
        "frames": records,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("scene", "frame_count", "total_detections", "all_legacy_fields_roundtrip_identical", "duration_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
