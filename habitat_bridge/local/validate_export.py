#!/usr/bin/env python3
"""Validate a Habitat export against the CGS v1 interface and its payload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument(
        "--schema",
        default=str(workspace / "docs" / "CGS_Habitat_Interface_v1.schema.json"),
    )
    parser.add_argument("--mark-ready", action="store_true")
    return parser.parse_args()


def load_poses(path: Path) -> np.ndarray:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        if values.size != 16:
            raise ValueError(f"traj.txt line {line_number} has {values.size}, expected 16")
        rows.append(values.reshape(4, 4))
    return np.asarray(rows, dtype=np.float64)


def verify_checksums(sequence_dir: Path) -> int:
    checksum_file = sequence_dir / "checksums.sha256"
    entries = 0
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        expected, separator, relative_text = line.partition("  ")
        if not separator or len(expected) != 64:
            raise ValueError(f"invalid checksum line {line_number}")
        relative = Path(relative_text)
        target = (sequence_dir / relative).resolve()
        if sequence_dir.resolve() not in target.parents:
            raise ValueError(f"checksum path escapes sequence: {relative}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        entries += 1
    return entries


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir).resolve()
    ready = sequence_dir / "READY"
    if args.mark_ready:
        ready.unlink(missing_ok=True)

    metadata = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(metadata)
    num_frames = metadata["num_frames"]

    rgb_files = sorted((sequence_dir / "results").glob("frame[0-9][0-9][0-9][0-9][0-9][0-9].jpg"))
    depth_files = sorted((sequence_dir / "results").glob("depth[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    semantic_files = sorted((sequence_dir / "semantic").glob("semantic[0-9][0-9][0-9][0-9][0-9][0-9].npy"))
    if len(rgb_files) != num_frames or len(depth_files) != num_frames:
        raise ValueError(
            f"frame count mismatch rgb={len(rgb_files)} depth={len(depth_files)} expected={num_frames}"
        )
    if semantic_files and len(semantic_files) != num_frames:
        raise ValueError(f"semantic count mismatch: {len(semantic_files)}")

    expected_shape = (metadata["rgb"]["height"], metadata["rgb"]["width"])
    nonzero_depth = []
    depth_medians = []
    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.shape[:2] != expected_shape:
            raise ValueError(f"invalid RGB shape: {rgb_path}")
        if depth is None or depth.shape != expected_shape or depth.dtype != np.uint16:
            raise ValueError(f"invalid uint16 depth: {depth_path}")
        valid = depth[depth > 0]
        nonzero_depth.append(valid.size / depth.size)
        if valid.size:
            depth_medians.append(float(np.median(valid)))
    if not depth_medians:
        raise ValueError("all depth images are empty")
    median_depth_mm = float(np.median(depth_medians))
    if not 10.0 <= median_depth_mm <= 65000.0:
        raise ValueError(f"implausible median depth: {median_depth_mm} mm")

    poses = load_poses(sequence_dir / metadata["poses"]["file"])
    if poses.shape != (num_frames, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"invalid pose array: {poses.shape}")
    for index, pose in enumerate(poses):
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"pose {index} rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"pose {index} rotation determinant is not 1")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError(f"pose {index} has invalid homogeneous row")
    first_pose = np.asarray(metadata["T_habitat_world_from_cg_map"], dtype=np.float64)
    if not np.allclose(poses[0], first_pose, atol=1e-7):
        raise ValueError("first pose differs from T_habitat_world_from_cg_map")

    intrinsics = json.loads((sequence_dir / "intrinsics.json").read_text(encoding="utf-8"))
    for key in ("fx", "fy", "cx", "cy"):
        if not np.isclose(intrinsics[key], metadata["camera"][key], atol=1e-7):
            raise ValueError(f"intrinsics mismatch: {key}")
    frame_lines = (sequence_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    if len(frame_lines) != num_frames:
        raise ValueError(f"frames.jsonl count mismatch: {len(frame_lines)}")
    for index, line in enumerate(frame_lines):
        if json.loads(line)["frame_index"] != index:
            raise ValueError(f"frames.jsonl index mismatch at {index}")

    checksum_entries = verify_checksums(sequence_dir)
    if args.mark_ready:
        ready.touch()
    print(f"sequence_id={metadata['sequence_id']}")
    print(f"num_frames={num_frames}")
    print(f"rgb_size={expected_shape[1]}x{expected_shape[0]}")
    print(f"median_depth_mm={median_depth_mm:.1f}")
    print(f"mean_nonzero_depth_ratio={np.mean(nonzero_depth):.6f}")
    print(f"semantic_available={metadata.get('semantic_available', False)}")
    print(f"checksum_entries={checksum_entries}")
    print(f"ready={ready.exists()}")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
