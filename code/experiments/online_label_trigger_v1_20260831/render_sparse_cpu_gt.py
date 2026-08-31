#!/usr/bin/env python3
"""Generate evaluation-only sparse ReplicaSSG instance GT with CPU ray casting.

The mapper is already complete when this script runs.  Only pixels touched by
kept processed masks (plus a fixed depth-audit grid) are rendered.  Ground truth
never enters the online mapper or any production trigger feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


SCENE_NAMES = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}
BG_LABELS = {"wall", "floor", "ceiling"}
STRICT_EXCLUDED = BG_LABELS | {"unknown", "undefined"}
PARITY_LIMITS = {
    "pixel_exact_agreement": 0.98,
    "labeled_pixel_exact_agreement": 0.98,
    "frame_median_pixel_agreement": 0.98,
    "frame_p05_pixel_agreement": 0.95,
    "mask_mixed_agreement": 0.98,
    "mask_mixed_positive_recall": 0.95,
    "mask_two_foreground_agreement": 0.98,
    "mask_two_foreground_positive_recall": 0.95,
    "min_within_5cm": 0.99,
}
VALID_MAPPER_STATUSES = {"completed", "MAP_COMPLETED_EVIDENCE_VALID"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=sorted(SCENE_NAMES), required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument(
        "--replica-root",
        type=Path,
        default=Path("/home/chenkejun/beauty/conceptgraphs/data/Replica"),
    )
    parser.add_argument(
        "--replica-ssg-root",
        type=Path,
        default=Path("/data/chenkejun/ReplicaSSG"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=680)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--depth-scale", type=float, default=6553.5)
    parser.add_argument("--reference-gt", type=Path)
    parser.add_argument("--parity-proof", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}") from error
    return rows


def source_frame_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"cannot parse source frame: {value}")
    return int(match.group(1))


def load_poses(path: Path) -> list[np.ndarray]:
    poses = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = np.fromstring(line, sep=" ", dtype=np.float64)
            if values.size != 16:
                raise ValueError(f"{path}:{line_number}: expected 16 values")
            poses.append(values.reshape(4, 4))
    return poses


def replica_c2w_to_habitat(c2w: np.ndarray) -> np.ndarray:
    # Exact matrices produced by Habitat's quat_from_two_vectors calls in the
    # reference EGL renderer.
    habitat_to_replica = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    replica_camera_to_habitat = np.diag([1.0, -1.0, -1.0, 1.0])
    return habitat_to_replica @ c2w @ replica_camera_to_habitat


def replica_c2w_to_semantic_asset(c2w: np.ndarray) -> np.ndarray:
    """Camera pose in the raw PLY asset frame used by the CPU ray caster.

    Habitat rotates the stage asset from Replica's Z-up coordinates into its
    Y-up world with the same leading matrix used by
    ``replica_c2w_to_habitat``.  Because the CPU caster reads the untransformed
    PLY directly, that common stage/world rotation cancels.
    """
    replica_camera_to_habitat = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ replica_camera_to_habitat


def object_labels(path: Path, source_scene: str) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scans = [item for item in payload["scans"] if item["scan"] == source_scene]
    if len(scans) != 1:
        raise ValueError(f"expected one objects entry for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def read_semantic_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"unterminated PLY header: {path}")
            line = raw.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        face_line = next(line for line in header_lines if line.startswith("element face "))
        vertex_count = int(vertex_line.split()[-1])
        face_count = int(face_line.split()[-1])
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
            align=False,
        )
        vertices_raw = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        if len(vertices_raw) != vertex_count:
            raise ValueError("truncated PLY vertex block")
        vertices = np.column_stack(
            (vertices_raw["x"], vertices_raw["y"], vertices_raw["z"])
        ).astype(np.float32, copy=False)
        face_dtype = np.dtype(
            [("count", "u1"), ("indices", "<u4", (4,)), ("object_id", "<u2")],
            align=False,
        )
        remaining = os.fstat(handle.fileno()).st_size - handle.tell()
        if face_dtype.itemsize != 19 or remaining != face_count * face_dtype.itemsize:
            raise ValueError("semantic mesh is not the expected packed-quad ReplicaSSG format")
        faces = np.fromfile(handle, dtype=face_dtype, count=face_count)
    if len(faces) != face_count or not np.all(faces["count"] == 4):
        raise ValueError("non-quad or truncated face block")
    quads = faces["indices"]
    triangles = np.empty((face_count * 2, 3), dtype=np.uint32)
    triangles[0::2] = quads[:, [0, 1, 2]]
    triangles[1::2] = quads[:, [0, 2, 3]]
    triangle_ids = np.repeat(faces["object_id"].astype(np.uint16), 2)
    return vertices, triangles, triangle_ids, {
        "vertices": vertex_count,
        "quads": face_count,
        "triangles": int(len(triangles)),
    }


def resolve_ref(exp_root: Path, ref: dict) -> Path:
    path = Path(str(ref["path"]))
    return path if path.is_absolute() else (exp_root / path).resolve()


def load_mask(exp_root: Path, ref: dict) -> tuple[np.ndarray, Path]:
    path = resolve_ref(exp_root, ref)
    if ref.get("sha256") and sha256_file(path) != ref["sha256"]:
        raise RuntimeError(f"processed-mask hash mismatch: {path}")
    with np.load(path) as handle:
        key = str(ref.get("key") or "mask")
        mask = np.asarray(handle[key])
        if ref.get("index") is not None:
            mask = mask[int(ref["index"])]
    return np.asarray(mask, dtype=bool), path


def mask_flags(mask: np.ndarray, semantic: np.ndarray, labels: dict[int, str]) -> dict:
    area = int(mask.sum())
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
        return {"mask_mixed": False, "mask_two_foreground": False}
    top_count, top_id = candidates[0]
    second_count, second_id = candidates[1] if len(candidates) > 1 else (0, None)
    top_label = labels[top_id]
    second_label = labels[second_id] if second_id is not None else None
    purity = top_count / max(area, 1)
    second_fraction = second_count / max(area, 1)
    eligible = top_count >= 25
    return {
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


def positive_recall(reference: list[bool], prediction: list[bool]) -> float:
    positives = sum(reference)
    if positives == 0:
        raise RuntimeError("DEV parity scene has no positive observations")
    return sum(r and p for r, p in zip(reference, prediction)) / positives


def raycast(
    scene: o3d.t.geometry.RaycastingScene,
    triangle_ids: np.ndarray,
    pose: np.ndarray,
    query_mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(query_mask)
    focal = width / (2.0 * np.tan(np.deg2rad(90.0) / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    camera_directions = np.column_stack(
        ((xs - cx) / focal, (cy - ys) / focal, -np.ones_like(xs))
    ).astype(np.float32)
    asset_pose = replica_c2w_to_semantic_asset(pose)
    world_directions = camera_directions @ asset_pose[:3, :3].T
    origins = np.repeat(asset_pose[None, :3, 3], len(xs), axis=0).astype(np.float32)
    rays = np.concatenate((origins, world_directions.astype(np.float32)), axis=1)
    t_hit = np.empty(len(rays), dtype=np.float32)
    primitive = np.empty(len(rays), dtype=np.uint32)
    chunk = 250_000
    for start in range(0, len(rays), chunk):
        answer = scene.cast_rays(o3d.core.Tensor(rays[start : start + chunk]))
        stop = min(start + chunk, len(rays))
        t_hit[start:stop] = answer["t_hit"].numpy()
        primitive[start:stop] = answer["primitive_ids"].numpy()
    instance_ids = np.zeros(len(rays), dtype=np.uint16)
    valid = np.isfinite(t_hit) & (primitive < len(triangle_ids))
    instance_ids[valid] = triangle_ids[primitive[valid]]
    return ys, xs, instance_ids, t_hit


def main() -> int:
    args = parse_args()
    if (args.start, args.end, args.stride, args.width, args.height, args.hfov) != (
        0, 2000, 5, 1200, 680, 90.0
    ):
        raise ValueError("this validation is frozen to 0:2000:5, 1200x680, HFOV=90")
    if args.reference_gt and args.parity_proof:
        raise ValueError("reference GT and parity proofs are mutually exclusive")
    if not args.reference_gt and len(args.parity_proof) < 2:
        raise ValueError("holdout CPU GT requires two DEV parity proofs")

    sequence_root = args.replica_root.resolve() / args.sequence
    trajectory_path = sequence_root / "traj.txt"
    results_root = sequence_root / "results"
    source_scene = SCENE_NAMES[args.sequence]
    mesh_path = (
        args.replica_ssg_root.resolve()
        / "Replica" / "data" / source_scene / "habitat" / "mesh_semantic.ply"
    )
    objects_path = args.replica_ssg_root.resolve() / "files" / "objects.json"
    exp_root = args.exp_root.resolve()
    evidence_root = exp_root / "evidence"
    for required in (
        trajectory_path, results_root, mesh_path, objects_path,
        evidence_root / "manifest.json", evidence_root / "frames.jsonl",
        evidence_root / "observations.jsonl",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    mapper_manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    if mapper_manifest.get("status") not in VALID_MAPPER_STATUSES:
        raise RuntimeError("mapper evidence is not completed")

    frames = list(range(args.start, args.end, args.stride))
    poses = load_poses(trajectory_path)
    if len(poses) < args.end:
        raise RuntimeError("trajectory is shorter than the frozen protocol")
    labels = object_labels(objects_path, source_scene)
    frame_rows = read_jsonl(evidence_root / "frames.jsonl")
    frame_by_uid = {row["frame_uid"]: source_frame_number(row["source_frame_id"]) for row in frame_rows}
    if sorted(frame_by_uid.values()) != frames:
        raise RuntimeError("mapper source frames are not exactly 0:2000:5")
    observations = [
        row for row in read_jsonl(evidence_root / "observations.jsonl")
        if row.get("status") == "kept"
    ]
    refs_by_frame: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for row in observations:
        ref = row.get("processed_mask_ref") or row.get("mask_ref")
        refs_by_frame[frame_by_uid[row["frame_uid"]]].append((row["obs_uid"], ref))

    proofs = []
    for proof_path in args.parity_proof:
        proof_path = proof_path.resolve()
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if not proof.get("parity_gates") or not all(proof["parity_gates"].values()):
            raise RuntimeError(f"failed DEV parity proof: {proof_path}")
        proofs.append(
            {
                "scene": proof["sequence"],
                "path": str(proof_path),
                "sha256": sha256_file(proof_path),
            }
        )
    if proofs and len({proof["scene"] for proof in proofs}) < 2:
        raise RuntimeError("parity proofs must come from two distinct DEV scenes")

    output = args.output_root.resolve() / args.sequence
    ready = output / "READY"
    if ready.exists() and not args.overwrite:
        raise FileExistsError(f"{ready} exists; pass --overwrite to rerun")
    output.mkdir(parents=True, exist_ok=True)
    ready.unlink(missing_ok=True)
    (output / "INCOMPLETE").write_text(f"started_at_unix={time.time()}\n", encoding="utf-8")

    print(f"loading semantic mesh: {mesh_path}", flush=True)
    vertices, triangles, triangle_ids, mesh_stats = read_semantic_mesh(mesh_path)
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
    )
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(tensor_mesh)
    del vertices, triangles, tensor_mesh

    alignment = []
    frame_agreements = []
    pixel_matches = pixel_total = 0
    labeled_matches = labeled_total = 0
    mixed_reference: list[bool] = []
    mixed_cpu: list[bool] = []
    strict_reference: list[bool] = []
    strict_cpu: list[bool] = []
    mask_hashes_verified = 0
    started = time.perf_counter()
    fixed_grid = np.zeros((args.height, args.width), dtype=bool)
    fixed_grid[5::10, 5::10] = True
    reference_root = args.reference_gt.resolve() if args.reference_gt else None

    for ordinal, raw_frame in enumerate(frames):
        union = np.zeros((args.height, args.width), dtype=bool)
        masks = []
        for obs_uid, ref in refs_by_frame.get(raw_frame, []):
            mask, _ = load_mask(exp_root, ref)
            if mask.shape != union.shape:
                raise RuntimeError(f"frame {raw_frame}: mask shape {mask.shape}")
            union |= mask
            masks.append((obs_uid, mask))
            mask_hashes_verified += 1
        query = union | fixed_grid
        ys, xs, ids, t_hit = raycast(
            ray_scene, triangle_ids, poses[raw_frame], query, args.width, args.height
        )
        semantic = np.zeros((args.height, args.width), dtype=np.uint16)
        semantic[ys, xs] = ids
        temporary = output / f"frame{raw_frame:06d}.npz.incomplete"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, semantic=semantic)
        temporary.replace(output / f"frame{raw_frame:06d}.npz")

        source_depth = np.asarray(
            Image.open(results_root / f"depth{raw_frame:06d}.png"), dtype=np.float32
        ) / args.depth_scale
        sampled_source = source_depth[ys, xs]
        valid = (sampled_source > 0) & np.isfinite(t_hit) & (t_hit > 0)
        if int(valid.sum()) < 1000:
            raise RuntimeError(f"frame {raw_frame}: too few valid depth audit pixels")
        errors = np.abs(sampled_source[valid] - t_hit[valid])
        alignment.append(
            {
                "ordinal": ordinal,
                "raw_frame": raw_frame,
                "semantic_rays": int(union.sum()),
                "depth_audit_rays": int(query.sum()),
                "valid_depth_pixels": int(valid.sum()),
                "median_abs_depth_m": float(np.median(errors)),
                "p90_abs_depth_m": float(np.quantile(errors, 0.90)),
                "p99_abs_depth_m": float(np.quantile(errors, 0.99)),
                "within_5cm": float(np.mean(errors <= 0.05)),
            }
        )

        if reference_root is not None:
            with np.load(reference_root / f"frame{raw_frame:06d}.npz") as handle:
                reference = np.asarray(handle["semantic"])
            if union.any():
                cpu_values = semantic[union]
                reference_values = reference[union]
                matches = cpu_values == reference_values
                pixel_matches += int(matches.sum())
                pixel_total += int(matches.size)
                frame_agreements.append(float(matches.mean()))
                labeled = np.isin(reference_values, np.fromiter(labels, dtype=np.int64))
                labeled_matches += int((matches & labeled).sum())
                labeled_total += int(labeled.sum())
            for _, mask in masks:
                cpu_flags = mask_flags(mask, semantic, labels)
                ref_flags = mask_flags(mask, reference, labels)
                mixed_cpu.append(cpu_flags["mask_mixed"])
                mixed_reference.append(ref_flags["mask_mixed"])
                strict_cpu.append(cpu_flags["mask_two_foreground"])
                strict_reference.append(ref_flags["mask_two_foreground"])

        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(frames):
            print(f"CPU GT {args.sequence}: {ordinal + 1}/{len(frames)}", flush=True)

    alignment_summary = {
        "max_median_abs_depth_m": max(row["median_abs_depth_m"] for row in alignment),
        "max_p90_abs_depth_m": max(row["p90_abs_depth_m"] for row in alignment),
        "max_p99_abs_depth_m": max(row["p99_abs_depth_m"] for row in alignment),
        "min_within_5cm": min(row["within_5cm"] for row in alignment),
    }
    parity = None
    parity_gates = None
    if reference_root is not None:
        parity = {
            "reference_gt": str(reference_root),
            "reference_manifest_sha256": sha256_file(reference_root / "manifest.json"),
            "pixel_exact_agreement": pixel_matches / pixel_total,
            "labeled_pixel_exact_agreement": labeled_matches / labeled_total,
            "frame_median_pixel_agreement": float(np.median(frame_agreements)),
            "frame_p05_pixel_agreement": float(np.quantile(frame_agreements, 0.05)),
            "mask_mixed_agreement": float(np.mean(np.equal(mixed_reference, mixed_cpu))),
            "mask_mixed_positive_recall": positive_recall(mixed_reference, mixed_cpu),
            "mask_two_foreground_agreement": float(
                np.mean(np.equal(strict_reference, strict_cpu))
            ),
            "mask_two_foreground_positive_recall": positive_recall(
                strict_reference, strict_cpu
            ),
            "observations_compared": len(mixed_cpu),
            "union_pixels_compared": pixel_total,
            "labeled_pixels_compared": labeled_total,
        }
        parity_gates = {
            key: parity[key] >= limit
            for key, limit in PARITY_LIMITS.items()
            if key != "min_within_5cm"
        }
        parity_gates["min_within_5cm"] = (
            alignment_summary["min_within_5cm"] >= PARITY_LIMITS["min_within_5cm"]
        )

    manifest = {
        "schema_version": "sparse-cpu-replicassg-gt/1.0",
        "generation_method": "cpu_sparse_raycast",
        "sequence": args.sequence,
        "source_scene": source_scene,
        "frames": frames,
        "frame_count": len(frames),
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "width": args.width,
        "height": args.height,
        "hfov_degrees": args.hfov,
        "depth_scale": args.depth_scale,
        "trajectory_sha256": sha256_file(trajectory_path),
        "semantic_mesh_sha256": sha256_file(mesh_path),
        "objects_sha256": sha256_file(objects_path),
        "mapper_manifest_sha256": sha256_file(evidence_root / "manifest.json"),
        "mesh": mesh_stats,
        "kept_observations": len(observations),
        "processed_mask_hashes_verified": mask_hashes_verified,
        "elapsed_seconds": time.perf_counter() - started,
        "alignment": alignment,
        "alignment_summary": alignment_summary,
        "parity_limits": PARITY_LIMITS,
        "parity": parity,
        "parity_gates": parity_gates,
        "parity_proofs": proofs,
        "parity_proof_count": len(proofs),
        "parity_all_passed": bool(proofs) and all(
            all(json.loads(Path(proof["path"]).read_text(encoding="utf-8"))["parity_gates"].values())
            for proof in proofs
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    if alignment_summary["min_within_5cm"] < PARITY_LIMITS["min_within_5cm"]:
        raise RuntimeError(f"CPU GT depth alignment failed: {alignment_summary}")
    if parity_gates is not None and not all(parity_gates.values()):
        raise RuntimeError(f"CPU/Habitat parity failed: {parity_gates}")
    (output / "INCOMPLETE").unlink(missing_ok=True)
    ready.write_text("ready\n", encoding="utf-8")
    print(json.dumps({"alignment": alignment_summary, "parity": parity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
