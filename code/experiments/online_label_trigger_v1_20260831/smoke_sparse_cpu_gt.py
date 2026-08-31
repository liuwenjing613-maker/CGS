#!/usr/bin/env python3
"""One-frame integration smoke test for the CPU ReplicaSSG ray caster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

import render_sparse_cpu_gt as cpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--reference-gt", type=Path, required=True)
    parser.add_argument("--raw-frame", type=int, default=0)
    parser.add_argument(
        "--mesh-variant",
        choices=("mesh_semantic.ply", "mesh_preseg_semantic.ply"),
        default="mesh_semantic.ply",
    )
    args = parser.parse_args()

    sequence_root = Path("/home/chenkejun/beauty/conceptgraphs/data/Replica") / args.sequence
    source_scene = cpu.SCENE_NAMES[args.sequence]
    ssg_root = Path("/data/chenkejun/ReplicaSSG")
    mesh_path = ssg_root / "Replica/data" / source_scene / "habitat" / args.mesh_variant
    labels = cpu.object_labels(ssg_root / "files/objects.json", source_scene)
    poses = cpu.load_poses(sequence_root / "traj.txt")
    evidence = args.exp_root.resolve() / "evidence"
    frames = cpu.read_jsonl(evidence / "frames.jsonl")
    raw_by_uid = {
        row["frame_uid"]: cpu.source_frame_number(row["source_frame_id"])
        for row in frames
    }
    observations = [
        row
        for row in cpu.read_jsonl(evidence / "observations.jsonl")
        if row.get("status") == "kept"
        and raw_by_uid[row["frame_uid"]] == args.raw_frame
    ]
    union = np.zeros((680, 1200), dtype=bool)
    for row in observations:
        mask, _ = cpu.load_mask(
            args.exp_root.resolve(), row.get("processed_mask_ref") or row["mask_ref"]
        )
        union |= mask
    grid = np.zeros_like(union)
    grid[5::10, 5::10] = True
    vertices, triangles, triangle_ids, mesh_stats = cpu.read_semantic_mesh(mesh_path)
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
    )
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(tensor_mesh)
    ys, xs, ids, t_hit = cpu.raycast(
        ray_scene, triangle_ids, poses[args.raw_frame], union | grid, 1200, 680
    )
    semantic = np.zeros_like(union, dtype=np.uint16)
    semantic[ys, xs] = ids
    with np.load(args.reference_gt / f"frame{args.raw_frame:06d}.npz") as handle:
        reference = np.asarray(handle["semantic"])
    exact = semantic[union] == reference[union]
    descriptor_path = mesh_path.parent / "info_semantic.json"
    descriptor_objects = json.loads(descriptor_path.read_text(encoding="utf-8"))["objects"]

    def agreement_after(mapping: dict[int, int]) -> float:
        values = semantic[union].astype(np.int64, copy=True)
        mapped = np.fromiter(
            (mapping.get(int(value), -1) for value in values),
            dtype=np.int64,
            count=len(values),
        )
        return float(np.mean(mapped == reference[union]))

    descriptor_mapping_agreement = {
        "raw_plus_one": agreement_after({index: index + 1 for index in range(2048)}),
        "raw_minus_one": agreement_after({index: index - 1 for index in range(2048)}),
        "object_vector_index_to_id": agreement_after(
            {index: int(item["id"]) for index, item in enumerate(descriptor_objects)}
        ),
        "object_id_to_vector_index": agreement_after(
            {int(item["id"]): index for index, item in enumerate(descriptor_objects)}
        ),
        "object_id_to_vector_index_plus_one": agreement_after(
            {int(item["id"]): index + 1 for index, item in enumerate(descriptor_objects)}
        ),
    }
    transformed_agreement = {
        "direct": float(exact.mean()),
        "vertical_flip": float((semantic[union] == reference[::-1, :][union]).mean()),
        "horizontal_flip": float((semantic[union] == reference[:, ::-1][union]).mean()),
        "both_flips": float((semantic[union] == reference[::-1, ::-1][union]).mean()),
    }
    labeled = np.isin(reference[union], np.fromiter(labels, dtype=np.int64))
    pairs, pair_counts = np.unique(
        np.column_stack((semantic[union], reference[union])), axis=0, return_counts=True
    )
    top_pairs = [
        {"cpu": int(pairs[index, 0]), "reference": int(pairs[index, 1]), "pixels": int(pair_counts[index])}
        for index in np.argsort(pair_counts)[::-1][:20]
    ]
    cpu_majority = []
    majority_matches = 0
    for cpu_id in np.unique(pairs[:, 0]):
        selected = pairs[:, 0] == cpu_id
        local = np.argmax(pair_counts[selected])
        local_pairs = pairs[selected]
        local_counts = pair_counts[selected]
        majority_matches += int(local_counts[local])
        cpu_majority.append(
            {
                "cpu": int(cpu_id),
                "reference": int(local_pairs[local, 1]),
                "pixels": int(local_counts[local]),
                "purity": float(local_counts[local] / local_counts.sum()),
            }
        )
    cpu_majority.sort(key=lambda item: item["pixels"], reverse=True)
    depth = np.asarray(
        Image.open(sequence_root / "results" / f"depth{args.raw_frame:06d}.png"),
        dtype=np.float32,
    ) / 6553.5
    sampled = depth[ys, xs]
    valid = (sampled > 0) & np.isfinite(t_hit) & (t_hit > 0)
    error = np.abs(sampled[valid] - t_hit[valid])
    result = {
        "sequence": args.sequence,
        "raw_frame": args.raw_frame,
        "mesh_path": str(mesh_path),
        "mesh": mesh_stats,
        "observations": len(observations),
        "union_pixels": int(union.sum()),
        "pixel_exact_agreement": float(exact.mean()),
        "semantic_transform_agreement": transformed_agreement,
        "descriptor_mapping_agreement": descriptor_mapping_agreement,
        "labeled_pixel_exact_agreement": float(exact[labeled].mean()),
        "top_id_pairs": top_pairs,
        "best_cpu_id_remap_agreement": float(majority_matches / exact.size),
        "cpu_id_majority_mapping": cpu_majority[:30],
        "depth_valid": int(valid.sum()),
        "depth_median_abs_m": float(np.median(error)),
        "depth_p90_abs_m": float(np.quantile(error, 0.90)),
        "depth_within_5cm": float(np.mean(error <= 0.05)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["depth_within_5cm"] < 0.99 or result["pixel_exact_agreement"] < 0.95:
        raise RuntimeError("one-frame CPU GT smoke gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
