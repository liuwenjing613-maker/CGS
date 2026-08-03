#!/usr/bin/env python3
"""Query a packaged ConceptGraphs object map with OpenCLIP text features."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import open_clip
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_points(points: np.ndarray, count: int) -> list[list[float]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return []
    if len(points) <= count:
        selected = points
    else:
        indices = np.linspace(0, len(points) - 1, num=count, dtype=np.int64)
        selected = points[indices]
    selected = selected[np.isfinite(selected).all(axis=1)]
    return selected.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-points", type=int, default=128)
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result_path = Path(args.result_path).resolve()
    output = Path(args.output).resolve()
    if not result_path.is_file():
        raise SystemExit(f"object map missing: {result_path}")
    if not args.query.strip():
        raise SystemExit("query must not be empty")
    if args.top_k < 1 or args.sample_points < 1:
        raise SystemExit("top-k and sample-points must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA query requested but torch.cuda.is_available() is false")

    with gzip.open(result_path, "rb") as handle:
        data = pickle.load(handle)
    objects = data.get("objects", []) if isinstance(data, dict) else []
    if not objects:
        raise SystemExit("object map contains no objects")

    features = []
    valid_objects = []
    valid_indices = []
    for index, obj in enumerate(objects):
        feature = np.asarray(obj.get("clip_ft"), dtype=np.float32).reshape(-1)
        if feature.size == 0 or not np.isfinite(feature).all():
            continue
        norm = float(np.linalg.norm(feature))
        if norm <= 0:
            continue
        features.append(feature / norm)
        valid_objects.append(obj)
        valid_indices.append(index)
    if not features:
        raise SystemExit("object map contains no finite CLIP features")

    print(f"Loading OpenCLIP {args.model}/{args.pretrained} on {args.device}...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    with torch.inference_mode():
        tokens = tokenizer([args.query]).to(args.device)
        text_feature = model.encode_text(tokens)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        object_features = torch.from_numpy(np.stack(features)).to(args.device)
        similarities = (object_features @ text_feature[0]).detach().cpu().numpy()

    order = np.argsort(-similarities, kind="stable")[: min(args.top_k, len(objects))]
    candidates = []
    for rank, feature_index in enumerate(order, 1):
        obj = valid_objects[int(feature_index)]
        object_index = valid_indices[int(feature_index)]
        bbox = np.asarray(obj.get("bbox_np"), dtype=np.float64)
        pcd = np.asarray(obj.get("pcd_np"), dtype=np.float64)
        if bbox.ndim != 2 or bbox.shape[1] != 3 or not np.isfinite(bbox).all():
            continue
        center = bbox.mean(axis=0)
        candidates.append(
            {
                "rank": rank,
                "object_index": object_index,
                "object_id": str(obj.get("id", object_index)),
                "class_name": str(obj.get("class_name", "unknown")),
                "clip_similarity": float(similarities[int(feature_index)]),
                "num_detections": int(obj.get("num_detections", 0)),
                "num_points": int(len(pcd)) if pcd.ndim == 2 else 0,
                "is_background": bool(obj.get("is_background", False)),
                "center_cg_map_m": center.tolist(),
                "bbox_corners_cg_map_m": bbox.tolist(),
                "sample_points_cg_map_m": sample_points(pcd, args.sample_points),
            }
        )
    if not candidates:
        raise SystemExit("query produced no candidates with finite geometry")

    payload = {
        "format_version": "cgs-object-query-v1",
        "query": args.query,
        "query_model": {
            "type": "openclip_cosine_similarity",
            "model": args.model,
            "pretrained": args.pretrained,
            "device": args.device,
        },
        "object_map": str(result_path),
        "object_map_sha256": sha256_file(result_path),
        "num_map_objects": len(objects),
        "top_k": len(candidates),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"QUERY_RESULT={output}")
    for item in candidates:
        print(
            f"rank={item['rank']} class={item['class_name']!r} "
            f"similarity={item['clip_similarity']:.6f} "
            f"detections={item['num_detections']} points={item['num_points']}"
        )


if __name__ == "__main__":
    main()
