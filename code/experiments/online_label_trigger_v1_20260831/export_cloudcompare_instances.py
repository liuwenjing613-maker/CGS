#!/usr/bin/env python3
"""Export final online object maps as a CloudCompare-friendly instance package.

Each instance is written as one binary PLY entity.  Entity filenames contain the
observation total and the three most frequent raw observation labels.  Complete
counts remain available in CSV/JSON sidecars.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import gzip
import json
import pickle
import re
import shutil
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCENE_SOURCES = {
    "room1": Path(
        "/home/chenkejun/beauty/conceptgraphs/data/Replica/room1/exps/"
        "online_label_trigger_v1_room1_holdout_onlinegen_pcd/"
        "pcd_online_label_trigger_v1_room1_holdout_onlinegen_pcd.pkl.gz"
    ),
    "office1": Path(
        "/home/chenkejun/beauty/conceptgraphs/data/Replica/office1/exps/"
        "online_label_trigger_v1_office1_holdout_onlinegen_pcd/"
        "pcd_online_label_trigger_v1_office1_holdout_onlinegen_pcd.pkl.gz"
    ),
}


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("instance_index", "<i4"),
        ("observation_count", "<i4"),
        ("distinct_label_count", "<i4"),
        ("dominant_label_fraction", "<f4"),
        ("is_background", "u1"),
    ]
)


def slug(text: str, limit: int = 24) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return (ascii_text or "label")[:limit].rstrip("-")


def instance_color(index: int, is_background: bool) -> tuple[int, int, int]:
    if is_background:
        shades = [(150, 150, 150), (185, 185, 185), (115, 125, 135), (165, 150, 135)]
        return shades[(index - 1) % len(shades)]
    hue = ((index - 1) * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return tuple(int(round(channel * 255)) for channel in rgb)


def observation_rows(obj: dict, class_names: list[str], stride: int) -> list[dict]:
    frames: dict[str, list[int]] = defaultdict(list)
    for class_id, frame_idx in zip(obj["class_id"], obj["image_idx"]):
        class_index = int(class_id)
        label = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"class_{class_index}"
        )
        frames[str(label)].append(int(frame_idx))

    total = max(1, sum(len(values) for values in frames.values()))
    rows = []
    for label, values in frames.items():
        count = len(values)
        rows.append(
            {
                "label": label,
                "count": count,
                "fraction": count / total,
                "first_processed_frame": min(values),
                "last_processed_frame": max(values),
                "first_raw_frame": min(values) * stride,
                "last_raw_frame": max(values) * stride,
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["label"]))
    return rows


def entity_filename(
    prefix: str,
    instance_id: str,
    stable_label: str,
    observation_count: int,
    labels: list[dict],
) -> str:
    top = "_".join(f"{slug(row['label'], 18)}-{row['count']}" for row in labels[:3])
    return (
        f"{prefix}_{instance_id}__{slug(stable_label)}__obs-{observation_count}"
        f"__{top or 'no-labels'}.ply"
    )


def finite_points(obj: dict) -> np.ndarray:
    points = np.asarray(obj["pcd_np"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unexpected pcd_np shape: {points.shape}")
    return points[np.isfinite(points).all(axis=1)]


def make_vertices(
    points: np.ndarray,
    color: tuple[int, int, int],
    instance_index: int,
    observation_count: int,
    distinct_label_count: int,
    dominant_fraction: float,
    is_background: bool,
) -> np.ndarray:
    vertices = np.empty(len(points), dtype=PLY_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = color
    vertices["instance_index"] = instance_index
    vertices["observation_count"] = observation_count
    vertices["distinct_label_count"] = distinct_label_count
    vertices["dominant_label_fraction"] = dominant_fraction
    vertices["is_background"] = int(is_background)
    return vertices


def write_ply(path: Path, vertices: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int instance_index\n"
        "property int observation_count\n"
        "property int distinct_label_count\n"
        "property float dominant_label_fraction\n"
        "property uchar is_background\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_scene(scene: str, source: Path, root: Path, stride: int) -> dict:
    with gzip.open(source, "rb") as handle:
        payload = pickle.load(handle)

    scene_root = root / scene
    foreground_root = scene_root / "01_foreground_instances"
    background_root = scene_root / "02_background_context"
    foreground_root.mkdir(parents=True, exist_ok=True)
    background_root.mkdir(parents=True, exist_ok=True)

    class_names = [str(value) for value in payload["class_names"]]
    objects = list(payload["objects"])
    objects.sort(
        key=lambda obj: (
            bool(obj["is_background"]),
            str(obj["class_name"]).lower(),
            str(obj["id"]),
        )
    )

    summary_rows: list[dict] = []
    label_rows: list[dict] = []
    metadata_objects: list[dict] = []
    all_vertices: list[np.ndarray] = []
    foreground_vertices: list[np.ndarray] = []
    background_vertices: list[np.ndarray] = []

    for index, obj in enumerate(objects, start=1):
        instance_id = f"I{index:02d}"
        stable_label = str(obj["class_name"])
        is_background = bool(obj["is_background"])
        points = finite_points(obj)
        labels = observation_rows(obj, class_names, stride)
        observation_count = len(obj["obs_uids"])
        if observation_count != sum(row["count"] for row in labels):
            raise ValueError(
                f"{scene}/{instance_id}: obs_uids={observation_count}, "
                f"label total={sum(row['count'] for row in labels)}"
            )
        dominant_fraction = labels[0]["fraction"] if labels else 0.0
        color = instance_color(index, is_background)
        vertices = make_vertices(
            points,
            color,
            index,
            observation_count,
            len(labels),
            dominant_fraction,
            is_background,
        )
        prefix = "BG" if is_background else "FG"
        filename = entity_filename(
            prefix, instance_id, stable_label, observation_count, labels
        )
        relative_file = Path(
            "02_background_context" if is_background else "01_foreground_instances"
        ) / filename
        write_ply(
            scene_root / relative_file,
            vertices,
            f"{scene} {instance_id}; exact observation counts are in instance_label_counts.csv",
        )

        all_vertices.append(vertices)
        (background_vertices if is_background else foreground_vertices).append(vertices)
        processed_frames = [int(frame) for frame in obj["image_idx"]]
        top_text = " | ".join(
            f"{row['label']}:{row['count']} ({row['fraction']:.1%})" for row in labels[:5]
        )
        summary_rows.append(
            {
                "instance_id": instance_id,
                "stable_label": stable_label,
                "is_background": is_background,
                "observation_count": observation_count,
                "distinct_label_count": len(labels),
                "dominant_label": labels[0]["label"] if labels else "",
                "dominant_count": labels[0]["count"] if labels else 0,
                "dominant_fraction": f"{dominant_fraction:.6f}",
                "first_raw_frame": min(processed_frames) * stride if processed_frames else "",
                "last_raw_frame": max(processed_frames) * stride if processed_frames else "",
                "point_count": len(points),
                "top5_labels": top_text,
                "object_uid": str(obj["id"]),
                "ply_file": str(relative_file),
            }
        )
        for rank, row in enumerate(labels, start=1):
            label_rows.append(
                {
                    "instance_id": instance_id,
                    "stable_label": stable_label,
                    "is_background": is_background,
                    "rank": rank,
                    "observation_label": row["label"],
                    "count": row["count"],
                    "fraction": f"{row['fraction']:.6f}",
                    "first_processed_frame": row["first_processed_frame"],
                    "last_processed_frame": row["last_processed_frame"],
                    "first_raw_frame": row["first_raw_frame"],
                    "last_raw_frame": row["last_raw_frame"],
                }
            )
        metadata_objects.append(
            {
                **summary_rows[-1],
                "color_rgb": list(color),
                "labels": [
                    {
                        **row,
                        "fraction": round(row["fraction"], 6),
                    }
                    for row in labels
                ],
            }
        )

    def concatenated(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=PLY_DTYPE)

    write_ply(
        scene_root / f"00_{scene}_ALL_instances_colored.ply",
        concatenated(all_vertices),
        f"{scene}; all final online instances; color=instance",
    )
    write_ply(
        scene_root / f"00_{scene}_FOREGROUND_only_colored.ply",
        concatenated(foreground_vertices),
        f"{scene}; foreground final online instances; color=instance",
    )
    write_ply(
        scene_root / f"00_{scene}_BACKGROUND_only_gray.ply",
        concatenated(background_vertices),
        f"{scene}; background context only",
    )

    write_csv(
        scene_root / "instance_summary.csv",
        summary_rows,
        [
            "instance_id",
            "stable_label",
            "is_background",
            "observation_count",
            "distinct_label_count",
            "dominant_label",
            "dominant_count",
            "dominant_fraction",
            "first_raw_frame",
            "last_raw_frame",
            "point_count",
            "top5_labels",
            "object_uid",
            "ply_file",
        ],
    )
    write_csv(
        scene_root / "instance_label_counts.csv",
        label_rows,
        [
            "instance_id",
            "stable_label",
            "is_background",
            "rank",
            "observation_label",
            "count",
            "fraction",
            "first_processed_frame",
            "last_processed_frame",
            "first_raw_frame",
            "last_raw_frame",
        ],
    )
    (scene_root / "metadata.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "source_map": str(source),
                "frame_protocol": {"start": 0, "end": 2000, "stride": stride},
                "instances": metadata_objects,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_points = sum(row["point_count"] for row in summary_rows)
    return {
        "scene": scene,
        "instance_count": len(summary_rows),
        "foreground_count": sum(not row["is_background"] for row in summary_rows),
        "background_count": sum(row["is_background"] for row in summary_rows),
        "observation_count": sum(row["observation_count"] for row in summary_rows),
        "point_count": total_points,
        "label_count": len(label_rows),
    }


def write_readme(root: Path, summaries: list[dict]) -> None:
    scene_lines = "\n".join(
        f"- {row['scene']}: {row['instance_count']} instances "
        f"({row['foreground_count']} foreground + {row['background_count']} background), "
        f"{row['observation_count']} observations, {row['point_count']} points"
        for row in summaries
    )
    text = f"""# CloudCompare：instance 与 observation 标签统计

{scene_lines}

## 最直观的打开方式

1. 先打开某场景的 `00_<scene>_ALL_instances_colored.ply`，确认完整空间结构。
2. 想逐个检查 instance 时，关闭合并点云，把 `01_foreground_instances` 目录中的 PLY 全选后拖入 CloudCompare。
3. 需要墙、地面、天花板作参照时，再把 `02_background_context` 中的 PLY 拖入；默认可先不加载，避免遮挡。
4. 左侧 DB Tree 中每个实体名都按以下格式显示：
   `FG_I01__stable-label__obs-总数__标签1-次数_标签2-次数_标签3-次数`。
5. 点击实体后按空格键可切换显示/隐藏；只保留一个实体可清楚查看它的形状和位置。
6. `instance_summary.csv` 每个 instance 一行；`instance_label_counts.csv` 每个 observation 标签一行，包含精确次数、比例和首次/末次帧。

## 点云中的标量字段

每个点还保存了 `instance_index`、`observation_count`、`distinct_label_count`、
`dominant_label_fraction` 和 `is_background`。这些字段在 CloudCompare 的 Properties / Scalar fields 中可选择着色。

## 解释边界

- observation 次数表示最终 object 吸收的历史 observation 数，不是点数。
- 文件名只展示前三个标签；CSV/JSON 保存全部标签，统计没有截断。
- 颜色仅用于区分 instance，不代表语义类别或错误类型。
- 本包来自最终在线地图；没有利用未来帧重建或修改 instance。
"""
    (root / "README_CN.md").write_text(text, encoding="utf-8")


def make_zip(root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENE_SOURCES), default=sorted(SCENE_SOURCES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    summaries = [
        export_scene(scene, SCENE_SOURCES[scene], args.output, args.stride)
        for scene in args.scenes
    ]
    write_readme(args.output, summaries)
    manifest = {
        "schema_version": "cloudcompare-instance-observation/1.0",
        "frame_protocol": {"start": 0, "end": 2000, "stride": args.stride},
        "scenes": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.zip_path:
        make_zip(args.output, args.zip_path)
    print(json.dumps({**manifest, "zip": str(args.zip_path) if args.zip_path else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
