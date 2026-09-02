#!/usr/bin/env python3
"""Export a lightweight per-frame GT-vs-online-owner reference viewer.

The viewer is evaluation-only. It combines the already-rendered full-frame
ReplicaSSG instance GT with the frozen online mapper evidence. No GT is fed
back into mapping or trigger scoring.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import gzip
import json
import pickle
import shutil
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path("/home/chenkejun/beauty/conceptgraphs")
EXPERIMENT = BASE / "results/experiments/online_label_trigger_v1_20260831"
SCENES = {
    "room0": {
        "map": BASE
        / "results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/"
        "online_label_trigger_v1_room0_dev_pcd/pcd_online_label_trigger_v1_room0_dev_pcd.pkl.gz",
        "gt": BASE
        / "results/experiments/oracle_three_error_20260828/pilot/gt_full/room0",
        "observations": EXPERIMENT / "dev/room0/observation_gt.jsonl",
        "rgb": BASE
        / "results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/results",
    },
    "room1": {
        "map": BASE
        / "data/Replica/room1/exps/online_label_trigger_v1_room1_holdout_onlinegen_pcd/"
        "pcd_online_label_trigger_v1_room1_holdout_onlinegen_pcd.pkl.gz",
        "gt": EXPERIMENT / "holdout_gt_habitat/room1",
        "observations": EXPERIMENT / "holdout/room1/observation_gt.jsonl",
        "rgb": BASE / "data/Replica/room1/results",
    },
    "office1": {
        "map": BASE
        / "data/Replica/office1/exps/online_label_trigger_v1_office1_holdout_onlinegen_pcd/"
        "pcd_online_label_trigger_v1_office1_holdout_onlinegen_pcd.pkl.gz",
        "gt": EXPERIMENT / "holdout_gt_habitat/office1",
        "observations": EXPERIMENT / "holdout/office1/observation_gt.jsonl",
        "rgb": BASE / "data/Replica/office1/results",
    },
}
FRAMES = list(range(0, 2000, 5))
BACKGROUND_LABELS = {"floor", "wall", "ceiling", "unknown", "undefined"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_color(index: int, muted: bool = False) -> tuple[int, int, int]:
    if muted:
        shades = [(126, 132, 140), (155, 160, 168), (178, 174, 166), (110, 120, 130)]
        return shades[index % len(shades)]
    hue = (index * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.96)
    return tuple(int(round(value * 255)) for value in rgb)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def boundary_map(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def label_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0
    return int(np.median(xs)), int(np.median(ys))


def draw_tag(image: Image.Image, position: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(image)
    x, y = position
    box = draw.textbbox((x, y), text, font=font, stroke_width=1)
    pad = 3
    rect = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))


def gt_overlay(
    rgb: np.ndarray,
    gt: np.ndarray,
    gt_labels: dict[int, str],
) -> Image.Image:
    color = np.zeros_like(rgb, dtype=np.uint8)
    visible = np.unique(gt)
    for instance_id in visible.tolist():
        label = gt_labels.get(int(instance_id), "unknown")
        muted = label in BACKGROUND_LABELS
        color[gt == instance_id] = stable_color(int(instance_id), muted)
    output = rgb.astype(np.float32)
    valid = gt > 0
    output[valid] = 0.52 * output[valid] + 0.48 * color[valid]
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[boundary_map(gt)] = (255, 255, 255)
    image = Image.fromarray(output, mode="RGB")
    font = load_font(16)
    for instance_id in visible.tolist():
        mask = gt == instance_id
        if int(mask.sum()) < 1800:
            continue
        x, y = label_centroid(mask)
        draw_tag(image, (x, y), f"GT{instance_id}:{gt_labels.get(int(instance_id), 'unknown')}", font)
    return image


def load_mask(path: Path) -> np.ndarray:
    with np.load(path) as handle:
        if "mask" in handle.files:
            mask = handle["mask"]
        elif len(handle.files) == 1:
            mask = handle[handle.files[0]]
        else:
            raise KeyError(f"cannot identify mask key in {path}: {handle.files}")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"unexpected mask shape {mask.shape}: {path}")
    return mask


def owner_overlay(
    rgb: np.ndarray,
    frame_rows: list[dict],
    owner_by_obs: dict[str, dict],
) -> tuple[Image.Image, int, int]:
    unions: dict[str, np.ndarray] = {}
    owner_meta: dict[str, dict] = {}
    unmatched = 0
    for row in frame_rows:
        owner = owner_by_obs.get(row["obs_uid"])
        if owner is None:
            unmatched += 1
            continue
        mask = load_mask(Path(row["mask_path"]))
        owner_id = owner["instance_id"]
        unions[owner_id] = mask if owner_id not in unions else (unions[owner_id] | mask)
        owner_meta[owner_id] = owner

    output = rgb.astype(np.float32)
    label_map = np.zeros(rgb.shape[:2], dtype=np.int32)
    for owner_id in sorted(unions):
        owner = owner_meta[owner_id]
        mask = unions[owner_id]
        color = np.asarray(owner["color"], dtype=np.float32)
        output[mask] = 0.52 * output[mask] + 0.48 * color
        label_map[mask] = owner["instance_index"]
    output = np.clip(output, 0, 255).astype(np.uint8)
    output[boundary_map(label_map) & (label_map > 0)] = (255, 255, 255)
    image = Image.fromarray(output, mode="RGB")
    font = load_font(16)
    for owner_id in sorted(unions):
        mask = unions[owner_id]
        if int(mask.sum()) < 1000:
            continue
        owner = owner_meta[owner_id]
        x, y = label_centroid(mask)
        draw_tag(image, (x, y), f"{owner_id}:{owner['stable_label']}", font)
    return image, len(unions), unmatched


def make_composite(
    scene: str,
    raw_frame: int,
    rgb_image: Image.Image,
    gt_image: Image.Image,
    owner_image: Image.Image,
    gt_visible: int,
    owner_visible: int,
) -> Image.Image:
    panel_size = (600, 340)
    panels = [
        rgb_image.resize(panel_size, Image.Resampling.LANCZOS),
        gt_image.resize(panel_size, Image.Resampling.LANCZOS),
        owner_image.resize(panel_size, Image.Resampling.LANCZOS),
    ]
    canvas = Image.new("RGB", (1800, 420), (20, 22, 26))
    title_font = load_font(22)
    small_font = load_font(17)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), f"{scene} | raw frame {raw_frame:06d} | processed {raw_frame // 5 + 1}/400", fill="white", font=title_font)
    titles = (
        "A  Original RGB",
        f"B  Full ReplicaSSG GT instances ({gt_visible} visible IDs)",
        f"C  ali-dev observations grouped by final owner ({owner_visible} owners)",
    )
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * 600
        canvas.paste(panel, (x, 52))
        draw.rectangle((x, 52, x + 600, 78), fill=(0, 0, 0))
        draw.text((x + 8, 55), title, fill="white", font=small_font)
    draw.text(
        (18, 397),
        "Reference rule: different GT IDs = annotated as separate instances; same GT ID across views = annotated as one instance. Confirm across multiple frames.",
        fill=(220, 224, 230),
        font=small_font,
    )
    return canvas


def sorted_objects(payload: dict) -> list[dict]:
    objects = list(payload["objects"])
    objects.sort(
        key=lambda obj: (
            bool(obj["is_background"]),
            str(obj["class_name"]).lower(),
            str(obj["id"]),
        )
    )
    return objects


def build_owner_index(map_path: Path) -> tuple[dict[str, dict], list[dict]]:
    with gzip.open(map_path, "rb") as handle:
        payload = pickle.load(handle)
    owner_by_obs: dict[str, dict] = {}
    owner_rows = []
    for index, obj in enumerate(sorted_objects(payload), start=1):
        owner = {
            "instance_id": f"I{index:02d}",
            "instance_index": index,
            "stable_label": str(obj["class_name"]),
            "object_uid": str(obj["id"]),
            "is_background": bool(obj["is_background"]),
            "num_observations": len(obj["obs_uids"]),
            "color": stable_color(index, bool(obj["is_background"])),
        }
        owner_rows.append(owner)
        for obs_uid in obj["obs_uids"]:
            if obs_uid in owner_by_obs:
                raise RuntimeError(f"duplicate final owner for {obs_uid}")
            owner_by_obs[str(obs_uid)] = owner
    return owner_by_obs, owner_rows


def ownership_statistics(
    scene: str,
    observations: list[dict],
    owner_by_obs: dict[str, dict],
    owner_rows: list[dict],
    output: Path,
) -> tuple[list[dict], list[dict]]:
    by_owner: dict[str, list[dict]] = defaultdict(list)
    for row in observations:
        owner = owner_by_obs.get(row["obs_uid"])
        if owner is None:
            continue
        by_owner[owner["instance_id"]].append(row)

    summary = []
    candidates_by_gt: dict[int, list[dict]] = defaultdict(list)
    for owner in owner_rows:
        rows = by_owner.get(owner["instance_id"], [])
        pure = [
            row
            for row in rows
            if row.get("gt_assignment_eligible")
            and float(row.get("gt_purity", 0.0)) >= 0.8
            and not row.get("mask_mixed")
            and row.get("gt_top_id") is not None
        ]
        counts = Counter(int(row["gt_top_id"]) for row in pure)
        labels = {}
        for row in pure:
            labels[int(row["gt_top_id"])] = str(row.get("gt_top_label") or "unknown")
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        dominant_id = ordered[0][0] if ordered else None
        dominant_count = ordered[0][1] if ordered else 0
        dominant_fraction = dominant_count / len(pure) if pure else 0.0
        supported_foreground = [
            (gt_id, count)
            for gt_id, count in ordered
            if count >= 2 and labels.get(gt_id, "unknown") not in BACKGROUND_LABELS
        ]
        row = {
            "scene": scene,
            "instance_id": owner["instance_id"],
            "stable_label": owner["stable_label"],
            "is_background": owner["is_background"],
            "observation_count": len(rows),
            "mixed_mask_count": sum(bool(item.get("mask_mixed")) for item in rows),
            "mixed_mask_fraction": f"{sum(bool(item.get('mask_mixed')) for item in rows) / max(len(rows), 1):.6f}",
            "pure_observation_count": len(pure),
            "dominant_gt_id": dominant_id if dominant_id is not None else "",
            "dominant_gt_label": labels.get(dominant_id, "") if dominant_id is not None else "",
            "dominant_pure_count": dominant_count,
            "dominant_pure_fraction": f"{dominant_fraction:.6f}",
            "pure_gt_distribution": " | ".join(
                f"GT{gt_id}:{labels.get(gt_id, 'unknown')}={count}" for gt_id, count in ordered
            ),
            "supported_foreground_gt_ids_ge2": " | ".join(
                f"GT{gt_id}:{labels.get(gt_id, 'unknown')}={count}"
                for gt_id, count in supported_foreground
            ),
            "potential_false_merge": len(supported_foreground) >= 2,
            "object_uid": owner["object_uid"],
        }
        summary.append(row)
        if (
            dominant_id is not None
            and dominant_count >= 2
            and dominant_fraction >= 0.5
            and labels.get(dominant_id, "unknown") not in BACKGROUND_LABELS
        ):
            candidates_by_gt[dominant_id].append(row)

    split_pairs = []
    for gt_id, candidates in sorted(candidates_by_gt.items()):
        for first, second in combinations(candidates, 2):
            split_pairs.append(
                {
                    "scene": scene,
                    "gt_instance_id": gt_id,
                    "gt_label": first["dominant_gt_label"],
                    "instance_a": first["instance_id"],
                    "label_a": first["stable_label"],
                    "support_a": first["dominant_pure_count"],
                    "fraction_a": first["dominant_pure_fraction"],
                    "instance_b": second["instance_id"],
                    "label_b": second["stable_label"],
                    "support_b": second["dominant_pure_count"],
                    "fraction_b": second["dominant_pure_fraction"],
                    "potential_false_split": True,
                }
            )

    write_csv(
        output / "instance_to_gt_summary.csv",
        summary,
        list(summary[0].keys()) if summary else [],
    )
    split_fields = [
        "scene", "gt_instance_id", "gt_label", "instance_a", "label_a",
        "support_a", "fraction_a", "instance_b", "label_b", "support_b",
        "fraction_b", "potential_false_split",
    ]
    write_csv(output / "potential_false_split_pairs.csv", split_pairs, split_fields)
    return summary, split_pairs


def export_scene(scene: str, output_root: Path) -> dict:
    scene_output = output_root / scene
    composite_output = scene_output / "frames"
    gt_png_output = scene_output / "gt_instance_id_png16"
    composite_output.mkdir(parents=True)
    gt_png_output.mkdir(parents=True)

    gt_root = SCENES[scene]["gt"]
    gt_manifest = json.loads((gt_root / "manifest.json").read_text(encoding="utf-8"))
    if gt_manifest["frames"] != FRAMES or gt_manifest["width"] != 1200 or gt_manifest["height"] != 680:
        raise RuntimeError(f"unexpected GT protocol for {scene}")
    gt_labels = {
        int(item["id"]): str(item["label"])
        for item in gt_manifest["visible_instances"]
    }
    observations = read_jsonl(SCENES[scene]["observations"])
    if len(observations) == 0:
        raise RuntimeError(f"no observations for {scene}")
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in observations:
        by_frame[int(row["raw_frame"])].append(row)

    owner_by_obs, owner_rows = build_owner_index(SCENES[scene]["map"])
    if set(owner_by_obs) != {row["obs_uid"] for row in observations}:
        raise RuntimeError(f"final owner/GT observation mismatch for {scene}")
    summary, split_pairs = ownership_statistics(
        scene, observations, owner_by_obs, owner_rows, scene_output
    )

    unmatched_total = 0
    visible_owner_counts = []
    visible_gt_counts = []
    rgb_root = SCENES[scene]["rgb"]
    for ordinal, raw_frame in enumerate(FRAMES):
        rgb_path = rgb_root / f"frame{raw_frame:06d}.jpg"
        gt_path = gt_root / f"frame{raw_frame:06d}.npz"
        with Image.open(rgb_path) as handle:
            rgb_image = handle.convert("RGB")
        rgb = np.asarray(rgb_image)
        with np.load(gt_path) as handle:
            gt = np.asarray(handle["semantic"], dtype=np.uint16)
        if rgb.shape[:2] != gt.shape:
            raise RuntimeError(f"shape mismatch {scene} frame {raw_frame}: {rgb.shape} vs {gt.shape}")

        Image.fromarray(gt).save(
            gt_png_output / f"frame{raw_frame:06d}.png", optimize=True
        )
        rendered_gt = gt_overlay(rgb, gt, gt_labels)
        rendered_owner, owner_count, unmatched = owner_overlay(
            rgb, by_frame.get(raw_frame, []), owner_by_obs
        )
        unmatched_total += unmatched
        gt_count = len(np.unique(gt))
        visible_owner_counts.append(owner_count)
        visible_gt_counts.append(gt_count)
        composite = make_composite(
            scene, raw_frame, rgb_image, rendered_gt, rendered_owner, gt_count, owner_count
        )
        composite.save(
            composite_output / f"frame{raw_frame:06d}.jpg",
            quality=82,
            optimize=True,
            progressive=True,
        )
        if (ordinal + 1) % 40 == 0 or ordinal + 1 == len(FRAMES):
            print(f"viewer {scene}: {ordinal + 1}/{len(FRAMES)}", flush=True)

    return {
        "scene": scene,
        "frame_count": len(FRAMES),
        "instance_count": len(owner_rows),
        "observation_count": len(observations),
        "unmatched_observations": unmatched_total,
        "potential_false_merge_instances": sum(
            bool(row["potential_false_merge"]) for row in summary
        ),
        "potential_false_split_pairs": len(split_pairs),
        "mean_visible_gt_ids": float(np.mean(visible_gt_counts)),
        "mean_visible_online_owners": float(np.mean(visible_owner_counts)),
        "gt_alignment_summary": gt_manifest["alignment_summary"],
    }


def write_index(output: Path) -> None:
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT instance reference viewer</title>
<style>
body{margin:0;background:#11151b;color:#eef2f6;font-family:system-ui,"Microsoft YaHei",sans-serif}header{position:sticky;top:0;background:#171d26;padding:12px 18px;box-shadow:0 2px 12px #0008;z-index:2}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}select,input,button{font-size:16px}input[type=range]{width:min(520px,62vw)}input[type=number]{width:92px}button{cursor:pointer;padding:3px 12px}.error{color:#ff8d8d;min-width:260px}#frame{font-variant-numeric:tabular-nums;color:#75d5ff;font-weight:700}.wrap{padding:16px}.viewer{max-width:1800px;margin:auto;background:#090b0e;border:1px solid #313946;border-radius:10px;overflow:hidden}.viewer img{display:block;width:100%;height:auto}.note{max-width:1200px;margin:16px auto;line-height:1.7;color:#c7d0dc}kbd{background:#2a3340;padding:2px 7px;border-radius:5px}.warn{color:#ffd479}
</style></head><body><header><div class="controls"><label>场景 <select id="scene"><option>room0</option><option>room1</option><option>office1</option></select></label><label>处理帧 <input id="slider" type="range" min="0" max="399" value="0"></label><label>原始帧号 <input id="raw-input" type="number" min="0" max="1995" step="5" value="0"></label><button id="jump" type="button">查找</button><span id="frame"></span><span id="error" class="error"></span></div></header>
<div class="wrap"><div class="viewer"><img id="image" alt="GT comparison"></div><div class="note"><p>查找：输入原始帧号后按 <kbd>Enter</kbd> 或“查找”。本实验在线处理的是 <code>0, 5, …, 1995</code> 共 400 帧，未处理的中间帧不会被静默取整。</p><p>操作：<kbd>←</kbd>/<kbd>→</kbd> 切换一帧，<kbd>PageUp</kbd>/<kbd>PageDown</kbd> 跳 20 个处理帧。页面一次只加载当前图片，不会像 3D 网页那样卡。</p><p><b>判断原则：</b>同一帧里不同 GT ID 表示数据集标为不同物理实例；跨多个视角始终是同一 GT ID，说明应视为一个实例。<span class="warn">不要根据一帧、一个标签字符串或少量边界错位直接拆分/合并。</span></p><p>精确 16-bit GT instance ID 图在各场景的 <code>gt_instance_id_png16</code>；自动候选见两个 CSV。</p></div></div>
<script>
const scene=document.querySelector('#scene'),slider=document.querySelector('#slider'),rawInput=document.querySelector('#raw-input'),jump=document.querySelector('#jump'),image=document.querySelector('#image'),frame=document.querySelector('#frame'),error=document.querySelector('#error');
function pad(n){return String(n).padStart(6,'0')} function show(){const raw=Number(slider.value)*5;rawInput.value=raw;error.textContent='';frame.textContent=`raw frame ${pad(raw)} · ${Number(slider.value)+1}/400`;image.src=`${scene.value}/frames/frame${pad(raw)}.jpg`;for(const d of [-1,1]){const i=Number(slider.value)+d;if(i>=0&&i<400){new Image().src=`${scene.value}/frames/frame${pad(i*5)}.jpg`}}}
function findRawFrame(){const raw=Number(rawInput.value);if(!Number.isInteger(raw)||raw<0||raw>1995||raw%5!==0){error.textContent='请输入 0–1995 且能被 5 整除的帧号';return}slider.value=raw/5;show()}
scene.onchange=show;slider.oninput=show;jump.onclick=findRawFrame;rawInput.addEventListener('keydown',e=>{if(e.key==='Enter')findRawFrame()});addEventListener('keydown',e=>{if(e.target===rawInput)return;let d=0;if(e.key==='ArrowRight')d=1;if(e.key==='ArrowLeft')d=-1;if(e.key==='PageDown')d=20;if(e.key==='PageUp')d=-20;if(d){e.preventDefault();slider.value=Math.max(0,Math.min(399,Number(slider.value)+d));show()}});show();
</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def write_readme(output: Path) -> None:
    text = """# 每帧 GT instance 对照查看器

打开 `index.html`。页面只加载当前帧，可输入原始帧号查找，也可用方向键逐帧切换。

每张图从左到右：原始 RGB、完整 ReplicaSSG GT instance、ali-dev observation 按最终 3D instance owner 合并后的 2D 显示。

## 怎么判断 split / merge

- 一个 ali-dev `Ixx` 在多个清晰视角反复覆盖两个不同前景 GT ID：疑似 false merge。
- 两个 ali-dev `Ixx` 在多个视角都稳定对应同一个前景 GT ID：疑似 false split。
- 只在一个边界帧出现、GT ID 属于墙/地面/天花板、或 mask 本身混合：先标记不确定，不直接修复。
- GT 是离线评测参考，不是在线 mapper 或 trigger 的输入。

`instance_to_gt_summary.csv` 给出每个 Ixx 的纯 observation→GT 分布；`potential_false_split_pairs.csv` 给出共享主 GT 的候选对。候选仍需结合多帧 RGB/GT 检查。
"""
    (output / "README_CN.md").write_text(text, encoding="utf-8")


def make_zip(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), default=tuple(SCENES))
    args = parser.parse_args()
    if args.output.exists():
        if args.overwrite and args.append:
            raise ValueError("--overwrite and --append are mutually exclusive")
        if args.overwrite:
            shutil.rmtree(args.output)
        elif not args.append:
            raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.append)
    prior_scenes = {}
    prior_manifest_path = args.output / "manifest.json"
    if args.append and prior_manifest_path.is_file():
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_scenes = {row["scene"]: row for row in prior_manifest.get("scenes", [])}
    for scene in args.scenes:
        if (args.output / scene).exists():
            raise FileExistsError(args.output / scene)
        prior_scenes[scene] = export_scene(scene, args.output)
    scenes = [prior_scenes[name] for name in SCENES if name in prior_scenes]
    write_index(args.output)
    write_readme(args.output)
    manifest = {
        "schema_version": "gt-reference-viewer/1.0",
        "protocol": {"start": 0, "end": 2000, "stride": 5, "frames": 400},
        "panels": ["original_rgb", "full_replicassg_instance_gt", "online_observations_by_final_owner"],
        "evaluation_only": True,
        "scenes": scenes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.zip_path:
        make_zip(args.output, args.zip_path)
    print(json.dumps({**manifest, "zip": str(args.zip_path) if args.zip_path else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
