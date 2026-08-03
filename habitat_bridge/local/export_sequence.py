#!/usr/bin/env python3
"""Export a Replica-compatible RGB-D+pose sequence from Habitat-Sim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

from pose_utils import habitat_sensor_pose_to_opencv_c2w


SEQUENCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def camera_spec(uuid: str, sensor_type, args: argparse.Namespace):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [args.height, args.width]
    spec.position = [0.0, args.sensor_height, 0.0]
    spec.hfov = args.hfov
    return spec


def write_checksums(sequence_dir: Path) -> None:
    relative_files = []
    for directory in ("results", "semantic"):
        relative_files.extend(
            path.relative_to(sequence_dir)
            for path in (sequence_dir / directory).glob("*")
            if path.is_file()
        )
    relative_files.extend(
        Path(name)
        for name in ("traj.txt", "intrinsics.json", "metadata.json", "frames.jsonl")
    )
    lines = []
    for relative in sorted(relative_files, key=lambda path: path.as_posix()):
        digest = hashlib.sha256((sequence_dir / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}")
    (sequence_dir / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("panorama-smoke",), default="panorama-smoke")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SEQUENCE_ID_RE.fullmatch(args.sequence_id):
        raise ValueError("sequence-id must match ^[A-Za-z0-9_.-]+$")
    if args.num_frames < 1:
        raise ValueError("num-frames must be positive")
    scene = Path(args.scene).resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)

    sequence_dir = Path(args.output_root).resolve() / args.sequence_id
    if sequence_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {sequence_dir}")
        shutil.rmtree(sequence_dir)
    results_dir = sequence_dir / "results"
    semantic_dir = sequence_dir / "semantic"
    results_dir.mkdir(parents=True)
    semantic_dir.mkdir()

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.enable_physics = False
    backend.gpu_device_id = args.gpu_device_id
    sensors = [
        camera_spec("color_sensor", habitat_sim.SensorType.COLOR, args),
        camera_spec("depth_sensor", habitat_sim.SensorType.DEPTH, args),
        camera_spec("semantic_sensor", habitat_sim.SensorType.SEMANTIC, args),
    ]
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))

    poses: list[np.ndarray] = []
    frame_records = []
    try:
        sim.seed(args.seed)
        if not sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings):
                raise RuntimeError("failed to generate NavMesh")
        sim.pathfinder.seed(args.seed)
        position = np.asarray(
            sim.pathfinder.get_random_navigable_point(), dtype=np.float32
        )
        if not np.isfinite(position).all():
            raise RuntimeError("PathFinder returned an invalid navigable point")

        semantic_objects = [obj for obj in sim.semantic_scene.objects if obj is not None]
        semantic_available = bool(semantic_objects)
        agent = sim.get_agent(0)
        for index in range(args.num_frames):
            yaw = index * (2.0 * math.pi / args.num_frames)
            state = habitat_sim.AgentState()
            state.position = position
            state.rotation = quat_from_angle_axis(
                yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            )
            agent.set_state(state, reset_sensors=True)
            observations = sim.get_sensor_observations()

            rgb = np.asarray(observations["color_sensor"])[..., :3]
            depth_m = np.asarray(observations["depth_sensor"], dtype=np.float32)
            semantic = np.asarray(
                observations.get(
                    "semantic_sensor", np.zeros(depth_m.shape, dtype=np.int32)
                ),
                dtype=np.int32,
            )
            depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(
                np.rint(depth_clean * 1000.0), 0, 65535
            ).astype(np.uint16)

            rgb_name = f"results/frame{index:06d}.jpg"
            depth_name = f"results/depth{index:06d}.png"
            semantic_name = f"semantic/semantic{index:06d}.npy"
            if not cv2.imwrite(
                str(sequence_dir / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write {rgb_name}")
            if not cv2.imwrite(str(sequence_dir / depth_name), depth_mm):
                raise RuntimeError(f"failed to write {depth_name}")
            np.save(sequence_dir / semantic_name, semantic)

            sensor_state = agent.get_state().sensor_states["color_sensor"]
            pose = habitat_sensor_pose_to_opencv_c2w(
                sensor_state.position, sensor_state.rotation
            )
            poses.append(pose)
            frame_records.append(
                {
                    "frame_index": index,
                    "rgb": rgb_name,
                    "depth": depth_name,
                    "semantic": semantic_name,
                    "yaw_rad": yaw,
                    "pose_row_major": pose.reshape(-1).tolist(),
                }
            )
    finally:
        sim.close()

    fx = args.width / (2.0 * math.tan(math.radians(args.hfov) / 2.0))
    intrinsics = {
        "width": args.width,
        "height": args.height,
        "hfov_deg": args.hfov,
        "fx": fx,
        "fy": fx,
        "cx": (args.width - 1) / 2.0,
        "cy": (args.height - 1) / 2.0,
    }
    metadata = {
        "format_version": "cgs-habitat-sequence-v1",
        "sequence_id": args.sequence_id,
        "scene_id": args.scene_id,
        "num_frames": args.num_frames,
        "rgb": {
            "width": args.width,
            "height": args.height,
            "pattern": "results/frame%06d.jpg",
        },
        "depth": {
            "pattern": "results/depth%06d.png",
            "storage_dtype": "uint16",
            "storage_unit": "millimeter",
            "png_depth_scale": 1000.0,
        },
        "camera": {
            "hfov_deg": args.hfov,
            "fx": fx,
            "fy": fx,
            "cx": (args.width - 1) / 2.0,
            "cy": (args.height - 1) / 2.0,
            "sensor_height_m": args.sensor_height,
        },
        "poses": {
            "file": "traj.txt",
            "type": "camera_to_world",
            "camera_frame": "opencv",
            "world_frame": "habitat",
        },
        "conceptgraphs": {
            "relative_pose": True,
            "map_frame": "first_opencv_camera",
        },
        "T_habitat_world_from_cg_map": poses[0].tolist(),
        "semantic_available": semantic_available,
        "export": {"mode": args.mode, "seed": args.seed},
    }
    (sequence_dir / "traj.txt").write_text(
        "\n".join(
            " ".join(f"{value:.10f}" for value in pose.reshape(-1))
            for pose in poses
        )
        + "\n",
        encoding="utf-8",
    )
    for name, value in (("intrinsics.json", intrinsics), ("metadata.json", metadata)):
        (sequence_dir / name).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    (sequence_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(record) for record in frame_records) + "\n",
        encoding="utf-8",
    )
    write_checksums(sequence_dir)
    print(f"sequence_dir={sequence_dir}")
    print(f"num_frames={len(poses)}")
    print(f"semantic_available={semantic_available}")
    print("EXPORT_COMPLETE (run validate_export.py before creating READY)")


if __name__ == "__main__":
    main()
