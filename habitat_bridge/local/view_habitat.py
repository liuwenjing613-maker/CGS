#!/usr/bin/env python3
"""Interactive first-person Habitat-Sim viewer for local scene debugging."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis


def build_sensor(uuid: str, sensor_type, args: argparse.Namespace):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [args.height, args.width]
    spec.position = [0.0, args.sensor_height, 0.0]
    spec.hfov = args.hfov
    return spec


def make_sim(args: argparse.Namespace):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(Path(args.scene).resolve())
    backend.enable_physics = False
    backend.gpu_device_id = args.gpu_device_id
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [
        build_sensor("color_sensor", habitat_sim.SensorType.COLOR, args),
        build_sensor("depth_sensor", habitat_sim.SensorType.DEPTH, args),
    ]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))
    sim.seed(args.seed)
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            sim.close()
            raise RuntimeError("Habitat could not generate a NavMesh for this scene")
    sim.pathfinder.seed(args.seed)
    return sim


def reset_agent(sim, agent, seed: int) -> None:
    sim.pathfinder.seed(seed)
    state = agent.get_state()
    point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
    if not np.isfinite(point).all():
        raise RuntimeError("PathFinder returned a non-finite point")
    state.position = point
    state.rotation = quat_from_angle_axis(
        0.0, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    agent.set_state(state, reset_sensors=True)


def compose_view(observations, position, mode: str, autoplay: str) -> np.ndarray:
    rgb = np.asarray(observations["color_sensor"])[..., :3]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    max_depth = float(np.percentile(valid, 98)) if valid.size else 1.0
    max_depth = max(max_depth, 1e-3)
    depth_norm = np.clip(depth / max_depth, 0.0, 1.0)
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    panel = np.hstack((rgb_bgr, depth_bgr))
    cv2.putText(
        panel,
        f"pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})  mode={mode} auto={autoplay}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "W/S move  A/D turn  R reset  SPACE autoplay  P save  Q/ESC quit",
        (12, panel.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def save_observation(observations, output_dir: Path, frame_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(observations["color_sensor"])[..., :3]
    depth_m = np.asarray(observations["depth_sensor"], dtype=np.float32)
    depth_mm = np.clip(
        np.rint(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0),
        0,
        65535,
    ).astype(np.uint16)
    cv2.imwrite(
        str(output_dir / f"frame{frame_index:06d}.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(output_dir / f"depth{frame_index:06d}.png"), depth_mm)
    print(f"saved frame {frame_index} -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--autoplay", choices=("none", "rotate", "walk"), default="none")
    parser.add_argument("--save-dir", default="results/habitat/viewer_captures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Path(args.scene).resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)
    sim = make_sim(args)
    agent = sim.get_agent(0)
    reset_agent(sim, agent, args.seed)
    output_dir = Path(args.save_dir).resolve()
    autoplay = args.autoplay
    frame_index = 0
    last_auto = time.monotonic()
    try:
        while True:
            if autoplay != "none" and time.monotonic() - last_auto >= 0.15:
                agent.act("turn_right" if autoplay == "rotate" else "move_forward")
                last_auto = time.monotonic()
            observations = sim.get_sensor_observations()
            position = agent.get_state().position
            panel = compose_view(observations, position, "RGB | Depth", autoplay)
            cv2.imshow("CGS Habitat Viewer", panel)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("w"):
                agent.act("move_forward")
            elif key == ord("s"):
                # Stock Habitat actions have no backward action; turn 180° twice.
                for _ in range(18):
                    agent.act("turn_right")
                agent.act("move_forward")
                for _ in range(18):
                    agent.act("turn_right")
            elif key == ord("a"):
                agent.act("turn_left")
            elif key == ord("d"):
                agent.act("turn_right")
            elif key == ord("r"):
                reset_agent(sim, agent, args.seed + frame_index + 1)
            elif key == ord(" "):
                autoplay = "none" if autoplay != "none" else "walk"
            elif key == ord("p"):
                save_observation(observations, output_dir, frame_index)
                frame_index += 1
    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
