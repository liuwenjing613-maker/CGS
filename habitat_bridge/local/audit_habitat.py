#!/usr/bin/env python3
"""Minimal Habitat/MP3D audit: load scene, sensors, navmesh, save observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--sensor-height", "--height", dest="sensor_height", type=float, default=1.25)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = args.scene
    backend.enable_physics = False
    backend.gpu_device_id = args.gpu_device_id

    def cam(uuid: str, stype):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [args.image_height, args.width]
        s.position = [0.0, args.sensor_height, 0.0]
        s.hfov = args.hfov
        return s

    sensors = [
        cam("color_sensor", habitat_sim.SensorType.COLOR),
        cam("depth_sensor", habitat_sim.SensorType.DEPTH),
    ]
    # Semantic only if scene supports it; still declare for API completeness
    sensors.append(cam("semantic_sensor", habitat_sim.SensorType.SEMANTIC))

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    sim.seed(args.seed)
    sim.pathfinder.seed(args.seed)

    print("pathfinder_loaded:", sim.pathfinder.is_loaded)
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        ok = sim.recompute_navmesh(sim.pathfinder, settings)
        print("recompute_navmesh:", ok, "loaded:", sim.pathfinder.is_loaded)
        if sim.pathfinder.is_loaded:
            nav_path = out / "generated.navmesh"
            sim.pathfinder.save_nav_mesh(str(nav_path))
            print("saved_navmesh:", nav_path)

    if not sim.pathfinder.is_loaded:
        raise RuntimeError("NavMesh not loaded")

    point = sim.pathfinder.get_random_navigable_point()
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = point
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]
    depth = obs["depth_sensor"]
    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", depth)

    semantic = obs.get("semantic_sensor")
    semantic_ids = np.unique(semantic) if semantic is not None else np.asarray([])
    semantic_objects = [obj for obj in sim.semantic_scene.objects if obj is not None]
    semantic_available = bool(semantic_objects)
    if semantic is not None:
        np.save(out / "semantic.npy", semantic.astype(np.int32))
    print("semantic_available:", semantic_available)
    print("semantic_unique:", int(semantic_ids.size))

    print("rgb_mean:", float(rgb.mean()))
    print("depth_finite:", int(np.isfinite(depth).sum()), "/", depth.size)
    print("agent_pose:", agent.get_state().position, agent.get_state().rotation)
    for s in agent.get_state().sensor_states.values():
        print("sensor_pose:", s.position, s.rotation)
    summary = {
        "scene": str(Path(args.scene).resolve()),
        "pathfinder_loaded": bool(sim.pathfinder.is_loaded),
        "rgb_mean": float(rgb.mean()),
        "depth_finite": int(np.isfinite(depth).sum()),
        "depth_total": int(depth.size),
        "semantic_available": semantic_available,
        "semantic_observation_nontrivial": bool(semantic_ids.size > 1),
        "semantic_unique": int(semantic_ids.size),
    }
    (out / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("AUDIT_OK")
    sim.close()


if __name__ == "__main__":
    main()
