#!/usr/bin/env python3
"""Run one deterministic predicted-object navigation episode in Habitat-Sim."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

from pose_utils import habitat_sensor_pose_to_opencv_c2w


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_tree(root: Path) -> None:
    files = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", "COMPLETE"}
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def camera_spec(uuid: str, sensor_type, args: argparse.Namespace):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [args.height, args.width]
    spec.position = [0.0, args.sensor_height, 0.0]
    spec.hfov = args.hfov
    return spec


def build_simulator(args: argparse.Namespace):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(Path(args.scene).resolve())
    backend.enable_physics = False
    backend.gpu_device_id = args.gpu_device_id
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [
        camera_spec("color_sensor", habitat_sim.SensorType.COLOR, args),
        camera_spec("depth_sensor", habitat_sim.SensorType.DEPTH, args),
    ]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_config]))
    sim.seed(args.seed)
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            sim.close()
            raise RuntimeError("Habitat could not generate a NavMesh")
    sim.pathfinder.seed(args.seed)
    return sim


def transform_points(transform: np.ndarray, points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def look_at_yaw(source, target) -> float:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    return math.atan2(-float(direction[0]), -float(direction[2]))


def set_agent_pose(agent, position, yaw: float) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_angle_axis(
        yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    agent.set_state(state, reset_sensors=True)


def finite_point(point) -> bool:
    return bool(np.isfinite(np.asarray(point, dtype=np.float64)).all())


def shortest_path(pathfinder, start, end):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    found = pathfinder.find_path(path)
    return found, path


def project_target_points(
    agent,
    observations,
    world_points: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    sensor_state = agent.get_state().sensor_states["color_sensor"]
    c2w = habitat_sensor_pose_to_opencv_c2w(
        sensor_state.position, sensor_state.rotation
    )
    w2c = np.linalg.inv(c2w)
    camera_points = transform_points(w2c, world_points)
    z = camera_points[:, 2]
    u = args.fx * camera_points[:, 0] / np.maximum(z, 1e-12) + args.cx
    v = args.fy * camera_points[:, 1] / np.maximum(z, 1e-12) + args.cy
    inside = (
        (z > 0.05)
        & (u >= 0)
        & (u < args.width)
        & (v >= 0)
        & (v < args.height)
    )
    depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
    matched = np.zeros(len(world_points), dtype=bool)
    projected = []
    for index in np.flatnonzero(inside):
        px = int(np.clip(round(float(u[index])), 0, args.width - 1))
        py = int(np.clip(round(float(v[index])), 0, args.height - 1))
        observed = float(depth[py, px])
        tolerance = max(args.visibility_tolerance_m, 0.04 * float(z[index]))
        is_match = observed > 0 and abs(observed - float(z[index])) <= tolerance
        matched[index] = is_match
        projected.append(
            {
                "point_index": int(index),
                "pixel": [px, py],
                "point_depth_m": float(z[index]),
                "sensor_depth_m": observed,
                "matches_depth": bool(is_match),
            }
        )
    return {
        "num_samples": int(len(world_points)),
        "num_in_view": int(inside.sum()),
        "num_depth_matches": int(matched.sum()),
        "in_view_ratio": float(inside.mean()) if len(inside) else 0.0,
        "visibility_ratio": float(matched.mean()) if len(matched) else 0.0,
        "projected": projected,
    }


def create_candidate_goals(
    sim,
    agent,
    start: np.ndarray,
    target_center: np.ndarray,
    target_points: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    candidates = []
    seen = set()
    for radius in args.candidate_radii:
        for angle_index in range(args.candidate_angles):
            angle = 2.0 * math.pi * angle_index / args.candidate_angles
            raw = np.asarray(
                [
                    target_center[0] + radius * math.cos(angle),
                    start[1],
                    target_center[2] + radius * math.sin(angle),
                ],
                dtype=np.float32,
            )
            snapped = np.asarray(sim.pathfinder.snap_point(raw), dtype=np.float32)
            if not finite_point(snapped):
                continue
            key = tuple(np.round(snapped, 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            horizontal_snap = float(np.linalg.norm((snapped - raw)[[0, 2]]))
            if horizontal_snap > args.max_snap_distance_m:
                continue
            found, path = shortest_path(sim.pathfinder, start, snapped)
            if not found or not math.isfinite(float(path.geodesic_distance)):
                continue
            if float(path.geodesic_distance) < args.min_episode_distance_m:
                continue
            yaw = look_at_yaw(snapped, target_center)
            set_agent_pose(agent, snapped, yaw)
            observations = sim.get_sensor_observations()
            visibility = project_target_points(agent, observations, target_points, args)
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "radius_requested_m": float(radius),
                    "angle_rad": float(angle),
                    "raw_position_habitat": raw.astype(float).tolist(),
                    "position_habitat": snapped.astype(float).tolist(),
                    "yaw_rad": float(yaw),
                    "horizontal_snap_distance_m": horizontal_snap,
                    "geodesic_distance_m": float(path.geodesic_distance),
                    "path_points": [np.asarray(point, dtype=float).tolist() for point in path.points],
                    "visibility": {
                        key: value for key, value in visibility.items() if key != "projected"
                    },
                }
            )
    return candidates


def select_goal(candidates: list[dict], args: argparse.Namespace) -> dict:
    visible = [
        item for item in candidates
        if item["visibility"]["num_depth_matches"] >= args.min_visible_points
    ]
    if not visible:
        best = sorted(
            candidates,
            key=lambda item: (
                -item["visibility"]["num_depth_matches"],
                -item["visibility"]["visibility_ratio"],
                item["geodesic_distance_m"],
                item["candidate_index"],
            ),
        )[:5]
        raise RuntimeError(
            "no candidate observation pose passed visibility; best="
            + json.dumps(best, ensure_ascii=False)
        )
    visible.sort(
        key=lambda item: (
            -item["visibility"]["num_depth_matches"],
            -item["visibility"]["visibility_ratio"],
            item["geodesic_distance_m"],
            item["candidate_index"],
        )
    )
    return visible[0]


def signed_yaw_error(agent, target) -> float:
    state = agent.get_state()
    forward = quat_rotate_vector(
        state.rotation, np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    )
    desired = np.asarray(target, dtype=np.float64) - np.asarray(
        state.position, dtype=np.float64
    )
    forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float64)
    desired_xz = np.asarray([desired[0], desired[2]], dtype=np.float64)
    forward_xz /= max(float(np.linalg.norm(forward_xz)), 1e-12)
    desired_xz /= max(float(np.linalg.norm(desired_xz)), 1e-12)
    cross = forward_xz[1] * desired_xz[0] - forward_xz[0] * desired_xz[1]
    dot = float(np.clip(forward_xz @ desired_xz, -1.0, 1.0))
    return math.atan2(float(cross), dot)


def compose_panel(observations, step: int, action: str, query: str, target_class: str,
                  visibility=None) -> np.ndarray:
    rgb = np.asarray(observations["color_sensor"])[..., :3]
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if visibility is not None:
        for item in visibility.get("projected", []):
            px, py = item["pixel"]
            color = (0, 255, 0) if item["matches_depth"] else (0, 0, 255)
            cv2.circle(rgb_bgr, (px, py), 2, color, -1)
    depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    high = float(np.percentile(valid, 98)) if valid.size else 1.0
    depth_u8 = (255.0 * (1.0 - np.clip(depth / max(high, 1e-3), 0, 1))).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    panel = np.hstack([rgb_bgr, depth_bgr])
    cv2.putText(
        panel,
        f"query={query} selected={target_class} step={step} action={action}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def quaternion_list(value) -> list[float]:
    return quaternion.as_float_array(value).astype(float).tolist()


def record_state(agent, step: int, action: str, collided: bool, geodesic: float) -> dict:
    state = agent.get_state()
    return {
        "step": step,
        "action": action,
        "collided": bool(collided),
        "position_habitat": np.asarray(state.position, dtype=float).tolist(),
        "rotation_wxyz": quaternion_list(state.rotation),
        "geodesic_distance_to_goal_m": float(geodesic),
    }


def verify_expected_class(selected_class: str, expected: str) -> bool:
    expected_tokens = [token for token in expected.lower().replace("_", " ").split() if token]
    value = selected_class.lower()
    return bool(expected_tokens) and all(token in value for token in expected_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--query-result", required=True)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--expected-class", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--success-distance-m", type=float, default=0.35)
    parser.add_argument("--min-episode-distance-m", type=float, default=1.0)
    parser.add_argument("--max-snap-distance-m", type=float, default=0.75)
    parser.add_argument("--visibility-tolerance-m", type=float, default=0.25)
    parser.add_argument("--min-visible-points", type=int, default=3)
    parser.add_argument("--candidate-radii", type=float, nargs="+", default=[0.8, 1.1, 1.4, 1.7, 2.0])
    parser.add_argument("--candidate-angles", type=int, default=36)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.fx = args.width / (2.0 * math.tan(math.radians(args.hfov) / 2.0))
    args.fy = args.fx
    args.cx = (args.width - 1) / 2.0
    args.cy = (args.height - 1) / 2.0
    return args


def main() -> None:
    args = parse_args()
    scene = Path(args.scene).resolve()
    bundle = Path(args.bundle).resolve()
    query_path = Path(args.query_result).resolve()
    output = Path(args.output_dir).resolve()
    for path in (scene, bundle / "COMPLETE", bundle / "object_map.pkl.gz", query_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest = json.loads((bundle / "map_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "sequence_metadata.json").read_text(encoding="utf-8"))
    query = json.loads(query_path.read_text(encoding="utf-8"))
    local_map_sha = sha256_file(bundle / "object_map.pkl.gz")
    if query["object_map_sha256"] != local_map_sha:
        raise RuntimeError("server query used a different object map than the local bundle")
    selected_matches = [
        item for item in query["candidates"] if int(item["rank"]) == args.candidate_rank
    ]
    if len(selected_matches) != 1:
        raise RuntimeError(f"candidate rank {args.candidate_rank} is unavailable")
    selected = selected_matches[0]
    class_correct = verify_expected_class(selected["class_name"], args.expected_class)
    if not class_correct:
        raise RuntimeError(
            f"selected class {selected['class_name']!r} does not match expected "
            f"class {args.expected_class!r}"
        )

    transform = np.asarray(manifest["T_habitat_world_from_cg_map"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError("invalid T_habitat_world_from_cg_map")
    target_center = transform_points(transform, [selected["center_cg_map_m"]])[0]
    target_points = transform_points(transform, selected["sample_points_cg_map_m"])
    if len(target_points) < args.min_visible_points:
        raise RuntimeError("selected object has too few sampled points")

    print("\n[LOCAL 1/5] Coordinate transform")
    print(f"  CG map center: {selected['center_cg_map_m']}")
    print(f"  Habitat world center: {target_center.tolist()}")
    print("  Effect: the server map target is now expressed in Habitat world coordinates.")

    sim = build_simulator(args)
    agent = sim.get_agent(0)
    video = None
    try:
        raw_start = np.asarray(transform[:3, 3], dtype=np.float32)
        raw_start[1] -= float(metadata["camera"]["sensor_height_m"])
        start = np.asarray(sim.pathfinder.snap_point(raw_start), dtype=np.float32)
        if not finite_point(start):
            raise RuntimeError(f"fixed start cannot be snapped to NavMesh: {raw_start}")
        start_snap_distance = float(np.linalg.norm(start - raw_start))
        set_agent_pose(agent, start, 0.0)

        print("\n[LOCAL 2/5] NavMesh candidate observation poses")
        print(f"  Fixed start raw/snapped: {raw_start.tolist()} -> {start.tolist()}")
        print(f"  Start snap distance: {start_snap_distance:.6f} m")
        candidates = create_candidate_goals(
            sim, agent, start, target_center, target_points, args
        )
        if not candidates:
            raise RuntimeError("no reachable NavMesh candidate goals were generated")
        goal = select_goal(candidates, args)
        goal_position = np.asarray(goal["position_habitat"], dtype=np.float32)
        print(f"  Reachable candidates: {len(candidates)}")
        print(f"  Selected goal: {goal_position.tolist()}")
        print(f"  Shortest distance: {goal['geodesic_distance_m']:.6f} m")
        print(f"  Visible target samples: {goal['visibility']['num_depth_matches']}/{len(target_points)}")
        print("  Effect: the goal is on NavMesh, reachable, and can geometrically see the mapped object.")

        json_dump(output / "goal_candidates.json", {"candidates": candidates})
        json_dump(
            output / "goal_pose.json",
            {
                "format_version": "cgs-objectnav-goal-v1",
                "selected_object": selected,
                "T_habitat_world_from_cg_map": transform.tolist(),
                "target_center_habitat": target_center.tolist(),
                "start_position_raw_habitat": raw_start.astype(float).tolist(),
                "start_position_habitat": start.astype(float).tolist(),
                "start_yaw_rad": 0.0,
                "goal": goal,
            },
        )
        json_dump(
            output / "shortest_path.json",
            {
                "geodesic_distance_m": goal["geodesic_distance_m"],
                "points_habitat": goal["path_points"],
            },
        )

        set_agent_pose(agent, start, 0.0)
        observations = sim.get_sensor_observations()
        video_path = output / "navigation.mp4"
        video_fps = 12.0
        visualization_delay_ms = max(1, int(round(1000.0 / video_fps)))
        video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps,
            (args.width * 2, args.height),
        )
        if not video.isOpened():
            raise RuntimeError("OpenCV could not create navigation.mp4")

        print("\n[LOCAL 3/5] Automatic shortest-path navigation")
        follower = sim.make_greedy_follower(
            0,
            args.success_distance_m,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        trajectory = []
        collisions = 0
        actual_path_length = 0.0
        stopped = False
        user_aborted = False
        previous_position = np.asarray(agent.get_state().position, dtype=np.float64)
        for step in range(args.max_steps):
            found, remaining_path = shortest_path(
                sim.pathfinder, agent.get_state().position, goal_position
            )
            remaining = float(remaining_path.geodesic_distance) if found else float("inf")
            action = follower.next_action_along(goal_position)
            if action == "stop":
                stopped = True
                trajectory.append(record_state(agent, step, "stop", False, remaining))
                panel = compose_panel(
                    observations, step, "stop", query["query"], selected["class_name"]
                )
                video.write(panel)
                break
            observations = sim.step(action)
            collided = bool(observations.get("collided", False))
            collisions += int(collided)
            current_position = np.asarray(agent.get_state().position, dtype=np.float64)
            actual_path_length += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position
            trajectory.append(record_state(agent, step, str(action), collided, remaining))
            panel = compose_panel(
                observations, step, str(action), query["query"], selected["class_name"]
            )
            video.write(panel)
            if args.visualize:
                cv2.imshow("CGS ObjectNav: RGB | Depth", panel)
                if cv2.waitKey(visualization_delay_ms) & 0xFF in (ord("q"), 27):
                    user_aborted = True
                    break
        if not stopped and not user_aborted:
            raise RuntimeError(f"navigation exceeded max_steps={args.max_steps}")

        print(f"  Actions executed: {len(trajectory)}")
        print(f"  Actual translation length: {actual_path_length:.6f} m")
        print(f"  Collisions: {collisions}")

        print("\n[LOCAL 4/5] Face the target and verify visibility")
        turn_steps = 0
        while turn_steps < 40:
            error = signed_yaw_error(agent, target_center)
            if abs(math.degrees(error)) <= 5.0:
                break
            action = "turn_left" if error > 0 else "turn_right"
            observations = sim.step(action)
            turn_steps += 1
            trajectory.append(
                record_state(agent, len(trajectory), action, False, 0.0)
            )
            panel = compose_panel(
                observations, len(trajectory), action, query["query"], selected["class_name"]
            )
            video.write(panel)
            if args.visualize:
                cv2.imshow("CGS ObjectNav: RGB | Depth", panel)
                cv2.waitKey(visualization_delay_ms)
        observations = sim.get_sensor_observations()
        final_visibility = project_target_points(agent, observations, target_points, args)
        final_panel = compose_panel(
            observations,
            len(trajectory),
            "final_visibility",
            query["query"],
            selected["class_name"],
            final_visibility,
        )
        for _ in range(12):
            video.write(final_panel)
        if args.visualize:
            cv2.imshow("CGS ObjectNav: RGB | Depth", final_panel)
            cv2.waitKey(1000)

        found, final_path = shortest_path(
            sim.pathfinder, agent.get_state().position, goal_position
        )
        final_distance = float(final_path.geodesic_distance) if found else float("inf")
        target_visible = final_visibility["num_depth_matches"] >= args.min_visible_points
        success = bool(
            stopped
            and not user_aborted
            and class_correct
            and final_distance <= args.success_distance_m + 1e-6
            and target_visible
        )
        shortest_distance = float(goal["geodesic_distance_m"])
        spl = (
            shortest_distance / max(shortest_distance, actual_path_length)
            if success and shortest_distance > 0
            else 0.0
        )

        rgb = np.asarray(observations["color_sensor"])[..., :3]
        depth_m = np.asarray(observations["depth_sensor"], dtype=np.float32)
        depth_mm = np.clip(
            np.rint(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000),
            0,
            65535,
        ).astype(np.uint16)
        cv2.imwrite(str(output / "final_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / "final_depth.png"), depth_mm)
        cv2.imwrite(str(output / "final_visibility.jpg"), final_panel)
        video.release()
        video = None
        (output / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(item, sort_keys=True) for item in trajectory) + "\n",
            encoding="utf-8",
        )

        episode_spec = {
            "format_version": "cgs-objectnav-episode-v1",
            "episode_id": args.episode_id,
            "scene": str(scene),
            "scene_id": metadata["scene_id"],
            "sequence_id": manifest["sequence_id"],
            "query": query["query"],
            "expected_class": args.expected_class,
            "selected_rank": args.candidate_rank,
            "seed": args.seed,
            "fixed_start_position_habitat": start.astype(float).tolist(),
            "navigator": "habitat_sim_greedy_geodesic_follower",
            "evaluation_protocol": "predicted_object_visibility_v1",
            "semantic_gt_available": bool(metadata.get("semantic_available", False)),
        }
        map_ref = {
            "bundle": str(bundle),
            "object_map_sha256": local_map_sha,
            "map_manifest_sha256": sha256_file(bundle / "map_manifest.json"),
            "query_result_sha256": sha256_file(query_path),
        }
        metrics = {
            "format_version": "cgs-objectnav-metrics-v1",
            "evaluation_protocol": "predicted_object_visibility_v1",
            "official_habitat_objectnav_gt": False,
            "success": success,
            "spl": float(spl),
            "query_selected_expected_class": class_correct,
            "selected_class": selected["class_name"],
            "selected_object_id": selected["object_id"],
            "clip_similarity": selected["clip_similarity"],
            "goal_on_navmesh": True,
            "path_found": True,
            "stopped": stopped,
            "user_aborted": user_aborted,
            "target_visible_at_stop": target_visible,
            "target_visible_points": final_visibility["num_depth_matches"],
            "target_sample_points": len(target_points),
            "shortest_path_length_m": shortest_distance,
            "actual_path_length_m": actual_path_length,
            "final_geodesic_distance_to_goal_m": final_distance,
            "success_distance_m": args.success_distance_m,
            "steps": len(trajectory),
            "collisions": collisions,
            "final_turn_steps": turn_steps,
        }
        signature_payload = {
            "scene_id": episode_spec["scene_id"],
            "sequence_id": episode_spec["sequence_id"],
            "seed": args.seed,
            "object_map_sha256": local_map_sha,
            "query": query["query"],
            "selected_class": selected["class_name"],
            "selected_object_id": selected["object_id"],
            "start_position_habitat": np.round(start, 6).tolist(),
            "goal_position_habitat": np.round(goal_position, 6).tolist(),
            "shortest_path_length_m": round(shortest_distance, 6),
            "actual_path_length_m": round(actual_path_length, 6),
            "actions": [item["action"] for item in trajectory],
            "success": success,
            "spl": round(float(spl), 6),
        }
        signature_text = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()

        json_dump(output / "episode_spec.json", episode_spec)
        json_dump(output / "environment.json", {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "habitat_sim": habitat_sim.__version__,
            "opencv": cv2.__version__,
            "display": os.environ.get("DISPLAY"),
        })
        json_dump(output / "map_bundle_ref.json", map_ref)
        shutil.copy2(query_path, output / "query_result.json")
        json_dump(output / "metrics.json", metrics)
        json_dump(output / "reproducibility_signature.json", {
            "sha256": signature,
            "payload": signature_payload,
        })

        print("\n[LOCAL 5/5] Episode evaluation")
        print(f"  Selected class correct: {class_correct}")
        print(f"  Goal on NavMesh / path found: True / True")
        print(f"  Final target visible: {target_visible} ({final_visibility['num_depth_matches']} points)")
        print(f"  Success: {success}")
        print(f"  SPL: {spl:.6f}")
        print(f"  Reproducibility signature: {signature}")
        if success:
            checksum_tree(output)
            (output / "COMPLETE").touch()
            print(f"OBJECTNAV_COMPLETE={output}")
        else:
            print(f"OBJECTNAV_FAILED={output}")
            raise SystemExit(1)
    finally:
        if video is not None:
            video.release()
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
