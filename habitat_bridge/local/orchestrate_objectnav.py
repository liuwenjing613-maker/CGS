#!/usr/bin/env python3
"""One-command local/server orchestrator for a CGS predicted-object episode."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def stage(index: int, total: int, title: str, explanation: str) -> None:
    print(f"\n{'=' * 78}\nSTAGE {index}/{total}: {title}\n{'=' * 78}")
    print(explanation)


def run(command, *, cwd=None, env=None) -> None:
    print("$ " + " ".join(map(str, command)))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_.-")
    return slug[:40] or "query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run server OpenCLIP query and local Habitat ObjectNav in one command."
    )
    parser.add_argument("--query", default="sofa")
    parser.add_argument("--expected-class", default="sofa")
    parser.add_argument("--sequence-id", default="2azQ1b91cZZ_smoke_v001")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scene-id", default="2azQ1b91cZZ")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--select-rank", type=int)
    parser.add_argument("--yes", action="store_true", help="Select rank 1 without prompting.")
    parser.add_argument("--no-server-query", action="store_true", help="Reuse the cached local query JSON.")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compare-signature", help="Previous reproducibility_signature.json to compare.")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"required environment variable is missing: {name}")
    return value


def main() -> None:
    args = parse_args()
    if not args.query.strip() or not args.expected_class.strip():
        raise SystemExit("query and expected-class must not be empty")
    cg_local = Path(require_env("CG_LOCAL")).resolve()
    cg_repo = Path(require_env("CG_REPO")).resolve()
    server_alias = require_env("CG_SERVER_ALIAS")
    remote_root = require_env("CG_REMOTE")
    local_result_root = Path(require_env("LOCAL_RESULT_ROOT")).resolve()
    habitat_env = require_env("HABITAT_CONDA_ENV")
    run_id = args.run_id or f"{args.sequence_id}_smoke"
    scene = Path(args.scene).expanduser().resolve() if args.scene else (
        cg_local / "data/scenes/mp3d" / args.scene_id / f"{args.scene_id}.glb"
    )
    bundle = local_result_root / args.sequence_id / run_id / "map_bundle"
    object_map = bundle / "object_map.pkl.gz"
    episode_id = args.episode_id or f"{slugify(args.query)}_objectnav_v001"
    output = cg_local / "results/objectnav" / args.scene_id / episode_id
    query_key = hashlib.sha256(
        (args.query + "\0" + sha256_file(object_map)).encode("utf-8")
    ).hexdigest()[:16]
    query_id = f"q_{slugify(args.query)}_{query_key}"
    query_dir = cg_local / "results/objectnav/_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    query_path = query_dir / f"{query_id}.json"

    stage(
        1, 6, "Validate the fixed inputs and map bundle",
        "Input: fixed scene, seed, sequence and map bundle.\n"
        "Output: a verified map payload. No query or navigation is allowed if a checksum fails.",
    )
    if not scene.is_file():
        raise FileNotFoundError(scene)
    for path in (bundle / "COMPLETE", bundle / "checksums.sha256", object_map):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(["sha256sum", "-c", "checksums.sha256"], cwd=bundle)
    print(f"Scene: {scene}")
    print(f"Seed: {args.seed}")
    print(f"Bundle: {bundle}")
    print(f"Object-map SHA-256: {sha256_file(object_map)}")

    stage(
        2, 6, "Run the server OpenCLIP text query",
        "Input: the user's text plus the exact server object map.\n"
        "Output: Top-K ConceptGraphs objects, similarities, 3D centers, bboxes and surface points.",
    )
    if args.no_server_query:
        if not query_path.is_file():
            raise FileNotFoundError(
                f"--no-server-query requested but cache is missing: {query_path}"
            )
        print(f"Reusing cached query result: {query_path}")
    else:
        query_base64 = base64.b64encode(args.query.encode("utf-8")).decode("ascii")
        run([
            "ssh", server_alias,
            f"{remote_root}/scripts/server_query_habitat_map.sh",
            args.sequence_id,
            run_id,
            query_base64,
            query_id,
        ])
        remote_query = (
            f"{server_alias}:{remote_root}/results/HabitatMP3D/"
            f"{args.sequence_id}/{run_id}/queries/{query_id}.json"
        )
        run(["rsync", "-av", remote_query, str(query_path)])
    query = json.loads(query_path.read_text(encoding="utf-8"))
    if query["object_map_sha256"] != sha256_file(object_map):
        raise RuntimeError("server query map SHA differs from the local navigation map")
    print("\nTop-K candidates returned by the server:")
    print(" rank | class                    | CLIP similarity | detections | points")
    print("------+--------------------------+-----------------+------------+-------")
    for item in query["candidates"]:
        print(
            f" {item['rank']:>4} | {item['class_name'][:24]:<24} | "
            f"{item['clip_similarity']:>15.6f} | {item['num_detections']:>10} | "
            f"{item['num_points']:>6}"
        )

    stage(
        3, 6, "Let the operator confirm the selected object",
        "You can participate here: inspect Top-K and choose the candidate rank.\n"
        "For deterministic automated reproduction, pass --yes or --select-rank.",
    )
    if args.select_rank is not None:
        selected_rank = args.select_rank
    elif args.yes:
        selected_rank = 1
    else:
        answer = input("Select candidate rank [1]: ").strip()
        selected_rank = int(answer) if answer else 1
    selected_list = [
        item for item in query["candidates"] if int(item["rank"]) == selected_rank
    ]
    if len(selected_list) != 1:
        raise SystemExit(f"rank {selected_rank} is not present in the Top-K result")
    selected = selected_list[0]
    print(
        f"Selected rank {selected_rank}: {selected['class_name']} "
        f"(CLIP={selected['clip_similarity']:.6f})"
    )
    print(
        "The local runner will fail closed if this class does not contain "
        f"the expected class text {args.expected_class!r}."
    )

    stage(
        4, 6, "Transform coordinates and generate a reachable observation goal",
        "Input: the selected object's ConceptGraphs coordinates and map-to-Habitat transform.\n"
        "Output: fixed start, NavMesh candidates, a reachable shortest path and a visible goal pose.",
    )
    runner = cg_repo / "habitat_bridge/local/run_objectnav.py"
    command = [
        "conda", "run", "--no-capture-output", "-n", habitat_env,
        "python", str(runner),
        "--scene", str(scene),
        "--bundle", str(bundle),
        "--query-result", str(query_path),
        "--candidate-rank", str(selected_rank),
        "--expected-class", args.expected_class,
        "--episode-id", episode_id,
        "--output-dir", str(output),
        "--seed", str(args.seed),
    ]
    if not args.no_visualize:
        if not os.environ.get("DISPLAY"):
            raise SystemExit("DISPLAY is missing; use a Ubuntu desktop terminal or --no-visualize")
        command.append("--visualize")
    if args.overwrite:
        command.append("--overwrite")
    child_env = os.environ.copy()
    child_env.pop("LD_LIBRARY_PATH", None)
    child_env.pop("PYTHONPATH", None)
    run(command, cwd=cg_local, env=child_env)

    stage(
        5, 6, "Inspect navigation outputs and metrics",
        "The agent has executed Habitat actions. This stage verifies video, trajectory, visibility, "
        "Success, SPL and all required evidence files.",
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    signature = json.loads(
        (output / "reproducibility_signature.json").read_text(encoding="utf-8")
    )
    required = [
        "COMPLETE", "episode_spec.json", "map_bundle_ref.json", "query_result.json",
        "goal_candidates.json", "goal_pose.json", "shortest_path.json",
        "trajectory.jsonl", "metrics.json", "navigation.mp4", "final_rgb.jpg",
        "final_depth.png", "final_visibility.jpg", "checksums.sha256",
        "reproducibility_signature.json",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"episode evidence files missing: {missing}")
    run(["sha256sum", "-c", "checksums.sha256"], cwd=output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    stage(
        6, 6, "Final acceptance checklist",
        "Only this checklist is the completion criterion. Intermediate windows and pkl loading are not enough.",
    )
    finite_spl = math.isfinite(float(metrics["spl"]))
    checklist = [
        ("One command launched the workflow", True),
        ("Fixed scene, seed and start", bool(signature["payload"]["start_position_habitat"])),
        ("Map bundle checksum passed", True),
        ("Text query selected the expected class", metrics["query_selected_expected_class"]),
        ("Object coordinates transformed to Habitat world", (output / "goal_pose.json").is_file()),
        ("Observation goal is on NavMesh", metrics["goal_on_navmesh"]),
        ("A path exists from start to goal", metrics["path_found"]),
        ("The agent automatically executed navigation", metrics["steps"] > 0),
        ("The target is visible at stop", metrics["target_visible_at_stop"]),
        ("success = 1", metrics["success"] is True),
        ("SPL is finite and calculable", finite_spl),
        ("Video, trajectory, config and metrics are saved", not missing),
    ]
    all_pass = True
    for label, passed in checklist:
        all_pass &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    if args.compare_signature:
        previous = json.loads(Path(args.compare_signature).read_text(encoding="utf-8"))
        same = previous["sha256"] == signature["sha256"]
        all_pass &= same
        print(
            f"[{'PASS' if same else 'FAIL'}] Same input produced the same signature "
            f"({previous['sha256']} vs {signature['sha256']})"
        )
    else:
        print(
            "[PENDING] Reproducibility comparison needs a second run. "
            "Pass --compare-signature on that run."
        )
    print(f"\nEpisode result: {output}")
    print(f"Reproducibility signature: {signature['sha256']}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

