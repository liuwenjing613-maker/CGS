#!/usr/bin/env python3
"""Generate the Replica-compatible ConceptGraphs config for one sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", required=True)
    args = parser.parse_args()
    sequence_dir = Path(args.sequence_dir).resolve()
    metadata = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    config = {
        "dataset_name": "replica",
        "camera_params": {
            "image_height": metadata["rgb"]["height"],
            "image_width": metadata["rgb"]["width"],
            "fx": metadata["camera"]["fx"],
            "fy": metadata["camera"]["fy"],
            "cx": metadata["camera"]["cx"],
            "cy": metadata["camera"]["cy"],
            "png_depth_scale": metadata["depth"]["png_depth_scale"],
            "crop_edge": 0,
        },
    }
    output = sequence_dir / "conceptgraphs_dataset.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"generated={output}")
    print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
