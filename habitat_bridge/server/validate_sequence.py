#!/usr/bin/env python3
"""Server entry point for validating an uploaded CGS Habitat sequence."""
from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve()
LOCAL_BRIDGE = HERE.parents[1] / "local"
SCHEMA = HERE.parents[1] / "interface_schema.json"
sys.path.insert(0, str(LOCAL_BRIDGE))

import validate_export  # noqa: E402


def main() -> None:
    if "--sequence-dir" not in sys.argv:
        raise SystemExit("--sequence-dir is required")
    sequence_dir = Path(sys.argv[sys.argv.index("--sequence-dir") + 1]).resolve()
    if not (sequence_dir / "READY").is_file():
        raise SystemExit(f"READY is missing: {sequence_dir}")
    if "--schema" not in sys.argv:
        sys.argv.extend(["--schema", str(SCHEMA)])
    validate_export.main()
    (sequence_dir / "VALIDATED").touch()
    print(f"validated_marker={sequence_dir / 'VALIDATED'}")


if __name__ == "__main__":
    main()
