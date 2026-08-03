#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

SEQ="${1:-${MP3D_SCENE_ID}_smoke_v001}"
BRIDGE="$CG_REPO/habitat_bridge/local"
EXPORT_OPTIONS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXPORT_OPTIONS+=(--overwrite)
fi

test -f "$MP3D_SCENE"
test -f "$HABITAT_INTERFACE_SCHEMA"

env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n "$HABITAT_CONDA_ENV" \
  python "$BRIDGE/export_sequence.py" \
  --scene "$MP3D_SCENE" \
  --scene-id "$MP3D_SCENE_ID" \
  --sequence-id "$SEQ" \
  --output-root "$HABITAT_EXPORT_ROOT" \
  --mode panorama-smoke \
  --num-frames 20 \
  --width 640 \
  --height 480 \
  --hfov 90 \
  --sensor-height 1.25 \
  --seed 2027 \
  "${EXPORT_OPTIONS[@]}"

env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n "$HABITAT_CONDA_ENV" \
  python "$BRIDGE/validate_export.py" \
  --sequence-dir "$HABITAT_EXPORT_ROOT/$SEQ" \
  --schema "$HABITAT_INTERFACE_SCHEMA" \
  --mark-ready

echo "SMOKE_READY=$HABITAT_EXPORT_ROOT/$SEQ"
