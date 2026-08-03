#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

SCENE="${1:-$MP3D_SCENE}"
AUTO="${2:-none}"
test -f "$SCENE"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Run this from the Ubuntu desktop session, not a headless SSH shell." >&2
  exit 2
fi

env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n "$HABITAT_CONDA_ENV" \
  python "$CG_REPO/habitat_bridge/local/view_habitat.py" \
  --scene "$SCENE" \
  --width 640 \
  --height 480 \
  --hfov 90 \
  --sensor-height 1.25 \
  --seed 2027 \
  --autoplay "$AUTO" \
  --save-dir "$CG_LOCAL/results/habitat/viewer_captures"
