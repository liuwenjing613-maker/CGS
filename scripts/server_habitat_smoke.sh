#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/server_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sequence_id>" >&2
  exit 2
fi
SEQ="$1"
SEQ_DIR="$HABITAT_SEQUENCE_ROOT/$SEQ"
CFG="$SEQ_DIR/conceptgraphs_dataset.yaml"
read -r NUM_FRAMES IMAGE_HEIGHT IMAGE_WIDTH < <(
  "$CG_ALI_PYTHON" -c \
    'import json,sys; m=json.load(open(sys.argv[1])); print(m["num_frames"], m["rgb"]["height"], m["rgb"]["width"])' \
    "$SEQ_DIR/metadata.json"
)
MAP_SUFFIX="${SEQ}_smoke_mapping"
DET_SUFFIX="${SEQ}_smoke_detections"
LOG="$CG_WORK/logs/habitat/${SEQ}_smoke_mapping.log"

test -f "$SEQ_DIR/READY"
test -f "$SEQ_DIR/VALIDATED"
test -f "$CFG"
mkdir -p "$(dirname "$LOG")"
cd "$CG_ALI_FOLDER/conceptgraph"

"$CG_ALI_PYTHON" slam/rerun_realtime_mapping.py \
  dataset_root="$HABITAT_SEQUENCE_ROOT" \
  dataset_config="$CFG" \
  scene_id="$SEQ" \
  image_height="$IMAGE_HEIGHT" image_width="$IMAGE_WIDTH" \
  start=0 end="$NUM_FRAMES" stride=1 \
  make_edges=false use_rerun=false save_rerun=false \
  force_detection=true save_detections=true \
  detections_exp_suffix="$DET_SUFFIX" \
  exp_suffix="$MAP_SUFFIX" \
  save_video=false save_objects_all_frames=false \
  obj_pcd_max_points=3000 \
  2>&1 | tee "$LOG"

"$CG_ALI_PYTHON" "$CG_WORK/habitat_bridge/server/package_map_bundle.py" \
  --sequence-dir "$SEQ_DIR" \
  --run-id "${SEQ}_smoke" \
  --mapping-suffix "$MAP_SUFFIX" \
  --detection-suffix "$DET_SUFFIX" \
  --result-root "$HABITAT_RESULT_ROOT/$SEQ"

echo "SERVER_SMOKE_COMPLETE=$HABITAT_RESULT_ROOT/$SEQ/${SEQ}_smoke/map_bundle"
