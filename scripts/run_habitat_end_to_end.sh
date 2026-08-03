#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

SEQ="${1:-${MP3D_SCENE_ID}_smoke_v001}"
if [[ ! "$SEQ" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid sequence_id: $SEQ" >&2
  exit 2
fi
RUN="${SEQ}_smoke"
SEQ_DIR="$HABITAT_EXPORT_ROOT/$SEQ"
BUNDLE="$LOCAL_RESULT_ROOT/$SEQ/$RUN/map_bundle"

if [[ -f "$SEQ_DIR/READY" && "${REEXPORT:-0}" != "1" ]]; then
  echo "Reusing validated local sequence: $SEQ_DIR"
  env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n "$HABITAT_CONDA_ENV" \
    python "$CG_REPO/habitat_bridge/local/validate_export.py" \
    --sequence-dir "$SEQ_DIR" \
    --schema "$HABITAT_INTERFACE_SCHEMA" \
    --mark-ready
else
  if [[ -d "$SEQ_DIR" ]]; then
    OVERWRITE=1 "$ROOT/scripts/run_habitat_smoke.sh" "$SEQ"
  else
    "$ROOT/scripts/run_habitat_smoke.sh" "$SEQ"
  fi
fi

"$ROOT/scripts/upload_sequence.sh" "$SEQ"
ssh "$CG_SERVER_ALIAS" "$CG_REMOTE/scripts/server_habitat_pipeline.sh" "$SEQ"
"$ROOT/scripts/download_map_bundle.sh" "$SEQ" "$RUN"

env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n conceptgraph \
  python -c 'import gzip,pickle,sys; data=pickle.load(gzip.open(sys.argv[1], "rb")); print("LOCAL_OBJECTS=" + str(len(data["objects"])))' \
  "$BUNDLE/object_map.pkl.gz"

echo "END_TO_END_COMPLETE=$BUNDLE"
echo "View it with: $ROOT/scripts/view_habitat_map.sh $SEQ $RUN"
