#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/server_env.sh"

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <sequence_id> <run_id> <query_base64> <query_id>" >&2
  exit 2
fi
SEQ="$1"
RUN="$2"
QUERY_BASE64="$3"
QUERY_ID="$4"
for VALUE in "$SEQ" "$RUN" "$QUERY_ID"; do
  if [[ ! "$VALUE" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid identifier: $VALUE" >&2
    exit 2
  fi
done

QUERY="$(printf '%s' "$QUERY_BASE64" | base64 --decode)"
BUNDLE="$HABITAT_RESULT_ROOT/$SEQ/$RUN/map_bundle"
OUTPUT="$HABITAT_RESULT_ROOT/$SEQ/$RUN/queries/$QUERY_ID.json"

test -f "$BUNDLE/COMPLETE"
(cd "$BUNDLE" && sha256sum -c checksums.sha256)

"$CG_ALI_PYTHON" "$CG_WORK/habitat_bridge/server/query_object_map.py" \
  --result-path "$BUNDLE/object_map.pkl.gz" \
  --query "$QUERY" \
  --output "$OUTPUT" \
  --top-k 5 \
  --sample-points 128 \
  --device cuda

echo "SERVER_QUERY_COMPLETE=$OUTPUT"
