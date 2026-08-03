#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/server_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sequence_id>" >&2
  exit 2
fi
SEQ="$1"
if [[ ! "$SEQ" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid sequence_id: $SEQ" >&2
  exit 2
fi

SEQ_DIR="$HABITAT_SEQUENCE_ROOT/$SEQ"
test -f "$SEQ_DIR/READY"

"$CG_ALI_PYTHON" "$CG_WORK/habitat_bridge/server/validate_sequence.py" \
  --sequence-dir "$SEQ_DIR"

"$CG_ALI_PYTHON" "$CG_WORK/habitat_bridge/server/generate_dataset_config.py" \
  --sequence-dir "$SEQ_DIR"

"$ROOT/scripts/server_habitat_smoke.sh" "$SEQ"
