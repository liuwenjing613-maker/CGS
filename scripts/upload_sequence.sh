#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sequence_id>" >&2
  exit 2
fi

SEQ="$1"
SRC="$HABITAT_EXPORT_ROOT/$SEQ"
DST="$CG_SERVER_ALIAS:$CG_REMOTE/data/HabitatMP3D/sequences/$SEQ/"

test -f "$SRC/READY"
rsync -avP --delete-delay "$SRC/" "$DST"
