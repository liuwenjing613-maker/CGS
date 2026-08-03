#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <sequence_id> <run_id>" >&2
  exit 2
fi

SEQ="$1"
RUN="$2"
DST="$LOCAL_RESULT_ROOT/$SEQ/$RUN/map_bundle"
mkdir -p "$DST"

rsync -avP \
  "$CG_SERVER_ALIAS:$CG_REMOTE/results/HabitatMP3D/$SEQ/$RUN/map_bundle/" \
  "$DST/"

(cd "$DST" && sha256sum -c checksums.sha256)
test -f "$DST/COMPLETE"
echo "MAP_BUNDLE_READY=$DST"
