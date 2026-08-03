#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

SEQ="${1:-${MP3D_SCENE_ID}_smoke_v001}"
RUN="${2:-${SEQ}_smoke}"
BUNDLE="$LOCAL_RESULT_ROOT/$SEQ/$RUN/map_bundle"
MAP="$BUNDLE/object_map.pkl.gz"

test -f "$BUNDLE/COMPLETE"
test -f "$MAP"
(cd "$BUNDLE" && sha256sum -c checksums.sha256)

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Run this from the Ubuntu desktop terminal." >&2
  exit 2
fi

export XDG_CACHE_HOME="$CG_LOCAL/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"

cd "$CG_REPO/code/concept-graphs-ali"
exec env -u LD_LIBRARY_PATH -u PYTHONPATH conda run --no-capture-output -n conceptgraph \
  python conceptgraph/scripts/visualize_cfslam_results.py \
  --result_path "$MAP" \
  --edge_file "$BUNDLE/edges.json" \
  --no_clip
