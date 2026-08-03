#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/local_env.sh"

exec python3 "$CG_REPO/habitat_bridge/local/orchestrate_objectnav.py" "$@"

