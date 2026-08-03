#!/usr/bin/env bash

export CG_LOCAL="$HOME/conceptgraphs"
export CG_REPO="$CG_LOCAL/CGS"

# 与 ~/.ssh/config 中 chenkejun 主机对应。
export CG_SERVER_ALIAS="cg-server"
export CG_REMOTE="/home/chenkejun/beauty/conceptgraphs"

export HABITAT_EXPORT_ROOT="$CG_LOCAL/data/habitat_exports"
export LOCAL_RESULT_ROOT="$CG_LOCAL/results/habitat"

# 本地 MP3D（指南默认 17DRP5sb8fy 本地没有；先用已有场景跑通 smoke）
export MP3D_ROOT="$CG_LOCAL/data/scenes/mp3d"
export MP3D_SCENE_ID="${MP3D_SCENE_ID:-2azQ1b91cZZ}"
export MP3D_SCENE="$MP3D_ROOT/$MP3D_SCENE_ID/$MP3D_SCENE_ID.glb"
export HABITAT_CONDA_ENV="${HABITAT_CONDA_ENV:-habitat}"
export HABITAT_INTERFACE_SCHEMA="$CG_LOCAL/docs/CGS_Habitat_Interface_v1.schema.json"

# 激活 habitat 前请先清掉 conceptgraph 污染的 LD_LIBRARY_PATH，否则 EGL 会崩：
#   unset LD_LIBRARY_PATH
#   conda activate habitat
#   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
