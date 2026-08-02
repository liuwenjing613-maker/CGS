#!/usr/bin/env bash

# Server-side ConceptGraphs workspace for this host.
export CG_WORK=/home/chenkejun/beauty/conceptgraphs

# Keep model/config caches inside the writable server workspace.
export XDG_CACHE_HOME="$CG_WORK/.cache"
export XDG_CONFIG_HOME="$CG_WORK/.config"
export CONDA_PKGS_DIRS="$CG_WORK/.conda/pkgs"
export PIP_CACHE_DIR="$CG_WORK/.cache/pip"
export TORCH_HOME="$CG_WORK/models/torch"
export HF_HOME="$CG_WORK/models/huggingface"
export YOLO_CONFIG_DIR="$CG_WORK/.config/Ultralytics"
export MPLCONFIGDIR="$CG_WORK/.config/matplotlib"
export WANDB_DIR="$CG_WORK/logs/wandb"

# One host GPU (physical index 3) currently fails CUDA initialization when
# all eight devices are visible. Keep a working default while allowing an
# explicit caller override, e.g. CUDA_VISIBLE_DEVICES=1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export REPLICA_ROOT="$CG_WORK/data/Replica"
export REPLICA_SEMANTIC_ROOT="$CG_WORK/data/ReplicaSemanticGT"
export CG_FOLDER="$CG_WORK/code/concept-graphs-main"
export REPLICA_CONFIG_PATH="$CG_FOLDER/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
export GSA_PATH="$CG_WORK/code/Grounded-Segment-Anything"
export LLAVA_PYTHON_PATH="$CG_WORK/code/LLaVA"
export LLAVA_CKPT_PATH="$CG_WORK/models/llava/LLaVA-7B-v0"

if [[ -d "$GSA_PATH" ]]; then
  export PYTHONPATH="$GSA_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -d "$LLAVA_PYTHON_PATH" ]]; then
  export PYTHONPATH="$LLAVA_PYTHON_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi
