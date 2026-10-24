#!/usr/bin/env bash
# Source this file from bash. Keep all generated state on the project data disk.
MINIMIND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$MINIMIND_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$MINIMIND_ROOT/.cache/uv"
export XDG_CACHE_HOME="$MINIMIND_ROOT/.cache"
export TMPDIR="$MINIMIND_ROOT/.cache/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
