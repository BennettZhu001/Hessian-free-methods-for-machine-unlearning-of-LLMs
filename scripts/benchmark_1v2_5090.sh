#!/usr/bin/env bash
set -euo pipefail

first_gpu="${FIRST_GPU:-0}"
gpu_pair="${GPU_PAIR:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${first_gpu}" python scripts/run_scalable_experiment.py \
  --config configs/1m_1x5090.json \
  --require-world-size 1 \
  --output artifacts/1m_1x5090_metrics.json

CUDA_VISIBLE_DEVICES="${gpu_pair}" torchrun --standalone --nproc_per_node=2 \
  scripts/run_scalable_experiment.py \
  --config configs/1m_2x5090.json \
  --require-world-size 2 \
  --output artifacts/1m_2x5090_metrics.json

python scripts/compare_scaling.py \
  artifacts/1m_1x5090_metrics.json \
  artifacts/1m_2x5090_metrics.json

