#!/usr/bin/env bash
set -euo pipefail

# Override CUDA_VISIBLE_DEVICES when the two 5090s do not have indices 0 and 1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/check_hardware.py
torchrun --standalone --nproc_per_node=2 \
  scripts/run_scalable_experiment.py \
  --config configs/1m_2x5090.json \
  --require-world-size 2 \
  --output artifacts/1m_2x5090_metrics.json

