#!/usr/bin/env bash
# HeteroFL on CIFAR-100 — full training run
# Mirrors the federated setting used by all other methods:
#   100 clients / 10 per round / 2000 rounds / 1 epoch / Zipf(1.2) / γ=1
#
# Usage:
#   ./baselines/heterofl/run_heterofl.sh [--comm_round N] [--gpu G]
#
# Defaults: 2000 rounds, GPU 0.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

DATA_DIR="${PROJECT_ROOT}/data"

# Parse optional overrides
COMM_ROUND=2000
GPU=0
RESUME=""
for arg in "$@"; do
  case $arg in
    --comm_round=*) COMM_ROUND="${arg#*=}" ;;
    --gpu=*)        GPU="${arg#*=}" ;;
    --resume)       RESUME="--resume" ;;
  esac
done

exec "$PYTHON" -m baselines.heterofl.trainer \
    --data_dir        "$DATA_DIR"       \
    --num_clients     100               \
    --clients_per_round 10              \
    --comm_round      "$COMM_ROUND"     \
    --local_epochs    1                 \
    --lr              0.025             \
    --batch_size      64                \
    --partition_alpha 0.1               \
    --validation_split 0.1             \
    --corr_gamma      1.0               \
    --max_training_mac 600000000        \
    --resource_max_mac 1500000000       \
    --zipf_alpha      1.2               \
    --seed            0                 \
    --eval_freq       20                \
    --checkpoint_dir  "${PROJECT_ROOT}/checkpoints/heterofl" \
    --save_freq       100               \
    $RESUME                             \
    --gpu             "$GPU"            \
    --wandb_project   "hetero_fednas_baselines" \
    --wandb_run_name  "d2-heterofl-gamma1"
