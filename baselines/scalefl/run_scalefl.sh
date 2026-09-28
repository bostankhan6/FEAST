#!/usr/bin/env bash
# ScaleFL on CIFAR-100 — full training run
# 100 clients / 10 per round / 2000 rounds / 1 epoch / Zipf(1.2) / gamma=1
#
# Usage:
#   ./baselines/scalefl/run_scalefl.sh [--comm_round N] [--gpu G]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"
DATA_DIR="${PROJECT_ROOT}/data"

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

exec "$PYTHON" -m baselines.scalefl.trainer \
    --data_dir        "$DATA_DIR"    \
    --num_clients     100            \
    --clients_per_round 10           \
    --comm_round      "$COMM_ROUND"  \
    --local_epochs    1              \
    --lr              0.025          \
    --batch_size      64             \
    --partition_alpha 0.1            \
    --validation_split 0.1          \
    --corr_gamma      1.0            \
    --max_training_mac 600000000     \
    --resource_max_mac 1500000000    \
    --zipf_alpha      1.2            \
    --seed            0              \
    --beta            0.1            \
    --tau             3.0            \
    --eval_freq       20             \
    --checkpoint_dir  "${PROJECT_ROOT}/checkpoints/scalefl" \
    --save_freq       100            \
    $RESUME                          \
    --gpu             "$GPU"         \
    --wandb_project   "hetero_fednas_baselines" \
    --wandb_run_name  "d1-scalefl-gamma1"
