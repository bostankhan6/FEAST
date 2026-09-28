#!/usr/bin/env bash
# FIARSE Dirichlet-sensitivity run for CIFAR-100 mixaug.
#
# Usage:
#   ./experiments/09_dirichlet_sensitivity/run_fiarse_cifar100_dirichlet.sh 0.3
#
# Optional environment overrides:
#   GPU=0
#   COMM_ROUND=4000
#
# Notes:
#   - gamma fixed at 1.0; resource Zipf alpha fixed at canonical 1.2.
#   - Matches FIARSE CIFAR-100 mixaug protocol; only --partition_alpha and
#     run/checkpoint naming change.
#   - FIARSE keeps lr=0.05, lr_global=1.0, no weight decay (paper Table 3).

set -euo pipefail

DIR_ALPHA="${1:?Usage: $0 <dirichlet_alpha, e.g. 0.3>}"
DIR_TAG="${DIR_ALPHA//./}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

GPU="${GPU:-0}"
COMM_ROUND="${COMM_ROUND:-4000}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-dirichlet-sensitivity-cifar100-mixaug}"
RUN_NAME="${RUN_NAME:-fiarse-cifar100-mixaug-dir-a${DIR_TAG}}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/dirichlet_sensitivity_cifar100_mixaug/fiarse_dir_a${DIR_TAG}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/checkpoints/dirichlet_sensitivity_cifar100_mixaug/logs}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
cd "$PROJECT_ROOT"

echo "Running FIARSE CIFAR-100 mixaug Dirichlet sensitivity"
echo "  dirichlet alpha: $DIR_ALPHA"
echo "  zipf alpha:      1.2 (fixed)"
echo "  gamma:           1.0 (fixed)"
echo "  rounds:          $COMM_ROUND"
echo "  run name:        $RUN_NAME"

"$PYTHON" -m baselines.fiarse.trainer \
    --dataset        cifar100                                  \
    --data_dir       "${PROJECT_ROOT}/data/cifar100"           \
    --num_clients    100                                       \
    --clients_per_round 10                                     \
    --comm_round     "$COMM_ROUND"                             \
    --local_epochs   1                                         \
    --lr             0.05                                      \
    --lr_global      1.0                                       \
    --batch_size     64                                        \
    --partition_alpha "$DIR_ALPHA"                             \
    --validation_split 0.1                                     \
    --corr_gamma     1.0                                       \
    --max_training_mac 600000000                               \
    --resource_max_mac 1500000000                              \
    --zipf_alpha     1.2                                       \
    --seed           0                                         \
    --eval_freq      20                                        \
    --checkpoint_dir "$CKPT_DIR"                               \
    --save_freq      100                                       \
    --gpu            "$GPU"                                    \
    --wandb_project  "$WANDB_PROJECT_NAME"                     \
    --wandb_run_name "$RUN_NAME"                               \
    --augmentation   mixaug                                    \
    --mix_aug_mode   alternating                               \
    --mixup_alpha    0.4                                       \
    --cutmix_alpha   1.0                                       \
    --label_smoothing 0.0                                      \
    --lr_cosine 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
