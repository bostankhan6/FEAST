#!/usr/bin/env bash
# HeteroFL on CIFAR-100, Mix-Aug, γ=0 (v3)
#
# Identical to the main-comparison run (run_heterofl_cifar100.sh) except
# --corr_gamma is 0.0 and the wandb / checkpoint names carry the gamma0_v3
# suffix. Gamma=0 arm of the gamma sensitivity comparison; the gamma=1 arm
# is run_heterofl_cifar100.sh itself.
#
# Usage:
#   ./experiments/07_gamma_ablation/heterofl_cifar100_gamma0_v3.sh [--comm_round=N] [--gpu=G] [--resume]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

COMM_ROUND=4000
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
    --dataset        cifar100                                       \
    --data_dir       "${PROJECT_ROOT}/data/cifar100"                \
    --num_clients    100                                            \
    --clients_per_round 10                                          \
    --comm_round     "$COMM_ROUND"                                  \
    --local_epochs   1                                              \
    --lr             0.025                                          \
    --wd             1e-4                                           \
    --batch_size     64                                             \
    --partition_alpha 0.1                                           \
    --validation_split 0.1                                          \
    --corr_gamma     0.0                                            \
    --max_training_mac 600000000                                    \
    --resource_max_mac 1500000000                                   \
    --zipf_alpha     1.2                                            \
    --seed           0                                              \
    --eval_freq      20                                             \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/cifar100_baselines_gamma0_v3_reruns/heterofl_cifar100_mixaug_gamma0_v3" \
    --save_freq      100                                            \
    $RESUME                                                         \
    --gpu            "$GPU"                                         \
    --wandb_project  "hetero_fednas_cifar100_gamma0_v3"                   \
    --wandb_run_name "heterofl-cifar100-mixaug-gamma0-v3"              \
    --augmentation   mixaug                                         \
    --mix_aug_mode   alternating                                    \
    --mixup_alpha    0.4                                            \
    --cutmix_alpha   1.0                                            \
    --label_smoothing 0.0                                           \
    --lr_cosine
