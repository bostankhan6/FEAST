#!/usr/bin/env bash
# FIARSE on CIFAR-100, Mix-Aug
#
# Matches FEAST_cifar100.sh augmentation package for fair comparison:
#   --augmentation mixaug    RandAugment(2,6), reflect-pad crop, random flip
#   --mix_aug_mode alternating   Mixup(α=0.4) / CutMix(α=1.0) alternating per batch
#   --lr_cosine              cosine LR decay
#   --label_smoothing 0.0    Mixup already softens labels
#
# Note: --wd intentionally omitted. FIARSE paper (Table 3) specifies no weight
# decay — not changed from vanilla to keep faithful to the method.
# lr=0.05, lr_global=1.0: FIARSE-specific (unchanged from vanilla).
#
# No stem_stride: CIFAR-100 is 32×32.
# validation_split=0.1: CIFAR-100 has no separate valid/ directory.
#
# Usage:
#   ./experiments/03_cifar100/run_fiarse_cifar100.sh [--comm_round=N] [--gpu=G] [--resume]

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

exec "$PYTHON" -m baselines.fiarse.trainer \
    --dataset        cifar100                                       \
    --data_dir       "${PROJECT_ROOT}/data/cifar100"                \
    --num_clients    100                                            \
    --clients_per_round 10                                          \
    --comm_round     "$COMM_ROUND"                                  \
    --local_epochs   1                                              \
    --lr             0.05                                           \
    --lr_global      1.0                                            \
    --batch_size     64                                             \
    --partition_alpha 0.1                                           \
    --validation_split 0.1                                          \
    --corr_gamma     1.0                                            \
    --max_training_mac 600000000                                    \
    --resource_max_mac 1500000000                                   \
    --zipf_alpha     1.2                                            \
    --seed           0                                              \
    --eval_freq      20                                             \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/fiarse_cifar100_mixaug_v3" \
    --save_freq      100                                            \
    $RESUME                                                         \
    --gpu            "$GPU"                                         \
    --wandb_project  "hetero_fednas_cifar100_v3"                 \
    --wandb_run_name "fiarse-cifar100-mixaug-v3"                       \
    --augmentation   mixaug                                         \
    --mix_aug_mode   alternating                                    \
    --mixup_alpha    0.4                                            \
    --cutmix_alpha   1.0                                            \
    --label_smoothing 0.0                                           \
    --lr_cosine
