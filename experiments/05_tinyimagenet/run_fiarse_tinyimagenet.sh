#!/usr/bin/env bash
# FIARSE on TinyImageNet-200, Mix-Aug
#
# Matches FEAST_tinyimagenet.sh augmentation package for fair comparison:
#   --stem_stride 2      64×64 input → 32×32 post-stem (same body MACs as CIFAR/CINIC)
#   --augmentation mixaug  RandAugment(2,6), reflect-pad crop, random flip
#   --mix_aug_mode alternating  Mixup(α=0.4) / CutMix(α=1.0) alternating per batch
#   --lr_cosine          cosine LR decay
#   --label_smoothing 0.0  Mixup already softens labels
#
# Note: FIARSE paper (Table 3) specifies no weight_decay and no momentum — not changed.
#
# Usage:
#   ./experiments/05_tinyimagenet/run_fiarse_tinyimagenet.sh [--comm_round=N] [--gpu=G] [--resume]

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
    --dataset        tinyimagenet                               \
    --data_dir       "${PROJECT_ROOT}/data/tinyimagenet"        \
    --stem_stride    2                                          \
    --num_clients    100                                        \
    --clients_per_round 10                                      \
    --comm_round     "$COMM_ROUND"                              \
    --local_epochs   1                                          \
    --lr             0.05                                       \
    --lr_global      1.0                                        \
    --batch_size     64                                         \
    --partition_alpha 0.1                                       \
    --validation_split 0.1                                      \
    --corr_gamma     1.0                                        \
    --max_training_mac 600000000                                \
    --resource_max_mac 1500000000                               \
    --zipf_alpha     1.2                                        \
    --seed           0                                          \
    --eval_freq      20                                         \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/fiarse_tinyimagenet_mixaug" \
    --save_freq      100                                        \
    $RESUME                                                     \
    --gpu            "$GPU"                                     \
    --wandb_project  "hetero_fed_nsa_tinyimagenet"              \
    --wandb_run_name "fiarse-tinyimagenet-mixaug"               \
    --augmentation   mixaug                                     \
    --mix_aug_mode   alternating                                \
    --mixup_alpha    0.4                                        \
    --cutmix_alpha   1.0                                        \
    --label_smoothing 0.0                                       \
    --lr_cosine
