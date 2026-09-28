#!/usr/bin/env bash
# FIARSE on CIFAR-100, Mix-Aug, matched-compute control.
#
# Extends FIARSE's round budget so its total training MACs match FEAST's
# canonical run (31.01 PMACs, Supplementary Sec. E.4 "Training-Computation
# Controls" -- reproduce via scripts/reproduce_training_compute_proxy.py).
#
# FIARSE has two MAC conventions, since its masks feed ordinary dense
# convolutions rather than an ideal sparse kernel:
#   logical active (ideal sparse exec):  19.12 PMACs -> ratio 1.6219 -> 6487 rounds
#   executed dense-conv (actual code):   29.97 PMACs -> ratio 1.0347 -> 4139 rounds
# This script targets executed dense-conv (4139 rounds), the honest
# apples-to-apples figure for what this repo's PyTorch implementation actually
# runs on hardware. Pass --comm_round=6487 for the logical-active target
# instead.
#
# --lr_cosine reads args.comm_round as its horizon T directly
# (baselines/fiarse/trainer.py:305), so --comm_round=4139 recomputes the
# FULL cosine anneal over the new horizon. Fresh run, not a resume past an
# already-annealed 4000-round schedule.
#
# Everything else identical to
# experiments/03_cifar100/run_fiarse_cifar100.sh
# (same seed=0, partition, budgets, augmentation package, no --wd per the
# FIARSE paper's Table 3) so the ONLY variable is round count / total compute.
#
# Usage:
#   ./experiments/10_matched_training_compute/run_fiarse_cifar100_matched_compute.sh [--comm_round=N] [--gpu=G] [--resume]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

COMM_ROUND=4139
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
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/fiarse_cifar100_mixaug_matched_compute" \
    --save_freq      100                                            \
    $RESUME                                                         \
    --gpu            "$GPU"                                         \
    --wandb_project  "hetero_fednas_cifar100_v3"                    \
    --wandb_run_name "fiarse-cifar100-mixaug-matched-compute-4139r" \
    --augmentation   mixaug                                         \
    --mix_aug_mode   alternating                                    \
    --mixup_alpha    0.4                                            \
    --cutmix_alpha   1.0                                            \
    --label_smoothing 0.0                                           \
    --lr_cosine
