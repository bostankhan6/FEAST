#!/usr/bin/env bash
# ScaleFL on CIFAR-100, Mix-Aug, matched-compute control.
#
# Extends ScaleFL's round budget so its total training MACs match FEAST's
# canonical run (31.01 PMACs, Supplementary Sec. E.4 "Training-Computation
# Controls" -- reproduce via scripts/reproduce_training_compute_proxy.py).
# ScaleFL's existing population margin (+31.99pp) already dwarfs a 2x
# compute difference, so this control exists mainly for completeness.
#
#   ratio  = FEAST_total_PMACs / ScaleFL_total_PMACs(4000r) = 31.01 / 15.22 = 2.0375
#   rounds = round(4000 * 2.0375) = 8150
#
# --lr_cosine reads args.comm_round as its horizon T directly
# (baselines/scalefl/trainer.py:352), so --comm_round=8150 recomputes the
# FULL cosine anneal over the new horizon. Fresh run, not a resume past an
# already-annealed 4000-round schedule.
#
# Everything else identical to
# experiments/03_cifar100/run_scalefl_cifar100.sh
# (same seed=0, partition, budgets, augmentation package, beta=0.1, tau=3.0)
# so the ONLY variable is round count / total compute.
#
# Usage:
#   ./experiments/10_matched_training_compute/run_scalefl_cifar100_matched_compute.sh [--comm_round=N] [--gpu=G] [--resume]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

COMM_ROUND=8150
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
    --corr_gamma     1.0                                            \
    --max_training_mac 600000000                                    \
    --resource_max_mac 1500000000                                   \
    --zipf_alpha     1.2                                            \
    --seed           0                                              \
    --beta           0.1                                            \
    --tau            3.0                                            \
    --eval_freq      20                                             \
    --checkpoint_dir "${PROJECT_ROOT}/checkpoints/scalefl_cifar100_mixaug_matched_compute" \
    --save_freq      100                                            \
    $RESUME                                                         \
    --gpu            "$GPU"                                         \
    --wandb_project  "hetero_fednas_cifar100_v3"                    \
    --wandb_run_name "scalefl-cifar100-mixaug-matched-compute-8150r" \
    --augmentation   mixaug                                         \
    --mix_aug_mode   alternating                                    \
    --mixup_alpha    0.4                                            \
    --cutmix_alpha   1.0                                            \
    --label_smoothing 0.0                                           \
    --lr_cosine
