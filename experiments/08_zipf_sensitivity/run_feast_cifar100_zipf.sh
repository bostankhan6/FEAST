#!/bin/bash
# FEAST Zipf-sensitivity run for CIFAR-100 mixaug.
#
# Usage:
#   ./experiments/08_zipf_sensitivity/run_feast_cifar100_zipf.sh 0.8
#
# Optional environment overrides:
#   GPU=0
#   COMM_ROUND=4000
#   WANDB_MODE=offline
#
# Notes:
#   - gamma is fixed at 1.0.
#   - This follows the canonical CIFAR-100 mixaug FEAST protocol, changing only
#     --resource_zipf_alpha and run naming.
#   - This script is intentionally sequential and does not pass --multi_gpu.
#     Use it for canonical robustness runs.

set -euo pipefail

ZIPF_ALPHA="${1:?Usage: $0 <zipf_alpha, e.g. 0.8>}"
ZIPF_TAG="${ZIPF_ALPHA//./}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

GPU="${GPU:-0}"
COMM_ROUND="${COMM_ROUND:-4000}"
WANDB_GROUP="${WANDB_GROUP:-zipf-sensitivity-cifar100-mixaug}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-hetero_fednas_extended_range}"
RUN_NAME="${RUN_NAME:-feast-cifar100-mixaug-zipf-a${ZIPF_TAG}}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/checkpoints/zipf_sensitivity_cifar100_mixaug/logs}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "Running FEAST CIFAR-100 mixaug Zipf sensitivity"
echo "  project root: $PROJECT_ROOT"
echo "  zipf alpha:   $ZIPF_ALPHA"
echo "  gamma:        1.0"
echo "  rounds:       $COMM_ROUND"
echo "  gpu arg:      $GPU"
echo "  mode:         sequential canonical"
echo "  run name:     $RUN_NAME"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name "$WANDB_PROJECT_NAME" \
    --wandb_group "$WANDB_GROUP" \
    --wandb_run_name "$RUN_NAME" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/zipf_sensitivity_cifar100_mixaug/${RUN_NAME}" \
    --gpu "$GPU" \
    --dataset cifar100 \
    --data_dir "$PROJECT_ROOT/data/cifar100" \
    --partition_method hetero \
    --partition_alpha 0.1 \
    --validation_split 0.1 \
    --client_num_in_total 100 \
    --client_num_per_round 10 \
    --comm_round "$COMM_ROUND" \
    --epochs 1 \
    --batch_size 64 \
    --client_optimizer sgd \
    --lr 0.025 \
    --momentum 0.9 \
    --wd 1e-4 \
    --init_seed 0 \
    --max_norm 10.0 \
    --frequency_of_the_test 20 \
    --efficient_test \
    --subnet_dist_type TS_all_random \
    --supernet_num_stages 4 \
    --supernet_max_extra_blocks_per_stage 8 \
    --supernet_original_stage_base_channels '[128, 256, 512, 1024]' \
    --supernet_width_multiplier_choices '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]' \
    --supernet_expansion_ratio_choices '[0.1, 0.14, 0.18, 0.22, 0.25]' \
    --diverse_subnets '{"0": {"d": [3, 7, 6, 7], "e": [0.18, 0.18, 0.1, 0.1], "w_indices": [2, 0, 0, 0, 0]}, "1": {"d": [5, 6, 7, 8], "e": [0.18, 0.1, 0.14, 0.22], "w_indices": [2, 0, 1, 1, 0]}, "2": {"d": [8, 8, 8, 8], "e": [0.22, 0.1, 0.1, 0.1], "w_indices": [4, 2, 3, 4, 3]}, "3": {"d": [8, 8, 8, 8], "e": [0.1, 0.1, 0.14, 0.1], "w_indices": [6, 6, 6, 3, 5]}, "4": {"d": [8, 8, 8, 8], "e": [0.18, 0.1, 0.1, 0.1], "w_indices": [9, 4, 6, 7, 6]}}' \
    --subnet_cache_path "$PROJECT_ROOT/subnet_caches/extended_range_25M_1500M.csv" \
    --resource_heterogeneity \
    --resource_distribution_type zipf \
    --resource_zipf_alpha "$ZIPF_ALPHA" \
    --resource_max_mac 1500000000 \
    --training_strategy feast \
    --per_step_random \
    --kd_ratio 0.5 \
    --use_bn \
    --reset_bn_stats \
    --reset_bn_sample_size 0.1 \
    --skip_train_test \
    --weight_dataset \
    --weight_dataset_by_budget \
    --corr_gamma 1.0 \
    --max_training_mac 600000000 \
    --use_sub_supernet \
    --augmentation mixaug \
    --mix_aug_mode alternating \
    --mixup_alpha 0.4 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.0 \
    --lr_cosine \
    --verbose 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
