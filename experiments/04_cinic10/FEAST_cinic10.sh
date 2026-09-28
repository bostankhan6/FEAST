#!/bin/bash
# Experiment: FEAST — CINIC-10, Mix-Aug
#
# Canonical FEAST CINIC-10 run, using the same mixaug augmentation package as
# the CIFAR-100 and TinyImageNet FEAST runs for a consistent cross-dataset
# augmentation comparison:
#   --augmentation mixaug        RandAugment(2,6) — no Cutout, no ColorJitter
#   --mix_aug_mode alternating   Mixup(α=0.4) / CutMix(α=1.0) alternating per batch
#   --mixup_alpha 0.4
#   --cutmix_alpha 1.0
#   --wd 1e-4
#   --label_smoothing 0.0        Mixup already softens labels; avoid double-softening
#   --lr_cosine                  Cosine LR decay (matches other mixaug runs)
#
# No stem_stride change: CINIC-10 is 32×32, same as CIFAR-100.
# No validation_split: CINIC-10 has its own valid/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running FEAST CINIC-10 mixaug from: $PROJECT_ROOT"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name hetero_fednas_cinic10_v3 \
    --wandb_group "cinic10-mixaug-v3" \
    --wandb_run_name "feast-cinic10-mixaug-v3" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/feast-cinic10-mixaug-v3" \
    --gpu 0 \
    --dataset cinic10 \
    --data_dir "$PROJECT_ROOT/data/cinic10" \
    --partition_method hetero \
    --partition_alpha 0.005 \
    --client_num_in_total 100 \
    --client_num_per_round 10 \
    --comm_round 4000 \
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
    --resource_zipf_alpha 1.2 \
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
    --verbose
