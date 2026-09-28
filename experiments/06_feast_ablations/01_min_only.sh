#!/bin/bash
# FEAST component ablation — Variant: min-only (= standard FedAvg)
#
# Every client trains only the globally-shared smallest variant (25M MACs).
# No local-max step, no random-intermediate step, no KD. This is the
# --training_strategy min_only baseline (E0 in the codebase) and is
# structurally equivalent to standard FedAvg on a single fixed-size model:
# all clients share the same architecture, no elastic/multi-subnet mechanism.
#
# Identical protocol otherwise to the canonical FEAST_cifar100.sh
# (Zipf(1.2) resource skew, gamma=1 data allocation, mixaug augmentation,
# sub-supernet routing, BN-cal, 4000 rounds).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running FEAST ablation (min-only / standard FedAvg) from: $PROJECT_ROOT"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name hetero_fednas_extended_range \
    --wandb_group "cifar100-mixaug-ablation" \
    --wandb_run_name "feast-ablation-min-only" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/feast-ablation-min-only" \
    --gpu 0 \
    --dataset cifar100 \
    --data_dir "$PROJECT_ROOT/data/cifar100" \
    --partition_method hetero \
    --partition_alpha 0.1 \
    --validation_split 0.1 \
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
    --training_strategy min_only \
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
