#!/bin/bash
# Experiment: SuperFedNAS transfer — real MaxNet aggregation, budget-constrained
# clients (Zipf), CIFAR-100 mixaug.
#
# Real MaxNet transfer under FEAST's resource-heterogeneous setting
# (Supplementary Sec. E.2, "SFN and DFN Transfer Configuration"): SFN's
# uniform-random sampler (TS_all_random_constrained) plus real MaxNet
# cosine-annealed aggregation (--weighted_avg_schedule maxnet_cos_all_subnet),
# restricted to each client's affordable architectures under Zipf(1.2)
# budgets. See cifar100_unconstrained_maxnet.sh for the full explanation of
# the MaxNet aggregation mechanism.
#
# Differs from FEAST (the full method) by omitting sub-supernet communication
# (--use_sub_supernet) and per-step min/random/max training
# (--training_strategy feast) -- this is the "standard" per-client
# single-subnet training loop instead.
#
# --weight_dataset_by_budget --corr_gamma 1.0: the same correlated-data
# environment shared across all constrained methods in this comparison
# (HeteroFL, ScaleFL, FIARSE, the SFN/DFN baselines), not an algorithmic
# contribution being tested here.
#
# Usage:
#   bash experiments/01_superfednas/cifar100_constrained_maxnet_gamma1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running SuperFedNAS Constrained + REAL MaxNet aggregation (mixaug) from: $PROJECT_ROOT"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name deepfednas_superfednas_comparison \
    --wandb_group "constrained-maxnet-mixaug" \
    --wandb_run_name "superfednas-constrained-maxnet-mixaug" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/superfednas-constrained-maxnet-mixaug" \
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
    --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":3200,"init":0.9,"final":0.125}' \
    --subnet_dist_type TS_all_random_constrained \
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
    --max_training_mac 600000000 \
    --weight_dataset \
    --weight_dataset_by_budget \
    --corr_gamma 1.0 \
    --use_bn \
    --reset_bn_stats \
    --reset_bn_sample_size 0.1 \
    --skip_train_test \
    --augmentation mixaug \
    --mix_aug_mode alternating \
    --mixup_alpha 0.4 \
    --cutmix_alpha 1.0 \
    --label_smoothing 0.0 \
    --lr_cosine \
    --verbose
