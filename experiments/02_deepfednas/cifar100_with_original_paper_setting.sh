#!/bin/bash
# Experiment: the original-paper-scale DeepFedNAS setting -- 20 clients,
# 8/round, 2000 rounds, partition_alpha=100 (near-IID), the original 4-stage
# supernet definition (max_extra_blocks_per_stage=2, base channels
# [256,512,1024,2048]), TS_optimal_path sampling, and the non-canonical
# subnet_caches/4_stage_cache_60_subnets.csv cache -- the same cache as
# 01_superfednas/'s equivalent script; only the sampler differs between the
# two "original setting" scripts.
#
# Usage:
#   bash experiments/02_deepfednas/cifar100_with_original_paper_setting.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running experiment from root: $PROJECT_ROOT"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name hetero_fednas_50M_600M_ablations \
    --wandb_run_name "deepfednas_original_cifar100_pA-100" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/deepfednas-original-cifar100-pA-100" \
    --gpu 0 \
    --dataset cifar100 \
    --data_dir "$PROJECT_ROOT/data/cifar100" \
    --partition_method hetero \
    --partition_alpha 100 \
    --validation_split 0.1 \
    --client_num_in_total 20 \
    --client_num_per_round 8 \
    --comm_round 2000 \
    --epochs 5 \
    --batch_size 64 \
    --client_optimizer sgd \
    --lr 0.1 \
    --init_seed 0 \
    --max_norm 10.0 \
    --frequency_of_the_test 20 \
    --efficient_test \
    --weighted_avg_schedule '{"type":"maxnet_cos_all_subnet","num_steps":1600,"init":0.9,"final":0.125}' \
    --subnet_dist_type TS_optimal_path \
    --supernet_num_stages 4 \
    --supernet_max_extra_blocks_per_stage 2 \
    --supernet_original_stage_base_channels '[256, 512, 1024, 2048]' \
    --supernet_width_multiplier_choices '[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]' \
    --supernet_expansion_ratio_choices '[0.1, 0.14, 0.18, 0.22, 0.25]' \
    --diverse_subnets '{"0": {"d": [2, 2, 2, 2], "e": [0.18, 0.18, 0.14, 0.1], "w_indices": [9, 4, 4, 4, 4]}, "1": {"d": [2, 2, 2, 2], "e": [0.14, 0.18, 0.14, 0.14], "w_indices": [9, 8, 7, 8, 7]}, "2": {"d": [2, 2, 2, 2], "e": [0.25, 0.25, 0.18, 0.18], "w_indices": [9, 9, 9, 9, 9]}, "3": {"d": [2, 2, 2, 2], "e": [0.25, 0.25, 0.25, 0.25], "w_indices": [9, 9, 9, 9, 9]}}' \
    --subnet_cache_path "$PROJECT_ROOT/subnet_caches/4_stage_cache_60_subnets.csv" \
    --verbose