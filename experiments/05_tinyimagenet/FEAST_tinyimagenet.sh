#!/bin/bash
# Experiment: FEAST — TinyImageNet-200, Mix-Aug
#
# Canonical FEAST TinyImageNet run, using the same mixaug augmentation
# package as the CIFAR-100 and CINIC-10 FEAST runs for a consistent
# cross-dataset augmentation comparison:
#   --augmentation mixaug        RandAugment(2,6) only -- no Cutout, no ColorJitter
#   --mix_aug_mode alternating   Mixup/CutMix applied at batch level (same mixed
#                                batch used for all three min/random/max subnets)
#   --wd 1e-4
#   --classifier_dropout 0.0
#   --label_smoothing 0.0        Mixup already softens labels; avoid double-softening
#   --lr_cosine
#   --kd_ratio 0.5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_ROOT"
echo "Running FEAST TinyImageNet (mix-aug) from: $PROJECT_ROOT"

./venv/bin/python train.py \
    --model ofaresnet_generic \
    --wandb_project_name hetero_fed_nsa_tinyimagenet \
    --wandb_group "tinyimagenet-mixaug" \
    --wandb_run_name "feast-tinyimagenet-mixaug" \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/feast-tinyimagenet-mixaug" \
    --gpu 0 \
    --dataset tinyimagenet \
    --data_dir "$PROJECT_ROOT/data/tinyimagenet" \
    --partition_method hetero \
    --partition_alpha 0.1 \
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
    --supernet_initial_input_hw 64 \
    --supernet_stem_stride 2 \
    --diverse_subnets '{"0": {"d": [3, 7, 6, 7], "e": [0.18, 0.18, 0.1, 0.1], "w_indices": [2, 0, 0, 0, 0]}, "1": {"d": [5, 6, 7, 8], "e": [0.18, 0.1, 0.14, 0.22], "w_indices": [2, 0, 1, 1, 0]}, "2": {"d": [8, 8, 8, 8], "e": [0.22, 0.1, 0.1, 0.1], "w_indices": [4, 2, 3, 4, 3]}, "3": {"d": [8, 8, 8, 8], "e": [0.1, 0.1, 0.14, 0.1], "w_indices": [6, 6, 6, 3, 5]}, "4": {"d": [8, 8, 8, 8], "e": [0.18, 0.1, 0.1, 0.1], "w_indices": [9, 4, 6, 7, 6]}}' \
    --subnet_cache_path "$PROJECT_ROOT/subnet_caches/extended_range_tinyimagenet_25M_1500M.csv" \
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
    --validation_split 0.1 \
    --use_sub_supernet \
    --augmentation mixaug \
    --mix_aug_mode alternating \
    --mixup_alpha 0.4 \
    --cutmix_alpha 1.0 \
    --classifier_dropout 0.0 \
    --label_smoothing 0.0 \
    --lr_cosine \
    --verbose
