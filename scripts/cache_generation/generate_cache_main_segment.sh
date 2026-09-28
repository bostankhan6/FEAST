#!/bin/bash
# Generate the MAIN segment of the canonical FEAST subnet cache: 50M-600M MACs,
# 50 subnets, rho0=0.31. Paired with generate_cache_low_end_segment.sh
# (25M-75M) and merged by merge_extended_caches.py into the final
# extended_range_25M_1500M.csv (see build_canonical_cache.sh for the full
# pipeline).
#
# Config: 4-stage-supernet-cifar100-v2.json
#   - max_extra_blocks_per_stage=8, base_ch=[128,256,512,1024]
#   - beta_depth_penalty=10.0, n_classes=100
#   - Stage-level expansion ratios (4 elements per subnet)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

PYTHON_SCRIPT="$PROJECT_ROOT/src/feast/nas/generate_subnet_cache.py"
CONFIG_PATH="$PROJECT_ROOT/configs/supernets/4-stage-supernet-cifar100-v2.json"
OUTPUT_CSV="$PROJECT_ROOT/subnet_caches/8_blocks_supernet_cache_alpha_weights_all_1s_rho_0-31.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "========================================================================"
echo "Generating Main Segment Subnet Cache (50M-600M, cifar100-v2 config)"
echo "Root:   $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_CSV"
echo "========================================================================"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

./venv/bin/python "$PYTHON_SCRIPT" \
    --arch_config_path "$CONFIG_PATH" \
    --output_csv "$OUTPUT_CSV" \
    --bounds_mode relative \
    --sampling_mode equidistant \
    --macs_lower_bound 50000000 \
    --macs_upper_bound 600000000 \
    --num_samples 50 \
    --rho0_constraint 0.31 \
    --ga_pop_size 256 \
    --ga_generations 512 \
    --save_interval 10

echo ""
echo "========================================================================"
echo "DONE! Cache saved to: $OUTPUT_CSV"
echo "========================================================================"
