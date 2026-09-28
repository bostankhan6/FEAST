#!/bin/bash
# Generate the LOW-END segment of the canonical FEAST subnet cache: 25M-75M
# MACs, 6 subnets, rho0=0.40 (relaxed from 0.31 — the tighter ceiling has no
# feasible architecture at this scale). Paired with generate_cache_main_segment.sh
# (50M-600M) and merged by merge_extended_caches.py into the final
# extended_range_25M_1500M.csv (see build_canonical_cache.sh for the full
# pipeline).
#
# Client budgets extend up to 1500M (device capability axis); training caps
# subnet size at 600M, so clients with budgets >600M train the same max
# subnet as 600M clients — the cache itself never needs subnets above ~596M.
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
OUTPUT_CSV_LOW="$PROJECT_ROOT/subnet_caches/extended_range_low_10M_50M.csv"

mkdir -p "$(dirname "$OUTPUT_CSV_LOW")"

echo "========================================================================"
echo "Generating Extended Range Subnet Cache (Low-End: 25M-75M)"
echo "Root:   $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_CSV_LOW"
echo "========================================================================"
echo "Note: Extended range adjusted to [25M, 75M] for constraint feasibility."
echo "Range still covers: Jetson Nano (~50M) and lower-mid devices (~75M)."
echo "Provides ~6 subnets to extend below existing cache minimum (50M)."
echo ""

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Generate low-end subnets [25M, 75M) — below existing cache minimum of 50M
./venv/bin/python "$PYTHON_SCRIPT" \
    --arch_config_path "$CONFIG_PATH" \
    --output_csv "$OUTPUT_CSV_LOW" \
    --bounds_mode relative \
    --sampling_mode equidistant \
    --macs_lower_bound 25000000 \
    --macs_upper_bound 75000000 \
    --num_samples 6 \
    --rho0_constraint 0.40 \
    --ga_pop_size 256 \
    --ga_generations 512 \
    --save_interval 10

echo ""
echo "========================================================================"
echo "LOW-END CACHE GENERATED!"
echo "Output: $OUTPUT_CSV_LOW"
echo ""
echo "Next: run generate_cache_main_segment.sh (if not already done), then"
echo "merge_extended_caches.py — or just run build_canonical_cache.sh for the"
echo "full pipeline in one step."
echo "========================================================================"
