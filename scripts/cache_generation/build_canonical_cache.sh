#!/bin/bash
# Build the canonical FEAST subnet cache (subnet_caches/extended_range_25M_1500M.csv)
# from scratch: main segment + low-end segment + merge.
#
# 56 subnets total, 24.96M-595.67M MACs. Config: 4-stage-supernet-cifar100-v2.json
# for both segments; only the target MAC range and rho0 ceiling differ. See
# subnet_caches/README.md for the provenance of each merged segment.
#
#   Main segment:     50M-600M,  50 subnets, rho0=0.31 (generate_cache_main_segment.sh)
#   Low-end segment:  25M-75M,    6 subnets, rho0=0.40 (generate_cache_low_end_segment.sh)
#   Merge:            merge_extended_caches.py, sorted by MACs
#
# Usage:
#   bash scripts/cache_generation/build_canonical_cache.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

MAIN_CSV="$PROJECT_ROOT/subnet_caches/8_blocks_supernet_cache_alpha_weights_all_1s_rho_0-31.csv"
LOW_END_CSV="$PROJECT_ROOT/subnet_caches/extended_range_low_10M_50M.csv"
OUTPUT_CSV="$PROJECT_ROOT/subnet_caches/extended_range_25M_1500M.csv"

echo "========================================================================"
echo "Step 1/3: Generating main segment (50M-600M, 50 subnets, rho0=0.31)"
echo "========================================================================"
bash "$SCRIPT_DIR/generate_cache_main_segment.sh"

echo ""
echo "========================================================================"
echo "Step 2/3: Generating low-end segment (25M-75M, 6 subnets, rho0=0.40)"
echo "========================================================================"
bash "$SCRIPT_DIR/generate_cache_low_end_segment.sh"

echo ""
echo "========================================================================"
echo "Step 3/3: Merging into canonical cache"
echo "========================================================================"
"$PROJECT_ROOT/venv/bin/python" "$SCRIPT_DIR/merge_extended_caches.py" \
    --low_end "$LOW_END_CSV" \
    --original "$MAIN_CSV" \
    --output "$OUTPUT_CSV"

echo ""
echo "========================================================================"
echo "DONE! Canonical cache saved to: $OUTPUT_CSV"
echo "========================================================================"
