#!/bin/bash
set -e

# Post-hoc single-subnet search at a target MAC budget, matching the
# post-hoc extraction protocol (population 512, 512 generations,
# mutation probability 0.3, rho0=0.40 below 50M MACs / 0.31 at or above).
#
# Usage:
#   ./run_single_subnet_search.sh <target_macs> [--output_json PATH] [extra search_single_subnet.py args]
#
# Example:
#   ./run_single_subnet_search.sh 30e6
#   ./run_single_subnet_search.sh 500e6 --output_json subnet_caches/posthoc/500M.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

PYTHON_SCRIPT="$PROJECT_ROOT/src/feast/nas/search_single_subnet.py"
CONFIG_PATH="$PROJECT_ROOT/configs/supernets/4-stage-supernet-cifar100-v2.json"

TARGET_MACS="${1:?Usage: $0 <target_macs, e.g. 30e6 or 500e6> [extra args]}"
shift

OUTPUT_JSON="$PROJECT_ROOT/subnet_caches/posthoc/target_${TARGET_MACS}.json"

# rho0 threshold matches the cache-construction segments exactly:
# generate_cache_low_end_segment.sh (rho0=0.40, <50M) and
# generate_cache_main_segment.sh (rho0=0.31, >=50M).
RHO0=0.31
if awk "BEGIN { exit !($TARGET_MACS < 50000000) }"; then
    RHO0=0.40
fi

echo "========================================================================"
echo "Post-hoc single-subnet search"
echo "Root:      $PROJECT_ROOT"
echo "Config:    $CONFIG_PATH"
echo "Target:    $TARGET_MACS MACs"
echo "rho0:      $RHO0"
echo "Output:    $OUTPUT_JSON"
echo "========================================================================"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python "$PYTHON_SCRIPT" \
    --arch_config_path "$CONFIG_PATH" \
    --target_macs "$TARGET_MACS" \
    --rho0_constraint "$RHO0" \
    --ga_pop_size 512 \
    --ga_generations 512 \
    --ga_mutate_p 0.3 \
    --seed 42 \
    --output_json "$OUTPUT_JSON" \
    "$@"
