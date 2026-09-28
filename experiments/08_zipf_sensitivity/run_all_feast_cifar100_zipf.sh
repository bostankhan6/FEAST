#!/bin/bash
# Run all FEAST Zipf-sensitivity settings sequentially in one job.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 ./experiments/08_zipf_sensitivity/run_all_feast_cifar100_zipf.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for alpha in 0.8 1.0 1.5; do
    echo ""
    echo "=== Zipf sensitivity: alpha=${alpha} ==="
    "$SCRIPT_DIR/run_feast_cifar100_zipf.sh" "$alpha"
done
