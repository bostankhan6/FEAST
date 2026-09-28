#!/bin/bash
# Run all three Dirichlet-sensitivity settings sequentially on one GPU.
#
# Usage:
#   ./experiments/09_dirichlet_sensitivity/run_all_feast_cifar100_dirichlet.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/FEAST_cifar100_dir_005.sh"
"$SCRIPT_DIR/FEAST_cifar100_dir_03.sh"
"$SCRIPT_DIR/FEAST_cifar100_dir_10.sh"
