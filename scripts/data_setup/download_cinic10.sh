#!/bin/bash
set -e

# 1. Robustly find the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 2. Define the target directory
TARGET_DIR="$PROJECT_ROOT/data/cinic10"
mkdir -p "$TARGET_DIR"

echo "Data will be downloaded to: $TARGET_DIR"
cd "$TARGET_DIR"

# 3. Download using the direct link (Fixes the gzip error)
echo "Downloading CINIC-10.tar.gz..."
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz -O CINIC-10.tar.gz

# 4. Extract
echo "Extracting CINIC-10..."
tar -xvzf CINIC-10.tar.gz

# Optional: cleanup
# rm CINIC-10.tar.gz

echo "Done. CINIC-10 is ready at $TARGET_DIR"