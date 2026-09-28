#!/bin/bash
set -e

# 1. Find the project root (Go up 2 levels: scripts/ -> data_setup/ -> root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 2. Define where the data should actually go (The root 'data' folder)
TARGET_DIR="$PROJECT_ROOT/data/cifar100"

# 3. Create dir and download
echo "Downloading to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
wget -P "$TARGET_DIR" -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

# 4. Extract
tar -xzvf "$TARGET_DIR/cifar-100-python.tar.gz" -C "$TARGET_DIR"