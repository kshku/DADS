#!/usr/bin/env bash
# Dataset setup script for DADS
# Initializes submodule, checks dependencies, then runs Python setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/dataset"

echo "=== DADS Dataset Setup ==="

# Step 1: Initialize git submodule
echo ""
echo "[1/2] Initializing git submodule..."
if [ ! -f "$DATASET_DIR/SEP-28k_labels.csv" ]; then
    git -C "$SCRIPT_DIR" submodule update --init --recursive
else
    echo "  Submodule already initialized."
fi

# Step 2: Check ffmpeg
echo ""
echo "[2/2] Checking dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg is not installed."
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    exit 1
fi
echo "  ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"

# Step 3: Run Python setup
echo ""
python3 "$SCRIPT_DIR/setup_dataset.py" "$@"
