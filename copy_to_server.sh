#!/bin/bash

# ============================================================================
# Copy Custom Configuration Files to Server
# ============================================================================
# This script copies the custom conditioner, dataset, and experiment files
# from the git repository to the server training directory
# ============================================================================

set -e

# Source directory (git repo)
SRC_DIR="/home/user/cosmos2.5_wzh"

# Target directory (server)
TARGET_DIR="/mnt/zihanw/cosmos-transfer2.5"

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "❌ ERROR: Source directory not found: $SRC_DIR"
    exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ ERROR: Target directory not found: $TARGET_DIR"
    echo "Please ensure the server training directory exists."
    exit 1
fi

echo "======================================"
echo "Copying Custom Configuration Files"
echo "======================================"
echo "Source: $SRC_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR/cosmos_transfer2/experiments/multiview"

# Copy files
echo "Copying files..."
echo ""

echo "1. Copying __init__.py..."
cp "$SRC_DIR/cosmos_transfer2/experiments/multiview/__init__.py" \
   "$TARGET_DIR/cosmos_transfer2/experiments/multiview/__init__.py"
echo "   ✓ Done"

echo "2. Copying custom_conditioners.py (NEW FILE)..."
cp "$SRC_DIR/cosmos_transfer2/experiments/multiview/custom_conditioners.py" \
   "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_conditioners.py"
echo "   ✓ Done"

echo "3. Copying custom_datasets.py..."
cp "$SRC_DIR/cosmos_transfer2/experiments/multiview/custom_datasets.py" \
   "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_datasets.py"
echo "   ✓ Done"

echo "4. Copying custom_experiments.py..."
cp "$SRC_DIR/cosmos_transfer2/experiments/multiview/custom_experiments.py" \
   "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_experiments.py"
echo "   ✓ Done"

echo ""
echo "======================================"
echo "Verifying Files"
echo "======================================"
echo ""

# Verify files exist
if [ -f "$TARGET_DIR/cosmos_transfer2/experiments/multiview/__init__.py" ]; then
    echo "✓ __init__.py exists"
else
    echo "❌ __init__.py NOT FOUND"
    exit 1
fi

if [ -f "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_conditioners.py" ]; then
    echo "✓ custom_conditioners.py exists"
else
    echo "❌ custom_conditioners.py NOT FOUND"
    exit 1
fi

if [ -f "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_datasets.py" ]; then
    echo "✓ custom_datasets.py exists"
else
    echo "❌ custom_datasets.py NOT FOUND"
    exit 1
fi

if [ -f "$TARGET_DIR/cosmos_transfer2/experiments/multiview/custom_experiments.py" ]; then
    echo "✓ custom_experiments.py exists"
else
    echo "❌ custom_experiments.py NOT FOUND"
    exit 1
fi

echo ""
echo "======================================"
echo "Files copied successfully!"
echo "======================================"
echo ""
echo "You can now run training:"
echo "  cd $TARGET_DIR"
echo "  ./train_rgbcloud.sh"
echo ""
