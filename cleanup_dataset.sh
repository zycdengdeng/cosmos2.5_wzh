#!/bin/bash
# Cleanup old test data and auxiliary files from datasets

RGBCLOUD_DIR="/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
DEPTHSPARSE_DIR="/mnt/zihanw/cosmos-transfer2.5/datasets/DepthSparse"

echo "=========================================="
echo "Cleaning up RGBCloud dataset..."
echo "=========================================="

# Remove auxiliary directories and files from RGBCloud
if [ -d "$RGBCLOUD_DIR/captions" ]; then
    echo "Removing: $RGBCLOUD_DIR/captions"
    rm -rf "$RGBCLOUD_DIR/captions"
fi

if [ -d "$RGBCLOUD_DIR/control_input_hdmap_bbox" ]; then
    echo "Removing: $RGBCLOUD_DIR/control_input_hdmap_bbox"
    rm -rf "$RGBCLOUD_DIR/control_input_hdmap_bbox"
fi

if [ -d "$RGBCLOUD_DIR/videos" ]; then
    echo "Removing: $RGBCLOUD_DIR/videos"
    rm -rf "$RGBCLOUD_DIR/videos"
fi

if [ -f "$RGBCLOUD_DIR/README.md" ]; then
    echo "Removing: $RGBCLOUD_DIR/README.md"
    rm -f "$RGBCLOUD_DIR/README.md"
fi

echo ""
echo "=========================================="
echo "Cleaning up DepthSparse dataset..."
echo "=========================================="

# Remove auxiliary directories and files from DepthSparse (if any)
if [ -d "$DEPTHSPARSE_DIR/captions" ]; then
    echo "Removing: $DEPTHSPARSE_DIR/captions"
    rm -rf "$DEPTHSPARSE_DIR/captions"
fi

if [ -d "$DEPTHSPARSE_DIR/control_input_hdmap_bbox" ]; then
    echo "Removing: $DEPTHSPARSE_DIR/control_input_hdmap_bbox"
    rm -rf "$DEPTHSPARSE_DIR/control_input_hdmap_bbox"
fi

if [ -d "$DEPTHSPARSE_DIR/videos" ]; then
    echo "Removing: $DEPTHSPARSE_DIR/videos"
    rm -rf "$DEPTHSPARSE_DIR/videos"
fi

if [ -f "$DEPTHSPARSE_DIR/README.md" ]; then
    echo "Removing: $DEPTHSPARSE_DIR/README.md"
    rm -f "$DEPTHSPARSE_DIR/README.md"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
echo ""
echo "Verifying datasets..."
echo ""

# Count valid samples in RGBCloud
RGBCLOUD_COUNT=$(ls -d $RGBCLOUD_DIR/*_seg* 2>/dev/null | wc -l)
echo "RGBCloud samples: $RGBCLOUD_COUNT (expected: 45)"

# Count valid samples in DepthSparse
DEPTHSPARSE_COUNT=$(ls -d $DEPTHSPARSE_DIR/*_seg* 2>/dev/null | wc -l)
echo "DepthSparse samples: $DEPTHSPARSE_COUNT (expected: 45)"

echo ""
echo "Run verify_dataset.py to confirm everything is correct:"
echo "  python /mnt/zihanw/cosmos-transfer2.5/verify_dataset.py"
