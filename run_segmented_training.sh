#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Quick launch script for segmented multiview training

# Set default values
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-12340}
MODE=${1:-"debug"}

echo "========================================="
echo "Segmented Multiview Training Launcher"
echo "========================================="
echo "Mode: $MODE"
echo "GPUs: $NUM_GPUS"
echo "Port: $MASTER_PORT"
echo ""

case $MODE in
  "debug")
    echo "Running DEBUG mode (100 iterations)..."
    EXPERIMENT="segmented_multiview_720p_depth_debug"
    ;;
  "720p")
    echo "Running FULL 720p training (30 hours, 108k iterations)..."
    EXPERIMENT="segmented_multiview_720p_depth_30h"
    ;;
  "480p")
    echo "Running FULL 480p training (30 hours, 108k iterations)..."
    EXPERIMENT="segmented_multiview_480p_depth_30h"
    ;;
  *)
    echo "ERROR: Unknown mode '$MODE'"
    echo ""
    echo "Usage: $0 [mode]"
    echo ""
    echo "Available modes:"
    echo "  debug  - Run 100 iterations for testing (default)"
    echo "  720p   - Run full 30-hour training at 720p"
    echo "  480p   - Run full 30-hour training at 480p"
    echo ""
    echo "Example:"
    echo "  $0 debug           # Test configuration"
    echo "  $0 720p            # Full training"
    echo "  NUM_GPUS=4 $0 720p # Use 4 GPUs"
    exit 1
    ;;
esac

echo "Experiment: $EXPERIMENT"
echo ""
echo "Starting in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Launch training
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=$EXPERIMENT

echo ""
echo "Training completed!"
