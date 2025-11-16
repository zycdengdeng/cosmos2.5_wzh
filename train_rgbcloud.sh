#!/bin/bash

# ============================================================================
# RGBCloud Post-training Script
# ============================================================================
# This script trains the Cosmos Transfer2.5 model with RGBCloud dataset
# using control_input_vis (visibility/blur) control type
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# Number of GPUs to use (REQUIRED: 8 GPUs for multiview with 7 cameras)
export NUM_GPUS=8

# Output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/mnt/zihanw/cosmos-transfer2.5/training_zihanw/output_zihanw}"

# HuggingFace cache directory (optional, set if needed)
# export HF_HOME=/path/to/your/hf/cache

# Master port for distributed training
MASTER_PORT=12341

# Experiment name
EXPERIMENT_NAME="rgbcloud_posttrain"

# ============================================================================
# Logging
# ============================================================================

echo "======================================"
echo "RGBCloud Post-training Configuration"
echo "======================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Output Directory: ${IMAGINAIRE_OUTPUT_ROOT}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Dataset Path: /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
echo "Control Type: control_input_vis"
echo "======================================"
echo ""

# Verify dataset exists
if [ ! -d "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud" ]; then
    echo "❌ ERROR: RGBCloud dataset not found at /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
    echo "Please ensure the dataset is built using build_dataset.sh"
    exit 1
fi

# ============================================================================
# Training Command
# ============================================================================

echo "Starting training..."
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- \
    experiment=${EXPERIMENT_NAME}

# ============================================================================
# Post-training Information
# ============================================================================

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
echo ""
echo "Checkpoint location:"
echo "  ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints/"
echo ""
echo "To convert checkpoint to PyTorch format for inference:"
echo ""
echo "  CHECKPOINT_DIR=\${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints"
echo "  CHECKPOINT_ITER=\$(cat \$CHECKPOINT_DIR/latest_checkpoint.txt)"
echo "  python scripts/convert_distcp_to_pt.py \$CHECKPOINT_DIR/\$CHECKPOINT_ITER/model \$CHECKPOINT_DIR/\$CHECKPOINT_ITER"
echo ""
echo "This will create:"
echo "  - model.pt (full checkpoint)"
echo "  - model_ema_fp32.pt (EMA weights in FP32)"
echo "  - model_ema_bf16.pt (EMA weights in BF16, recommended for inference)"
echo ""
