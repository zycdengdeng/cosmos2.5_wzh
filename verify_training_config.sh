#!/bin/bash

# ============================================================================
# Configuration Verification Script
# ============================================================================
# This script verifies that custom configurations are properly registered
# ============================================================================

echo "======================================"
echo "Verifying Configuration Registration"
echo "======================================"
echo ""

# Test if configurations can be loaded
echo "Testing RGBCloud configuration..."
python -c "
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.config import make_config
from hydra.core.config_store import ConfigStore
from hydra import initialize_config_dir, compose
import os

# Get config directory
config_dir = os.path.join(os.getcwd(), 'cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer')

# Initialize Hydra
with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
    # Try to compose config with rgbcloud_posttrain
    try:
        cfg = compose(config_name='config', overrides=['experiment=rgbcloud_posttrain'])
        print('✓ RGBCloud configuration loaded successfully!')
        print(f'  - Experiment name: {cfg.job.name}')
        print(f'  - Dataset path: {cfg.dataloader_train.dataset.dataset_dir}')
        print(f'  - Control type: {cfg.dataloader_train.dataset.hint_key}')
    except Exception as e:
        print(f'✗ Failed to load RGBCloud configuration: {e}')
        exit(1)
" || exit 1

echo ""
echo "Testing DepthSparse configuration..."
python -c "
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.config import make_config
from hydra.core.config_store import ConfigStore
from hydra import initialize_config_dir, compose
import os

# Get config directory
config_dir = os.path.join(os.getcwd(), 'cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer')

# Initialize Hydra
with initialize_config_dir(config_dir=config_dir, version_base='1.1'):
    # Try to compose config with depthsparse_posttrain
    try:
        cfg = compose(config_name='config', overrides=['experiment=depthsparse_posttrain'])
        print('✓ DepthSparse configuration loaded successfully!')
        print(f'  - Experiment name: {cfg.job.name}')
        print(f'  - Dataset path: {cfg.dataloader_train.dataset.dataset_dir}')
        print(f'  - Control type: {cfg.dataloader_train.dataset.hint_key}')
    except Exception as e:
        print(f'✗ Failed to load DepthSparse configuration: {e}')
        exit(1)
" || exit 1

echo ""
echo "======================================"
echo "✓ All configurations verified!"
echo "======================================"
echo ""
echo "Available experiments:"
echo "  - rgbcloud_posttrain"
echo "  - depthsparse_posttrain"
echo ""
echo "You can now run the training scripts:"
echo "  ./train_rgbcloud.sh"
echo "  ./train_depthsparse.sh"
echo ""
