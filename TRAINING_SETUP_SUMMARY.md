# Cosmos Transfer2.5 Custom Training Setup Summary

## Overview

This document summarizes the complete setup for post-training Cosmos Transfer2.5 with custom datasets for **RGBCloud** (vis control) and **DepthSparse** (depth control).

## Configuration Architecture

### The Challenge We Solved

The official multiview example only supports `hdmap_bbox` control type. However, we need to train:
- **RGBCloud dataset** → `vis` (visibility) control branch
- **DepthSparse dataset** → `depth` control branch

### The Solution: Custom Conditioners

We created custom conditioners that:
1. **Read** control images from `control_input_hdmap_bbox/` directory (dataset level)
2. **Map** them to the appropriate control branch - `vis` or `depth` (model level)

This allows us to use the existing dataset structure while training the correct control branches.

## File Structure

```
cosmos_transfer2/experiments/multiview/
├── __init__.py                    # Registers all custom configurations
├── custom_conditioners.py         # Custom conditioner mappings (NEW)
├── custom_datasets.py             # Dataset loaders for RGBCloud and DepthSparse
└── custom_experiments.py          # Experiment configurations

datasets/
├── RGBCloud/                      # RGB colored point cloud dataset
│   ├── captions/
│   ├── control_input_hdmap_bbox/  # Actually contains vis control images
│   └── videos/
└── depthsparse/                   # Depth sparse point cloud dataset
    ├── captions/
    ├── control_input_hdmap_bbox/  # Actually contains depth control images
    └── videos/
```

## Key Configuration Mappings

### RGBCloud Training Flow

```
Dataset directory: control_input_hdmap_bbox/
         ↓ (dataset reads with hint_key="control_input_hdmap_bbox")
Custom Conditioner: CustomMultiViewVisConditioner
         ↓ (remaps to control_input_vis)
Model: hint_keys="vis"
         ↓
Trains the VIS control branch
```

### DepthSparse Training Flow

```
Dataset directory: control_input_hdmap_bbox/
         ↓ (dataset reads with hint_key="control_input_hdmap_bbox")
Custom Conditioner: CustomMultiViewDepthConditioner
         ↓ (remaps to control_input_depth)
Model: hint_keys="depth"
         ↓
Trains the DEPTH control branch
```

## Configuration Details

### 1. Custom Conditioners (`custom_conditioners.py`)

**CustomMultiViewVisConditioner:**
- Filters out all control types except vis
- Maps `control_input_hdmap_bbox` → `control_input_vis`
- Includes multiview-specific fields

**CustomMultiViewDepthConditioner:**
- Filters out all control types except depth
- Maps `control_input_hdmap_bbox` → `control_input_depth`
- Includes multiview-specific fields

### 2. Custom Datasets (`custom_datasets.py`)

**get_rgbcloud_multiview_dataset:**
```python
dataset_dir: "/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud"
hint_key: "control_input_hdmap_bbox"  # Directory to read from
```

**get_depthsparse_multiview_dataset:**
```python
dataset_dir: "/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse"
hint_key: "control_input_hdmap_bbox"  # Directory to read from
```

### 3. Custom Experiments (`custom_experiments.py`)

**rgbcloud_posttrain:**
```python
defaults:
  - /experiment/cosmos_multiview_control_transfer_v2p5
  - override /data_train: rgbcloud_multiview_train
  - override /conditioner: custom_multiview_vis_conditioner

model.config.hint_keys: "vis"  # Train vis control branch
context_parallel_size: 2       # Must match NUM_GPUS
```

**depthsparse_posttrain:**
```python
defaults:
  - /experiment/cosmos_multiview_control_transfer_v2p5
  - override /data_train: depthsparse_multiview_train
  - override /conditioner: custom_multiview_depth_conditioner

model.config.hint_keys: "depth"  # Train depth control branch
context_parallel_size: 2         # Must match NUM_GPUS
```

## Training Configuration

### GPU Settings
- **NUM_GPUS**: 2 (using 2 out of 8 available GPUs)
- **context_parallel_size**: 2 (must match NUM_GPUS)

### Training Parameters
- **Batch size**: 1 per GPU
- **Max iterations**: 10,000 (adjust based on dataset size)
- **Checkpoint save**: Every 500 iterations
- **Logging**: Every 50 iterations
- **Validation**: Disabled (run_validation=False)

### Output Configuration
- **S3 saving**: Disabled (all callbacks set save_s3=False)
- **Local output**: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/`
  - RGBCloud: `rgbcloud_vis_control/`
  - DepthSparse: `depthsparse_depth_control/`

## How to Run Training

### 1. RGBCloud Training (vis control)

```bash
./train_rgbcloud.sh
```

This will:
- Load the base Cosmos Transfer2.5 checkpoint
- Train the **vis control branch** with RGB colored point cloud data
- Save checkpoints to `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/`

### 2. DepthSparse Training (depth control)

```bash
./train_depthsparse.sh
```

This will:
- Load the base Cosmos Transfer2.5 checkpoint
- Train the **depth control branch** with depth sparse point cloud data
- Save checkpoints to `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/depthsparse_depth_control/`

## After Training: Checkpoint Conversion

After training completes, convert the distributed checkpoint to PyTorch format:

```bash
# For RGBCloud
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/$CHECKPOINT_ITER/model $CHECKPOINT_DIR/$CHECKPOINT_ITER

# For DepthSparse
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/depthsparse_depth_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/$CHECKPOINT_ITER/model $CHECKPOINT_DIR/$CHECKPOINT_ITER
```

This creates:
- `model.pt` - Full checkpoint
- `model_ema_fp32.pt` - EMA weights in FP32
- `model_ema_bf16.pt` - EMA weights in BF16 (recommended for inference)

## Verification

All custom configurations are properly registered:

```python
# Custom conditioners registered in ConfigStore
- conditioner/custom_multiview_vis_conditioner
- conditioner/custom_multiview_depth_conditioner

# Custom datasets registered in ConfigStore
- data_train/rgbcloud_multiview_train
- data_train/depthsparse_multiview_train

# Custom experiments registered in ConfigStore
- experiment/rgbcloud_posttrain
- experiment/depthsparse_posttrain
```

## Important Notes

1. **Directory Naming**: Dataset directories must be named `control_input_hdmap_bbox/` even though we're training different control types. The custom conditioner handles the remapping.

2. **GPU Configuration**: `NUM_GPUS` in training scripts must match `context_parallel_size` in experiment configs. Both are currently set to 2.

3. **Control Type Distinction**:
   - `hint_key` (dataset level): Which directory to read from
   - `hint_keys` (model level): Which control branch to train

4. **Complete Model Output**: Training produces a complete Cosmos Transfer2.5 model with the trained control branch, not just a ControlNet head.

5. **Checkpoint Format**: Training produces distributed checkpoints (DCP format) which must be converted to `.pt` format for inference.

## Troubleshooting

If you encounter `MissingConfigException`:
- Verify all files in `cosmos_transfer2/experiments/multiview/` are present
- Ensure `__init__.py` calls `register_custom_conditioners()`
- Check that the Python environment has all dependencies installed

If you encounter `world_size` errors:
- Verify `NUM_GPUS` matches `context_parallel_size`
- Check `CUDA_VISIBLE_DEVICES` if set

If you encounter `KeyError: 'hdmap_bbox'` or control type errors:
- Verify dataset `hint_key="control_input_hdmap_bbox"`
- Verify directory name is `control_input_hdmap_bbox/`
- Verify custom conditioner is being used in experiment config
