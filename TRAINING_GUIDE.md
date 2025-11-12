# Post-Training Guide for Custom Datasets

## Overview

This guide covers post-training Cosmos Transfer2.5 with your custom datasets:
- **RGBCloud**: Colored point cloud projections → uses `control_input_vis`
- **DepthSparse**: Depth sparse point cloud → uses `control_input_depth`

## 📋 Prerequisites

### 1. Dataset Preparation

Ensure both datasets are built and validated:

```bash
# On your server
cd /mnt/zihanw/cosmos-transfer2.5/datasets

# Verify RGBCloud dataset
ls -la RGBCloud/
# Should see: captions/, control_input_hdmap_bbox/, videos/

# Verify DepthSparse dataset
ls -la depthsparse/
# Should see: captions/, control_input_hdmap_bbox/, videos/
```

### 2. Environment Setup

```bash
# Set output directory (important!)
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
# Default is /tmp/imaginaire4-output (may not have enough space)

# Optional: Set HuggingFace cache directory
export HF_HOME=/path/to/your/hf/cache

# Optional: Login to HuggingFace (for downloading base model)
huggingface-cli login
```

### 3. Copy Training Scripts to Server

Upload these files to your server:
```bash
# On your local machine
scp train_rgbcloud.sh train_depthsparse.sh wzh@a100:/path/to/cosmos2.5_wzh/

# On server
chmod +x train_rgbcloud.sh train_depthsparse.sh
```

## 🚀 Training

### Training RGBCloud Dataset

```bash
# On server
cd /path/to/cosmos2.5_wzh

# Start training
./train_rgbcloud.sh
```

**Training Configuration:**
- **Dataset**: `/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud`
- **Control Type**: `control_input_vis` (visibility/blur)
- **GPUs**: 8 (adjust in script if needed)
- **Max Iterations**: 10,000 (adjust in config if needed)
- **Checkpoint Every**: 500 iterations
- **Output**: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/`

### Training DepthSparse Dataset

```bash
# On server
cd /path/to/cosmos2.5_wzh

# Start training
./train_depthsparse.sh
```

**Training Configuration:**
- **Dataset**: `/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse`
- **Control Type**: `control_input_depth`
- **GPUs**: 8 (adjust in script if needed)
- **Max Iterations**: 10,000 (adjust in config if needed)
- **Checkpoint Every**: 500 iterations
- **Output**: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/depthsparse_depth_control/`

## ⚙️ Customization Options

### Adjusting Training Parameters

Edit the experiment configs in:
- `cosmos_transfer2/experiments/multiview/custom_experiments.py`

**Common adjustments:**

```python
trainer=dict(
    max_iter=10_000,      # Total training iterations
    logging_iter=50,       # Log every N iterations
    validation_iter=500,   # Validate every N iterations
),
checkpoint=dict(
    save_iter=500,         # Save checkpoint every N iterations
),
```

### Adjusting Number of GPUs

Edit the training scripts:

```bash
# In train_rgbcloud.sh or train_depthsparse.sh
export NUM_GPUS=4  # Change from 8 to 4, for example
```

### Enabling Weights & Biases (W&B) Logging

1. Set up W&B:
```bash
export WANDB_API_KEY=your_api_key_here
```

2. Remove `job.wandb_mode=disabled` from training scripts

## 📊 Monitoring Training

### View Training Logs

```bash
# Training logs are printed to stdout
# Checkpoint progress:
tail -f ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints/latest_checkpoint.txt
```

### Check GPU Utilization

```bash
# On server
watch -n 1 nvidia-smi
```

### Training Metrics

The training script will log:
- Loss values every 50 iterations
- Iteration speed (it/s)
- GPU memory usage
- Sample generations every 500 iterations

## 🔄 Converting Checkpoints for Inference

After training completes, convert the DCP checkpoint to PyTorch format:

### For RGBCloud Model

```bash
# Set paths
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)

# Convert
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER
```

This creates:
- `model.pt` - Full checkpoint
- `model_ema_fp32.pt` - EMA weights (FP32)
- `model_ema_bf16.pt` - **EMA weights (BF16, recommended)**

### For DepthSparse Model

```bash
# Set paths
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/depthsparse_depth_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)

# Convert
python scripts/convert_distcp_to_pt.py \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINT_DIR/$CHECKPOINT_ITER
```

## 🎯 Inference with Trained Models

### RGBCloud Model Inference

```bash
# Prepare inference spec JSON (similar to multiview_spec.json)
# Make sure to provide RGB colored point cloud projection as control input

NUM_GPUS=8
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/rgbcloud_vis_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)

torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 \
    -m examples.multiview \
    -i your_inference_spec.json \
    -o outputs/rgbcloud_inference \
    --checkpoint_path $CHECKPOINT_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
    --experiment rgbcloud_posttrain
```

### DepthSparse Model Inference

```bash
# Prepare inference spec JSON
# Make sure to provide depth sparse point cloud as control input

NUM_GPUS=8
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/custom_datasets/depthsparse_depth_control/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINT_DIR/latest_checkpoint.txt)

torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 \
    -m examples.multiview \
    -i your_inference_spec.json \
    -o outputs/depthsparse_inference \
    --checkpoint_path $CHECKPOINT_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
    --experiment depthsparse_posttrain
```

## 📁 File Structure Summary

```
cosmos2.5_wzh/
├── cosmos_transfer2/
│   └── experiments/
│       └── multiview/
│           ├── custom_datasets.py      # Dataset configurations
│           ├── custom_experiments.py   # Experiment configurations
│           └── __init__.py             # Module initialization
│
├── train_rgbcloud.sh                   # RGBCloud training script
├── train_depthsparse.sh                # DepthSparse training script
├── TRAINING_GUIDE.md                   # This file
│
└── (On server: /mnt/zihanw/cosmos-transfer2.5/)
    └── datasets/
        ├── RGBCloud/                   # Your RGBCloud dataset
        └── depthsparse/                # Your DepthSparse dataset
```

## 🔧 Troubleshooting

### Out of Memory

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size (currently set to 1)
2. Reduce number of GPUs and use FSDP sharding
3. Reduce `num_frames` in dataset config

### Dataset Not Found

**Symptoms**: Error about missing dataset directory

**Solutions**:
1. Verify dataset path in `custom_datasets.py`
2. Ensure datasets are built using build scripts
3. Check file permissions

### Slow Training

**Symptoms**: Very low iteration speed

**Solutions**:
1. Check if data is on slow storage (use local SSD if possible)
2. Increase `num_workers` in dataloader (currently 8)
3. Increase `prefetch_factor` (currently 2)

### Checkpoint Loading Failed

**Symptoms**: Cannot load pretrained checkpoint

**Solutions**:
1. Ensure HuggingFace login is done: `huggingface-cli login`
2. Check network connection for downloading checkpoint
3. Set `HF_HOME` to a directory with enough space

## 📝 Training Configuration Details

### RGBCloud Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Dataset** | `/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud` | RGBCloud dataset path |
| **Control Type** | `control_input_vis` | Visibility/blur control |
| **Hint Keys** | `"vis"` | Control type identifier |
| **Resolution** | 720p | Video resolution |
| **Video Size** | 704x1280 | Actual video dimensions |
| **State T** | 8 | Temporal latent dimension |
| **Num Frames** | 29 | Number of frames per clip |
| **Cameras** | 7 | Number of camera views |

### DepthSparse Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Dataset** | `/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse` | DepthSparse dataset path |
| **Control Type** | `control_input_depth` | Depth control |
| **Hint Keys** | `"depth"` | Control type identifier |
| **Resolution** | 720p | Video resolution |
| **Video Size** | 704x1280 | Actual video dimensions |
| **State T** | 8 | Temporal latent dimension |
| **Num Frames** | 29 | Number of frames per clip |
| **Cameras** | 7 | Number of camera views |

## 🎓 Understanding Control Types

### What is `control_input_vis`?

- **Original Purpose**: Visibility/blur control
- **Your Use**: Colored RGB point cloud projections
- **Why it works**: RGB color information is similar to visibility/appearance guidance

### What is `control_input_depth`?

- **Original Purpose**: Depth map control
- **Your Use**: Depth sparse point cloud projections
- **Why it works**: Directly matches depth information

### Can I Use Both Models Together?

**Yes!** After training both models:
- RGBCloud model handles `control_input_vis`
- DepthSparse model handles `control_input_depth`
- You could theoretically combine them for multi-control inference

## ⏱️ Estimated Training Time

**Assumptions**: 8x A100 GPUs, 5 sequences, 7 cameras each

| Dataset | Iterations | Estimated Time |
|---------|-----------|----------------|
| RGBCloud | 10,000 | ~6-12 hours |
| DepthSparse | 10,000 | ~6-12 hours |

**Note**: Actual time depends on:
- GPU model and memory
- Data loading speed
- Network bandwidth (if using remote storage)
- System configuration

## 📚 Next Steps

1. ✅ **Prepare datasets** using build scripts
2. ✅ **Configure environment** (set `IMAGINAIRE_OUTPUT_ROOT`)
3. ✅ **Start training** using provided scripts
4. ⏳ **Monitor progress** and adjust if needed
5. ⏳ **Convert checkpoints** after training
6. ⏳ **Run inference** with trained models

## 💡 Tips

- **Start small**: Try training for 1000 iterations first to verify everything works
- **Monitor GPU memory**: Use `nvidia-smi` to ensure GPUs are fully utilized
- **Save checkpoints frequently**: Storage is cheaper than re-training
- **Use BF16 for inference**: `model_ema_bf16.pt` is faster and uses less memory
- **Compare with baseline**: Run inference with original pretrained model first

---

**Created**: 2025-11-12
**Version**: 1.0
**For**: RGBCloud and DepthSparse post-training
