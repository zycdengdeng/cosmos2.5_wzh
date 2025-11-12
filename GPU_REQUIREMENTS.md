# GPU Requirements for Multiview Training

## Critical Requirement

**For 7-camera multiview training, you MUST use 8 GPUs.**

This is not a configuration choice - it's an architectural requirement of the Cosmos Transfer2.5 multiview model.

## Why 8 GPUs?

The multiview model has a hard constraint in the code:

```python
# From multiview_vid2vid_model_control_vace_rectified_flow.py:116
assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
```

Where:
- **n_views** = number of camera views in your dataset (7 cameras)
- **cp_size** = context_parallel_size = number of GPUs

**Requirement**: `context_parallel_size > n_views`

Therefore:
- 7 cameras → need `cp_size ≥ 8` → **need 8 GPUs minimum**

## Error You'll See with 2 GPUs

If you try to train with only 2 GPUs:

```
AssertionError: n_views must be less than cp_size, got n_views=7 and cp_size=2
```

This error occurs during the encoding step when the model tries to process the 7 camera views across the parallel context.

## Why This Limitation?

The context parallel mechanism distributes the camera views across GPUs for efficient processing. The architecture requires:

1. Each camera view needs processing capacity
2. Context parallel distributes views across GPUs
3. The implementation requires `cp_size > n_views` for proper tensor distribution

## Configuration Summary

### What's Fixed:
```bash
# training scripts
NUM_GPUS=8  # REQUIRED for 7 cameras

# custom_experiments.py
context_parallel_size=8  # MUST be > n_views (7)
```

### What You Cannot Change:
- Number of cameras in dataset: 7 (fixed by your data)
- Minimum GPUs required: 8 (architectural constraint)

## Alternative: Reduce Camera Views (Not Recommended)

If you absolutely cannot use 8 GPUs, you would need to:

1. **Reduce to fewer camera views** (e.g., 4 or 5 cameras)
2. Rebuild your dataset with fewer cameras
3. Use fewer GPUs (e.g., 2 GPUs for 1 camera view only)

**However, this is NOT recommended because:**
- Multiview training benefits from multiple perspectives
- Your control inputs are designed for 7-camera setup
- Reducing cameras significantly impacts model quality

## Recommended Setup

**Use all 8 available GPUs:**

```bash
# Check available GPUs
nvidia-smi

# Training will use all 8 GPUs
cd /mnt/zihanw/cosmos-transfer2.5/
./train_rgbcloud.sh
```

This provides:
- ✅ Meets architectural requirements
- ✅ Full 7-camera multiview support
- ✅ Optimal training performance
- ✅ Best model quality

## Memory Requirements

Each GPU should have sufficient VRAM:
- **Minimum**: 40GB per GPU (A100 recommended)
- **Batch size**: 1 per GPU (total batch size = 8)
- **Mixed precision**: BF16 used to reduce memory

If you encounter OOM (Out of Memory) errors, check:
1. GPU memory usage: `nvidia-smi`
2. Reduce max_iter or checkpoint frequency if needed
3. Ensure no other processes are using GPU memory

## Summary

**Bottom line**: For your 7-camera multiview setup, **you must use 8 GPUs**. This is a hard requirement, not a configuration option.
