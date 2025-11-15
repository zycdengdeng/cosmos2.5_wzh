# Segmented Multiview Training Configuration

这个配置用于训练基于分段多视图视频数据集的Cosmos Transfer模型。

## 数据结构

数据按照以下结构组织：

```
/mnt/zihanw/cosmos-transfer2.5/data_prepa/
├── GT/
│   └── {scene_id}_GT_90frames_1280x720/
│       └── GT_segments/
│           ├── FL/
│           │   ├── FL_GT_seg01.mp4
│           │   ├── FL_GT_seg02.mp4
│           │   └── ...
│           ├── FR/
│           ├── RL/
│           ├── FN/
│           ├── RR/
│           ├── FW/
│           └── RN/
└── guidence/
    └── {scene_id}_90frames_1280x720/
        ├── color_segments/
        │   ├── FL/
        │   │   ├── FL_color_seg01.mp4
        │   │   ├── FL_color_seg02.mp4
        │   │   └── ...
        │   └── ...
        └── depth_segments/
            ├── FL/
            │   ├── FL_depth_seg01.mp4
            │   ├── FL_depth_seg02.mp4
            │   └── ...
            └── ...
```

### 场景列表
- 002
- 004
- 006
- 008
- 009

### 相机列表
- FL (Front Left)
- FR (Front Right)
- RL (Rear Left)
- FN (Front Normal) - 主前视相机
- RR (Rear Right)
- FW (Front Wide)
- RN (Rear Normal)

## 训练配置

### 主要训练设置 (30小时)

配置名称: `segmented_multiview_720p_depth_30h`

**训练参数:**
- 总训练时间: 30小时
- 每个文件处理时间: ~1秒
- 总迭代次数: 108,000
- 分辨率: 720p (720x1280)
- 控制输入: Depth (深度图)
- Batch size: 1

**Checkpoint保存:**
- 保存频率: 每18,000次迭代 (约5小时)
- 总保存次数: 6个checkpoints
- 保存位置:
  - iter_018000 (5小时)
  - iter_036000 (10小时)
  - iter_054000 (15小时)
  - iter_072000 (20小时)
  - iter_090000 (25小时)
  - iter_108000 (30小时, 最终)

**日志和验证:**
- 日志频率: 每500次迭代 (~8分钟)
- 验证频率: 每5,000次迭代 (~1.4小时)
- 采样频率: 每5,000次迭代

## 使用方法

### 1. 调试模式（测试配置）

```bash
torchrun --nproc_per_node=8 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=segmented_multiview_720p_depth_debug
```

调试配置会运行100次迭代，用于快速验证配置是否正确。

### 2. 完整30小时训练 (720p)

```bash
torchrun --nproc_per_node=8 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=segmented_multiview_720p_depth_30h
```

### 3. 480p训练（更快的训练速度）

```bash
torchrun --nproc_per_node=8 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=segmented_multiview_480p_depth_30h
```

### 4. 自定义数据路径

如果数据在不同位置，可以覆盖配置：

```bash
torchrun --nproc_per_node=8 --master_port=12340 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=segmented_multiview_720p_depth_30h \
  dataloader_train.dataset.data_root=/your/custom/path
```

## 文件说明

### 核心文件

1. **Dataset类**: `cosmos_transfer2/_src/transfer2/datasets/local_datasets/segmented_multiview_dataset.py`
   - 实现了分段视频的加载逻辑
   - 处理GT和guidance视频的配对
   - 支持多相机配置

2. **Dataloader配置**: `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/defaults/segmented_dataloader.py`
   - 注册720p和480p dataloader
   - 配置相机到视图ID的映射
   - 设置数据加载参数

3. **实验配置**: `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiment/segmented_multiview_training.py`
   - 定义30小时训练参数
   - 配置checkpoint保存策略
   - 提供debug和不同分辨率版本

4. **主配置**: `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py`
   - 已修改以注册新的dataloader

## 数据集统计

假设每个场景每个相机有9个segment:
- 总场景数: 5
- 总相机数: 7
- 每相机segments: ~9
- **总样本数**: ~315

训练/验证分割: 90%/10%
- 训练样本: ~283
- 验证样本: ~32

## 性能估算

基于1秒/文件的处理时间:
- 每个epoch时间: ~283秒 (~4.7分钟)
- 30小时内epochs数: ~382个epochs
- 每个样本会被训练约382次

## 监控训练

训练过程中，可以通过以下方式监控：

1. **日志输出**: 每500次迭代会打印训练损失
2. **Wandb**: 如果配置了wandb，可以在web界面查看
3. **Checkpoints**: 检查保存的checkpoints在 `{job.path_local}/checkpoints/`

## 故障排除

### 问题1: 找不到数据文件
- 检查数据路径是否正确
- 验证数据结构符合上述格式
- 查看日志中的警告信息

### 问题2: 内存不足
- 减少batch_size
- 减少num_workers
- 使用480p配置而非720p

### 问题3: 训练速度慢
- 检查`persistent_workers=True`是否设置
- 确保使用SSD而非HDD存储数据
- 增加`prefetch_factor`

## 修改配置

### 更改checkpoint保存频率

在experiment配置中修改`save_iter`:

```python
checkpoint=dict(
    save_iter=10000,  # 改为每10,000次迭代保存
    ...
)
```

### 更改总训练时间

修改`max_iter`:

```python
trainer=dict(
    max_iter=216000,  # 60小时训练
    ...
)
```

### 添加更多控制输入

修改dataset的`hint_key`:

```python
# 在segmented_dataloader.py中
hint_key="control_input_depth_color",  # 同时使用depth和color
```

## 联系

如有问题，请查看代码注释或联系开发团队。
