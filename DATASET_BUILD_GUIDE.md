# Cosmos Transfer2.5 数据集构建指南

## 概述

本指南帮助你将现有的点云投影数据（guidance）和 ground truth 视频组织成 Cosmos Transfer2.5 官方训练格式。

## 数据说明

### 源数据位置

**Guidance（彩色点云投影）:**
```
/mnt/zihanw/cosmos-transfer2.5/data_prepa/guidence/
├── 002_90frames_1280x720/color/
├── 004_90frames_1280x720/color/
├── 006_90frames_1280x720/color/
├── 008_90frames_1280x720/color/
└── 009_90frames_1280x720/color/
    ├── FL_color.mp4   # Front Left
    ├── FN_color.mp4   # Front Narrow/Tele
    ├── FR_color.mp4   # Front Right
    ├── FW_color.mp4   # Front Wide
    ├── RL_color.mp4   # Rear Left
    ├── RN_color.mp4   # Rear Narrow/Tele
    └── RR_color.mp4   # Rear Right
```

**Ground Truth:**
```
/mnt/zihanw/cosmos-transfer2.5/data_prepa/GT/
├── 002_GT_90frames_1280x720/
├── 004_GT_90frames_1280x720/
├── 006_GT_90frames_1280x720/
├── 008_GT_90frames_1280x720/
└── 009_GT_90frames_1280x720/
    ├── FL_GT.mp4
    ├── FN_GT.mp4
    ├── FR_GT.mp4
    ├── FW_GT.mp4
    ├── RL_GT.mp4
    ├── RN_GT.mp4
    └── RR_GT.mp4
```

### 目标数据集结构

```
/mnt/zihanw/cosmos-transfer2.5/datasets/self_Data_Training/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       ├── 002.json
│       ├── 004.json
│       ├── 006.json
│       ├── 008.json
│       └── 009.json
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/      # FL
│   ├── ftheta_camera_cross_right_120fov/     # FR
│   ├── ftheta_camera_front_wide_120fov/      # FW
│   ├── ftheta_camera_front_tele_30fov/       # FN
│   ├── ftheta_camera_rear_left_70fov/        # RL
│   ├── ftheta_camera_rear_right_70fov/       # RR
│   └── ftheta_camera_rear_tele_30fov/        # RN
│       ├── 002.mp4
│       ├── 004.mp4
│       ├── 006.mp4
│       ├── 008.mp4
│       └── 009.mp4
└── videos/
    ├── ftheta_camera_cross_left_120fov/
    ├── ftheta_camera_cross_right_120fov/
    ├── ftheta_camera_front_wide_120fov/
    ├── ftheta_camera_front_tele_30fov/
    ├── ftheta_camera_rear_left_70fov/
    ├── ftheta_camera_rear_right_70fov/
    └── ftheta_camera_rear_tele_30fov/
        ├── 002.mp4
        ├── 004.mp4
        ├── 006.mp4
        ├── 008.mp4
        └── 009.mp4
```

## 使用步骤

### 1. 上传脚本到服务器

将以下脚本上传到服务器的代码库目录：
- `build_dataset.sh` - 数据集构建脚本
- `verify_dataset.sh` - 数据集验证脚本

```bash
# 在你的本地机器上
scp build_dataset.sh verify_dataset.sh wzh@your-server:/path/to/cosmos2.5_wzh/
```

或者直接在服务器上创建这些文件。

### 2. 在服务器上运行构建脚本

```bash
# SSH 登录到服务器
ssh wzh@your-server

# 进入代码库目录
cd /path/to/cosmos2.5_wzh/

# 给脚本添加执行权限
chmod +x build_dataset.sh verify_dataset.sh

# 运行构建脚本
./build_dataset.sh
```

脚本会自动：
1. ✓ 创建目录结构
2. ✓ 复制 guidance 视频到 `control_input_hdmap_bbox/`
3. ✓ 复制 GT 视频到 `videos/`
4. ✓ 为每个序列创建空的 caption JSON 文件
5. ✓ 生成 README.md 说明文档
6. ✓ 显示统计信息

### 3. 验证数据集

```bash
# 运行验证脚本
./verify_dataset.sh
```

验证脚本会检查：
- ✓ 目录结构是否完整
- ✓ 视频文件是否存在且非空
- ✓ Caption 文件是否存在且格式正确
- ✓ 每个序列的文件配对是否完整
- ✓ 生成统计报告

### 4. 更新 Caption（可选但推荐）

初始的 caption 文件是占位符，你可以编辑它们以提供更准确的场景描述。

```bash
# 示例：编辑第一个 caption
vim /mnt/zihanw/cosmos-transfer2.5/datasets/self_Data_Training/captions/ftheta_camera_front_wide_120fov/002.json
```

Caption 格式示例：
```json
{
    "caption": "A vehicle driving on a highway with moderate traffic, clear weather conditions",
    "sequence_id": "002",
    "camera": "ftheta_camera_front_wide_120fov",
    "note": "Updated with actual scene description"
}
```

## 相机映射说明

| 源文件前缀 | 官方相机名称 | 说明 | FOV |
|-----------|-------------|------|-----|
| FL | ftheta_camera_cross_left_120fov | 前左交叉相机 | 120° |
| FR | ftheta_camera_cross_right_120fov | 前右交叉相机 | 120° |
| FW | ftheta_camera_front_wide_120fov | 前方广角相机 | 120° |
| FN | ftheta_camera_front_tele_30fov | 前方长焦相机 | 30° |
| RL | ftheta_camera_rear_left_70fov | 后左相机 | 70° |
| RR | ftheta_camera_rear_right_70fov | 后右相机 | 70° |
| RN | ftheta_camera_rear_tele_30fov | 后方长焦相机 | 30° |

## 常见问题

### Q1: 如果源数据路径不同怎么办？

编辑 `build_dataset.sh` 中的路径配置：

```bash
# 修改这两行
GUIDANCE_BASE="/your/path/to/guidence"
GT_BASE="/your/path/to/GT"
```

### Q2: 如果数据集已存在，会发生什么？

脚本会询问是否删除并重建。如果选择 "No"，脚本会退出而不修改现有数据。

### Q3: 如何添加更多序列？

编辑 `build_dataset.sh`，在 `SEQUENCES` 数组中添加新的序列号：

```bash
SEQUENCES=("002" "004" "006" "008" "009" "010" "011")
```

### Q4: 视频文件很大，可以用软链接吗？

可以！修改脚本中的复制命令为软链接：

在 `copy_videos()` 函数中，将：
```bash
cp "$guidance_file" "$control_dst"
cp "$gt_file" "$video_dst"
```

改为：
```bash
ln -s "$guidance_file" "$control_dst"
ln -s "$gt_file" "$video_dst"
```

### Q5: 如何检查单个视频文件？

```bash
# 检查视频信息
ffprobe /path/to/video.mp4

# 播放视频（如果有 GUI）
ffplay /path/to/video.mp4

# 获取视频时长、分辨率等
ffmpeg -i /path/to/video.mp4 2>&1 | grep "Duration\|Video"
```

## 脚本选项说明

### build_dataset.sh

**功能特性：**
- 自动创建符合官方格式的目录结构
- 批量复制/链接视频文件
- 生成占位符 caption 文件
- 创建详细的 README 文档
- 显示完整的统计信息

**输出信息：**
- 目录创建进度
- 文件复制进度
- 成功/失败统计
- 数据集大小统计

### verify_dataset.sh

**验证项目：**
- [x] 15 个必需目录的存在性
- [x] 视频文件完整性（非空检查）
- [x] Caption JSON 格式验证
- [x] 文件配对完整性（每个序列应该有 7 个相机 x 2 种类型 + 1 个 caption）

**输出信息：**
- 详细的检查报告
- 彩色状态指示（绿色=通过，红色=失败，黄色=警告）
- 完整的统计报告

## 下一步：开始训练

数据集构建完成后，参考以下文档开始训练：

1. **查看训练文档：**
   ```bash
   cat /home/user/cosmos2.5_wzh/docs/post-training_auto_multiview.md
   ```

2. **准备训练配置：**
   - 修改配置文件，指向你的数据集路径
   - 设置训练参数（batch size, learning rate 等）

3. **启动训练：**
   ```bash
   # 示例命令（具体参数需要根据文档调整）
   python scripts/train.py \
       --config configs/your_config.yaml \
       --data-path /mnt/zihanw/cosmos-transfer2.5/datasets/self_Data_Training
   ```

## 技术支持

如遇到问题：
1. 检查 `/mnt/zihanw/cosmos-transfer2.5/datasets/self_Data_Training/README.md`
2. 运行 `./verify_dataset.sh` 查看详细错误
3. 查看官方文档：`docs/post-training_auto_multiview.md`

## 数据集统计（预期）

- **序列数量：** 5 个（002, 004, 006, 008, 009）
- **相机数量：** 7 个
- **每序列视频数：** 14 个（7 个 guidance + 7 个 GT）
- **总视频数：** 70 个
- **Caption 文件数：** 5 个
- **视频规格：** 1280x720, 90 frames, MP4 格式

---

**创建日期：** 2025-11-12
**版本：** 1.0
**作者：** Claude Code Assistant
