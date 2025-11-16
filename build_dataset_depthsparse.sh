#!/bin/bash

# ============================================================================
# Cosmos Transfer2.5 数据集构建脚本 - Depth Sparse 版本
# 功能：将 depth sparse 和 GT 数据组织成官方训练格式
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 配置部分
# ============================================================================

# 源数据路径
GUIDANCE_BASE="/mnt/zihanw/cosmos-transfer2.5/data_prepa/guidence"
GT_BASE="/mnt/zihanw/cosmos-transfer2.5/data_prepa/GT"

# 目标数据集路径
DATASET_BASE="/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse"

# 所有数据集序列（根据你提供的列表）
SEQUENCES=("002" "004" "006" "008" "009")

# 相机名称映射：源文件名 -> 目标文件夹名
declare -A CAMERA_MAP=(
    ["FL"]="ftheta_camera_cross_left_120fov"
    ["FR"]="ftheta_camera_cross_right_120fov"
    ["FW"]="ftheta_camera_front_wide_120fov"
    ["FN"]="ftheta_camera_front_tele_30fov"
    ["RL"]="ftheta_camera_rear_left_70fov"
    ["RR"]="ftheta_camera_rear_right_70fov"
    ["RN"]="ftheta_camera_rear_tele_30fov"
)

# ============================================================================
# 函数定义
# ============================================================================

# 创建目录结构
create_directory_structure() {
    echo "======================================"
    echo "创建目录结构..."
    echo "======================================"

    # 创建 captions 目录
    mkdir -p "${DATASET_BASE}/captions/ftheta_camera_front_wide_120fov"

    # 创建 control_input_hdmap_bbox 和 videos 目录
    for cam_folder in "${CAMERA_MAP[@]}"; do
        mkdir -p "${DATASET_BASE}/control_input_hdmap_bbox/${cam_folder}"
        mkdir -p "${DATASET_BASE}/videos/${cam_folder}"
    done

    echo "✓ 目录结构创建完成"
    echo ""
}

# 复制视频文件
copy_videos() {
    echo "======================================"
    echo "复制视频文件..."
    echo "======================================"

    local total_files=0
    local success_count=0

    for seq in "${SEQUENCES[@]}"; do
        echo ""
        echo "处理序列: ${seq}"
        echo "--------------------------------------"

        # 构建源路径 - 注意这里使用 depth 目录而不是 color
        guidance_dir="${GUIDANCE_BASE}/${seq}_90frames_1280x720/depth"
        gt_dir="${GT_BASE}/${seq}_GT_90frames_1280x720"

        # 检查源目录是否存在
        if [ ! -d "$guidance_dir" ]; then
            echo "⚠ 警告: Depth 目录不存在: $guidance_dir"
            continue
        fi

        if [ ! -d "$gt_dir" ]; then
            echo "⚠ 警告: GT 目录不存在: $gt_dir"
            continue
        fi

        # 遍历每个相机
        for cam_src in "${!CAMERA_MAP[@]}"; do
            cam_dst="${CAMERA_MAP[$cam_src]}"

            # 源文件名 - depth 视频可能的命名格式
            # 尝试多种可能的命名格式
            guidance_file=""

            # 尝试不同的文件名格式
            if [ -f "${guidance_dir}/${cam_src}_depth.mp4" ]; then
                guidance_file="${guidance_dir}/${cam_src}_depth.mp4"
            elif [ -f "${guidance_dir}/${cam_src}.mp4" ]; then
                guidance_file="${guidance_dir}/${cam_src}.mp4"
            elif [ -f "${guidance_dir}/${cam_src}_sparse.mp4" ]; then
                guidance_file="${guidance_dir}/${cam_src}_sparse.mp4"
            elif [ -f "${guidance_dir}/${cam_src}_depth_sparse.mp4" ]; then
                guidance_file="${guidance_dir}/${cam_src}_depth_sparse.mp4"
            fi

            gt_file="${gt_dir}/${cam_src}_GT.mp4"

            # 目标文件名（使用序列号作为文件名）
            control_dst="${DATASET_BASE}/control_input_hdmap_bbox/${cam_dst}/${seq}.mp4"
            video_dst="${DATASET_BASE}/videos/${cam_dst}/${seq}.mp4"

            total_files=$((total_files + 2))

            # 复制 depth guidance (control input)
            if [ -n "$guidance_file" ] && [ -f "$guidance_file" ]; then
                cp "$guidance_file" "$control_dst"
                echo "  ✓ 复制: $(basename $guidance_file) -> control_input/${cam_dst}/${seq}.mp4"
                success_count=$((success_count + 1))
            else
                echo "  ✗ 缺失: depth 视频 for ${cam_src} (尝试了多种命名格式)"
            fi

            # 复制 GT (videos)
            if [ -f "$gt_file" ]; then
                cp "$gt_file" "$video_dst"
                echo "  ✓ 复制: ${cam_src}_GT.mp4 -> videos/${cam_dst}/${seq}.mp4"
                success_count=$((success_count + 1))
            else
                echo "  ✗ 缺失: $gt_file"
            fi
        done
    done

    echo ""
    echo "======================================"
    echo "视频复制完成: ${success_count}/${total_files} 文件"
    echo "======================================"
    echo ""
}

# 创建空的 caption JSON 文件
create_captions() {
    echo "======================================"
    echo "创建 Caption 文件..."
    echo "======================================"

    local caption_dir="${DATASET_BASE}/captions/ftheta_camera_front_wide_120fov"

    for seq in "${SEQUENCES[@]}"; do
        caption_file="${caption_dir}/${seq}.json"

        # 创建一个基本的空 caption JSON
        cat > "$caption_file" <<EOF
{
    "caption": "A driving scene with depth sparse point cloud guidance from vehicle cameras",
    "sequence_id": "${seq}",
    "camera": "ftheta_camera_front_wide_120fov",
    "guidance_type": "depth_sparse",
    "note": "This is a placeholder caption. Please update with actual description."
}
EOF

        echo "  ✓ 创建: ${seq}.json"
    done

    echo ""
    echo "✓ Caption 文件创建完成"
    echo ""
}

# 生成数据集统计信息
generate_summary() {
    echo "======================================"
    echo "数据集统计信息"
    echo "======================================"
    echo ""
    echo "数据集路径: ${DATASET_BASE}"
    echo "数据集类型: Depth Sparse Point Cloud"
    echo ""
    echo "视频文件数量:"

    for cam_folder in "${CAMERA_MAP[@]}"; do
        control_count=$(ls -1 "${DATASET_BASE}/control_input_hdmap_bbox/${cam_folder}"/*.mp4 2>/dev/null | wc -l)
        video_count=$(ls -1 "${DATASET_BASE}/videos/${cam_folder}"/*.mp4 2>/dev/null | wc -l)
        echo "  ${cam_folder}:"
        echo "    - control_input (depth): ${control_count} 文件"
        echo "    - videos (GT): ${video_count} 文件"
    done

    caption_count=$(ls -1 "${DATASET_BASE}/captions/ftheta_camera_front_wide_120fov"/*.json 2>/dev/null | wc -l)
    echo ""
    echo "Caption 文件数量: ${caption_count}"
    echo ""

    # 计算总大小
    total_size=$(du -sh "${DATASET_BASE}" | cut -f1)
    echo "总数据集大小: ${total_size}"
    echo ""
}

# 创建数据集配置说明文件
create_readme() {
    readme_file="${DATASET_BASE}/README.md"

    cat > "$readme_file" <<'EOF'
# Depth Sparse Dataset

## 数据集结构

```
depthsparse/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       └── *.json                  # Caption 文件（每个序列一个）
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/      # FL: Front Left
│   ├── ftheta_camera_cross_right_120fov/     # FR: Front Right
│   ├── ftheta_camera_front_wide_120fov/      # FW: Front Wide
│   ├── ftheta_camera_front_tele_30fov/       # FN: Front Narrow/Tele
│   ├── ftheta_camera_rear_left_70fov/        # RL: Rear Left
│   ├── ftheta_camera_rear_right_70fov/       # RR: Rear Right
│   └── ftheta_camera_rear_tele_30fov/        # RN: Rear Narrow/Tele
└── videos/
    ├── ftheta_camera_cross_left_120fov/
    ├── ftheta_camera_cross_right_120fov/
    ├── ftheta_camera_front_wide_120fov/
    ├── ftheta_camera_front_tele_30fov/
    ├── ftheta_camera_rear_left_70fov/
    ├── ftheta_camera_rear_right_70fov/
    └── ftheta_camera_rear_tele_30fov/
```

## 数据说明

- **control_input_hdmap_bbox/**: 包含深度稀疏点云投影生成的 guidance 视频
- **videos/**: 包含对应的 ground truth 视频
- **captions/**: 包含视频描述（当前为占位符，需要更新）

## 与 rgbCloud 数据集的区别

| 特性 | rgbCloud | depthsparse |
|------|----------|-------------|
| Guidance 来源 | 彩色点云投影 (color/) | 深度稀疏点云 (depth/) |
| GT 视频 | 相同 | 相同 |
| 用途 | RGB 颜色信息引导 | 深度信息引导 |

## 序列列表

- 002_90frames_1280x720
- 004_90frames_1280x720
- 006_90frames_1280x720
- 008_90frames_1280x720
- 009_90frames_1280x720

## 相机配置

| 原始名称 | 官方名称 | 说明 |
|---------|---------|------|
| FL | ftheta_camera_cross_left_120fov | 前左交叉相机 (120° FOV) |
| FR | ftheta_camera_cross_right_120fov | 前右交叉相机 (120° FOV) |
| FW | ftheta_camera_front_wide_120fov | 前方广角相机 (120° FOV) |
| FN | ftheta_camera_front_tele_30fov | 前方长焦相机 (30° FOV) |
| RL | ftheta_camera_rear_left_70fov | 后左相机 (70° FOV) |
| RR | ftheta_camera_rear_right_70fov | 后右相机 (70° FOV) |
| RN | ftheta_camera_rear_tele_30fov | 后方长焦相机 (30° FOV) |

## 视频规格

- 分辨率: 1280x720
- 帧数: 90 frames
- 格式: MP4
- Guidance 类型: Depth Sparse Point Cloud

## 下一步

1. **更新 Captions**: 编辑 caption JSON 文件
2. **验证数据**: 使用验证脚本检查数据集
3. **开始训练**: 使用此数据集进行模型训练

## 生成信息

- 生成时间: $(date)
- 数据类型: Depth Sparse
- 脚本版本: 1.0
EOF

    echo "✓ README.md 已创建: ${readme_file}"
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   Cosmos Transfer2.5 数据集构建 - Depth Sparse                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # 检查源目录是否存在
    if [ ! -d "$GUIDANCE_BASE" ]; then
        echo "❌ 错误: Guidance 源目录不存在: $GUIDANCE_BASE"
        exit 1
    fi

    if [ ! -d "$GT_BASE" ]; then
        echo "❌ 错误: GT 源目录不存在: $GT_BASE"
        exit 1
    fi

    # 如果目标目录已存在，询问是否覆盖
    if [ -d "$DATASET_BASE" ]; then
        echo "⚠ 警告: 目标目录已存在: $DATASET_BASE"
        read -p "是否删除并重新创建？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "正在删除旧数据集..."
            rm -rf "$DATASET_BASE"
        else
            echo "操作已取消"
            exit 0
        fi
    fi

    # 执行构建步骤
    create_directory_structure
    copy_videos
    create_captions
    create_readme
    generate_summary

    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   ✓ Depth Sparse 数据集构建完成！                             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "数据集位置: ${DATASET_BASE}"
    echo ""
    echo "下一步操作建议："
    echo "  1. 检查数据集结构: ls -lR ${DATASET_BASE}"
    echo "  2. 查看 README: cat ${DATASET_BASE}/README.md"
    echo "  3. 验证数据集: ./verify_dataset_depthsparse.sh"
    echo "  4. 更新 caption 文件"
    echo "  5. 开始训练"
    echo ""
}

# 运行主程序
main "$@"
