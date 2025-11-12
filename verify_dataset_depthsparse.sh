#!/bin/bash

# ============================================================================
# Cosmos Transfer2.5 数据集验证脚本 - Depth Sparse 版本
# 功能：验证 depth sparse 数据集结构和文件完整性
# ============================================================================

set -e

# 数据集路径
DATASET_BASE="/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 验证函数
# ============================================================================

check_directory_structure() {
    echo "======================================"
    echo "检查目录结构..."
    echo "======================================"

    local all_good=true

    # 检查主目录
    if [ ! -d "$DATASET_BASE" ]; then
        echo -e "${RED}✗ 数据集根目录不存在: $DATASET_BASE${NC}"
        return 1
    fi

    # 定义应该存在的目录
    declare -a required_dirs=(
        "captions/ftheta_camera_front_wide_120fov"
        "control_input_hdmap_bbox/ftheta_camera_cross_left_120fov"
        "control_input_hdmap_bbox/ftheta_camera_cross_right_120fov"
        "control_input_hdmap_bbox/ftheta_camera_front_wide_120fov"
        "control_input_hdmap_bbox/ftheta_camera_front_tele_30fov"
        "control_input_hdmap_bbox/ftheta_camera_rear_left_70fov"
        "control_input_hdmap_bbox/ftheta_camera_rear_right_70fov"
        "control_input_hdmap_bbox/ftheta_camera_rear_tele_30fov"
        "videos/ftheta_camera_cross_left_120fov"
        "videos/ftheta_camera_cross_right_120fov"
        "videos/ftheta_camera_front_wide_120fov"
        "videos/ftheta_camera_front_tele_30fov"
        "videos/ftheta_camera_rear_left_70fov"
        "videos/ftheta_camera_rear_right_70fov"
        "videos/ftheta_camera_rear_tele_30fov"
    )

    for dir in "${required_dirs[@]}"; do
        if [ -d "${DATASET_BASE}/${dir}" ]; then
            echo -e "${GREEN}✓${NC} ${dir}"
        else
            echo -e "${RED}✗${NC} ${dir} (缺失)"
            all_good=false
        fi
    done

    echo ""
    if [ "$all_good" = true ]; then
        echo -e "${GREEN}✓ 目录结构完整${NC}"
        return 0
    else
        echo -e "${RED}✗ 目录结构不完整${NC}"
        return 1
    fi
}

check_video_files() {
    echo "======================================"
    echo "检查视频文件..."
    echo "======================================"

    declare -a cameras=(
        "ftheta_camera_cross_left_120fov"
        "ftheta_camera_cross_right_120fov"
        "ftheta_camera_front_wide_120fov"
        "ftheta_camera_front_tele_30fov"
        "ftheta_camera_rear_left_70fov"
        "ftheta_camera_rear_right_70fov"
        "ftheta_camera_rear_tele_30fov"
    )

    local all_good=true

    for cam in "${cameras[@]}"; do
        echo ""
        echo "相机: ${cam}"
        echo "--------------------------------------"

        # 检查 control input 视频 (depth)
        control_dir="${DATASET_BASE}/control_input_hdmap_bbox/${cam}"
        control_count=$(ls -1 "${control_dir}"/*.mp4 2>/dev/null | wc -l)

        if [ "$control_count" -eq 0 ]; then
            echo -e "  ${RED}✗${NC} control_input (depth): 没有视频文件"
            all_good=false
        else
            echo -e "  ${GREEN}✓${NC} control_input (depth): ${control_count} 个视频文件"

            # 检查视频文件是否可读且大小 > 0
            for video in "${control_dir}"/*.mp4; do
                if [ ! -s "$video" ]; then
                    echo -e "    ${YELLOW}⚠${NC} 文件为空: $(basename $video)"
                    all_good=false
                fi
            done
        fi

        # 检查 GT 视频
        video_dir="${DATASET_BASE}/videos/${cam}"
        video_count=$(ls -1 "${video_dir}"/*.mp4 2>/dev/null | wc -l)

        if [ "$video_count" -eq 0 ]; then
            echo -e "  ${RED}✗${NC} videos (GT): 没有视频文件"
            all_good=false
        else
            echo -e "  ${GREEN}✓${NC} videos (GT): ${video_count} 个视频文件"

            # 检查视频文件是否可读且大小 > 0
            for video in "${video_dir}"/*.mp4; do
                if [ ! -s "$video" ]; then
                    echo -e "    ${YELLOW}⚠${NC} 文件为空: $(basename $video)"
                    all_good=false
                fi
            done
        fi

        # 检查数量是否匹配
        if [ "$control_count" -ne "$video_count" ]; then
            echo -e "  ${YELLOW}⚠${NC} control_input 和 videos 数量不匹配"
            all_good=false
        fi
    done

    echo ""
    if [ "$all_good" = true ]; then
        echo -e "${GREEN}✓ 视频文件检查通过${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ 视频文件检查发现问题${NC}"
        return 1
    fi
}

check_caption_files() {
    echo "======================================"
    echo "检查 Caption 文件..."
    echo "======================================"

    local caption_dir="${DATASET_BASE}/captions/ftheta_camera_front_wide_120fov"
    local caption_count=$(ls -1 "${caption_dir}"/*.json 2>/dev/null | wc -l)

    if [ "$caption_count" -eq 0 ]; then
        echo -e "${RED}✗ 没有找到 caption 文件${NC}"
        return 1
    fi

    echo -e "${GREEN}✓${NC} 找到 ${caption_count} 个 caption 文件"

    # 检查每个 caption 文件是否是有效的 JSON
    local all_valid=true
    for json_file in "${caption_dir}"/*.json; do
        if command -v python3 &> /dev/null; then
            if python3 -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $(basename $json_file)"
            else
                echo -e "  ${RED}✗${NC} $(basename $json_file) (无效的 JSON)"
                all_valid=false
            fi
        else
            # 如果没有 python，只检查文件是否为空
            if [ ! -s "$json_file" ]; then
                echo -e "  ${RED}✗${NC} $(basename $json_file) (文件为空)"
                all_valid=false
            else
                echo -e "  ${YELLOW}?${NC} $(basename $json_file) (无法验证 JSON 格式)"
            fi
        fi
    done

    echo ""
    if [ "$all_valid" = true ]; then
        echo -e "${GREEN}✓ Caption 文件格式正确${NC}"
        return 0
    else
        echo -e "${RED}✗ 部分 caption 文件有问题${NC}"
        return 1
    fi
}

check_file_pairing() {
    echo "======================================"
    echo "检查文件配对..."
    echo "======================================"

    local all_good=true

    # 获取所有 control input 文件的序列号
    local control_dir="${DATASET_BASE}/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov"

    if [ ! -d "$control_dir" ]; then
        echo -e "${RED}✗ 参考目录不存在${NC}"
        return 1
    fi

    for control_file in "${control_dir}"/*.mp4; do
        [ -f "$control_file" ] || continue

        local seq_id=$(basename "$control_file" .mp4)
        echo ""
        echo "序列: ${seq_id}"
        echo "--------------------------------------"

        local has_error=false

        # 检查所有相机的 control input 和 video 是否存在
        declare -a cameras=(
            "ftheta_camera_cross_left_120fov"
            "ftheta_camera_cross_right_120fov"
            "ftheta_camera_front_wide_120fov"
            "ftheta_camera_front_tele_30fov"
            "ftheta_camera_rear_left_70fov"
            "ftheta_camera_rear_right_70fov"
            "ftheta_camera_rear_tele_30fov"
        )

        for cam in "${cameras[@]}"; do
            local ctrl="${DATASET_BASE}/control_input_hdmap_bbox/${cam}/${seq_id}.mp4"
            local vid="${DATASET_BASE}/videos/${cam}/${seq_id}.mp4"

            if [ ! -f "$ctrl" ]; then
                echo -e "  ${RED}✗${NC} 缺失 control_input (depth): ${cam}"
                has_error=true
            fi

            if [ ! -f "$vid" ]; then
                echo -e "  ${RED}✗${NC} 缺失 video (GT): ${cam}"
                has_error=true
            fi
        done

        # 检查 caption
        local caption="${DATASET_BASE}/captions/ftheta_camera_front_wide_120fov/${seq_id}.json"
        if [ ! -f "$caption" ]; then
            echo -e "  ${RED}✗${NC} 缺失 caption"
            has_error=true
        fi

        if [ "$has_error" = false ]; then
            echo -e "  ${GREEN}✓${NC} 所有文件完整"
        else
            all_good=false
        fi
    done

    echo ""
    if [ "$all_good" = true ]; then
        echo -e "${GREEN}✓ 文件配对检查通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 文件配对不完整${NC}"
        return 1
    fi
}

generate_report() {
    echo ""
    echo "======================================"
    echo "数据集统计报告 - Depth Sparse"
    echo "======================================"
    echo ""

    # 统计总文件数
    local total_control=$(find "${DATASET_BASE}/control_input_hdmap_bbox" -name "*.mp4" 2>/dev/null | wc -l)
    local total_videos=$(find "${DATASET_BASE}/videos" -name "*.mp4" 2>/dev/null | wc -l)
    local total_captions=$(find "${DATASET_BASE}/captions" -name "*.json" 2>/dev/null | wc -l)

    echo "文件统计:"
    echo "  - Control Input (depth) 视频: ${total_control}"
    echo "  - GT 视频: ${total_videos}"
    echo "  - Caption 文件: ${total_captions}"
    echo ""

    # 计算数据集大小
    if [ -d "$DATASET_BASE" ]; then
        local total_size=$(du -sh "$DATASET_BASE" 2>/dev/null | cut -f1)
        echo "总大小: ${total_size}"
    fi

    echo ""

    # 列出所有序列
    echo "序列列表:"
    local control_dir="${DATASET_BASE}/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov"
    if [ -d "$control_dir" ]; then
        for file in "${control_dir}"/*.mp4; do
            [ -f "$file" ] || continue
            echo "  - $(basename "$file" .mp4)"
        done
    fi

    echo ""
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   Cosmos Transfer2.5 数据集验证 - Depth Sparse                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "数据集路径: ${DATASET_BASE}"
    echo "数据类型: Depth Sparse Point Cloud"
    echo ""

    local status=0

    # 执行各项检查
    check_directory_structure || status=1
    echo ""

    check_video_files || status=1
    echo ""

    check_caption_files || status=1
    echo ""

    check_file_pairing || status=1
    echo ""

    generate_report

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    if [ $status -eq 0 ]; then
        echo -e "║   ${GREEN}✓ Depth Sparse 数据集验证通过！可以开始训练${NC}             ║"
    else
        echo -e "║   ${YELLOW}⚠ 数据集存在问题，请检查并修复${NC}                         ║"
    fi
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    exit $status
}

# 运行主程序
main "$@"
