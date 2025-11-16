#!/bin/bash
# Check the actual frame count of control_input videos in RGBCloud dataset

echo "Checking control_input video frame counts in RGBCloud dataset..."
echo ""

# Get first sample directory
sample_dir=$(ls -d /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/*/ 2>/dev/null | head -1)

if [ -z "$sample_dir" ]; then
    echo "ERROR: Cannot find dataset at /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/"
    exit 1
fi

echo "Sample directory: $sample_dir"
echo ""

# Check control_input for front camera
control_video="${sample_dir}ftheta_camera_front_wide_120fov/control_input_hdmap_bbox.mp4"

if [ -f "$control_video" ]; then
    frame_count=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$control_video" 2>/dev/null)
    echo "Control video: $control_video"
    echo "Frame count: $frame_count"
else
    echo "ERROR: Cannot find control video at $control_video"
fi

echo ""
echo "Checking a few more samples..."
for dir in $(ls -d /mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud/*/ 2>/dev/null | head -3); do
    control="${dir}ftheta_camera_front_wide_120fov/control_input_hdmap_bbox.mp4"
    if [ -f "$control" ]; then
        frames=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$control" 2>/dev/null)
        echo "  $control: $frames frames"
    fi
done
