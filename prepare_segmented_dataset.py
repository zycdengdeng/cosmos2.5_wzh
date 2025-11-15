#!/usr/bin/env python3
"""
Convert segmented video data to Cosmos Transfer2.5 training format.

Input structure:
  GT/{scene}_GT_90frames_1280x720/GT_segments/{camera}/{camera}_GT_seg{N}.mp4
  guidence/{scene}_90frames_1280x720/color_segments/{camera}/{camera}_color_seg{N}.mp4
  guidence/{scene}_90frames_1280x720/depth_segments/{camera}/{camera}_depth_seg{N}.mp4

Output structure (RGBCloud):
  datasets/RGBCloud/{scene}_seg{N}/
    {camera_name}/
      video.mp4  (GT)
      control_input_hdmap_bbox.mp4  (color guidance)

Output structure (DepthSparse):
  datasets/DepthSparse/{scene}_seg{N}/
    {camera_name}/
      video.mp4  (GT)
      control_input_hdmap_bbox.mp4  (depth guidance)
"""

import os
import shutil
from pathlib import Path
from typing import Dict

# Camera name mapping from abbreviated to full name
CAMERA_MAPPING = {
    "FL": "ftheta_camera_cross_left_120fov",
    "FR": "ftheta_camera_cross_right_120fov",
    "RL": "ftheta_camera_rear_left_70fov",
    "FN": "ftheta_camera_front_tele_30fov",
    "RR": "ftheta_camera_rear_right_70fov",
    "FW": "ftheta_camera_front_wide_120fov",
    "RN": "ftheta_camera_rear_tele_30fov",
}

def create_dataset(
    data_root: str,
    output_root: str,
    dataset_type: str,  # "RGBCloud" or "DepthSparse"
    control_type: str,  # "color" or "depth"
):
    """
    Create training dataset from segmented videos.

    Args:
        data_root: Root directory containing GT and guidence folders
        output_root: Output directory for training datasets
        dataset_type: "RGBCloud" or "DepthSparse"
        control_type: "color_segments" or "depth_segments"
    """
    data_root = Path(data_root)
    output_root = Path(output_root) / dataset_type

    gt_root = data_root / "GT"
    guidance_root = data_root / "guidence"

    # Find all scene directories
    scene_dirs = sorted([d for d in gt_root.iterdir() if d.is_dir()])

    print(f"\n{'='*60}")
    print(f"Creating {dataset_type} dataset with {control_type} control")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Found {len(scene_dirs)} scenes")

    total_samples = 0

    for scene_dir in scene_dirs:
        # Extract scene ID (e.g., "002" from "002_GT_90frames_1280x720")
        scene_id = scene_dir.name.split("_")[0]
        print(f"\nProcessing scene {scene_id}...")

        # Find corresponding guidance directory
        guidance_scene = guidance_root / f"{scene_id}_90frames_1280x720"
        if not guidance_scene.exists():
            print(f"  WARNING: Guidance directory not found: {guidance_scene}")
            continue

        gt_segments = scene_dir / "GT_segments"
        control_segments = guidance_scene / control_type

        if not gt_segments.exists():
            print(f"  WARNING: GT segments not found: {gt_segments}")
            continue
        if not control_segments.exists():
            print(f"  WARNING: Control segments not found: {control_segments}")
            continue

        # Process each camera
        camera_dirs = sorted([d for d in gt_segments.iterdir() if d.is_dir()])

        for camera_dir in camera_dirs:
            camera_abbr = camera_dir.name
            camera_full_name = CAMERA_MAPPING.get(camera_abbr)

            if not camera_full_name:
                print(f"  WARNING: Unknown camera abbreviation: {camera_abbr}")
                continue

            # Find all segment files for this camera
            gt_camera_dir = gt_segments / camera_abbr
            control_camera_dir = control_segments / camera_abbr

            if not control_camera_dir.exists():
                print(f"  WARNING: Control camera dir not found: {control_camera_dir}")
                continue

            gt_segments_files = sorted(gt_camera_dir.glob(f"{camera_abbr}_GT_seg*.mp4"))

            for gt_file in gt_segments_files:
                # Extract segment number (e.g., "01" from "FL_GT_seg01.mp4")
                seg_num = gt_file.stem.split("seg")[-1]

                # Find corresponding control file
                control_file = control_camera_dir / f"{camera_abbr}_{control_type.split('_')[0]}_seg{seg_num}.mp4"

                if not control_file.exists():
                    print(f"  WARNING: Control file not found: {control_file}")
                    continue

                # Create output directory structure
                sample_name = f"{scene_id}_seg{seg_num}"
                output_sample_dir = output_root / sample_name / camera_full_name
                output_sample_dir.mkdir(parents=True, exist_ok=True)

                # Copy files
                output_video = output_sample_dir / "video.mp4"
                output_control = output_sample_dir / "control_input_hdmap_bbox.mp4"

                if not output_video.exists():
                    shutil.copy2(gt_file, output_video)
                if not output_control.exists():
                    shutil.copy2(control_file, output_control)

                total_samples += 1

        print(f"  Completed scene {scene_id}")

    # Calculate total samples (divide by 7 cameras to get actual sample count)
    num_samples = total_samples // 7
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Total files created: {total_samples}")
    print(f"Total training samples: {num_samples} (each with 7 cameras)")
    print(f"Output location: {output_root}")
    print(f"{'='*60}\n")


def main():
    """Main function to create both RGBCloud and DepthSparse datasets."""

    # Configuration
    data_root = "/mnt/zihanw/cosmos-transfer2.5/data_prepa"
    output_root = "/mnt/zihanw/cosmos-transfer2.5/datasets"

    # Create RGBCloud dataset (using color guidance)
    create_dataset(
        data_root=data_root,
        output_root=output_root,
        dataset_type="RGBCloud",
        control_type="color_segments",
    )

    # Create DepthSparse dataset (using depth guidance)
    create_dataset(
        data_root=data_root,
        output_root=output_root,
        dataset_type="DepthSparse",
        control_type="depth_segments",
    )

    print("\n✅ All datasets created successfully!\n")
    print("Next steps:")
    print("1. Verify dataset structure:")
    print(f"   ls -R {output_root}/RGBCloud | head -50")
    print(f"   ls -R {output_root}/DepthSparse | head -50")
    print("\n2. Check a sample video:")
    print(f"   ffprobe {output_root}/RGBCloud/002_seg01/ftheta_camera_front_wide_120fov/video.mp4")
    print("\n3. Start training:")
    print("   bash train_rgbcloud.sh")


if __name__ == "__main__":
    main()
