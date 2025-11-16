#!/usr/bin/env python3
"""
Convert segmented video data to Cosmos Transfer2.5 training format.

Input structure:
  GT/{scene}_GT_90frames_1280x720/GT_segments/{camera}/{camera}_GT_seg{N}.mp4
  guidence/{scene}_90frames_1280x720/color_segments/{camera}/{camera}_color_seg{N}.mp4
  guidence/{scene}_90frames_1280x720/depth_segments/{camera}/{camera}_depth_seg{N}.mp4

Output structure (expected by MultiviewTransferDataset):
  datasets/RGBCloud/
    videos/
      {camera_full_name}/
        {scene}_seg{N}.mp4
        ...
    control_input_hdmap_bbox/
      {camera_full_name}/
        {scene}_seg{N}.mp4
        ...
    captions/
      {front_camera_full_name}/
        {scene}_seg{N}.json
        ...
"""

import json
import os
import shutil
from pathlib import Path

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

# Default caption for all videos
DEFAULT_CAPTION = (
    "This multi-camera perspective captures a drive along a multi-lane urban freeway during the daytime "
    "under a hazy or partly cloudy sky. The vehicle travels in one of the right lanes, flanked on one side "
    "by a high retaining wall featuring a concrete base and a brown, brick-patterned upper section with some "
    "climbing vines, and on the other side by a concrete median barrier."
)


def create_dataset(
    data_root: str,
    output_root: str,
    dataset_type: str,  # "RGBCloud" or "DepthSparse"
    control_type: str,  # "color" or "depth"
):
    """
    Create training dataset from segmented videos in MultiviewTransferDataset format.

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

    # Create output directory structure
    videos_dir = output_root / "videos"
    control_dir = output_root / "control_input_hdmap_bbox"
    captions_dir = output_root / "captions"

    # Create camera subdirectories
    for camera_full_name in CAMERA_MAPPING.values():
        (videos_dir / camera_full_name).mkdir(parents=True, exist_ok=True)
        (control_dir / camera_full_name).mkdir(parents=True, exist_ok=True)

    # Create caption directory for front camera
    front_camera = "ftheta_camera_front_wide_120fov"
    (captions_dir / front_camera).mkdir(parents=True, exist_ok=True)

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
        for camera_abbr, camera_full_name in CAMERA_MAPPING.items():
            gt_camera_dir = gt_segments / camera_abbr
            control_camera_dir = control_segments / camera_abbr

            if not gt_camera_dir.exists():
                print(f"  WARNING: GT camera dir not found: {gt_camera_dir}")
                continue
            if not control_camera_dir.exists():
                print(f"  WARNING: Control camera dir not found: {control_camera_dir}")
                continue

            # Process all segments for this camera
            gt_segments_files = sorted(gt_camera_dir.glob(f"{camera_abbr}_GT_seg*.mp4"))

            for gt_file in gt_segments_files:
                # Extract segment number (e.g., "01" from "FL_GT_seg01.mp4")
                seg_num = gt_file.stem.split("seg")[-1]

                # Find corresponding control file
                control_file = control_camera_dir / f"{camera_abbr}_{control_type.split('_')[0]}_seg{seg_num}.mp4"

                if not control_file.exists():
                    print(f"  WARNING: Control file not found: {control_file}")
                    continue

                # Output filenames: {scene}_seg{N}.mp4
                output_filename = f"{scene_id}_seg{seg_num}.mp4"

                # Copy GT video
                output_video = videos_dir / camera_full_name / output_filename
                if not output_video.exists():
                    shutil.copy2(gt_file, output_video)

                # Copy control video
                output_control = control_dir / camera_full_name / output_filename
                if not output_control.exists():
                    shutil.copy2(control_file, output_control)

                # Create caption (only for front camera, once per sample)
                if camera_full_name == front_camera:
                    caption_file = captions_dir / front_camera / f"{scene_id}_seg{seg_num}.json"
                    if not caption_file.exists():
                        caption_data = {"caption": DEFAULT_CAPTION}
                        with open(caption_file, "w") as f:
                            json.dump(caption_data, f, indent=2)
                    total_samples += 1

        print(f"  Completed scene {scene_id}")

    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Total training samples: {total_samples}")
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
    print(f"   ls {output_root}/RGBCloud/videos/ftheta_camera_front_wide_120fov/ | head -10")
    print(f"   ls {output_root}/RGBCloud/control_input_hdmap_bbox/ftheta_camera_front_wide_120fov/ | head -10")
    print("\n2. Check a sample video:")
    print(f"   ffprobe {output_root}/RGBCloud/videos/ftheta_camera_front_wide_120fov/002_seg01.mp4")
    print("\n3. Start training:")
    print("   bash train_rgbcloud.sh")


if __name__ == "__main__":
    main()
