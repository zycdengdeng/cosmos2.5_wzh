#!/usr/bin/env python3
"""
Verify the prepared dataset structure and content.
"""

import subprocess
from pathlib import Path
from typing import Tuple


def check_video_frames(video_path: Path) -> Tuple[int, str]:
    """
    Check number of frames in a video file.

    Returns:
        (frame_count, resolution) tuple
    """
    try:
        # Get frame count
        cmd_frames = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path),
        ]
        frames_result = subprocess.run(cmd_frames, capture_output=True, text=True, check=True)
        frame_count = int(frames_result.stdout.strip())

        # Get resolution
        cmd_res = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path),
        ]
        res_result = subprocess.run(cmd_res, capture_output=True, text=True, check=True)
        width, height = res_result.stdout.strip().split(",")
        resolution = f"{width}x{height}"

        return frame_count, resolution

    except Exception as e:
        return -1, f"Error: {e}"


def verify_dataset(dataset_root: str, dataset_name: str, expected_frames: int = 10):
    """
    Verify dataset structure and content.

    Args:
        dataset_root: Root directory containing the dataset
        dataset_name: Name of dataset (e.g., "RGBCloud")
        expected_frames: Expected number of frames per video (default: 10 for 1s @ 10fps)
    """
    dataset_path = Path(dataset_root) / dataset_name

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name} Dataset")
    print(f"{'='*60}")
    print(f"Location: {dataset_path}")

    # Expected cameras
    expected_cameras = [
        "ftheta_camera_front_wide_120fov",
        "ftheta_camera_cross_left_120fov",
        "ftheta_camera_cross_right_120fov",
        "ftheta_camera_rear_left_70fov",
        "ftheta_camera_rear_right_70fov",
        "ftheta_camera_rear_tele_30fov",
        "ftheta_camera_front_tele_30fov",
    ]

    # Find all sample directories
    sample_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(sample_dirs)} samples")

    if len(sample_dirs) == 0:
        print("❌ No samples found!")
        return

    # Check first few samples in detail
    samples_to_check = min(3, len(sample_dirs))
    print(f"\nChecking first {samples_to_check} samples in detail...\n")

    issues_found = 0
    total_videos_checked = 0

    for i, sample_dir in enumerate(sample_dirs[:samples_to_check]):
        print(f"Sample {i+1}: {sample_dir.name}")

        # Check each camera
        for camera in expected_cameras:
            camera_dir = sample_dir / camera

            if not camera_dir.exists():
                print(f"  ❌ Missing camera directory: {camera}")
                issues_found += 1
                continue

            # Check video.mp4
            video_file = camera_dir / "video.mp4"
            if not video_file.exists():
                print(f"  ❌ Missing video.mp4 in {camera}")
                issues_found += 1
                continue

            # Check control_input
            control_file = camera_dir / "control_input_hdmap_bbox.mp4"
            if not control_file.exists():
                print(f"  ❌ Missing control_input_hdmap_bbox.mp4 in {camera}")
                issues_found += 1
                continue

            # Verify video properties (only for front camera to save time)
            if camera == "ftheta_camera_front_wide_120fov":
                video_frames, video_res = check_video_frames(video_file)
                control_frames, control_res = check_video_frames(control_file)

                total_videos_checked += 2

                print(f"  ✓ {camera}:")
                print(f"    video.mp4: {video_frames} frames, {video_res}")
                print(f"    control_input: {control_frames} frames, {control_res}")

                if video_frames != expected_frames:
                    print(f"    ⚠️  Warning: Expected {expected_frames} frames, got {video_frames}")
                    issues_found += 1

                if control_frames != expected_frames:
                    print(f"    ⚠️  Warning: Control expected {expected_frames} frames, got {control_frames}")
                    issues_found += 1

                if video_res != "1280x720":
                    print(f"    ⚠️  Warning: Expected 1280x720, got {video_res}")
                    issues_found += 1

        print()

    # Quick check remaining samples (just count cameras)
    if len(sample_dirs) > samples_to_check:
        print(f"Quick checking remaining {len(sample_dirs) - samples_to_check} samples...")
        for sample_dir in sample_dirs[samples_to_check:]:
            camera_count = sum(1 for c in expected_cameras if (sample_dir / c).exists())
            if camera_count != 7:
                print(f"  ⚠️  {sample_dir.name}: only {camera_count}/7 cameras")
                issues_found += 1

    print(f"\n{'='*60}")
    print(f"Verification Summary for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(sample_dirs)}")
    print(f"Samples checked in detail: {samples_to_check}")
    print(f"Videos verified: {total_videos_checked}")
    print(f"Issues found: {issues_found}")

    if issues_found == 0:
        print(f"\n✅ {dataset_name} dataset looks good!")
    else:
        print(f"\n⚠️  Found {issues_found} issues that need attention")

    print(f"{'='*60}\n")


def main():
    """Verify both datasets."""
    dataset_root = "/mnt/zihanw/cosmos-transfer2.5/datasets"

    # Verify RGBCloud
    verify_dataset(dataset_root, "RGBCloud", expected_frames=10)

    # Verify DepthSparse
    verify_dataset(dataset_root, "DepthSparse", expected_frames=10)

    print("\n✅ Verification complete!\n")


if __name__ == "__main__":
    main()
