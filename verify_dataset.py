#!/usr/bin/env python3
"""
Verify the prepared dataset structure and content for MultiviewTransferDataset format.

Expected structure:
  datasets/{dataset_name}/
    videos/
      {camera_name}/
        {scene}_seg{N}.mp4
        ...
    control_input_hdmap_bbox/
      {camera_name}/
        {scene}_seg{N}.mp4
        ...
    captions/
      {front_camera_name}/
        {scene}_seg{N}.json
        ...
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
    Verify dataset structure and content for MultiviewTransferDataset format.

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

    front_camera = "ftheta_camera_front_wide_120fov"

    # Check directories exist
    videos_dir = dataset_path / "videos"
    control_dir = dataset_path / "control_input_hdmap_bbox"
    captions_dir = dataset_path / "captions"

    if not videos_dir.exists():
        print(f"❌ Missing videos directory: {videos_dir}")
        return
    if not control_dir.exists():
        print(f"❌ Missing control_input_hdmap_bbox directory: {control_dir}")
        return
    if not captions_dir.exists():
        print(f"❌ Missing captions directory: {captions_dir}")
        return

    print("\n✓ Directory structure looks good")

    # Check front camera videos to get sample count
    front_video_dir = videos_dir / front_camera
    if not front_video_dir.exists():
        print(f"❌ Missing front camera video directory: {front_video_dir}")
        return

    video_files = sorted(list(front_video_dir.glob("*.mp4")))
    print(f"\nFound {len(video_files)} samples (based on front camera videos)")

    if len(video_files) == 0:
        print("❌ No video files found!")
        return

    # Check first few samples in detail
    samples_to_check = min(3, len(video_files))
    print(f"\nChecking first {samples_to_check} samples in detail...\n")

    issues_found = 0
    total_videos_checked = 0

    for i, video_file in enumerate(video_files[:samples_to_check]):
        sample_name = video_file.stem  # e.g., "002_seg01"
        print(f"Sample {i+1}: {sample_name}")

        # Check all cameras for this sample
        cameras_ok = 0
        for camera in expected_cameras:
            # Check video exists
            video_path = videos_dir / camera / f"{sample_name}.mp4"
            control_path = control_dir / camera / f"{sample_name}.mp4"

            if not video_path.exists():
                print(f"  ❌ Missing video: {camera}/{sample_name}.mp4")
                issues_found += 1
                continue

            if not control_path.exists():
                print(f"  ❌ Missing control: {camera}/{sample_name}.mp4")
                issues_found += 1
                continue

            cameras_ok += 1

        # Check caption
        caption_path = captions_dir / front_camera / f"{sample_name}.json"
        if not caption_path.exists():
            print(f"  ❌ Missing caption: {sample_name}.json")
            issues_found += 1

        if cameras_ok == len(expected_cameras):
            print(f"  ✓ All {cameras_ok} cameras present")

        # Verify video properties for front camera
        video_path = videos_dir / front_camera / f"{sample_name}.mp4"
        control_path = control_dir / front_camera / f"{sample_name}.mp4"

        video_frames, video_res = check_video_frames(video_path)
        control_frames, control_res = check_video_frames(control_path)

        total_videos_checked += 2

        print(f"  ✓ {front_camera}:")
        print(f"    video: {video_frames} frames, {video_res}")
        print(f"    control: {control_frames} frames, {control_res}")

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

    # Quick check remaining samples
    if len(video_files) > samples_to_check:
        print(f"Quick checking remaining {len(video_files) - samples_to_check} samples...")
        for video_file in video_files[samples_to_check:]:
            sample_name = video_file.stem
            camera_count = sum(
                1
                for c in expected_cameras
                if (videos_dir / c / f"{sample_name}.mp4").exists()
                and (control_dir / c / f"{sample_name}.mp4").exists()
            )
            if camera_count != 7:
                print(f"  ⚠️  {sample_name}: only {camera_count}/7 cameras")
                issues_found += 1

    print(f"\n{'='*60}")
    print(f"Verification Summary for {dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(video_files)}")
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

    # Verify RGBCloud (9 frames for state_t=3)
    verify_dataset(dataset_root, "RGBCloud", expected_frames=9)

    # Verify DepthSparse (9 frames for state_t=3)
    verify_dataset(dataset_root, "DepthSparse", expected_frames=9)

    print("\n✅ Verification complete!\n")


if __name__ == "__main__":
    main()
