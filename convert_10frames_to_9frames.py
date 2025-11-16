#!/usr/bin/env python3
"""
Convert existing 10-frame videos to 9-frame videos to match state_t=3 requirement.

Formula: expected_frames = (state_t - 1) * 4 + 1
For state_t=3: expected_frames = (3-1)*4+1 = 9 frames
"""

import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def convert_video_to_9frames(input_path: Path, output_path: Path) -> bool:
    """
    Convert a 10-frame video to 9-frame video by extracting first 9 frames.

    Args:
        input_path: Path to input 10-frame video
        output_path: Path to output 9-frame video

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use ffmpeg to extract first 9 frames
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", "select='lt(n,9)'",  # Select first 9 frames
            "-vsync", "0",  # Don't duplicate or drop frames
            "-c:v", "libx264",  # Use H.264 codec
            "-crf", "18",  # High quality
            "-preset", "fast",
            "-y",  # Overwrite output
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr}")
        return False


def convert_dataset(dataset_root: str, dataset_name: str, max_workers: int = 8):
    """
    Convert all videos in a dataset from 10 frames to 9 frames.

    Args:
        dataset_root: Root directory containing datasets
        dataset_name: Name of dataset (e.g., "RGBCloud")
        max_workers: Number of parallel workers
    """
    dataset_path = Path(dataset_root) / dataset_name

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"\n{'='*60}")
    print(f"Converting {dataset_name} dataset: 10 frames → 9 frames")
    print(f"{'='*60}")

    # Process both videos and control_input directories
    for subdir_name in ["videos", "control_input_hdmap_bbox"]:
        subdir_path = dataset_path / subdir_name

        if not subdir_path.exists():
            print(f"Subdirectory not found: {subdir_path}")
            continue

        print(f"\nProcessing {subdir_name}...")

        # Find all mp4 files
        all_videos = list(subdir_path.rglob("*.mp4"))
        print(f"Found {len(all_videos)} videos")

        # Create temporary output directory
        temp_dir = dataset_path / f"{subdir_name}_9frames"
        temp_dir.mkdir(exist_ok=True)

        # Convert videos in parallel
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for video_path in all_videos:
                # Get relative path from subdir
                rel_path = video_path.relative_to(subdir_path)
                output_path = temp_dir / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                task = executor.submit(convert_video_to_9frames, video_path, output_path)
                tasks.append((task, video_path, output_path))

            # Monitor progress
            successful = 0
            failed = 0
            for task, input_path, output_path in tqdm(tasks, desc=f"Converting {subdir_name}"):
                if task.result():
                    successful += 1
                else:
                    failed += 1

        print(f"  Converted: {successful} videos")
        if failed > 0:
            print(f"  Failed: {failed} videos")

        # Replace original directory with converted one
        if failed == 0:
            print(f"  Replacing original {subdir_name} with converted videos...")
            import shutil
            backup_dir = dataset_path / f"{subdir_name}_10frames_backup"
            shutil.move(str(subdir_path), str(backup_dir))
            shutil.move(str(temp_dir), str(subdir_path))
            print(f"  Original directory backed up to: {backup_dir}")
        else:
            print(f"  ⚠️  Some conversions failed. Original directory unchanged.")
            print(f"  Converted videos are in: {temp_dir}")

    print(f"\n{'='*60}")
    print(f"{dataset_name} conversion complete!")
    print(f"{'='*60}\n")


def main():
    """Convert both RGBCloud and DepthSparse datasets."""
    dataset_root = "/mnt/zihanw/cosmos-transfer2.5/datasets"

    # Convert RGBCloud
    convert_dataset(dataset_root, "RGBCloud", max_workers=8)

    # Convert DepthSparse
    convert_dataset(dataset_root, "DepthSparse", max_workers=8)

    print("\n✅ All datasets converted successfully!\n")
    print("Next steps:")
    print("1. Verify converted videos:")
    print("   python verify_dataset.py")
    print("\n2. Update configuration to use state_t=3:")
    print("   - Edit cosmos_transfer2/experiments/multiview/custom_datasets.py")
    print("   - Change: state_t=3, num_frames=9")
    print("\n3. Start training:")
    print("   bash train_rgbcloud.sh")


if __name__ == "__main__":
    main()
