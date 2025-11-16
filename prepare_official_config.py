#!/usr/bin/env python3
"""
Prepare dataset using OFFICIAL configuration: state_t=8, num_frames=29

This matches the pretrained model's configuration exactly.
"""

import subprocess
from pathlib import Path
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Camera mapping
CAMERA_MAPPING = {
    "FL": "ftheta_camera_cross_left_120fov",
    "FR": "ftheta_camera_cross_right_120fov",
    "RL": "ftheta_camera_rear_left_70fov",
    "FN": "ftheta_camera_front_tele_30fov",
    "RR": "ftheta_camera_rear_right_70fov",
    "FW": "ftheta_camera_front_wide_120fov",
    "RN": "ftheta_camera_rear_tele_30fov",
}

DEFAULT_CAPTION = "This multi-camera perspective captures a drive along a multi-lane urban freeway."


def extract_frames_to_video(input_video: Path, output_video: Path, start_frame: int, num_frames: int):
    """Extract specific frames from video."""
    cmd = [
        "ffmpeg",
        "-i", str(input_video),
        "-vf", f"select='gte(n,{start_frame})*lt(n,{start_frame + num_frames})'",
        "-vsync", "0",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-y",
        str(output_video),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def concatenate_and_segment(scene_id: str, camera_abbr: str, camera_full: str,
                            gt_base: Path, control_base: Path,
                            output_videos: Path, output_control: Path):
    """
    Concatenate 10-frame segments and re-segment to 29 frames.
    90 frames total -> 3 segments of 29 frames each (87 frames used, 3 frames discarded)
    """
    import tempfile

    # Find all segment files
    gt_files = sorted(gt_base.glob(f"{camera_abbr}_GT_seg*.mp4"))
    control_files = sorted(control_base.glob(f"{camera_abbr}_*_seg*.mp4"))

    if len(gt_files) != 9 or len(control_files) != 9:
        print(f"Warning: {scene_id}/{camera_abbr} has {len(gt_files)} GT and {len(control_files)} control files")
        return False

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Concatenate GT segments
        concat_list = temp_dir / "gt_concat.txt"
        with open(concat_list, 'w') as f:
            for gt_file in gt_files:
                f.write(f"file '{gt_file.absolute()}'\n")

        gt_concat = temp_dir / "gt_full.mp4"
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c", "copy", "-y", str(gt_concat)]
        subprocess.run(cmd, capture_output=True, check=True)

        # Concatenate control segments
        concat_list = temp_dir / "control_concat.txt"
        with open(concat_list, 'w') as f:
            for control_file in control_files:
                f.write(f"file '{control_file.absolute()}'\n")

        control_concat = temp_dir / "control_full.mp4"
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-c", "copy", "-y", str(control_concat)]
        subprocess.run(cmd, capture_output=True, check=True)

        # Create output directories
        (output_videos / camera_full).mkdir(parents=True, exist_ok=True)
        (output_control / camera_full).mkdir(parents=True, exist_ok=True)

        # Extract 3 segments of 29 frames each
        # Segment 1: frames 0-28
        # Segment 2: frames 29-57
        # Segment 3: frames 58-86
        for seg_idx in range(3):
            start_frame = seg_idx * 29

            # GT video
            gt_out = output_videos / camera_full / f"{scene_id}_seg{seg_idx+1:02d}.mp4"
            extract_frames_to_video(gt_concat, gt_out, start_frame, 29)

            # Control video
            control_out = output_control / camera_full / f"{scene_id}_seg{seg_idx+1:02d}.mp4"
            extract_frames_to_video(control_concat, control_out, start_frame, 29)

        # Cleanup
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        print(f"Error processing {scene_id}/{camera_abbr}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def main():
    data_root = Path("/mnt/zihanw/cosmos-transfer2.5/data_prepa")
    output_root = Path("/mnt/zihanw/cosmos-transfer2.5/datasets_official")

    print(f"\n{'='*60}")
    print(f"Preparing dataset with OFFICIAL configuration")
    print(f"state_t=8, num_frames=29 (matches pretrained model)")
    print(f"{'='*60}\n")

    for dataset_type in ["RGBCloud", "DepthSparse"]:
        control_type = "color_segments" if dataset_type == "RGBCloud" else "depth_segments"

        print(f"\nProcessing {dataset_type}...")

        output_dataset = output_root / dataset_type
        output_videos = output_dataset / "videos"
        output_control = output_dataset / "control_input_hdmap_bbox"
        output_captions = output_dataset / "captions" / "ftheta_camera_front_wide_120fov"

        output_captions.mkdir(parents=True, exist_ok=True)

        # Find scenes
        gt_root = data_root / "GT"
        guidance_root = data_root / "guidence"

        scenes = sorted([d.name.split("_")[0] for d in gt_root.iterdir() if d.is_dir()])

        print(f"  Found {len(scenes)} scenes")
        print(f"  90 frames -> 3 segments of 29 frames each")
        print(f"  Total samples: {len(scenes)} * 3 = {len(scenes)*3}")

        # Process all scene/camera combinations
        tasks = []
        for scene_id in scenes:
            gt_segments = gt_root / f"{scene_id}_GT_90frames_1280x720" / "GT_segments"
            control_segments = guidance_root / f"{scene_id}_90frames_1280x720" / control_type

            for camera_abbr, camera_full in CAMERA_MAPPING.items():
                tasks.append((scene_id, camera_abbr, camera_full,
                            gt_segments, control_segments,
                            output_videos, output_control))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concatenate_and_segment, *task) for task in tasks]
            for future in tqdm(futures, desc=f"  Processing {dataset_type}"):
                future.result()

        # Create captions
        for scene_id in scenes:
            for seg_idx in range(3):
                caption_file = output_captions / f"{scene_id}_seg{seg_idx+1:02d}.json"
                with open(caption_file, 'w') as f:
                    json.dump({"caption": DEFAULT_CAPTION}, f, indent=2)

        print(f"  ✅ {dataset_type} complete!")

    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"Total samples: {len(scenes) * 3}")
    print(f"Configuration: state_t=8, num_frames=29 (OFFICIAL)")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("1. Backup old dataset and use new one:")
    print("   mv /mnt/zihanw/cosmos-transfer2.5/datasets /mnt/zihanw/cosmos-transfer2.5/datasets_old")
    print("   mv /mnt/zihanw/cosmos-transfer2.5/datasets_official /mnt/zihanw/cosmos-transfer2.5/datasets")
    print("\n2. Verify:")
    print("   python verify_dataset.py")
    print("\n3. Start training:")
    print("   bash train_rgbcloud.sh")


if __name__ == "__main__":
    main()
