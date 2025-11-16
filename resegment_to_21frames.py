#!/usr/bin/env python3
"""
Re-segment videos into 21-frame segments for state_t=6.

Formula: expected_frames = (state_t - 1) * 4 + 1
For state_t=6: expected_frames = (6-1)*4+1 = 21 frames

This script will:
1. Concatenate all 9 segments of each scene (9*10frames = 90 frames total)
2. Re-segment into 21-frame chunks (90 frames -> 4 segments of 21 frames each)
"""

import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import shutil


def concatenate_segments(scene_id: str, camera_abbr: str, base_dir: Path, num_segments: int = 9) -> Path:
    """
    Concatenate all segments of a scene/camera into one video.

    Returns:
        Path to concatenated video
    """
    # Create concat file list
    temp_dir = Path(tempfile.mkdtemp())
    concat_file = temp_dir / f"concat_{scene_id}_{camera_abbr}.txt"

    with open(concat_file, 'w') as f:
        for seg_num in range(1, num_segments + 1):
            seg_file = base_dir / f"{camera_abbr}_*_seg{seg_num:02d}.mp4"
            matching = list(base_dir.glob(f"{camera_abbr}_*_seg{seg_num:02d}.mp4"))
            if matching:
                f.write(f"file '{matching[0].absolute()}'\n")

    # Concatenate videos
    output_path = temp_dir / f"concatenated_{scene_id}_{camera_abbr}.mp4"
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        "-y",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def segment_video_21frames(input_video: Path, output_dir: Path, scene_id: str, camera_abbr: str, segment_size: int = 21):
    """
    Segment a video into 21-frame chunks.
    """
    # Get total frames
    cmd_frames = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(input_video),
    ]
    result = subprocess.run(cmd_frames, capture_output=True, text=True, check=True)
    total_frames = int(result.stdout.strip())

    num_segments = total_frames // segment_size

    for seg_idx in range(num_segments):
        start_frame = seg_idx * segment_size
        output_file = output_dir / f"{camera_abbr}_{scene_id}_seg{seg_idx+1:02d}.mp4"

        cmd = [
            "ffmpeg",
            "-i", str(input_video),
            "-vf", f"select='gte(n,{start_frame})*lt(n,{start_frame + segment_size})'",
            "-vsync", "0",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-y",
            str(output_file),
        ]

        subprocess.run(cmd, capture_output=True, check=True)


def process_scene_camera(scene_id: str, camera_abbr: str, camera_full_name: str,
                         gt_dir: Path, control_dir: Path,
                         output_videos_dir: Path, output_control_dir: Path):
    """
    Process one scene/camera: concatenate 10-frame segments, then re-segment to 21 frames.
    """
    try:
        # Process GT videos
        gt_concat = concatenate_segments(scene_id, camera_abbr, gt_dir / camera_abbr)
        gt_output_dir = output_videos_dir / camera_full_name
        gt_output_dir.mkdir(parents=True, exist_ok=True)
        segment_video_21frames(gt_concat, gt_output_dir, scene_id, camera_full_name.split('_')[-2], segment_size=21)

        # Process control videos
        control_concat = concatenate_segments(scene_id, camera_abbr, control_dir / camera_abbr)
        control_output_dir = output_control_dir / camera_full_name
        control_output_dir.mkdir(parents=True, exist_ok=True)
        segment_video_21frames(control_concat, control_output_dir, scene_id, camera_full_name.split('_')[-2], segment_size=21)

        # Cleanup temp files
        shutil.rmtree(gt_concat.parent)
        shutil.rmtree(control_concat.parent)

        return True
    except Exception as e:
        print(f"Error processing {scene_id}/{camera_abbr}: {e}")
        return False


def main():
    """
    Main function to re-segment all videos from 10-frame to 21-frame segments.
    """
    data_root = Path("/mnt/zihanw/cosmos-transfer2.5/data_prepa")
    output_root = Path("/mnt/zihanw/cosmos-transfer2.5/datasets_21frames")

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

    print(f"\n{'='*60}")
    print(f"Re-segmenting videos: 9x10frames -> 4x21frames")
    print(f"{'='*60}\n")

    # Find all scenes
    gt_root = data_root / "GT"
    guidance_root = data_root / "guidence"

    scene_dirs = sorted([d for d in gt_root.iterdir() if d.is_dir()])
    scene_ids = [d.name.split("_")[0] for d in scene_dirs]

    print(f"Found {len(scene_ids)} scenes: {scene_ids}")
    print(f"Each scene: 90 frames -> 4 segments of 21 frames")
    print(f"Total segments: {len(scene_ids)} scenes * 4 segments = {len(scene_ids)*4}\n")

    for dataset_type in ["RGBCloud", "DepthSparse"]:
        control_type = "color_segments" if dataset_type == "RGBCloud" else "depth_segments"

        print(f"\nProcessing {dataset_type} dataset...")

        output_dataset_dir = output_root / dataset_type
        output_videos_dir = output_dataset_dir / "videos"
        output_control_dir = output_dataset_dir / "control_input_hdmap_bbox"
        output_captions_dir = output_dataset_dir / "captions"

        # Create directories
        output_videos_dir.mkdir(parents=True, exist_ok=True)
        output_control_dir.mkdir(parents=True, exist_ok=True)
        (output_captions_dir / "ftheta_camera_front_wide_120fov").mkdir(parents=True, exist_ok=True)

        # Process all scene/camera combinations
        tasks = []
        for scene_id in scene_ids:
            gt_segments = gt_root / f"{scene_id}_GT_90frames_1280x720" / "GT_segments"
            control_segments = guidance_root / f"{scene_id}_90frames_1280x720" / control_type

            for camera_abbr, camera_full_name in CAMERA_MAPPING.items():
                tasks.append((scene_id, camera_abbr, camera_full_name,
                             gt_segments, control_segments,
                             output_videos_dir, output_control_dir))

        print(f"Processing {len(tasks)} scene/camera combinations...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_scene_camera, *task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

        # Create captions (4 per scene instead of 9)
        print(f"Creating captions...")
        caption_dir = output_captions_dir / "ftheta_camera_front_wide_120fov"
        for scene_id in scene_ids:
            for seg_idx in range(1, 5):  # 4 segments
                caption_file = caption_dir / f"FW_{scene_id}_seg{seg_idx:02d}.json"
                import json
                caption_data = {
                    "caption": "This multi-camera perspective captures a drive along a multi-lane urban freeway."
                }
                with open(caption_file, 'w') as f:
                    json.dump(caption_data, f, indent=2)

        print(f"✅ {dataset_type} complete!")

    print(f"\n{'='*60}")
    print(f"All datasets re-segmented successfully!")
    print(f"Total samples: {len(scene_ids) * 4} (was {len(scene_ids) * 9})")
    print(f"Output location: {output_root}")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("1. Move old datasets:")
    print("   mv /mnt/zihanw/cosmos-transfer2.5/datasets /mnt/zihanw/cosmos-transfer2.5/datasets_9frames_old")
    print("   mv /mnt/zihanw/cosmos-transfer2.5/datasets_21frames /mnt/zihanw/cosmos-transfer2.5/datasets")
    print("\n2. Update config: state_t=6, num_frames=21")
    print("\n3. Start training")


if __name__ == "__main__":
    main()
