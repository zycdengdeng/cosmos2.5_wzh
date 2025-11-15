# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset for segmented multi-view videos with GT and guidance (color/depth) videos.
Designed for training with 1 second per file processing time.
"""

import glob
import os
import traceback
import warnings
from typing import Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.modules.input_handling.utils import detect_aspect_ratio
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
)
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import get_hdmap_augmentor_for_local_datasets


DEFAULT_PROMPT = "Multi-camera perspective of an autonomous driving scenario with various camera views providing comprehensive spatial coverage."


class SegmentedMultiviewDataset(Dataset):
    """
    Dataset for loading segmented multi-view videos.

    Data structure expected:
    - GT videos: {data_root}/GT/{scene_id}_GT_90frames_1280x720/GT_segments/{camera}/{camera}_GT_seg{num:02d}.mp4
    - Color guidance: {data_root}/guidence/{scene_id}_90frames_1280x720/color_segments/{camera}/{camera}_color_seg{num:02d}.mp4
    - Depth guidance: {data_root}/guidence/{scene_id}_90frames_1280x720/depth_segments/{camera}/{camera}_depth_seg{num:02d}.mp4
    """

    def __init__(
        self,
        data_root: str,
        camera_keys: List[str],
        camera_to_view_id: Dict[str, int],
        num_frames: int = 90,
        resolution: str = "720p",
        video_size: tuple = (720, 1280),
        hint_key: str = "control_input_depth",
        front_camera_key: str = None,
        state_t: int = 8,
        is_train: bool = True,
        train_val_split: float = 0.9,
        **kwargs,
    ):
        """
        Args:
            data_root: Root directory containing GT and guidence folders
            camera_keys: List of camera names (e.g., ['FL', 'FR', 'RL', 'FN', 'RR', 'FW', 'RN'])
            camera_to_view_id: Mapping from camera name to view index
            num_frames: Number of frames per video segment
            resolution: Resolution string (e.g., '720p')
            video_size: (H, W) tuple for video size
            hint_key: Control input key name
            front_camera_key: Front camera key for main view
            state_t: State t parameter
            is_train: Whether this is training set
            train_val_split: Train/val split ratio
        """
        super().__init__()
        self.data_root = data_root
        self.camera_keys = camera_keys
        self.camera_to_view_id = camera_to_view_id
        self.num_frames = num_frames
        self.resolution = resolution
        self.H, self.W = video_size
        self.hint_key = hint_key
        self.front_camera_key = front_camera_key or camera_keys[0]
        self.state_t = state_t
        self.is_train = is_train

        # Parse control type from hint_key
        self.ctrl_type = hint_key.replace("control_input_", "")

        # Setup augmentor
        augmentor_cfg = get_hdmap_augmentor_for_local_datasets(
            resolution=resolution,
            control_input_type=self.ctrl_type
        )
        self.augmentor = {k: instantiate(v) for k, v in augmentor_cfg.items()}

        # Find all GT video segments
        self.samples = self._init_samples()

        # Train/val split
        cutoff = int(len(self.samples) * train_val_split)
        self.samples = self.samples[:cutoff] if is_train else self.samples[cutoff:]

        log.info(f"Loaded {len(self.samples)} samples for {'training' if is_train else 'validation'}")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((self.H, self.W))])

    def _init_samples(self) -> List[Dict]:
        """
        Initialize list of all video segments across all scenes and cameras.
        """
        samples = []
        gt_base = os.path.join(self.data_root, "GT")

        # Find all scene directories
        scene_dirs = sorted(glob.glob(os.path.join(gt_base, "*_GT_90frames_1280x720")))

        for scene_dir in scene_dirs:
            scene_id = os.path.basename(scene_dir).split("_")[0]

            # For each camera, find all segment files
            for camera in self.camera_keys:
                gt_camera_dir = os.path.join(scene_dir, "GT_segments", camera)

                if not os.path.exists(gt_camera_dir):
                    log.warning(f"GT camera dir not found: {gt_camera_dir}")
                    continue

                # Find all segment files for this camera
                segment_files = sorted(glob.glob(os.path.join(gt_camera_dir, f"{camera}_GT_seg*.mp4")))

                for gt_file in segment_files:
                    # Extract segment number
                    seg_num = os.path.basename(gt_file).replace(f"{camera}_GT_seg", "").replace(".mp4", "")

                    sample = {
                        "scene_id": scene_id,
                        "camera": camera,
                        "segment_num": seg_num,
                        "gt_file": gt_file,
                    }
                    samples.append(sample)

        log.info(f"Found {len(samples)} total video segments")
        return samples

    def _load_video(self, video_path: str) -> tuple:
        """Load video and return frames as numpy array and fps."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        # Load all frames
        frame_ids = list(range(min(n_frames, self.num_frames)))
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 24

        return frame_data, fps

    def _get_guidance_paths(self, scene_id: str, camera: str, segment_num: str) -> Dict[str, str]:
        """Get paths for color and depth guidance videos."""
        base_dir = os.path.join(self.data_root, "guidence", f"{scene_id}_90frames_1280x720")

        paths = {
            "color": os.path.join(base_dir, "color_segments", camera, f"{camera}_color_seg{segment_num}.mp4"),
            "depth": os.path.join(base_dir, "depth_segments", camera, f"{camera}_depth_seg{segment_num}.mp4"),
        }

        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            scene_id = sample["scene_id"]
            camera = sample["camera"]
            segment_num = sample["segment_num"]

            # Load GT video
            gt_frames, fps = self._load_video(sample["gt_file"])

            # Get guidance paths
            guidance_paths = self._get_guidance_paths(scene_id, camera, segment_num)

            # Load guidance videos based on control type
            control_data = {}
            if "depth" in self.ctrl_type:
                if os.path.exists(guidance_paths["depth"]):
                    depth_frames, _ = self._load_video(guidance_paths["depth"])
                    depth_frames_t = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
                    depth_frames_t = self.preprocess(depth_frames_t)
                    depth_frames_t = torch.clamp(depth_frames_t * 255.0, 0, 255).to(torch.uint8)
                    control_data["depth"] = {
                        "video": depth_frames_t.permute(1, 0, 2, 3),  # T,C,H,W -> C,T,H,W
                        "frame_start": 0,
                        "frame_end": len(depth_frames) - 1,
                    }
                else:
                    raise FileNotFoundError(f"Depth guidance not found: {guidance_paths['depth']}")

            # Process GT video frames
            gt_frames_t = torch.from_numpy(gt_frames.astype(np.uint8)).permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
            gt_frames_t = self.preprocess(gt_frames_t)
            gt_frames_t = torch.clamp(gt_frames_t * 255.0, 0, 255).to(torch.uint8)
            video = gt_frames_t.permute(1, 0, 2, 3)  # T,C,H,W -> C,T,H,W

            aspect_ratio = detect_aspect_ratio((self.W, self.H))

            # Prepare data for augmentor
            data_for_augmentor = {
                "video": video,
                "frame_start": 0,
                "frame_end": len(gt_frames) - 1,
                "frame_indices": list(range(len(gt_frames))),
                "aspect_ratio": aspect_ratio,
                "fps": fps,
                "ai_caption": DEFAULT_PROMPT,
                "video_name": {
                    "video_path": sample["gt_file"],
                    "scene_id": scene_id,
                    "camera": camera,
                    "segment": segment_num,
                }
            }

            # Add control data
            data_for_augmentor.update(control_data)

            # Apply augmentor
            for _, aug_fn in self.augmentor.items():
                augmented_data = aug_fn(data_for_augmentor)

            # Prepare final output
            final_data = {
                "video": augmented_data["video"],
                self.hint_key: augmented_data[self.hint_key],
                "image_size": torch.tensor([self.H, self.W, self.H, self.W]),
                "fps": fps,
                "sample_n_views": 1,  # Single view per sample
                "num_video_frames_per_view": len(gt_frames),
                "view_indices": torch.tensor([self.camera_to_view_id[camera]]).repeat(len(gt_frames)),
                "latent_view_indices_B_T": torch.tensor([self.camera_to_view_id[camera]]).repeat(self.state_t),
                "video_name": data_for_augmentor["video_name"],
                "aspect_ratio": aspect_ratio,
                "ai_caption": DEFAULT_PROMPT,
                "padding_mask": torch.zeros(1, self.H, self.W),
                "ref_cam_view_idx_sample_position": -1,
                "front_cam_view_idx_sample_position": torch.tensor([0]),
            }

            return final_data

        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load sample {index} (failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            # Return a random sample instead
            return self[np.random.randint(len(self))]


if __name__ == "__main__":
    # Example usage
    from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
    from torch.utils.data import DataLoader

    camera_keys = ["FL", "FR", "RL", "FN", "RR", "FW", "RN"]
    camera_to_view_id = {cam: idx for idx, cam in enumerate(camera_keys)}

    dataset = L(SegmentedMultiviewDataset)(
        data_root="/mnt/zihanw/cosmos-transfer2.5/data_prepa",
        camera_keys=camera_keys,
        camera_to_view_id=camera_to_view_id,
        num_frames=90,
        resolution="720p",
        video_size=(720, 1280),
        hint_key="control_input_depth",
        front_camera_key="FN",
        state_t=8,
        is_train=True,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=4, pin_memory=True, drop_last=True)
    data = next(iter(dataloader))
    print(f"Video shape: {data['video'].shape}")
    print(f"Control input shape: {data['control_input_depth'].shape}")
