# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom training configurations for RGBCloud and DepthSparse datasets
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.multiview_dataset import MultiviewTransferDataset


def get_rgbcloud_multiview_dataset(is_train=True):
    """
    Dataset configuration for RGBCloud (colored point cloud projection)
    Uses control_input_vis control type
    """
    camera_keys = [
        "ftheta_camera_front_wide_120fov",
        "ftheta_camera_cross_left_120fov",
        "ftheta_camera_cross_right_120fov",
        "ftheta_camera_rear_left_70fov",
        "ftheta_camera_rear_right_70fov",
        "ftheta_camera_rear_tele_30fov",
        "ftheta_camera_front_tele_30fov",
    ]
    camera_to_view_id = {
        "ftheta_camera_front_wide_120fov": 0,
        "ftheta_camera_cross_left_120fov": 1,
        "ftheta_camera_cross_right_120fov": 2,
        "ftheta_camera_rear_left_70fov": 3,
        "ftheta_camera_rear_right_70fov": 4,
        "ftheta_camera_rear_tele_30fov": 5,
        "ftheta_camera_front_tele_30fov": 6,
    }

    dataset = L(MultiviewTransferDataset)(
        dataset_dir="/mnt/zihanw/cosmos-transfer2.5/datasets/RGBCloud",
        hint_key="control_input_hdmap_bbox",  # Must match directory name
        resolution="720",
        state_t=8,
        num_frames=25,  # Match actual control_input video frame count
        sequence_interval=1,
        camera_keys=camera_keys,
        video_size=(704, 1280),
        front_camera_key="ftheta_camera_front_wide_120fov",
        camera_to_view_id=camera_to_view_id,
        front_view_caption_only=True,
        is_train=is_train,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=1,
        drop_last=True,
        num_workers=6,  # Increased from 4 to 6 for better GPU feeding
        prefetch_factor=2,  # Restored to 2 for smoother data pipeline
        pin_memory=True,
    )


def get_depthsparse_multiview_dataset(is_train=True):
    """
    Dataset configuration for DepthSparse (depth sparse point cloud projection)
    Uses control_input_depth control type
    """
    camera_keys = [
        "ftheta_camera_front_wide_120fov",
        "ftheta_camera_cross_left_120fov",
        "ftheta_camera_cross_right_120fov",
        "ftheta_camera_rear_left_70fov",
        "ftheta_camera_rear_right_70fov",
        "ftheta_camera_rear_tele_30fov",
        "ftheta_camera_front_tele_30fov",
    ]
    camera_to_view_id = {
        "ftheta_camera_front_wide_120fov": 0,
        "ftheta_camera_cross_left_120fov": 1,
        "ftheta_camera_cross_right_120fov": 2,
        "ftheta_camera_rear_left_70fov": 3,
        "ftheta_camera_rear_right_70fov": 4,
        "ftheta_camera_rear_tele_30fov": 5,
        "ftheta_camera_front_tele_30fov": 6,
    }

    dataset = L(MultiviewTransferDataset)(
        dataset_dir="/mnt/zihanw/cosmos-transfer2.5/datasets/depthsparse",
        hint_key="control_input_hdmap_bbox",  # Must match directory name
        resolution="720",
        state_t=8,
        num_frames=25,  # Match actual control_input video frame count
        sequence_interval=1,
        camera_keys=camera_keys,
        video_size=(704, 1280),
        front_camera_key="ftheta_camera_front_wide_120fov",
        camera_to_view_id=camera_to_view_id,
        front_view_caption_only=True,
        is_train=is_train,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=1,
        drop_last=True,
        num_workers=6,  # Increased from 4 to 6 for better GPU feeding
        prefetch_factor=2,  # Restored to 2 for smoother data pipeline
        pin_memory=True,
    )


def register_custom_datasets():
    """Register custom datasets to Hydra ConfigStore"""
    cs = ConfigStore()

    # Register RGBCloud dataset
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="rgbcloud_multiview_train",
        node=get_rgbcloud_multiview_dataset(is_train=True),
    )

    # Register DepthSparse dataset
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="depthsparse_multiview_train",
        node=get_depthsparse_multiview_dataset(is_train=True),
    )
