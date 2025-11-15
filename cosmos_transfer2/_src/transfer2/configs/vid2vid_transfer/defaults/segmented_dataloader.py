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
Dataloader configuration for segmented multi-view dataset.
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.segmented_multiview_dataset import (
    SegmentedMultiviewDataset,
)


# Camera configuration for the 7-camera setup
CAMERA_KEYS = ["FL", "FR", "RL", "FN", "RR", "FW", "RN"]
CAMERA_TO_VIEW_ID = {
    "FN": 0,  # Front Normal - main front view
    "FL": 1,  # Front Left
    "FR": 2,  # Front Right
    "RL": 3,  # Rear Left
    "RR": 4,  # Rear Right
    "FW": 5,  # Front Wide
    "RN": 6,  # Rear Normal
}


def register_segmented_multiview_dataloader():
    """Register the segmented multiview dataloader configurations."""
    cs = ConfigStore.instance()

    # 720p configuration
    segmented_dataset_720p = L(SegmentedMultiviewDataset)(
        data_root="/mnt/zihanw/cosmos-transfer2.5/data_prepa",
        camera_keys=CAMERA_KEYS,
        camera_to_view_id=CAMERA_TO_VIEW_ID,
        num_frames=90,
        resolution="720p",
        video_size=(720, 1280),
        hint_key="control_input_depth",
        front_camera_key="FN",
        state_t=8,
        is_train=True,
        train_val_split=0.9,
    )

    # Training dataloader
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="segmented_multiview_720p",
        node=L(get_generic_dataloader)(
            dataset=segmented_dataset_720p,
            sampler=L(get_sampler)(dataset=segmented_dataset_720p),
            batch_size=1,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    # Validation dataloader (same dataset with is_train=False)
    segmented_dataset_720p_val = L(SegmentedMultiviewDataset)(
        data_root="/mnt/zihanw/cosmos-transfer2.5/data_prepa",
        camera_keys=CAMERA_KEYS,
        camera_to_view_id=CAMERA_TO_VIEW_ID,
        num_frames=90,
        resolution="720p",
        video_size=(720, 1280),
        hint_key="control_input_depth",
        front_camera_key="FN",
        state_t=8,
        is_train=False,
        train_val_split=0.9,
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="segmented_multiview_720p",
        node=L(get_generic_dataloader)(
            dataset=segmented_dataset_720p_val,
            sampler=L(get_sampler)(dataset=segmented_dataset_720p_val),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    # 480p configuration (optional, for lower resolution training)
    segmented_dataset_480p = L(SegmentedMultiviewDataset)(
        data_root="/mnt/zihanw/cosmos-transfer2.5/data_prepa",
        camera_keys=CAMERA_KEYS,
        camera_to_view_id=CAMERA_TO_VIEW_ID,
        num_frames=90,
        resolution="480p",
        video_size=(480, 832),
        hint_key="control_input_depth",
        front_camera_key="FN",
        state_t=8,
        is_train=True,
        train_val_split=0.9,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="segmented_multiview_480p",
        node=L(get_generic_dataloader)(
            dataset=segmented_dataset_480p,
            sampler=L(get_sampler)(dataset=segmented_dataset_480p),
            batch_size=1,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    segmented_dataset_480p_val = L(SegmentedMultiviewDataset)(
        data_root="/mnt/zihanw/cosmos-transfer2.5/data_prepa",
        camera_keys=CAMERA_KEYS,
        camera_to_view_id=CAMERA_TO_VIEW_ID,
        num_frames=90,
        resolution="480p",
        video_size=(480, 832),
        hint_key="control_input_depth",
        front_camera_key="FN",
        state_t=8,
        is_train=False,
        train_val_split=0.9,
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name="segmented_multiview_480p",
        node=L(get_generic_dataloader)(
            dataset=segmented_dataset_480p_val,
            sampler=L(get_sampler)(dataset=segmented_dataset_480p_val),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
