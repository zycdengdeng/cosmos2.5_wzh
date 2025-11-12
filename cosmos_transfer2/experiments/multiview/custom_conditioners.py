# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom conditioner configurations for RGBCloud (vis) and DepthSparse (depth)
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.conditioner import ReMapkey
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import _SHARED_CONFIG
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import (
    MultiViewControlVideo2WorldConditioner,
)

# Create custom config for vis control
_CUSTOM_CONFIG_VIS = {
    key: value for key, value in _SHARED_CONFIG.items()
    if key not in ['control_input_edge', 'control_input_depth', 'control_input_seg',
                   'control_input_hdmap_bbox', 'control_input_inpaint',
                   'control_input_edge_mask', 'control_input_depth_mask',
                   'control_input_seg_mask', 'control_input_hdmap_bbox_mask',
                   'control_input_inpaint_mask']
}

# Add vis control mapping
_CUSTOM_CONFIG_VIS["control_input_vis"] = L(ReMapkey)(
    input_key="control_input_hdmap_bbox",  # Read from hdmap_bbox directory
    output_key="control_input_vis",        # Map to vis control
    dropout_rate=0.0,
    dtype=None,
)

# Add multiview-specific fields
_CUSTOM_CONFIG_VIS["view_indices_B_T"] = L(ReMapkey)(
    input_key="latent_view_indices_B_T",
    output_key="view_indices_B_T",
    dropout_rate=0.0,
    dtype=None,
)

_CUSTOM_CONFIG_VIS["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
    input_key="ref_cam_view_idx_sample_position",
    output_key="ref_cam_view_idx_sample_position",
    dropout_rate=0.0,
    dtype=None,
)

# Create custom config for depth control
_CUSTOM_CONFIG_DEPTH = {
    key: value for key, value in _SHARED_CONFIG.items()
    if key not in ['control_input_edge', 'control_input_vis', 'control_input_seg',
                   'control_input_hdmap_bbox', 'control_input_inpaint',
                   'control_input_edge_mask', 'control_input_vis_mask',
                   'control_input_seg_mask', 'control_input_hdmap_bbox_mask',
                   'control_input_inpaint_mask']
}

# Add depth control mapping
_CUSTOM_CONFIG_DEPTH["control_input_depth"] = L(ReMapkey)(
    input_key="control_input_hdmap_bbox",  # Read from hdmap_bbox directory
    output_key="control_input_depth",      # Map to depth control
    dropout_rate=0.0,
    dtype=None,
)

# Add multiview-specific fields
_CUSTOM_CONFIG_DEPTH["view_indices_B_T"] = L(ReMapkey)(
    input_key="latent_view_indices_B_T",
    output_key="view_indices_B_T",
    dropout_rate=0.0,
    dtype=None,
)

_CUSTOM_CONFIG_DEPTH["ref_cam_view_idx_sample_position"] = L(ReMapkey)(
    input_key="ref_cam_view_idx_sample_position",
    output_key="ref_cam_view_idx_sample_position",
    dropout_rate=0.0,
    dtype=None,
)

# Create custom conditioners
CustomMultiViewVisConditioner = L(MultiViewControlVideo2WorldConditioner)(
    **_CUSTOM_CONFIG_VIS,
)

CustomMultiViewDepthConditioner = L(MultiViewControlVideo2WorldConditioner)(
    **_CUSTOM_CONFIG_DEPTH,
)


def register_custom_conditioners():
    """Register custom conditioners to Hydra ConfigStore"""
    cs = ConfigStore.instance()

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="custom_multiview_vis_conditioner",
        node=CustomMultiViewVisConditioner,
    )

    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="custom_multiview_depth_conditioner",
        node=CustomMultiViewDepthConditioner,
    )
