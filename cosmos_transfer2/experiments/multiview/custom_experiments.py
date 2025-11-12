# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom experiment configurations for RGBCloud and DepthSparse post-training
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

# ============================================================================
# RGBCloud Post-training Configuration
# ============================================================================

rgbcloud_posttrain = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "rgbcloud_multiview_train"},
    ],
    job=dict(
        project="cosmos_transfer_v2p5",
        group="custom_datasets",
        name="rgbcloud_vis_control",
    ),
    checkpoint=dict(
        save_iter=500,  # Save checkpoint every 500 iterations
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem, not S3
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            base_load_from=None,
            hint_keys="vis",  # Using vis control for RGB colored point cloud
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=10_000,  # Adjust based on your dataset size
        validation_iter=500,
        run_validation=False,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=500,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=500,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
            frame_loss_log=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,  # Must match number of GPUs (set to 2, 4, or 8)
    ),
)

# ============================================================================
# DepthSparse Post-training Configuration
# ============================================================================

depthsparse_posttrain = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "depthsparse_multiview_train"},
    ],
    job=dict(
        project="cosmos_transfer_v2p5",
        group="custom_datasets",
        name="depthsparse_depth_control",
    ),
    checkpoint=dict(
        save_iter=500,  # Save checkpoint every 500 iterations
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem, not S3
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            base_load_from=None,
            hint_keys="depth",  # Using depth control for depth sparse point cloud
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=10_000,  # Adjust based on your dataset size
        validation_iter=500,
        run_validation=False,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=500,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=500,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
            frame_loss_log=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=2,  # Must match number of GPUs (set to 2, 4, or 8)
    ),
)

# ============================================================================
# Register Experiments to Hydra ConfigStore
# ============================================================================

cs = ConfigStore.instance()

for _item in [
    rgbcloud_posttrain,
    depthsparse_posttrain,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
