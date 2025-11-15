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
Experiment configuration for segmented multiview dataset training.

Training setup:
- Total training time: 30 hours
- Processing time per file: 1 second
- Total iterations: 108,000 (30 hours * 3600 seconds)
- Checkpoints: 6 saves (every 18,000 iterations)
- Resolution: 720p
- Control input: Depth
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict


# Training configuration for 30-hour run with 6 checkpoints
TRAINING_CONFIG_30H = dict(
    max_iter=108000,  # 30 hours * 3600 seconds
    logging_iter=500,  # Log every 500 iterations (~8 minutes)
    validation_iter=5000,  # Validate every 5000 iterations (~1.4 hours)
    run_validation=True,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=5000,  # Sample from regular model every 5000 iterations
        ),
        every_n_sample_ema=dict(
            every_n=5000,  # Sample from EMA model every 5000 iterations
        ),
    ),
)

# Checkpoint configuration - save 6 checkpoints evenly spaced
CHECKPOINT_CONFIG = dict(
    save_iter=18000,  # Save every 18,000 iterations (5 hours)
    # This gives us 6 checkpoints: iter 18k, 36k, 54k, 72k, 90k, 108k
    load_training_state=True,
    strict_resume=True,
)


# Main experiment: 720p depth-controlled training
segmented_multiview_720p_depth_30h = LazyDict(
    dict(
        defaults=[
            "_self_",
            {"override /data_train": "segmented_multiview_720p"},
            {"override /data_val": "segmented_multiview_720p"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /model": "ddp"},
            {"override /callbacks": "basic"},
            {"override /conditioner": "video_prediction_control_conditioner"},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /checkpoint": "s3"},
            {"override /ckpt_type": "dummy"},
        ],
        job=dict(
            project="cosmos_transfer2",
            group="segmented_multiview_training",
            name="segmented_mv_720p_depth_30h",
        ),
        trainer=TRAINING_CONFIG_30H,
        checkpoint=CHECKPOINT_CONFIG,
        model=dict(
            config=dict(
                state_t=8,  # Latent video length
                resolution="720p",
                preset_hint_keys=["control_input_depth"],  # Only depth control
                fsdp_shard_size=16,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=dict(
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        ),
    ),
    flags={"allow_objects": True},
)


# Debug version - short run for testing
segmented_multiview_720p_depth_debug = LazyDict(
    dict(
        defaults=[
            "segmented_multiview_720p_depth_30h",
            "_self_",
        ],
        job=dict(
            group="segmented_multiview_debug",
            name="segmented_mv_720p_depth_debug_${now:%Y-%m-%d}_${now:%H-%M-%S}",
        ),
        trainer=dict(
            max_iter=100,
            logging_iter=10,
            validation_iter=50,
            run_validation=True,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=50),
                every_n_sample_ema=dict(every_n=50),
            ),
        ),
        checkpoint=dict(
            save_iter=50,
            load_training_state=False,
            strict_resume=False,
        ),
    ),
)


# 480p version - lower resolution for faster training/testing
segmented_multiview_480p_depth_30h = LazyDict(
    dict(
        defaults=[
            "segmented_multiview_720p_depth_30h",
            {"override /data_train": "segmented_multiview_480p"},
            {"override /data_val": "segmented_multiview_480p"},
            "_self_",
        ],
        job=dict(
            group="segmented_multiview_training",
            name="segmented_mv_480p_depth_30h",
        ),
        model=dict(
            config=dict(
                resolution="480p",
            ),
        ),
    ),
)


# Register all experiment configurations
cs = ConfigStore.instance()

experiments = [
    segmented_multiview_720p_depth_30h,
    segmented_multiview_720p_depth_debug,
    segmented_multiview_480p_depth_30h,
]

for exp in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=exp["job"]["name"],
        node=exp,
    )
