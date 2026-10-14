# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Any, List

import attrs

from cosmos_framework.trainer import ImaginaireTrainer as Trainer
from cosmos_framework.utils import config


@attrs.define(slots=False)
class DataSetting:
    """Configuration for data.

    Attributes:
        qwen_max_video_token_length: Maximum video token length.
        qwen_target_fps: Target fps for video sampling.
        text_chat_order: Order of text items in user messages.
    """

    qwen_max_video_token_length: int = 8192


@attrs.define(slots=False)
class Config(config.Config):
    data_setting: DataSetting = attrs.field(factory=DataSetting)
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"model": "mot_fsdp"},
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "adamw"},
            {"scheduler": "warmup_cosine_lr"},
            {"checkpoint": "s3"},
            {"callbacks": ["basic", "optimization", "job_monitor"]},
            {"ema": "power"},
            {"tokenizer": "wan2pt2_tokenizer"},
            {"cluster": "default"},
            {"vlm_config": None},
            {"ckpt_type": "dcp"},
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
    )

    c.job.project = "cosmos3"
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 20
    c.trainer.callbacks = None

    c.upload_reproducible_setup = False

    from cosmos_framework.configs.base.defaults.callbacks import register_callbacks
    from cosmos_framework.configs.base.defaults.checkpointer import register_checkpoint, register_ckpt_type
    from cosmos_framework.configs.base.defaults.cluster import register_cluster
    from cosmos_framework.configs.base.defaults.ema import register_ema

    from cosmos_framework.configs.base.defaults.model import register_model
    from cosmos_framework.configs.base.defaults.optimizer import register_optimizer, register_scheduler
    from cosmos_framework.configs.base.defaults.tokenizer import register_tokenizer
    from cosmos_framework.configs.base.defaults.vlm import register_vlm

    register_model()
    register_checkpoint()
    register_ckpt_type()
    register_optimizer()
    register_scheduler()
    register_callbacks()
    register_tokenizer()
    register_ema()
    register_cluster()
    register_vlm()

    import cosmos_framework.configs.base.experiment.sft.vision_sft_nano
    import cosmos_framework.configs.base.experiment.sft.vpm
    return c
