# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import PLACEHOLDER, LazyDict
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.model.vfm.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface

Wan2pt2VAEConfig: LazyDict = L(Wan2pt2VAEInterface)(
    bucket_name=PLACEHOLDER,
    object_store_credential_path_pretrained=PLACEHOLDER,
    vae_path="${oc.env:WAN_VAE_PATH,checkpoints/wan22_vae/Wan2.2_VAE.pth}",
    spatial_compression_factor=16,
    temporal_compression_factor=4,
)


def register_tokenizer():
    cs = ConfigStore.instance()
    cs.store(group="tokenizer", package="model.config.tokenizer", name="wan2pt2_tokenizer", node=Wan2pt2VAEConfig)
