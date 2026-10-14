# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing parallelism degrees shared by VFM and VLM trainers."""

import attrs
import torch

PRECISION_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@attrs.define(slots=False)
class ParallelismConfig:
    data_parallel_shard_degree: int = -1

    data_parallel_replicate_degree: int = 1

    context_parallel_shard_degree: int = 1

    cfg_parallel_shard_degree: int = 1

    enable_inference_mode: bool = False

    fsdp_master_dtype: str = "float32"
