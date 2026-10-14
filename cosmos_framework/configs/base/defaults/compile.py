# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""User-facing torch.compile knobs for VFM and VLM training paths."""

import attrs


@attrs.define(slots=False)
class CompileConfig:
    enabled: bool = False

    compiled_region: str = attrs.field(
        default="language",
        validator=attrs.validators.in_({"all", "language"}),
    )

    compile_dynamic: bool = True

    use_cuda_graphs: bool = False

    max_autotune_pointwise: bool = False

    coordinate_descent_tuning: bool = False
