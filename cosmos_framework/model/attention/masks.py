# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Mask utilities
"""

from enum import Enum


class CausalType(Enum):
    """
    Different types of causal masking supported by backends of interest.
    """

    TopLeft = 0

    BottomRight = 1

    DontCare = 2
