# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-view point-tracking head (MVTrack4Gen / MV-TAP) bolted onto the Cosmos3 DiT.

This subpackage vendors the *tracking head* of MV-TAP [Koo et al., 2026] and the
training objective of MVTrack4Gen [Lee et al., 2026], adapted from the paper's
monocular (ref+tgt) setting to our genuinely multi-view (V symmetric views) world
model. The DiT itself is UNCHANGED: we only tap the post-RoPE query/key features of
ONE intermediate 3D-attention layer and route them into an auxiliary
``MultiViewTrackingHead`` that builds local 4D correlation volumes and refines point
tracks — exactly MVTrack4Gen's recipe, but with V views packed on our mRoPE view axis.

Public entry point: :class:`MultiViewTrackingHead`.
"""

from cosmos_framework.model.vfm.tracking.head import MultiViewTrackingHead

__all__ = ["MultiViewTrackingHead"]
