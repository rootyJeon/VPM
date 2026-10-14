# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Plucker camera-code adapter for multi-view generation.

Rays are evaluated analytically on the VAE latent grid (one ray per spatial
16x16 image cell), packed in groups of four input frames, and reduced by one
learned 2x2 convolution to the transformer's spatial token grid.  The compact
code is shared by the additive camera branch and every layer's RoCE heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from cosmos_framework.model.vfm.mot.camera_geometry import (
    compute_plucker_rays,
    pack_plucker_rays_temporal,
)


class CameraConditioner(nn.Module):
    """Convert calibrated cameras into one compact code per vision token."""

    def __init__(self, code_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.ray_aggregator = nn.Conv2d(24, self.code_dim, kernel_size=2, stride=2)
        self.code_norm = nn.LayerNorm(self.code_dim)
        self.additive_projection = nn.Linear(self.code_dim, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the stem normally and the behavior-changing output at zero."""

        self.ray_aggregator.reset_parameters()
        self.code_norm.reset_parameters()
        self.additive_projection.reset_parameters()
        nn.init.zeros_(self.additive_projection.weight)
        nn.init.zeros_(self.additive_projection.bias)

    def forward(
        self,
        camera_conditioning: list[dict[str, torch.Tensor]],
        vision_tokens: list[torch.Tensor],
        token_shapes: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build compact and additive codes in patchify's item-major T/H/W order.

        Args:
            camera_conditioning: One calibrated camera dictionary per real
                sample.  Its tensors retain the view axis.
            vision_tokens: Flat sample-major/view-minor VAE tensors, each
                ``[1,C,T_lat,H_lat,W_lat]``.
            token_shapes: Corresponding transformer shapes
                ``(T_lat,ceil(H_lat/2),ceil(W_lat/2))``.

        Returns:
            ``(camera_code, additive_embedding)`` with shapes
            ``[N_vision_tokens,code_dim]`` and
            ``[N_vision_tokens,hidden_size]``.
        """

        if len(vision_tokens) != len(token_shapes):
            raise ValueError(
                f"Camera conditioning received {len(vision_tokens)} vision items but {len(token_shapes)} token shapes."
            )

        item_offset = 0
        packed_codes: list[torch.Tensor] = []
        for sample_idx, camera in enumerate(camera_conditioning):
            c2w = camera["camera_c2w"]
            num_views = int(c2w.shape[0])
            sample_tokens = vision_tokens[item_offset : item_offset + num_views]
            sample_shapes = token_shapes[item_offset : item_offset + num_views]
            if len(sample_tokens) != num_views:
                raise ValueError(
                    f"Camera sample {sample_idx} declares {num_views} views, but only "
                    f"{len(sample_tokens)} flattened vision items remain."
                )

            latent_shapes = [tuple(int(x) for x in token.shape[-3:]) for token in sample_tokens]
            if len(set(latent_shapes)) != 1:
                raise ValueError(
                    f"All views in camera sample {sample_idx} must share latent T/H/W; got {latent_shapes}."
                )
            latent_t, latent_h, latent_w = latent_shapes[0]
            expected_raw_t = 4 * (latent_t - 1) + 1
            if int(c2w.shape[1]) != expected_raw_t:
                raise ValueError(
                    f"Camera sample {sample_idx} has {c2w.shape[1]} poses, but latent T={latent_t} "
                    f"requires {expected_raw_t} input frames for 4x temporal packing."
                )

            rays = compute_plucker_rays(
                intrinsics=camera["camera_intrinsics"],
                distortion=camera["camera_distortion"],
                camera_model=camera["camera_model"],
                c2w=c2w,
                image_size=camera["camera_image_size"],
                grid_h=latent_h,
                grid_w=latent_w,
            )
            rays = pack_plucker_rays_temporal(rays)
            if tuple(rays.shape[:2]) != (num_views, latent_t):
                raise RuntimeError(
                    f"Unexpected packed ray shape {tuple(rays.shape)} for V={num_views}, T_lat={latent_t}."
                )

            token_t, token_h, token_w = sample_shapes[0]
            expected_token_shape = (latent_t, math.ceil(latent_h / 2), math.ceil(latent_w / 2))
            shapes_match = tuple(sample_shapes[0]) == expected_token_shape and all(
                shape == sample_shapes[0] for shape in sample_shapes
            )
            if not shapes_match:
                raise ValueError(
                    f"Camera sample {sample_idx} token shapes {sample_shapes} do not match "
                    f"latent-derived shape {expected_token_shape}."
                )
            del token_t
            pad_h = 2 * token_h - latent_h
            pad_w = 2 * token_w - latent_w
            rays = rays.reshape(num_views * latent_t, 24, latent_h, latent_w)
            rays = F.pad(rays, (0, pad_w, 0, pad_h))
            rays = rays.to(device=self.ray_aggregator.weight.device, dtype=self.ray_aggregator.weight.dtype)

            code = self.ray_aggregator(rays)
            if tuple(code.shape[-2:]) != (token_h, token_w):
                raise RuntimeError(
                    f"Camera adapter produced spatial shape {tuple(code.shape[-2:])}, expected {(token_h, token_w)}."
                )
            code = code.reshape(num_views, latent_t, self.code_dim, token_h, token_w)
            code = code.permute(0, 1, 3, 4, 2).reshape(-1, self.code_dim)
            packed_codes.append(self.code_norm(code))
            item_offset += num_views

        if item_offset != len(vision_tokens):
            raise ValueError(
                f"Camera metadata covers {item_offset} views, but the packed batch contains "
                f"{len(vision_tokens)} vision items."
            )

        camera_code = torch.cat(packed_codes, dim=0)
        return camera_code, self.additive_projection(camera_code)
