# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Plücker-ray camera conditioning for the multi-view tracking head (vendored from MV-TAP).

Faithful port of MV-TAP's ``models/cam_embed.py`` (``camera_center_normalization`` +
``get_plucker_coords``), the conditioning the pretrained ``cam_proj`` / ``cam_norm`` weights
expect. Given per-(view,frame) intrinsics + world-to-camera extrinsics and the running 2D track
estimate, it returns a 6D Plücker ray ``[dir | origin × dir]`` per track, which the head projects
and ADDS to the Refiner input — giving the cross-view attention explicit camera geometry.

Scale note: Plücker directions depend only on ``(u-cx)/fx`` ratios, so intrinsics may be supplied
in ANY units consistent with ``coords`` (we use image-normalized [0,1], matching the head's
``coords / [W0,H0]``); the moment term uses camera-center-normalized world poses. This reproduces
MV-TAP's pixel-unit embedding exactly, so the warm-started ``cam_proj`` stays valid.
"""

import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c: torch.Tensor, nframe: int, camera_scale: float = 2.0) -> torch.Tensor:
    """MV-TAP/SEVA camera normalization. ``w2c``: [F*V, 4, 4] world-to-camera, flattened FRAME-major
    (idx = f*V + v); ``nframe`` = number of views V. Makes each frame's view-0 the identity, centers
    the camera cloud, and scales translations so the first camera sits at radius ``camera_scale``."""
    c2w_view0 = w2c[::nframe].inverse()
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)
    w2c = c2w_view0 @ w2c

    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(
        c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1
    )
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    return w2c


@torch.amp.autocast("cuda", enabled=False)
def plucker_from_tracks(coords01: torch.Tensor, intrinsics: torch.Tensor,
                        extrinsics: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """6D Plücker ray per track point (MV-TAP ``get_plucker_coords``).

    Args:
        coords01:   [V, T, N, 2] track positions in image-normalized [0,1] (x,y).
        intrinsics: [V, T, 3, 3] image-normalized pinhole (fx,fy,cx,cy in [0,1] units).
        extrinsics: [V, T, 4, 4] world-to-camera (OpenCV w2c).
        normalize:  apply MV-TAP camera-center normalization (recommended).
    Returns:
        [V, T, N, 6] Plücker = concat(dir_world, origin_world × dir_world).
    """
    V, T, N, _ = coords01.shape
    dev = coords01.device
    coords = coords01.permute(1, 0, 2, 3).float()
    intr = intrinsics.permute(1, 0, 2, 3).reshape(T * V, 3, 3).float()
    extr = extrinsics.permute(1, 0, 2, 3).reshape(T * V, 4, 4).float()
    if normalize:
        extr = camera_center_normalization(extr, nframe=V)

    c2w = torch.inverse(extr)[:, :3, :4].to(dev)
    u, v = coords[..., 0], coords[..., 1]
    fx = intr[:, 0, 0].reshape(T, V, 1)
    fy = intr[:, 1, 1].reshape(T, V, 1)
    cx = intr[:, 0, 2].reshape(T, V, 1)
    cy = intr[:, 1, 2].reshape(T, V, 1)
    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    R = c2w[:, :3, :3].view(T, V, 3, 3)
    t = c2w[:, :3, 3].view(T, V, 1, 3)
    dirs_world = torch.matmul(R.unsqueeze(2), dirs_cam.unsqueeze(-1)).squeeze(-1)
    orig_world = t.expand_as(dirs_world)
    cross = torch.cross(orig_world, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, cross], dim=-1)
    return plucker.permute(1, 0, 2, 3).contiguous()
