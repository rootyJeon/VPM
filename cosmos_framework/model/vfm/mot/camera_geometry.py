# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure-Torch camera geometry helpers for multi-view conditioning.

The public ray API deliberately accepts *processed* intrinsics and camera-to-world
poses whose translations are already expressed in meters.  Dataset-native unit
conversion, resize/crop adjustment of intrinsics, and choice of the clip anchor
belong to the data layer; silently repeating any of those operations here would
make the action and ray encodings disagree.

Pixel coordinates use an image-boundary convention: an image spans ``[0, W]`` by
``[0, H]`` and the center of grid cell ``(row, col)`` is
``((col + 0.5) * W / grid_w, (row + 0.5) * H / grid_h)``.  In particular, a
full-resolution grid matches the half-pixel convention used by ReDirector.
"""

from __future__ import annotations

import torch

CAMERA_MODEL_PINHOLE = 0
CAMERA_MODEL_BROWN4 = 1


def _check_shape(name: str, value: torch.Tensor, expected: tuple[int | None, ...]) -> None:
    if value.ndim != len(expected):
        raise ValueError(f"{name} must have {len(expected)} dimensions, got shape {tuple(value.shape)}")
    for dim, (actual, wanted) in enumerate(zip(value.shape, expected, strict=True)):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} dimension {dim} must be {wanted}, got shape {tuple(value.shape)}")


def canonicalize_camera_poses(
    c2w: torch.Tensor,
    *,
    anchor_view: int = 0,
    anchor_time: int = 0,
) -> torch.Tensor:
    """Express every camera pose in one fixed anchor-camera coordinate frame.

    Args:
        c2w: Camera-to-world transforms with shape ``[V, T, 4, 4]``.  Translation
            units are preserved; callers should convert them to meters first when
            the result will be passed to :func:`compute_plucker_rays`.
        anchor_view: View whose coordinate frame is used as the fixed gauge.
        anchor_time: Time of that view used as the fixed gauge.

    Returns:
        ``anchor_c2w.inverse() @ c2w`` with shape ``[V, T, 4, 4]``.
    """

    _check_shape("c2w", c2w, (None, None, 4, 4))
    num_views, num_frames = c2w.shape[:2]
    if not 0 <= anchor_view < num_views:
        raise ValueError(f"anchor_view={anchor_view} is outside [0, {num_views})")
    if not 0 <= anchor_time < num_frames:
        raise ValueError(f"anchor_time={anchor_time} is outside [0, {num_frames})")

    poses = c2w.float()
    anchor_inv = torch.linalg.inv(poses[anchor_view, anchor_time])
    return anchor_inv.view(1, 1, 4, 4) @ poses


def inverse_brown4(
    distorted_xy: torch.Tensor,
    distortion: torch.Tensor,
    *,
    num_iterations: int = 8,
) -> torch.Tensor:
    """Invert OpenCV's four-coefficient Brown-Conrady distortion in fp32.

    ``distortion`` follows OpenCV order ``[k1, k2, p1, p2]`` and must be
    broadcastable to ``distorted_xy[..., :1]``.  A small Newton solve is used
    instead of interpolation or a lookup table, so arbitrary analytic grid-cell
    centers can be evaluated directly.
    """

    if distorted_xy.shape[-1] != 2:
        raise ValueError(f"distorted_xy must end in 2 coordinates, got {tuple(distorted_xy.shape)}")
    if distortion.shape[-1] != 4:
        raise ValueError(f"distortion must end in 4 coefficients, got {tuple(distortion.shape)}")
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    xy_d = distorted_xy.float()
    coeff = distortion.to(device=xy_d.device, dtype=torch.float32)
    k1, k2, p1, p2 = coeff.unbind(dim=-1)
    x_d, y_d = xy_d.unbind(dim=-1)
    x = x_d.clone()
    y = y_d.clone()

    eps = torch.finfo(torch.float32).eps
    for _ in range(num_iterations):
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4

        delta_x = 2.0 * p1 * xy + p2 * (3.0 * x2 + y2)
        delta_y = p1 * (x2 + 3.0 * y2) + 2.0 * p2 * xy
        residual_x = x * radial + delta_x - x_d
        residual_y = y * radial + delta_y - y_d

        radial_dx = 2.0 * k1 * x + 4.0 * k2 * r2 * x
        radial_dy = 2.0 * k1 * y + 4.0 * k2 * r2 * y
        j11 = radial + x * radial_dx + 2.0 * p1 * y + 6.0 * p2 * x
        j12 = x * radial_dy + 2.0 * p1 * x + 2.0 * p2 * y
        j21 = y * radial_dx + 2.0 * p1 * x + 2.0 * p2 * y
        j22 = radial + y * radial_dy + 6.0 * p1 * y + 2.0 * p2 * x

        det = j11 * j22 - j12 * j21
        safe_det = torch.where(det.abs() >= eps, det, torch.where(det >= 0, eps, -eps))
        step_x = (residual_x * j22 - residual_y * j12) / safe_det
        step_y = (j11 * residual_y - j21 * residual_x) / safe_det
        x = x - step_x
        y = y - step_y

    return torch.stack((x, y), dim=-1)


def compute_plucker_rays(
    intrinsics: torch.Tensor,
    distortion: torch.Tensor,
    camera_model: torch.Tensor,
    c2w: torch.Tensor,
    image_size: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Compute analytic per-cell Plucker rays for every view and frame.

    Args:
        intrinsics: Processed camera matrices ``[V, 3, 3]`` matching the
            resized/cropped images.
        distortion: OpenCV-order ``[k1, k2, p1, p2]`` coefficients ``[V, 4]``.
            Values are ignored for pinhole views.
        camera_model: Integer model IDs ``[V]``: 0 = pinhole, 1 = Brown4.
        c2w: Fixed-anchor, meter-valued camera-to-world poses ``[V, T, 4, 4]``.
        image_size: Processed ``[height, width]`` for each view, shape ``[V, 2]``.
        grid_h: Number of analytic ray samples vertically.
        grid_w: Number of analytic ray samples horizontally.

    Returns:
        Float32 tensor ``[V, T, 6, grid_h, grid_w]`` with convention
        ``[origin cross direction, direction]`` and unit-length directions.
    """

    _check_shape("intrinsics", intrinsics, (None, 3, 3))
    _check_shape("distortion", distortion, (None, 4))
    _check_shape("camera_model", camera_model, (None,))
    _check_shape("c2w", c2w, (None, None, 4, 4))
    _check_shape("image_size", image_size, (None, 2))
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_h and grid_w must be positive, got {grid_h}x{grid_w}")

    num_views, num_frames = c2w.shape[:2]
    for name, value in (
        ("intrinsics", intrinsics),
        ("distortion", distortion),
        ("camera_model", camera_model),
        ("image_size", image_size),
    ):
        if value.shape[0] != num_views:
            raise ValueError(f"{name} has {value.shape[0]} views but c2w has {num_views}")
    if num_frames < 1:
        raise ValueError("c2w must contain at least one frame")

    device = c2w.device
    k = intrinsics.to(device=device, dtype=torch.float32)
    coeff = distortion.to(device=device, dtype=torch.float32)
    model = camera_model.to(device=device)
    poses = c2w.float()
    sizes = image_size.to(device=device, dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(k)).item()):
        raise ValueError("intrinsics must be finite")
    if not bool(torch.all(torch.isfinite(coeff)).item()):
        raise ValueError("distortion must be finite")
    if not bool(torch.all(torch.isfinite(poses)).item()):
        raise ValueError("c2w must be finite")
    if not bool(torch.all(torch.isfinite(sizes) & (sizes > 0)).item()):
        raise ValueError("image_size entries must be finite and positive")
    valid_models = (model == CAMERA_MODEL_PINHOLE) | (model == CAMERA_MODEL_BROWN4)
    if not bool(torch.all(valid_models).item()):
        invalid = torch.unique(model[~valid_models]).detach().cpu().tolist()
        raise ValueError(f"camera_model contains unsupported IDs {invalid}; expected 0 or 1")

    rows = torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5
    cols = torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5
    y = rows.view(1, grid_h, 1) * (sizes[:, 0] / float(grid_h)).view(num_views, 1, 1)
    x = cols.view(1, 1, grid_w) * (sizes[:, 1] / float(grid_w)).view(num_views, 1, 1)
    x = x.expand(num_views, grid_h, grid_w)
    y = y.expand(num_views, grid_h, grid_w)
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=1).reshape(num_views, 3, -1)

    normalized_h = torch.linalg.solve(k, pixels).transpose(1, 2)
    distorted_xy = normalized_h[..., :2] / normalized_h[..., 2:].clamp_min(torch.finfo(torch.float32).tiny)
    undistorted_xy = inverse_brown4(distorted_xy, coeff[:, None, :])
    use_brown = (model == CAMERA_MODEL_BROWN4).view(num_views, 1, 1)
    normalized_xy = torch.where(use_brown, undistorted_xy, distorted_xy)

    camera_directions = torch.cat((normalized_xy, torch.ones_like(normalized_xy[..., :1])), dim=-1)
    camera_directions = torch.nn.functional.normalize(camera_directions, dim=-1)
    camera_directions = camera_directions.view(num_views, grid_h, grid_w, 3)

    rotation = poses[..., :3, :3]
    direction = torch.einsum("vtij,vhwj->vthwi", rotation, camera_directions)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    origin = poses[..., :3, 3].view(num_views, num_frames, 1, 1, 3).expand_as(direction)
    moment = torch.linalg.cross(origin, direction, dim=-1)
    rays = torch.cat((moment, direction), dim=-1)
    return rays.permute(0, 1, 4, 2, 3).contiguous()


def pack_plucker_rays_temporal(rays: torch.Tensor) -> torch.Tensor:
    """Pack raw-frame rays into the VAE's four-frame temporal channel layout.

    For ``T = 4n + 1``, the conceptual sequence is padded to
    ``[r0, r0, r0, r0, r1, ..., r4n]`` and split into groups of four.  Each
    group's output channel order matches ReDirector: ``component * 4 +
    temporal_offset``.  No temporal interpolation or averaging is performed.

    Args:
        rays: Plucker rays ``[V, T, 6, H, W]``.

    Returns:
        Packed rays ``[V, n + 1, 24, H, W]``.
    """

    _check_shape("rays", rays, (None, None, 6, None, None))
    num_views, num_frames, _, height, width = rays.shape
    if num_frames < 1 or (num_frames - 1) % 4 != 0:
        raise ValueError(f"rays time dimension must be T=4n+1, got T={num_frames}")

    padded = torch.cat((rays[:, :1].expand(-1, 4, -1, -1, -1), rays[:, 1:]), dim=1)
    latent_frames = padded.shape[1] // 4
    grouped = padded.reshape(num_views, latent_frames, 4, 6, height, width)
    return grouped.permute(0, 1, 3, 2, 4, 5).reshape(num_views, latent_frames, 24, height, width)
