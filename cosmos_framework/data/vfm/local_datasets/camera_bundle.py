# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""9D framewise camera action and per-sample camera bundle for ray conditioning.

The camera action is the 9D framewise block (translation + rot6d); inter-view geometry is carried by the
per-token ray code (``model/vfm/mot/camera_conditioning.CameraConditioner``). Bundle contract (consumed by
``compute_plucker_rays`` via ``get_data_and_condition``)::

    camera_c2w         [V,T,4,4]  float32, metres, canonicalised to view 0 at frame 0
    camera_intrinsics  [V,3,3]    float32, output-pixel K (after resize and crop)
    camera_distortion  [V,4]      float32, OpenCV [k1,k2,p1,p2] (zeros for pinhole)
    camera_model       [V]        int64, 0 = pinhole, 1 = Brown4
    camera_image_size  [V,2]      int64, output [height, width]

``T`` is the pixel frame count of the clip (``T == 4 * (latent_t - 1) + 1``).
"""

from __future__ import annotations

import numpy as np
import torch

from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel
from cosmos_framework.data.vfm.action.transforms import pad_action_to_max_dim

CAMERA_MODEL_PINHOLE = 0
CAMERA_MODEL_BROWN4 = 1


def build_framewise_camera_action(c2w: np.ndarray, *, translation_scale: float, max_action_dim: int) -> torch.Tensor:
    """The 9D camera action: framewise translation(3)+rot6d(6), then zero padding."""
    framewise = np.asarray(
        pose_abs_to_rel(
            c2w,
            rotation_format="rot6d",
            pose_convention="backward_framewise",
            translation_scale=translation_scale,
        )
    )
    if framewise.ndim != 2 or framewise.shape[1] != 9:
        raise ValueError(f"expected framewise camera action [T-1,9], got {framewise.shape}")
    return pad_action_to_max_dim(torch.from_numpy(framewise).float(), max_action_dim)


def build_camera_bundle(
    *,
    c2w_views: "list[np.ndarray]",
    intrinsics_views: "list[np.ndarray]",
    native_sizes: "list[tuple[int, int]]",
    unit_to_meters: float,
    input_height: int,
    input_width: int,
    resize_height: int,
    resize_width: int,
    crop_y: int,
    crop_x: int,
    output_height: int,
    output_width: int,
    distortion_views: "list[np.ndarray] | None" = None,
    model_views: "list[int] | None" = None,
) -> "dict[str, torch.Tensor]":
    """Build the model-facing camera bundle in VIDEO ITEM ORDER.

    Args:
        c2w_views: per view, camera-to-world trajectory ``[T,4,4]`` sampled at the SAME frame
            indices as the decoded clip (native translation units).
        intrinsics_views: per view, 3x3 pixel K at that view's ``native_sizes`` (h, w).
        native_sizes: the (height, width) each K is expressed at.
        unit_to_meters: multiply native translations by this to get metres (1.0 for metric data).
        input_*/resize_*/crop_*/output_*: the exact integer resize/crop affine applied to the RGB
            frames (``input`` = decoded size, i.e. the jsonl width/height).
        distortion_views / model_views: optional per-view OpenCV [k1,k2,p1,p2] + model id; default
            pinhole with zero distortion.
    """
    V = len(c2w_views)
    if V == 0 or len(intrinsics_views) != V or len(native_sizes) != V:
        raise ValueError(f"inconsistent view counts: c2w={V} K={len(intrinsics_views)} sizes={len(native_sizes)}")
    if min(input_height, input_width, resize_height, resize_width, output_height, output_width) <= 0:
        raise ValueError("camera image dimensions must be positive")

    resize_sx = float(resize_width) / float(input_width)
    resize_sy = float(resize_height) / float(input_height)
    input_to_output = np.array(
        [[resize_sx, 0.0, -float(crop_x)], [0.0, resize_sy, -float(crop_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    Ks = []
    for K, (native_h, native_w) in zip(intrinsics_views, native_sizes):
        K = np.asarray(K, np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"bad intrinsics {K.shape}")
        native_to_input = np.array(
            [[input_width / float(native_w), 0.0, 0.0],
             [0.0, input_height / float(native_h), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Ks.append(input_to_output @ native_to_input @ K)

    c2w = np.stack([np.asarray(t, np.float64) for t in c2w_views])
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"bad c2w stack {c2w.shape}")
    c2w = c2w.copy()
    c2w[..., :3, 3] *= float(unit_to_meters)
    c2w = np.linalg.inv(c2w[0, 0]) @ c2w

    dist = (np.stack([np.asarray(d, np.float32) for d in distortion_views])
            if distortion_views is not None else np.zeros((V, 4), np.float32))
    if dist.shape != (V, 4):
        raise ValueError(f"bad distortion {dist.shape}")
    models = (np.asarray(model_views, np.int64) if model_views is not None
              else np.full(V, CAMERA_MODEL_PINHOLE, np.int64))

    return {
        "camera_c2w": torch.from_numpy(np.ascontiguousarray(c2w.astype(np.float32))),
        "camera_intrinsics": torch.from_numpy(np.ascontiguousarray(np.stack(Ks).astype(np.float32))),
        "camera_distortion": torch.from_numpy(np.ascontiguousarray(dist)),
        "camera_model": torch.from_numpy(models),
        "camera_image_size": torch.from_numpy(
            np.tile(np.asarray([[output_height, output_width]], np.int64), (V, 1))),
    }

