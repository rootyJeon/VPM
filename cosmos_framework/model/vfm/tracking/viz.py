# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Draw point tracks on video frames (adapted from MV-TAP ``visualize.py``). No text is rendered."""

import cv2
import numpy as np


def track_colors(n: int) -> np.ndarray:
    """``(n, 3)`` uint8 RGB colors evenly spaced over the hue circle."""
    n = max(int(n), 1)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    hsv = np.stack([hues, np.full(n, 255, np.uint8), np.full(n, 255, np.uint8)], axis=1)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def draw_tracks(
    frames: np.ndarray,
    tracks: np.ndarray,
    visible: "np.ndarray | None" = None,
    colors: "np.ndarray | None" = None,
    trail: int = 12,
    radius: int = 4,
) -> np.ndarray:
    """Draw every track as a dot with a short trail.

    Args:
        frames: ``[T, H, W, 3]`` uint8 RGB.
        tracks: ``[T, N, 2]`` pixel ``(x, y)`` in the frames' resolution.
        visible: ``[T, N]`` bool; points are drawn where true (default: everywhere).
        colors: ``[N, 3]`` uint8 RGB (default: :func:`track_colors`).

    Returns:
        ``[T, H, W, 3]`` uint8 RGB.
    """
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    T, H, W, _ = frames.shape
    tracks = np.asarray(tracks, np.float32)
    N = tracks.shape[1]
    vis = np.ones((T, N), bool) if visible is None else np.asarray(visible, bool)
    colors = track_colors(N) if colors is None else np.asarray(colors, np.uint8)
    pts = np.round(np.nan_to_num(tracks, nan=-1e4)).clip(-1e4, 1e4).astype(np.int64).tolist()
    out = np.empty_like(frames)
    for t in range(T):
        img = frames[t].copy()
        for n in range(N):
            col = tuple(int(c) for c in colors[n])
            for s in range(max(1, t - trail + 1), t + 1):
                if vis[s, n] and vis[s - 1, n]:
                    cv2.line(img, tuple(pts[s - 1][n]), tuple(pts[s][n]), col, 2, cv2.LINE_AA)
            x, y = pts[t][n]
            if vis[t, n] and 0 <= x < W and 0 <= y < H:
                cv2.circle(img, (x, y), radius, col, -1, cv2.LINE_AA)
        out[t] = img
    return out
