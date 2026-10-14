"""Generated videos (predictions) and the matching reference frames on the protocol grid.

Prediction layout (docs/PROTOCOL.md):
    <pred_root>/<Subset>/<id>/view{k}.mp4      one video per view
    <pred_root>/<Subset>/<id>/meta.json        optional: {"fps": 24.0, "first_frame": "ref" | "future"}

Without meta.json the video must already be on the window grid: frame j = GT frame window_indices[j], so frame 0 is the
conditioning frame t_ref and at least horizon+1 frames are present. With "fps", frames are taken by time
(t - t_ref) / fps_gt: window frame t sits at position x = (t - t_ref) * fps / fps_gt of the video, the nearest frame
when fps >= 12 (ties round up: frame floor(x + 0.5)), else a linear blend of the two neighbours. x is computed exactly
(`window_positions`: each rate is read as the nearest fraction with a denominator <= 10**6, so 29.97 = 2997/100 and
30000/1001 stay exact), so a tie such as 20 -> 30 fps at x = 61.5 takes frame 62. With first_frame = "future" the
video starts at the first scored frame and the GT conditioning frame is prepended as frame 0 (time 0; with "fps" the
video's frame i is at time (i + 1) / fps, so below 12 fps the first scored position can blend the given conditioning
frame with the video's first frame).
meta.json is checked by every loader (`check_meta`): a JSON object, "first_frame" exactly 'ref' or 'future', "fps" a
finite number > 0; anything else raises ValueError.
Any resolution / aspect: frames are mapped to the protocol eval grid with the protocol resize mode.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from numbers import Real
from pathlib import Path

import numpy as np

from .geometry import image_map, warp_image
from .io.video import read_frames
from .protocol import Protocol


def prediction_dir(pred_root, sample) -> Path:
    return Path(pred_root) / sample.subset / sample.id


def check_meta(meta) -> dict:
    """`meta` if it is a usable prediction meta.json (module docstring), else ValueError. "fps" must be a JSON number:
    a string ("24") or a boolean (true would read as 1 fps) is refused, not converted."""
    if not isinstance(meta, dict):
        raise ValueError("meta.json must hold a JSON object")
    if meta.get("first_frame", "ref") not in ("ref", "future"):
        raise ValueError(f"meta.json first_frame must be 'ref' or 'future', got {meta.get('first_frame')!r}")
    fps = meta.get("fps")
    if fps is not None:
        try:
            ok = isinstance(fps, Real) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise ValueError(f"meta.json fps must be a number > 0, got {fps!r}")
    return meta


def prediction_meta(pred_root, sample) -> dict:
    """The sample's meta.json ({} without one), checked with `check_meta` (ValueError when unusable)."""
    p = prediction_dir(pred_root, sample) / "meta.json"
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"meta.json is not valid JSON: {e}") from e
    return check_meta(meta)


def has_prediction(pred_root, sample, views) -> bool:
    d = prediction_dir(pred_root, sample)
    return all((d / f"view{k}.mp4").exists() for k in views)


def _exact_rate(fps) -> Fraction:
    return Fraction(float(fps)).limit_denominator(10 ** 6)


def window_positions(protocol: Protocol, fps_gt: float, fps: float) -> list:
    """Exact positions (Fractions) of the window frames in a video at `fps` whose frame 0 is t_ref:
    (t - t_ref) * fps / fps_gt for t in protocol.window_indices."""
    r = _exact_rate(fps) / _exact_rate(fps_gt)
    return [(t - protocol.ref_frame) * r for t in protocol.window_indices]


def _resample(frames: np.ndarray, protocol: Protocol, fps_gt: float, meta: dict, ref_frame: np.ndarray | None):
    check_meta(meta)
    win = protocol.window_indices
    first = meta.get("first_frame", "ref")
    if first == "future":
        if ref_frame is None:
            raise ValueError("first_frame='future' needs the GT conditioning frame")
        frames = np.concatenate([ref_frame[None], frames], 0)
    fps = meta.get("fps")
    if fps is None:
        if len(frames) < len(win):
            raise ValueError(f"prediction has {len(frames)} frames, needs {len(win)} (t_ref + horizon)")
        return frames[: len(win)]
    fps = float(fps)
    n = len(frames)
    pos = window_positions(protocol, fps_gt, fps)
    if pos[-1] > n - Fraction(1, 2):
        raise ValueError(f"prediction too short: needs {float(pos[-1]):.1f} s*fps frames, has {n}")
    if fps >= 12:
        return frames[np.asarray([min(max(math.floor(x + Fraction(1, 2)), 0), n - 1) for x in pos], int)]
    out = []
    for x in pos:
        i0 = min(math.floor(x), n - 1)
        i1 = min(i0 + 1, n - 1)
        w = np.float64(x - math.floor(x))
        out.append((frames[i0].astype(np.float32) * (1 - w) + frames[i1].astype(np.float32) * w).round().astype(np.uint8))
    return np.stack(out)


def load_prediction(pred_root, sample, view: int, protocol: Protocol, ref_frame: np.ndarray | None = None) -> np.ndarray:
    """(len(window), H, W, 3) uint8 on the protocol eval grid of the sample's subset."""
    d = prediction_dir(pred_root, sample)
    meta = prediction_meta(pred_root, sample)
    frames = read_frames(d / f"view{view}.mp4")
    frames = _resample(frames, protocol, sample.fps, meta, ref_frame)
    W, H = protocol.eval_size_for(sample.subset)
    if frames.shape[1:3] != (H, W):
        m = image_map((frames.shape[2], frames.shape[1]), (W, H), protocol.resize_mode_for(sample.subset))
        frames = np.stack([warp_image(f, m) for f in frames])
    return frames


def load_reference(sample, view: int, protocol: Protocol) -> np.ndarray:
    """The real video of `view` on the window grid at the protocol eval grid (the reference for every metric)."""
    if protocol.reference_frames == "_reproduce":
        p = sample.root / "_reproduce" / sample.subset / sample.id / "real_480sq" / f"view{view}.mp4"
        if p.exists():
            fr = read_frames(p)
            win = protocol.window_indices
            if win[0] != 16 or len(fr) < len(win) or protocol.frame_step != 1:
                raise ValueError("_reproduce frames cover GT 16..80 only (legacy protocol)")
            return fr[: len(win)]
    W, H = protocol.eval_size_for(sample.subset)
    return sample.frames(view, protocol.window_indices, size=(W, H), mode=protocol.resize_mode_for(sample.subset))
