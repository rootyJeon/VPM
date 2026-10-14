"""3D lifting of multi-view videos (the first step of the δ metric): the `Lifter` interface, the depth `Anchor`,
and the model-independent geometry shared by depth-based lifters (depth anchoring, scene normalisation, regridding).

A lifter turns the V synchronised videos of one sample into 3D point tracks in the WORLD frame:

    xyz, vis = lifter.track(frames, K, w2c, queries, anchor)

    frames   (V, n, H, W, 3) uint8   RGB, every view on one pixel grid and one time grid ("window positions" 0..n-1;
                                     position 0 is the anchor frame = t_ref of the generation protocol)
    K        (V, n, 3, 3)    float   intrinsics in pixels of that grid (integer pixel coordinates are pixel centres)
    w2c      (V, n, 4, 4)    float   world-to-camera extrinsics, OpenCV axes (x right, y down, z forward), metres
    queries  (N, 4)          float   [t, x, y, z]: integer window position t and the world point (m) to track from t
    anchor   Anchor                  how the metric scale of the estimated depth is fixed (see `Anchor`)
    ->
    xyz      (n, N, 3)       float32 predicted world positions (m) at every window position
    vis      (n, N)          bool    predicted visibility

The world is right-handed, metres, +z up (docs/DATA_FORMAT.md); the scene normalisation relies on +z being up.

`DepthAnchoredLifter` implements the generic pipeline used by the reference lifter (lifters/mvtracker.py):
1. (optional) resample frames, K and anchor depth maps to `track_size` (e.g. to undo an aspect-changing eval grid);
2. per view, metric depth for every frame from a monocular estimator (`estimate_depth`), anchored on window position 0
   (`anchor_depth`: scale+shift against GT depth, or a median ratio at visible GT points), depth outside
   (0, far_clip_m] set to 0 (= invalid);
3. scene normalisation (`scene_normalization`) computed from `anchor.norm_depth` (the sample's REFERENCE depth, so it is
   identical for every source scored against that sample), applied to depth, extrinsics and queries;
4. one tracker call with all queries (`run_tracker`) in the normalised frame; predictions mapped back to the world.
"""
from __future__ import annotations

import abc
import importlib
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from ..geometry import image_map, warp_image

ANCHOR_MODES = ("gt_depth", "gt_points", "none")

TARGET_RADIUS = 6.3
FLOOR_QUANTILE = 0.12
NORM_MIN_PIXELS = 100

FIT_MIN_GT_M = 0.1
FIT_MIN_PRED_M = 1e-3
FIT_MIN_PIXELS = 1000
FIT_TRIM_QUANTILES = (0.02, 0.98)

RATIO_MIN_PRED_M = 1e-3
RATIO_MIN_POINTS = 5

NON_SQUARE_WARN = 0.05


@dataclass
class Anchor:
    """How a lifter fixes the metric scale of estimated depth, and which depth defines the scene normalisation.

    mode        "gt_depth":  per view, (a, b) = least squares of `depth ≈ a*est + b` on window position 0 against
                             `depth` (see `fit_scale_shift`), applied to every frame of that view.
                "gt_points": per view, a = median(z_gt / est) over the GT points visible in that view at window
                             position 0 (see `median_ratio_scale`); b = 0.
                "none":      the estimator's metric depth as is (a = 1, b = 0).
    depth       (V, H, W) float, metres, 0 = invalid: GT (or reconstruction) planar depth of window position 0 on the
                frame grid of the frames passed to `track` (or on `track_size`). Required for "gt_depth".
    points      (M, 3) float world metres: GT points at window position 0 (NaN allowed where unknown).
    points_vis  (V, M) bool: points[m] is visible in view v at window position 0. Both required for "gt_points".
    far_clip_m  anchored depth outside (0, far_clip_m] becomes 0 (invalid); also the upper bound of the GT depth used
                by the fit and of the reference depth used by the scene normalisation.
    norm_depth  (V, H, W) float metres, 0 = invalid: depth of window position 0 that defines the scene normalisation.
                Use the sample's REFERENCE depth so the normalisation is identical for every source of a sample:
                the GT depth for "gt_depth", the anchored depth of the real video otherwise (`Lifter.prepare_anchor`
                fills it). None -> `depth` if set, else the lifter's anchored depth of the tracked source itself.
    """
    mode: str = "gt_depth"
    depth: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    points_vis: Optional[np.ndarray] = None
    far_clip_m: float = 80.0
    norm_depth: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mode not in ANCHOR_MODES:
            raise ValueError(f"anchor mode {self.mode!r} not in {ANCHOR_MODES}")
        self.far_clip_m = float(self.far_clip_m)
        if not self.far_clip_m > 0:
            raise ValueError("far_clip_m must be > 0")
        for name in ("depth", "norm_depth"):
            d = getattr(self, name)
            if d is not None:
                d = np.asarray(d, np.float32)
                if d.ndim != 3:
                    raise ValueError(f"anchor.{name} must be (V, H, W), got {d.shape}")
                setattr(self, name, d)
        if self.points is not None:
            self.points = np.asarray(self.points, np.float64).reshape(-1, 3)
        if self.points_vis is not None:
            self.points_vis = np.asarray(self.points_vis, bool)
            if self.points_vis.ndim != 2:
                raise ValueError(f"anchor.points_vis must be (V, M), got {self.points_vis.shape}")
            if self.points is not None and self.points_vis.shape[1] != len(self.points):
                raise ValueError(f"anchor.points_vis {self.points_vis.shape} does not match points {self.points.shape}")
        if self.mode == "gt_depth" and self.depth is None:
            raise ValueError("anchor mode 'gt_depth' needs `depth` (V, H, W)")
        if self.mode == "gt_points" and (self.points is None or self.points_vis is None):
            raise ValueError("anchor mode 'gt_points' needs `points` (M, 3) and `points_vis` (V, M)")


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray, far_clip_m: float = 80.0,
                    min_pixels: int = FIT_MIN_PIXELS) -> tuple:
    """Affine depth alignment `gt ≈ a * pred + b` (the pre-release `fit_scale_shift`).

    pred, gt: (H, W) depth maps in metres on the same grid; non-finite pred = invalid, gt <= 0 = invalid.
    Pixels used: finite pred > 1 mm and 0.1 m < gt < far_clip_m; fewer than `min_pixels` -> (1, 0). pred is trimmed to
    its open 2-98 % quantile range, then ordinary least squares on [pred, 1]. If a <= 0 (or pred has no spread, where
    the affine fit is undefined) the fit falls back to the median ratio median(gt) / median(pred) with b = 0.
    Returns (a: float, b: float metres, n_pixels: int used by the fit).
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} must have the same shape")
    with np.errstate(invalid="ignore"):
        m = np.isfinite(pred) & np.isfinite(gt) & (gt > FIT_MIN_GT_M) & (gt < far_clip_m) & (pred > FIT_MIN_PRED_M)
    if int(m.sum()) < min_pixels:
        return 1.0, 0.0, int(m.sum())
    x, y = pred[m].astype(np.float64), gt[m].astype(np.float64)
    keep = (x > np.quantile(x, FIT_TRIM_QUANTILES[0])) & (x < np.quantile(x, FIT_TRIM_QUANTILES[1]))
    if int(keep.sum()) >= 2:
        x, y = x[keep], y[keep]
    a, b = 0.0, 0.0
    if np.ptp(x) > 0:
        a, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    if not a > 0:
        a, b = float(np.median(y) / max(np.median(x), 1e-6)), 0.0
    return float(a), float(b), int(len(x))


def median_ratio_scale(depth0: np.ndarray, points: np.ndarray, points_vis: np.ndarray, K0: np.ndarray,
                       w2c0: np.ndarray, min_points: int = RATIO_MIN_POINTS) -> tuple:
    """Depth scale from GT points (the pre-release RB-Y1 `anchored_moge`).

    depth0 (H, W) estimated depth in metres (non-finite = invalid); points (M, 3) world metres; points_vis (M,) bool
    visibility of each point in this view; K0 (3, 3) and w2c0 (4, 4) of this view on the depth grid.
    Every visible point is projected, its pixel rounded (np.round) and clipped to the image; points behind the camera
    or more than 1 px outside the image are ignored. Points whose estimated depth is finite and > 1 mm are used:
    a = median(z_cam(point) / depth0[pixel]). Fewer than `min_points` -> a = 1.
    Returns (a: float, n_points: int).
    """
    depth0 = np.asarray(depth0)
    H, W = depth0.shape
    X = np.asarray(points, np.float64).reshape(-1, 3)
    E = np.asarray(w2c0, np.float64)
    Xc = X @ E[:3, :3].T + E[:3, 3]
    uvw = Xc @ np.asarray(K0, np.float64).T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
        m = np.asarray(points_vis, bool) & np.isfinite(uv).all(-1) & (Xc[:, 2] > 0)
        m &= (uv[:, 0] >= -1.0) & (uv[:, 0] <= W) & (uv[:, 1] >= -1.0) & (uv[:, 1] <= H)
    ui = np.clip(np.round(uv[m, 0]).astype(int), 0, W - 1)
    vi = np.clip(np.round(uv[m, 1]).astype(int), 0, H - 1)
    dm = depth0[vi, ui]
    with np.errstate(invalid="ignore"):
        good = np.isfinite(dm) & (dm > RATIO_MIN_PRED_M)
    n = int(good.sum())
    if n < min_points:
        return 1.0, n
    return float(np.median(Xc[m, 2][good] / dm[good])), n


def anchor_depth(depth: np.ndarray, view: int, K0: np.ndarray, w2c0: np.ndarray, anchor: Anchor) -> tuple:
    """Anchor one view's estimated depth (n, H, W) metres (non-finite = invalid) with `anchor` on window position 0.

    K0 (3, 3), w2c0 (4, 4): the view's camera at window position 0 on the depth grid. Returns (depth (n, H, W) float32
    metres with invalid / negative / > far_clip_m values set to 0, info {"mode", "scale", "shift", "anchors"}), where
    "anchors" is the number of pixels (gt_depth) or points (gt_points) behind the estimate.
    """
    depth = np.asarray(depth, np.float32)
    if anchor.mode == "gt_depth":
        gt = anchor.depth[view]
        if gt.shape != depth.shape[1:]:
            raise ValueError(f"anchor.depth[{view}] {gt.shape} is not on the frame grid {depth.shape[1:]}")
        a, b, n = fit_scale_shift(depth[0], gt, anchor.far_clip_m)
    elif anchor.mode == "gt_points":
        (a, n), b = median_ratio_scale(depth[0], anchor.points, anchor.points_vis[view], K0, w2c0), 0.0
    else:
        a, b, n = 1.0, 0.0, 0
    with np.errstate(invalid="ignore"):
        out = np.where(np.isfinite(depth), a * depth + b, 0.0).astype(np.float32)
        out[(out < 0) | (out > anchor.far_clip_m)] = 0.0
    return out, {"mode": anchor.mode, "scale": float(a), "shift": float(b), "anchors": int(n)}


def scene_normalization(norm_depth: np.ndarray, K0: np.ndarray, w2c0: np.ndarray, far_clip_m: float | None = None,
                        fallback_points: np.ndarray | None = None, target_radius: float = TARGET_RADIUS,
                        floor_quantile: float = FLOOR_QUANTILE, min_pixels: int = NORM_MIN_PIXELS) -> tuple:
    """MVTracker-style similarity normalisation X' = s X + T of a z-up world (the pre-release `scene_norm`).

    norm_depth (V, H, W) planar depth in metres (valid: finite, > 0 and <= far_clip_m); K0 (V, 3, 3), w2c0 (V, 4, 4) or
    (V, 3, 4): the cameras of the same frame. Like MVTracker's `compute_auto_scene_normalization` but with the TRUE
    camera centres C_v = -R^T t (MVTracker uses the translation column) and no confidence threshold:
        P = back-projection of all valid pixels (views with < `min_pixels` valid pixels are skipped)
        c = mean(P);  f = quantile(P_z - c_z, 0.12)                       (floor height; +z must be up)
        s = 6.3 / median_v ||C_v - c - (0, 0, f)||;  T = -s c - (0, 0, s f)
    `fallback_points` (M, 3) world replace P when no view has enough valid depth; without them the identity is used.
    Returns (s: float, T: (3,) float64, info {"source": "depth"|"points"|"identity", "n_points": int}).
    The arithmetic (K inverted in its own dtype, homogeneous 4x4 inverses) follows the pre-release implementation
    operation for operation: MVTracker amplifies even 1e-7 relative input differences into per-clip delta changes.
    """
    K0 = np.asarray(K0)
    w2c0 = np.asarray(w2c0)
    if w2c0.shape[-2:] == (4, 4):
        w2c0 = w2c0[..., :3, :]
    elif w2c0.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must be (V, 4, 4) or (V, 3, 4), got {w2c0.shape}")
    pts = []
    for v in range(len(norm_depth)):
        d = np.asarray(norm_depth[v])
        with np.errstate(invalid="ignore"):
            m = np.isfinite(d) & (d > 0)
            if far_clip_m is not None:
                m &= d <= far_clip_m
        if int(m.sum()) < min_pixels:
            continue
        yy, xx = np.nonzero(m)
        uv1 = np.stack([xx, yy, np.ones_like(xx)], 0).astype(np.float64)
        cam = (np.linalg.inv(K0[v]) @ uv1) * d[m][None]
        E = np.eye(4)
        E[:3] = w2c0[v]
        pts.append((np.linalg.inv(E) @ np.concatenate([cam, np.ones((1, cam.shape[1]))]))[:3].T)
    source = "depth"
    if pts:
        P = np.concatenate(pts)
    else:
        P = np.zeros((0, 3))
        if fallback_points is not None:
            P = np.asarray(fallback_points, np.float64).reshape(-1, 3)
            P = P[np.isfinite(P).all(-1)]
        source = "points"
    if len(P) == 0:
        return 1.0, np.zeros(3), {"source": "identity", "n_points": 0}
    centroid = P.mean(0)
    floor_z = float(np.quantile(P[:, 2] - centroid[2], floor_quantile))
    centers = np.stack([np.linalg.inv(np.vstack([w2c0[v], [0, 0, 0, 1]]))[:3, 3] for v in range(len(w2c0))])
    cc = centers - centroid
    cc[:, 2] -= floor_z
    scale = float(target_radius / max(float(np.median(np.linalg.norm(cc, axis=1))), 1e-3))
    trans = -scale * centroid
    trans[2] -= scale * floor_z
    return scale, trans, {"source": source, "n_points": int(len(P))}


def apply_normalization(scale: float, translation: np.ndarray, depths: np.ndarray, w2c: np.ndarray,
                        points: np.ndarray) -> tuple:
    """Apply X' = s X + T: depth (any shape, metres) -> s * depth; w2c (..., 4, 4) or (..., 3, 4) [R | t] ->
    [R | s t - R T]; world points (..., 3) -> s X + T. Pixels are unchanged. Computed in the dtypes of the inputs
    (the lifter passes float32, the tracker's precision, as the pre-release code did)."""
    w2c = np.asarray(w2c)
    T = np.asarray(translation)
    out = w2c.copy()
    rot = w2c[..., :3, :3]
    sub = "vtij,j->vti" if rot.ndim == 4 else "...ij,j->...i"
    out[..., :3, 3] = scale * w2c[..., :3, 3] - np.einsum(sub, rot, T)
    return np.asarray(depths) * scale, out, np.asarray(points) * scale + T


def invert_normalization(points: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    """Normalised points (..., 3) -> world metres: (X' - T) / s, in the dtype of the inputs."""
    return (np.asarray(points) - np.asarray(translation)) / scale


def fov_x_deg(K: np.ndarray, width: int) -> np.ndarray:
    """Horizontal field of view in degrees, 2 atan(W / (2 fx)), for K (..., 3, 3) on a grid `width` pixels wide."""
    return np.degrees(2.0 * np.arctan(width / (2.0 * np.asarray(K, np.float64)[..., 0, 0])))


def map_K(K: np.ndarray, src_size, dst_size, plain: bool = False) -> np.ndarray:
    """Intrinsics (..., 3, 3) on a (W, H) grid -> on a (w, h) grid covering the same field of view ('resize').
    plain=False: pixel-centre mapping (geometry.ImageMap.apply_K); plain=True: K rows scaled by (w/W, h/H) as in the
    pre-release code (protocol `plain_K_scaling`)."""
    K = np.asarray(K, np.float64)
    m = image_map(src_size, dst_size, "resize")
    if not plain:
        return m.apply_K(K)
    out = K.copy()
    out[..., 0, :] *= m.sx
    out[..., 1, :] *= m.sy
    return out


def resample_inputs(frames: np.ndarray, K: np.ndarray, size, plain_K_scaling: bool = False, depth_maps=()) -> tuple:
    """Resample every view to the (w, h) grid `size` covering the same field of view ('resize' mode).

    frames (V, n, H, W, 3) uint8 -> (V, n, h, w, 3) with INTER_AREA when shrinking an axis, else INTER_CUBIC
    (geometry.warp_image 'auto'); K (V, n, 3, 3) mapped with `map_K`; each depth map array (..., H, W) in `depth_maps`
    is resampled with nearest neighbour (arrays already on (h, w) and None are passed through).
    Returns (frames, K, [depth maps]).
    """
    V, n, H, W = frames.shape[:4]
    w, h = (int(x) for x in size)
    m = image_map((W, H), (w, h), "resize")
    out = np.empty((V, n, h, w, frames.shape[-1]), frames.dtype)
    for v in range(V):
        for i in range(n):
            out[v, i] = warp_image(frames[v, i], m)
    maps = [_depth_on_grid(d, (H, W), (h, w), m) for d in depth_maps]
    return out, map_K(K, (W, H), (w, h), plain_K_scaling), maps


def _depth_on_grid(d, src_hw, dst_hw, m):
    if d is None:
        return None
    d = np.asarray(d, np.float32)
    if d.shape[-2:] == tuple(dst_hw):
        return d
    if d.shape[-2:] != tuple(src_hw):
        raise ValueError(f"depth map {d.shape} is neither on the frame grid {src_hw} nor on the track grid {dst_hw}")
    flat = d.reshape((-1,) + d.shape[-2:])
    res = np.stack([warp_image(x, m, interp="nearest") for x in flat])
    return res.reshape(d.shape[:-2] + tuple(dst_hw))


def _as_4x4(E: np.ndarray) -> np.ndarray:
    if E.shape[-2:] == (4, 4):
        return E
    if E.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsics must be (..., 4, 4) or (..., 3, 4), got {E.shape}")
    out = np.zeros(E.shape[:-2] + (4, 4), E.dtype)
    out[..., :3, :] = E
    out[..., 3, 3] = 1.0
    return out


def check_world_transform(A) -> np.ndarray:
    """Validate a world transform X' = A X: (4, 4) with an orthogonal 3x3 block (rotation or reflection, no scale)
    and last row [0, 0, 0, 1]. Returns it as float64."""
    A = np.asarray(A, np.float64)
    if A.shape != (4, 4) or not np.allclose(A[3], [0, 0, 0, 1]) or \
            not np.allclose(A[:3, :3] @ A[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError("world_transform must be a 4x4 rigid or mirror transform (orthogonal 3x3, last row 0 0 0 1)")
    return A


_UNITS_PER_M = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def source_world_transform(to_source: dict) -> np.ndarray:
    """The sample's SOURCE world as a `track(world_transform=...)` argument.

    to_source: sample.json["world"]["to_source"] = {"A": (3, 3), "b": (3,), "source_units": "m" | "cm" | "mm"}, the
    map X_source = A X + b from the released world X (metres) to the source frame (in source units). Returns the
    (4, 4) float64 transform X' = (A X + b) / units_per_metre, i.e. the source frame expressed in metres; it must be
    rigid or a mirror (checked).
    The pre-release delta numbers were computed in exactly these frames: Simul-Robot in the left-handed Unreal world
    (a mirror of the released world, ~150 m from its origin), Simul-General in the un-centred vendor world, Real-* in
    the released world (identity). MVTracker is not mirror invariant: on 10 Simul-Robot clips the per-clip delta of the
    real video moved by ~5 points on average (up to 16 on a hard moving-camera clip) between the two handednesses, so
    reproducing the legacy numbers needs this transform; the default protocol tracks in the released right-handed world
    (no transform).
    """
    units = str(to_source.get("source_units", "m"))
    if units not in _UNITS_PER_M:
        raise ValueError(f"source_units {units!r} not in {sorted(_UNITS_PER_M)}")
    A = np.eye(4)
    A[:3, :3] = np.asarray(to_source["A"], np.float64).reshape(3, 3) / _UNITS_PER_M[units]
    A[:3, 3] = np.asarray(to_source["b"], np.float64).reshape(3) / _UNITS_PER_M[units]
    return check_world_transform(A)


def check_anchor(anchor: Anchor, V: int, H: int, W: int) -> None:
    """Raise ValueError unless the anchor maps are (V, H, W) and the point visibility has V rows."""
    for name in ("depth", "norm_depth"):
        d = getattr(anchor, name)
        if d is not None and d.shape != (V, H, W):
            raise ValueError(f"anchor.{name} {d.shape} must be (V, H, W) = {(V, H, W)}: one map per view on the "
                             f"frame grid (or on track_size); see Sample.depth(view, [t], size=..., mode=...)")
    if anchor.points_vis is not None and anchor.points_vis.shape[0] != V:
        raise ValueError(f"anchor.points_vis has {anchor.points_vis.shape[0]} views, the frames have {V}")


def check_inputs(frames, K, w2c, queries) -> tuple:
    """Validate and normalise lifter inputs. Accepts static cameras K (V, 3, 3) / w2c (V, 4, 4) or (V, 3, 4) and
    per-frame (V, n, ...) ones. Returns frames (V, n, H, W, 3) uint8, K (V, n, 3, 3) float64,
    w2c (V, n, 4, 4) float64, queries (N, 4) float64 with integer t in [0, n-1] and finite xyz."""
    frames = np.asarray(frames)
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (V, n, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 RGB, got {frames.dtype}")
    V, n = frames.shape[:2]
    K = np.asarray(K, np.float64)
    if K.shape == (V, 3, 3):
        K = np.repeat(K[:, None], n, 1)
    if K.shape != (V, n, 3, 3):
        raise ValueError(f"K must be (V, n, 3, 3) = {(V, n, 3, 3)}, got {K.shape}")
    w2c = _as_4x4(np.asarray(w2c, np.float64))
    if w2c.shape == (V, 4, 4):
        w2c = np.repeat(w2c[:, None], n, 1)
    if w2c.shape != (V, n, 4, 4):
        raise ValueError(f"w2c must be (V, n, 4, 4) = {(V, n, 4, 4)}, got {w2c.shape}")
    q = np.asarray(queries, np.float64).reshape(-1, 4)
    if len(q):
        t = q[:, 0]
        if not (np.all(np.isfinite(q)) and np.all(t == np.round(t)) and t.min() >= 0 and t.max() <= n - 1):
            raise ValueError(f"queries must be finite [t, x, y, z] with integer t in [0, {n - 1}]")
    return frames, K, w2c, q


class Lifter(abc.ABC):
    """Interface of a multi-view 3D lifter (see the module docstring for shapes and conventions)."""

    name = "lifter"

    @abc.abstractmethod
    def track(self, frames: np.ndarray, K: np.ndarray, w2c: np.ndarray, queries: np.ndarray, anchor: Anchor,
              **kwargs):
        """frames (V, n, H, W, 3) uint8, K (V, n, 3, 3), w2c (V, n, 4, 4) world->camera (m), queries (N, 4)
        [t, x, y, z] world (m), anchor `Anchor` -> (xyz (n, N, 3) float32 world m, vis (n, N) bool)."""

    def prepare_anchor(self, anchor: Anchor, frames: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                       **kwargs) -> Anchor:
        """Return `anchor` with `norm_depth` set from the REFERENCE (real) video of the sample, so that the scene
        normalisation is identical for every source tracked against it. The base implementation can only use the
        GT depth ("gt_depth"); depth-estimating lifters override it."""
        if anchor.norm_depth is None and anchor.depth is not None:
            return replace(anchor, norm_depth=anchor.depth)
        return anchor


class DepthAnchoredLifter(Lifter):
    """Generic lifter = monocular metric depth + anchoring + MVTracker-style scene normalisation + a 3D tracker.

    Subclasses implement `estimate_depth` and `run_tracker`.

    track_size       (w, h) or None: grid the views are resampled to before depth estimation and tracking (frames
                     INTER_AREA/INTER_CUBIC, K via `map_K`, anchor depth maps nearest). Use it to undo an
                     aspect-changing eval grid, e.g. (640, 480) for the legacy squashed 480x480 Real-General frames.
                     Can be overridden per `track` call.
    plain_K_scaling  K mapping used by that resampling (see `map_K`); True reproduces the pre-release code.
    """

    name = "depth_anchored"

    def __init__(self, track_size=None, plain_K_scaling: bool = False):
        self.track_size = None if track_size is None else tuple(int(x) for x in track_size)
        self.plain_K_scaling = bool(plain_K_scaling)
        self.last_info: dict = {}

    @abc.abstractmethod
    def estimate_depth(self, frames: np.ndarray, fov_x: np.ndarray) -> np.ndarray:
        """frames (n, H, W, 3) uint8 of ONE view, fov_x (n,) horizontal field of view in degrees ->
        metric planar depth (n, H, W) float32 in metres, non-finite where invalid."""

    @abc.abstractmethod
    def run_tracker(self, frames: np.ndarray, depths: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                    queries: np.ndarray) -> tuple:
        """All arrays in the NORMALISED scene: frames (V, n, H, W, 3) uint8, depths (V, n, H, W) float32
        (0 = invalid), K (V, n, 3, 3) float32, w2c (V, n, 3, 4) float32 [R | t], queries (N, 4) float32
        [t, x', y', z'] -> (xyz' (n, N, 3) normalised coordinates, vis (n, N) bool)."""

    def anchored_depth(self, frames: np.ndarray, K: np.ndarray, w2c: np.ndarray, anchor: Anchor,
                       fov_x=None) -> tuple:
        """Estimated + anchored depth of every view. frames (V, n, H, W, 3) uint8, K (V, n, 3, 3), w2c (V, n, 4, 4)
        on the same grid as the anchor maps; fov_x: horizontal FoV in degrees for the estimator, scalar, (V,) or
        (V, n) (None = 2 atan(W / 2 fx) from K) -> (depth (V, n, H, W) float32 metres, 0 = invalid, [info per view])."""
        V, n, H, W = frames.shape[:4]
        fov = fov_x_deg(K, W) if fov_x is None else np.broadcast_to(
            np.asarray(fov_x, np.float64).reshape((V, -1) if np.size(fov_x) > 1 else (1, 1)), (V, n))
        out = np.zeros((V, n, H, W), np.float32)
        info = []
        for v in range(V):
            d = np.asarray(self.estimate_depth(frames[v], fov[v]), np.float32)
            if d.shape != (n, H, W):
                raise ValueError(f"estimate_depth returned {d.shape}, expected {(n, H, W)}")
            out[v], inf = anchor_depth(d, v, K[v, 0], w2c[v, 0], anchor)
            info.append({"view": v, **inf})
        return out, info

    def _regrid(self, frames, K, anchor, track_size):
        size = self.track_size if track_size is None else tuple(int(x) for x in track_size)
        H, W = frames.shape[2:4]
        if size is None or size == (W, H):
            return frames, K, anchor, None
        frames, K, (depth, norm_depth) = resample_inputs(frames, K, size, self.plain_K_scaling,
                                                         (anchor.depth, anchor.norm_depth))
        return frames, K, replace(anchor, depth=depth, norm_depth=norm_depth), (W, H)

    def prepare_anchor(self, anchor: Anchor, frames: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                       track_size=None, fov_x=None, **kwargs) -> Anchor:
        """Fill `anchor.norm_depth` from the REFERENCE (real) video: the GT depth for "gt_depth", otherwise the
        anchored depth estimate of the reference video's window position 0 (frames[:, 0], estimated alone).
        frames (V, n, H, W, 3) uint8 (only position 0 is used), K / w2c / track_size / fov_x as for `track`.
        Returns a new Anchor (the norm depth is on the track grid when `track_size` resampling applies; `track`
        accepts either grid)."""
        if anchor.norm_depth is not None:
            return anchor
        if anchor.mode == "gt_depth":
            return replace(anchor, norm_depth=anchor.depth)
        frames, K, w2c, _ = check_inputs(np.asarray(frames)[:, :1], np.asarray(K)[:, :1] if np.ndim(K) == 4 else K,
                                         np.asarray(w2c)[:, :1] if np.ndim(w2c) == 4 else w2c, np.zeros((0, 4)))
        frames, K, anchor_g, _ = self._regrid(frames, K, anchor, track_size)
        check_anchor(anchor_g, *frames.shape[:1], *frames.shape[2:4])
        self._check_pixels(K)
        if fov_x is not None and np.ndim(fov_x) == 2:
            fov_x = np.asarray(fov_x)[:, :1]
        fov = fov_x_deg(K, frames.shape[3]) if fov_x is None else fov_x
        depth, _ = self.anchored_depth(frames, K.astype(np.float32), w2c.astype(np.float32), anchor_g, fov)
        return replace(anchor, norm_depth=depth[:, 0])

    def track(self, frames: np.ndarray, K: np.ndarray, w2c: np.ndarray, queries: np.ndarray, anchor: Anchor,
              track_size=None, depth: np.ndarray | None = None, world_transform: np.ndarray | None = None,
              fov_x=None, return_info: bool = False):
        """Lift a multi-view video to world tracks (shapes and conventions in the module docstring).

        Cameras, queries and the scene normalisation are handled in float32 (the tracker's precision) with the
        operation order of the pre-release implementation, so that legacy inputs give bit-identical tracker inputs.

        track_size       override of the instance `track_size` for this call.
        depth            optional metric depth (V, n, H, W) metres (0 / non-finite = invalid) on the frame grid used
                         INSTEAD of the estimator (e.g. GT depth); no anchoring, the far clip still applies.
        fov_x            optional horizontal FoV in degrees for the depth estimator: scalar, (V,) or (V, n)
                         (default 2 atan(W / 2 fx) from K; the pre-release runs used the source metadata, e.g. the
                         Simul-Robot `hfov_deg`, which differs from the K-derived value in the 6th digit).
        world_transform  optional (4, 4) rigid or mirror transform A: the lifter works in the world X' = A X (queries,
                         anchor points and extrinsics are mapped; predictions are mapped back, so outputs stay in the
                         input world). The scene normalisation needs +z up, so use it for y-up worlds. Distances and
                         therefore every track metric are invariant to A, the 3D tracker is not: the pre-release
                         Simul-Robot runs worked in the left-handed Unreal frame, and tracking the same inputs in its
                         mirror changes per-clip delta by several points. Legacy reproduction from the released
                         files: world_transform = source_world_transform(sample.meta["world"]["to_source"]).
        return_info      also return a JSON-friendly dict (depth alignment per view, scene normalisation, grid);
                         the same dict is kept in `self.last_info`.
        Returns (xyz (n, N, 3) float32 world metres, vis (n, N) bool[, info]).
        """
        frames, K, w2c, q = check_inputs(frames, K, w2c, queries)
        A = None
        if world_transform is not None:
            A = check_world_transform(world_transform)
            q = q.copy()
            q[:, 1:] = q[:, 1:] @ A[:3, :3].T + A[:3, 3]
            w2c = w2c @ np.linalg.inv(A)
            if anchor.points is not None:
                anchor = replace(anchor, points=anchor.points @ A[:3, :3].T + A[:3, 3])
        V, n, H0, W0 = frames.shape[:4]
        frames, K, anchor, src = self._regrid(frames, K, anchor, track_size)
        H, W = frames.shape[2:4]
        check_anchor(anchor, V, H, W)
        info = {"lifter": self.name, "grid": [W, H], "resampled_from": None if src is None else list(src),
                "n_queries": int(len(q)), "world_transform": None if A is None else A.tolist()}
        if len(q) == 0:
            self.last_info = info
            out = (np.zeros((n, 0, 3), np.float32), np.zeros((n, 0), bool))
            return out + (info,) if return_info else out
        self._check_pixels(K)
        fov = fov_x_deg(K, W) if fov_x is None else fov_x
        K, w2c = K.astype(np.float32), np.ascontiguousarray(w2c[..., :3, :], dtype=np.float32)
        if depth is None:
            depths, align = self.anchored_depth(frames, K, w2c, anchor, fov)
        else:
            depths = _depth_on_grid(np.asarray(depth, np.float32), (H0, W0), (H, W),
                                    image_map((W0, H0), (W, H), "resize"))
            if depths.shape != (V, n, H, W):
                raise ValueError(f"depth must be (V, n, H, W) = {(V, n, H0, W0)}, got {np.shape(depth)}")
            with np.errstate(invalid="ignore"):
                depths = np.where(np.isfinite(depths) & (depths > 0) & (depths <= anchor.far_clip_m),
                                  depths, 0.0).astype(np.float32)
            align = [{"view": v, "mode": "given", "scale": 1.0, "shift": 0.0, "anchors": 0} for v in range(V)]
        norm_src = "norm_depth"
        norm_depth = anchor.norm_depth
        if norm_depth is None:
            norm_depth, norm_src = (anchor.depth, "depth") if anchor.depth is not None else (depths[:, 0], "own")
        scale, trans, ninfo = scene_normalization(norm_depth, K[:, 0], w2c[:, 0], anchor.far_clip_m,
                                                  fallback_points=q[:, 1:])
        T32 = trans.astype(np.float32)
        dn, wn, qn = apply_normalization(scale, T32, depths, w2c, q[:, 1:].astype(np.float32))
        pred, vis = self.run_tracker(frames, dn, K, wn, np.concatenate([q[:, :1].astype(np.float32), qn], 1))
        pred = np.asarray(pred, np.float32).reshape(n, len(q), 3)
        xyz = invert_normalization(pred, scale, T32)
        if A is not None:
            Ai = np.linalg.inv(A)
            xyz = xyz.astype(np.float64) @ Ai[:3, :3].T + Ai[:3, 3]
        xyz = xyz.astype(np.float32)
        vis = np.asarray(vis).reshape(n, len(q)).astype(bool)
        info.update({"depth_align": align,
                     "scene_norm": {"scale": scale, "translation": [float(x) for x in trans],
                                    "from": norm_src if ninfo["source"] == "depth" else ninfo["source"],
                                    "n_points": ninfo["n_points"]}})
        self.last_info = info
        return (xyz, vis, info) if return_info else (xyz, vis)

    @staticmethod
    def _check_pixels(K):
        r = K[..., 1, 1] / K[..., 0, 0]
        if np.any(np.abs(r - 1.0) > NON_SQUARE_WARN):
            warnings.warn(f"non-square pixels (fy/fx in [{r.min():.3f}, {r.max():.3f}]); monocular depth models assume "
                          f"square pixels - pass track_size to resample to the native aspect", stacklevel=3)


_LIFTERS: dict = {"mvtracker": "vpm_worldbench.lifters.mvtracker:MVTrackerLifter"}


def register_lifter(name: str, factory) -> None:
    """Register a lifter class / factory (callable(**kwargs) -> Lifter) or an 'module:attr' path under `name`."""
    _LIFTERS[name] = factory


def get_lifter(name: str = "mvtracker", **kwargs) -> Lifter:
    """Instantiate a registered lifter (Protocol.lifter names it), or one given as an importable 'module:attr' path
    (e.g. a protocol YAML with ``lifter: my_package.lifters:MyLifter``). kwargs go to its constructor."""
    if name not in _LIFTERS and ":" not in name:
        raise KeyError(f"unknown lifter {name!r}; registered: {sorted(_LIFTERS)} (or give 'module:attr')")
    factory: Callable = _LIFTERS.get(name, name)
    if isinstance(factory, str):
        mod, attr = factory.split(":")
        factory = getattr(importlib.import_module(mod), attr)
    return factory(**kwargs)
