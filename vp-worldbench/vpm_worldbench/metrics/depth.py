"""Depth metrics: alignment of (relative) depth predictions to metric ground truth, and the standard error and
accuracy measures of monocular / video depth estimation (task `depth`, and the depth part of `recon4d`).

    aligned = align(pred, gt, mask, mode="scale_shift", per="sequence", space="disparity")
    m = depth_metrics(aligned, gt, mask)              # {"abs_rel", "sq_rel", "rmse", ..., "delta1", "coverage", ...}
    m = evaluate_depth(pred, gt, mask, align_mode="scale_shift", per="sequence", space="disparity")   # both steps

Conventions (docs/DATA_FORMAT.md)
- Depth is planar camera z in metres. Arrays are (T, H, W) videos of one view (frames = axis 0) or single (H, W)
  frames; any array with ndim >= 3 is read as (T, ...) with the trailing axes pooled inside a frame, and any array with
  ndim <= 2 is ONE frame. Per-frame sparse values, e.g. depth at N track points over T frames (see `sample_depth`),
  are therefore passed as (T, N, 1); a (T, N) array would be scored as one frame.
- Scored GT pixels: `mask` (optional, bool, same shape) is True, GT is finite and min_depth < GT (and GT < max_depth
  when max_depth is given). The benchmark stores invalid depth as 0, which the default min_depth = 1e-3 m excludes;
  min_depth must be >= 0 (GT <= 0 is never valid).
- Missing predictions: a predicted depth that is non-finite or <= 0 (the benchmark's "0 = invalid" convention) is
  *missing*. Missing pixels are excluded from alignment fits and from the metrics, and are reported through
  `coverage` = percent of scored GT pixels that have a prediction. Report coverage next to the errors.
- Before scoring, predictions are clipped to [min_depth, max_depth] (KITTI / Eigen practice). `align` never turns a
  fitted pixel into a missing one: its depth-space outputs are at least max(min_depth, MIN_PRED_DEPTH) > 0.
- Empty inputs (no frame or no pixel) give NaN metrics with zero counts.
- Summary dicts give shares (delta1..3, coverage) in percent (0-100) and errors in natural units.

Metrics (Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network", NeurIPS 2014;
the formulas of e.g. Monodepth2 `compute_errors`), with p = prediction, g = GT, d = ln p - ln g:

    abs_rel  = mean |p - g| / g                 sq_rel   = mean (p - g)^2 / g          [m]
    rmse     = sqrt(mean (p - g)^2)  [m]        rmse_log = sqrt(mean d^2)
    log10    = mean |log10 p - log10 g|         silog    = 100 * sqrt(mean d^2 - (mean d)^2)   (KITTI scale-invariant)
    delta_k  = 100 * mean(max(p / g, g / p) < 1.25^k),  k = 1, 2, 3   (strict <)

Reduction over a video (`reduce`): "frames" (default) computes every metric per frame and averages over the frames
that have scored pixels (the per-image KITTI / NYU practice, e.g. Monodepth2: `compute_errors` per image, then the
mean); "pixels" pools all scored pixels of the sequence into one set of metrics (MonST3R's `depth_evaluation`).
The DepthCrafter / Video Depth Anything `metric.py` reduces abs_rel, sq_rel, rmse, rmse_log and delta1..3 the same
way as "frames" (per-frame sums / n, then the mean over frames), but not log10 and silog: its log10 pools all valid
pixels of the video and its silog_rmse is 100 * sqrt(mean over frames of the per-frame variance), while here both are
per frame, then averaged (KITTI practice; a difference of < 1 % in random trials). Their evaluation scripts report
only abs_rel, rmse and delta1, which this module reproduces.

Alignment (affine-invariant evaluation of MiDaS, Ranftl et al., TPAMI 2022; video protocol of DepthCrafter, Hu et al.,
CVPR 2025, and Video Depth Anything, Chen et al., CVPR 2025): a least-squares scale (and shift) fitted on the scored
pixels, once per sequence (one fit over all frames: temporally consistent depth) or per frame, in disparity (1/depth)
or depth space. The video-depth standard is `mode="scale_shift", per="sequence", space="disparity"`: the fit is
`gt_disparity ~ s * pred_disparity + b`, the aligned disparity is clamped at >= 1e-3 (both reference implementations
do this) and inverted, then clipped to [min_depth, max_depth] and scored in depth space. The reference code also
clamps the raw predicted disparity at >= 1e-3 before the fit (`pred_min=1e-3` here; off by default because it is
tied to disparities normalised to [0, 1]) and adds 1e-8 to the GT before inverting it (not done here: a relative
change of < 1e-5 of the GT disparity). With a disparity prediction `pd` and GT depth `gd` (T, H, W):

    evaluate_depth(pd, gd, pred_space="disparity", max_depth=D, pred_min=1e-3)       # DepthCrafter / VDA eval
    evaluate_depth(pred_depth, gd, align_mode="median", per="frame", space="depth", max_depth=80.0)   # Monodepth2

reproduce DepthCrafter's `eval_single(domain="depth")` / Video Depth Anything's `eval_depthcrafter` (D = the dataset
max depth) and Monodepth2's monocular KITTI evaluation (median scaling per image; pass the Eigen crop as `mask`).
`evaluate_depth(pred, gt, align_mode="scale_shift" | "median", space="depth", min_depth=0.0, max_depth=D,
reduce="pixels")` reproduces MonST3R's `depth_evaluation` (lstsq / median alignment, max_depth=D, post_clip_max=D,
no custom_mask) when every prediction is > 0 and every aligned depth is >= MIN_PRED_DEPTH. Outside that, MonST3R
fits and scores predictions <= 0 (missing here), and computes abs_rel, sq_rel and RMSE on the unclamped aligned depth
(which can be <= 0), clamping at 1e-5 only before the log metrics and delta (here every metric sees the clamped
value; the log metrics and delta still agree). It also fits on all valid GT pixels and applies `custom_mask` only
afterwards (here `mask` restricts the fit too).

`mode="none"` performs no fit and, apart from an explicit `pred_min`, no clamping: the prediction is only converted
to depth (1 / disparity for a disparity input; a non-positive disparity is missing). `depth_metrics` still clips to
[min_depth, max_depth] when scoring.

Everything is numpy (float64 accumulation, processed frame by frame to bound memory); deterministic.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["align", "depth_metrics", "depth_metrics_per_frame", "evaluate_depth", "valid_gt_mask", "sample_depth",
           "ALIGN_MODES", "ALIGN_PER", "SPACES", "METRIC_KEYS", "DISP_FLOOR", "DEFAULT_MIN_DEPTH", "MIN_PRED_DEPTH"]

ALIGN_MODES = ("none", "median", "scale", "scale_shift")
ALIGN_PER = ("sequence", "frame")
SPACES = ("depth", "disparity")
REDUCE_MODES = ("pixels", "frames")
DISP_FLOOR = 1e-3
DEFAULT_MIN_DEPTH = 1e-3
MIN_PRED_DEPTH = 1e-5
DELTA_BASE = 1.25
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3")

_NAN = float("nan")
_N_VALID, _N_SCORED, _ABS, _SQ, _SE, _D, _D2, _L10, _C1, _C2, _C3 = range(11)
_N_SUMS = 11


def _check_choice(name: str, value, choices) -> None:
    if value not in choices:
        raise ValueError(f"{name}={value!r} not in {choices}")


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of products with numpy's pairwise summation (deterministic, independent of BLAS threading)."""
    return float(np.sum(a * b))


def _rows(a: np.ndarray) -> np.ndarray:
    """View/copy of `a` as (G, M) frames x values: axis 0 = frames for ndim >= 3; ndim <= 2 is a single frame.
    Explicit sizes, so empty inputs (T = 0 or no pixel) reshape too."""
    a = np.asarray(a)
    if a.ndim <= 2:
        return a.reshape(1, a.size)
    return a.reshape(a.shape[0], int(np.prod(a.shape[1:])))


def _check_range(min_depth: float, max_depth: Optional[float]) -> None:
    if not (np.isfinite(min_depth) and min_depth >= 0):
        raise ValueError(f"min_depth must be finite and >= 0 (metres), got {min_depth}")
    if max_depth is not None and not max_depth > min_depth:
        raise ValueError(f"max_depth ({max_depth}) must be > min_depth ({min_depth})")


def _depth_floor(min_depth: float) -> float:
    """Lower clamp of aligned depth: min_depth, or MIN_PRED_DEPTH when min_depth = 0 (always > 0)."""
    return float(min_depth) if min_depth > 0 else MIN_PRED_DEPTH


def _check_same_shape(pred: np.ndarray, gt: np.ndarray, mask) -> Optional[np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} must have the same shape (resample the prediction "
                         "to the GT grid first, nearest-neighbour for depth)")
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != gt.shape:
        raise ValueError(f"mask {mask.shape} must have the shape of gt {gt.shape}")
    return mask.astype(bool, copy=False)


def _valid_gt_rows(g: np.ndarray, m: Optional[np.ndarray], min_depth: float, max_depth: Optional[float]):
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(g) & (g > min_depth)
        if max_depth is not None:
            ok &= g < max_depth
    if m is not None:
        ok &= m
    return ok


def valid_gt_mask(gt, mask=None, min_depth: float = DEFAULT_MIN_DEPTH, max_depth: Optional[float] = None) -> np.ndarray:
    """Bool array (shape of `gt`): GT pixels that are scored, i.e. `mask` & finite & min_depth < gt (< max_depth)."""
    _check_range(min_depth, max_depth)
    gt = np.asarray(gt)
    m = None if mask is None else np.asarray(mask).astype(bool, copy=False)
    if m is not None and m.shape != gt.shape:
        raise ValueError(f"mask {m.shape} must have the shape of gt {gt.shape}")
    return _valid_gt_rows(gt, m, min_depth, max_depth)


def _to_space(pred: np.ndarray, pred_space: str, space: str) -> np.ndarray:
    """float64 copy of the prediction expressed in the alignment space; NaN marks missing predictions."""
    x = np.array(pred, dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        if pred_space == "depth":
            x[~(np.isfinite(x) & (x > 0))] = np.nan
            if space == "disparity":
                np.reciprocal(x, out=x)
        else:
            x[~np.isfinite(x)] = np.nan
            if space == "depth":
                x[~(x > 0)] = np.nan
                np.reciprocal(x, out=x)
    return x


def _from_space(x: np.ndarray, space: str, depth_floor: float) -> np.ndarray:
    """In place: aligned values -> depth >= a positive floor (NaN stays NaN): 1 / max(x, DISP_FLOOR) for disparity,
    max(x, depth_floor) for depth (depth_floor > 0, so every fitted pixel stays scored)."""
    if space == "disparity":
        np.maximum(x, DISP_FLOOR, out=x)
        np.reciprocal(x, out=x)
    else:
        np.maximum(x, depth_floor, out=x)
    return x


def _unaligned_depth(pred: np.ndarray, x: np.ndarray, pred_space: str, space: str, pred_min: Optional[float]):
    """mode="none": the prediction as depth, without any fit or floor. Without pred_min it is converted straight from
    `pred` (no disparity round trip); with pred_min, x (alignment space, already clamped) is converted, a non-positive
    disparity being missing (NaN)."""
    if pred_min is None:
        return _to_space(pred, pred_space, "depth")
    if space == "disparity":
        with np.errstate(invalid="ignore", divide="ignore"):
            x[~(x > 0)] = np.nan
            np.reciprocal(x, out=x)
    return x


def _fit(subsets: Callable, n_groups: int, mode: str):
    """Least-squares alignment y ~ s * x + b over the pixels yielded by subsets(g) for g in range(n_groups).

    Returns (s, b, n_pixels). Too few pixels -> (nan, nan, n). Two passes (means, then centred sums) for accuracy."""
    if mode == "none":
        n = sum(int(subsets(g)[0].size) for g in range(n_groups))
        return 1.0, 0.0, n
    if mode == "median":
        xs, ys = [], []
        for g in range(n_groups):
            x, y = subsets(g)
            keep = x > 0
            xs.append(x[keep])
            ys.append(y[keep])
        x = np.concatenate(xs) if xs else np.zeros(0)
        y = np.concatenate(ys) if ys else np.zeros(0)
        if x.size == 0:
            return _NAN, _NAN, 0
        mx = float(np.median(x))
        return (float(np.median(y)) / mx if mx > 0 else _NAN), 0.0, int(x.size)
    n, sx, sy, sxx, sxy = 0, 0.0, 0.0, 0.0, 0.0
    for g in range(n_groups):
        x, y = subsets(g)
        n += x.size
        sx += float(x.sum())
        sy += float(y.sum())
        sxx += _dot(x, x)
        sxy += _dot(x, y)
    if mode == "scale":
        if n == 0 or not sxx > 0:
            return _NAN, _NAN, n
        return sxy / sxx, 0.0, n
    if n < 2:
        return _NAN, _NAN, n
    mx, my = sx / n, sy / n
    cxx, cxy = 0.0, 0.0
    for g in range(n_groups):
        x, y = subsets(g)
        dx = x - mx
        cxx += _dot(dx, dx)
        cxy += _dot(dx, y - my)
    if not cxx > 1e-12 * sxx:
        return (sxy / sxx if sxx > 0 else _NAN), 0.0, n
    s = cxy / cxx
    return s, my - s * mx, n


def align(pred, gt, mask=None, mode: str = "scale_shift", per: str = "sequence", space: str = "disparity", *,
          pred_space: str = "depth", min_depth: float = DEFAULT_MIN_DEPTH, max_depth: Optional[float] = None,
          pred_min: Optional[float] = None, return_params: bool = False):
    """Align a depth (or disparity) prediction to metric GT depth.

    Args:
        pred: (T, H, W) video or (H, W) frame (see the module conventions); depth (any positive unit) if
            pred_space == "depth", disparity (e.g. affine-invariant inverse depth) if pred_space == "disparity".
        gt: GT depth in metres, same shape; values <= min_depth / non-finite / >= max_depth are not used.
        mask: optional bool, same shape: pixels allowed in the fit (combined with the GT validity).
        mode: "none" (no fit and no clamp; only converts disparity -> depth), "median" (s = median(y) / median(x),
            b = 0), "scale" (least-squares s, b = 0) or "scale_shift" (least-squares s and b).
        per: "sequence" (one fit over all frames) or "frame" (one fit per frame; a frame without enough usable
            pixels gets NaN parameters and a NaN (missing) output).
        space: where the fit is done: "disparity" (x = 1/depth, y = 1/gt) or "depth" (x = depth, y = gt).
        pred_space: what `pred` holds, "depth" or "disparity".
        min_depth, max_depth: GT validity range in metres (min_depth >= 0; also the lower clamp of depth-space
            outputs, see Returns).
        pred_min: optional lower clamp of the prediction expressed in the alignment space (x = max(x, pred_min)),
            applied before the fit and to the output; DepthCrafter / Video Depth Anything use 1e-3 on their
            [0, 1]-normalised disparity. None (default) = no clamp.
        return_params: also return the fitted parameters.

    Returns:
        aligned depth in metres, float64, shape of `pred`; NaN where the prediction is missing (non-finite, or <= 0
        for depth input) or its frame could not be fitted. Disparity-space outputs are 1 / max(s x + b, 1e-3);
        depth-space outputs are max(s x + b, min_depth), or max(s x + b, MIN_PRED_DEPTH) when min_depth = 0, so
        every fitted pixel is > 0 and is scored by `depth_metrics`. mode="none" returns the prediction as depth
        (pred, or 1 / pred for a disparity input with pred > 0; NaN otherwise), unclamped unless pred_min is given.
        With return_params: (aligned, params) where params = {"mode", "per", "space", "pred_space", "scale",
        "shift", "n_pixels"}; scale/shift/n_pixels are scalars for per="sequence" and arrays (T,) for per="frame".
    """
    _check_choice("mode", mode, ALIGN_MODES)
    _check_choice("per", per, ALIGN_PER)
    _check_choice("space", space, SPACES)
    _check_choice("pred_space", pred_space, SPACES)
    _check_range(min_depth, max_depth)
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask = _check_same_shape(pred, gt, mask)
    x = _to_space(pred, pred_space, space)
    if pred_min is not None:
        np.maximum(x, float(pred_min), out=x)
    X, G = _rows(x), _rows(gt)
    M = None if mask is None else _rows(mask)

    def subsets(g: int):
        ok = _valid_gt_rows(G[g], None if M is None else M[g], min_depth, max_depth) & np.isfinite(X[g])
        xs = X[g][ok]
        ys = G[g][ok].astype(np.float64)
        if space == "disparity":
            ys = 1.0 / ys
        return xs, ys

    n_groups = X.shape[0]
    if per == "sequence":
        s, b, n = _fit(subsets, n_groups, mode)
        scale, shift = np.full(n_groups, s), np.full(n_groups, b)
        params = {"scale": float(s), "shift": float(b), "n_pixels": int(n)}
    else:
        fits = [_fit(lambda _g, g=g: subsets(g), 1, mode) for g in range(n_groups)]
        scale = np.array([f[0] for f in fits], np.float64)
        shift = np.array([f[1] for f in fits], np.float64)
        params = {"scale": scale, "shift": shift, "n_pixels": np.array([f[2] for f in fits], np.int64)}
    if mode == "none":
        out = _unaligned_depth(pred, x, pred_space, space, pred_min).reshape(pred.shape)
    else:
        for g in range(n_groups):
            X[g] *= scale[g]
            X[g] += shift[g]
        out = _from_space(x, space, _depth_floor(min_depth)).reshape(pred.shape)
    if not return_params:
        return out
    params.update(mode=mode, per=per, space=space, pred_space=pred_space)
    return out, params


def _frame_sums(p: np.ndarray, g: np.ndarray, m: Optional[np.ndarray], min_depth: float,
                max_depth: Optional[float]) -> np.ndarray:
    """Accumulators of one frame (1-D rows): counts, error sums and threshold counts."""
    out = np.zeros(_N_SUMS, np.float64)
    ok = _valid_gt_rows(g, m, min_depth, max_depth)
    out[_N_VALID] = ok.sum()
    with np.errstate(invalid="ignore"):
        sc = ok & np.isfinite(p) & (p > 0)
    n = int(sc.sum())
    out[_N_SCORED] = n
    if n == 0:
        return out
    pp = np.clip(p[sc].astype(np.float64), min_depth, np.inf if max_depth is None else max_depth)
    gg = g[sc].astype(np.float64)
    diff = pp - gg
    out[_ABS] = np.sum(np.abs(diff) / gg)
    out[_SQ] = np.sum(diff * diff / gg)
    out[_SE] = _dot(diff, diff)
    d = np.log(pp) - np.log(gg)
    out[_D] = d.sum()
    out[_D2] = _dot(d, d)
    out[_L10] = np.sum(np.abs(np.log10(pp) - np.log10(gg)))
    ratio = np.maximum(pp / gg, gg / pp)
    out[_C1] = np.count_nonzero(ratio < DELTA_BASE)
    out[_C2] = np.count_nonzero(ratio < DELTA_BASE ** 2)
    out[_C3] = np.count_nonzero(ratio < DELTA_BASE ** 3)
    return out


def _from_sums(s: np.ndarray) -> dict:
    n = s[_N_SCORED]
    if n <= 0:
        return {k: _NAN for k in METRIC_KEYS}
    md = s[_D] / n
    var = max(s[_D2] / n - md * md, 0.0)
    return {
        "abs_rel": s[_ABS] / n,
        "sq_rel": s[_SQ] / n,
        "rmse": float(np.sqrt(s[_SE] / n)),
        "rmse_log": float(np.sqrt(s[_D2] / n)),
        "log10": s[_L10] / n,
        "silog": 100.0 * float(np.sqrt(var)),
        "delta1": 100.0 * s[_C1] / n,
        "delta2": 100.0 * s[_C2] / n,
        "delta3": 100.0 * s[_C3] / n,
    }


def _all_frame_sums(pred, gt, mask, min_depth, max_depth) -> np.ndarray:
    """(T, _N_SUMS) per-frame accumulators ((0, _N_SUMS) for an input without frames)."""
    _check_range(min_depth, max_depth)
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask = _check_same_shape(pred, gt, mask)
    P, G = _rows(pred), _rows(gt)
    M = None if mask is None else _rows(mask)
    if P.shape[0] == 0:
        return np.zeros((0, _N_SUMS), np.float64)
    return np.stack([_frame_sums(P[i], G[i], None if M is None else M[i], min_depth, max_depth)
                     for i in range(P.shape[0])])


def depth_metrics_per_frame(pred, gt, mask=None, *, min_depth: float = DEFAULT_MIN_DEPTH,
                            max_depth: Optional[float] = None) -> dict:
    """Per-frame depth metrics (frames = axis 0 of a (T, H, W) video; an (H, W) input is one frame).

    Args: as `depth_metrics` (pred is metric depth, e.g. the output of `align`).
    Returns: {metric: (T,) float64 array (NaN for frames without scored pixels)} for every key of METRIC_KEYS (delta in
        percent), plus "n_valid" / "n_scored" (T,) int64 and "coverage" (T,) percent.
    """
    sums = _all_frame_sums(pred, gt, mask, min_depth, max_depth)
    per = [_from_sums(s) for s in sums]
    out = {k: np.array([r[k] for r in per], np.float64) for k in METRIC_KEYS}
    out["n_valid"] = sums[:, _N_VALID].astype(np.int64)
    out["n_scored"] = sums[:, _N_SCORED].astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["coverage"] = np.where(sums[:, _N_VALID] > 0, 100.0 * sums[:, _N_SCORED] / np.maximum(sums[:, _N_VALID], 1),
                                   np.nan)
    return out


def depth_metrics(pred, gt, mask=None, *, min_depth: float = DEFAULT_MIN_DEPTH, max_depth: Optional[float] = None,
                  reduce: str = "frames") -> dict:
    """Standard depth metrics of a metric depth prediction against GT depth.

    Args:
        pred: predicted depth in metres, (T, H, W) video or (H, W) frame (module conventions); non-finite or <= 0 =
            missing (excluded, see "coverage"); other values are clipped to [min_depth, max_depth].
        gt: GT depth in metres, same shape; scored where finite and min_depth < gt (< max_depth).
        mask: optional bool, same shape; restricts the scored pixels further.
        min_depth, max_depth: evaluation range in metres (max_depth None = no upper limit).
        reduce: "frames" (default) computes the metrics per frame and averages over the frames that have scored
            pixels (per-image KITTI / NYU practice; the DepthCrafter / Video Depth Anything reduction except for
            log10 and silog, see the module docstring); "pixels" pools all scored pixels of all frames
            (MonST3R-style). Identical for a single frame.

    Returns:
        {"abs_rel", "sq_rel" [m], "rmse" [m], "rmse_log", "log10", "silog", "delta1", "delta2", "delta3" [percent],
         "n_valid" (scored GT pixels), "n_scored" (... with a prediction), "coverage" [percent of n_valid],
         "n_frames" (frames with scored pixels), "reduce"}. Metrics are NaN when nothing is scored.
    """
    _check_choice("reduce", reduce, REDUCE_MODES)
    sums = _all_frame_sums(pred, gt, mask, min_depth, max_depth)
    has = sums[:, _N_SCORED] > 0
    if reduce == "pixels":
        out = _from_sums(sums.sum(0))
    else:
        rows = [_from_sums(s) for s in sums[has]]
        out = {k: (float(np.mean([r[k] for r in rows])) if rows else _NAN) for k in METRIC_KEYS}
    out = {k: float(v) for k, v in out.items()}
    n_valid, n_scored = int(sums[:, _N_VALID].sum()), int(sums[:, _N_SCORED].sum())
    out.update(n_valid=n_valid, n_scored=n_scored, coverage=(100.0 * n_scored / n_valid if n_valid else _NAN),
               n_frames=int(has.sum()), reduce=reduce)
    return out


def evaluate_depth(pred, gt, mask=None, *, align_mode: str = "scale_shift", per: str = "sequence",
                   space: str = "disparity", pred_space: str = "depth", min_depth: float = DEFAULT_MIN_DEPTH,
                   max_depth: Optional[float] = None, pred_min: Optional[float] = None,
                   reduce: str = "frames") -> dict:
    """`align` followed by `depth_metrics` with the same mask and depth range (arguments as there; `align_mode` is
    `align`'s `mode`). The defaults are the DepthCrafter / Video Depth Anything protocol (one scale-shift fit in
    disparity per sequence, per-frame metrics averaged over frames), except pred_min and the log10 / silog reduction
    (see the module docstring).

    Returns the `depth_metrics` dict plus "align": {"mode", "per", "space", "pred_space", "scale", "shift",
    "n_pixels"} (JSON-friendly: lists for per="frame").
    """
    aligned, params = align(pred, gt, mask, mode=align_mode, per=per, space=space, pred_space=pred_space,
                            min_depth=min_depth, max_depth=max_depth, pred_min=pred_min, return_params=True)
    out = depth_metrics(aligned, gt, mask, min_depth=min_depth, max_depth=max_depth, reduce=reduce)
    out["align"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}
    return out


def sample_depth(depth, uv, interp: str = "nearest") -> np.ndarray:
    """Read a depth map at pixel positions (e.g. predicted depth at projected GT track points).

    Args:
        depth: (H, W) depth in metres (<= 0 or non-finite = invalid).
        uv: (..., 2) pixel coordinates on the depth grid, OpenCV convention (integer = pixel centre).
        interp: "nearest" (pixel floor(u + 0.5), floor(v + 0.5)) or "bilinear" (edge-replicated in the outer half
            pixel; NaN if any neighbour with non-zero weight is invalid, so no mixing with holes).

    Returns:
        (...,) float64 depth; NaN outside the image ([-0.5, W - 0.5) x [-0.5, H - 0.5)) or where invalid.
        Sparse per-frame values of T frames stack to (T, N); score them per frame as (T, N, 1) (module conventions).
    """
    _check_choice("interp", interp, ("nearest", "bilinear"))
    d = np.asarray(depth, np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {d.shape}")
    H, W = d.shape
    uv = np.asarray(uv, np.float64)
    if uv.shape[-1] != 2:
        raise ValueError(f"uv must be (..., 2), got {uv.shape}")
    u, v = uv[..., 0], uv[..., 1]
    out = np.full(u.shape, np.nan)
    with np.errstate(invalid="ignore"):
        ins = np.isfinite(u) & np.isfinite(v) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    uu, vv = u[ins], v[ins]
    if interp == "nearest":
        iu = np.clip(np.floor(uu + 0.5).astype(np.int64), 0, W - 1)
        iv = np.clip(np.floor(vv + 0.5).astype(np.int64), 0, H - 1)
        val = d[iv, iu]
        bad = ~(np.isfinite(val) & (val > 0))
    else:
        uc, vc = np.clip(uu, 0, W - 1), np.clip(vv, 0, H - 1)
        u0, v0 = np.floor(uc).astype(np.int64), np.floor(vc).astype(np.int64)
        u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
        wu, wv = uc - u0, vc - v0
        val = np.zeros(uc.shape)
        bad = np.zeros(uc.shape, bool)
        for iu, iv, w in ((u0, v0, (1 - wu) * (1 - wv)), (u1, v0, wu * (1 - wv)),
                          (u0, v1, (1 - wu) * wv), (u1, v1, wu * wv)):
            dc = d[iv, iu]
            okc = np.isfinite(dc) & (dc > 0)
            use = w > 0
            bad |= use & ~okc
            val += np.where(use & okc, w * np.where(okc, dc, 0.0), 0.0)
    val[bad] = np.nan
    out[ins] = val
    return out
