"""Point-tracking metrics: 3D tracks (MVTracker / TAP-Vid-3D style), 2D tracks (TAP-Vid) and trajectory errors.

Pure numpy, float64, deterministic. All arrays are time-major:

    positions   (T, N, D)  float   D = 3 (world metres) or 2 (pixels); NaN allowed where the point is unknown
    visibility  (T, N)     bool    True = visible. Numeric scores are accepted: occluded iff score < 0.5
    query_t     (N,)       int     frame index of each track's query (0 <= query_t < T)

Ground-truth entries whose position is not finite are treated as not visible.

3D: ``track3d_metrics``
    Re-implements ``mvtracker.evaluation.metrics.evaluate_predictions`` (MVTracker, Rajic et al., "Multi-View 3D Point
    Tracking", ICCV 2025, arXiv:2508.21060, github.com/ethz-vlg/mvtracker), i.e. the TAP-Vid metrics (Doersch et al.,
    NeurIPS 2022 D&B, arXiv:2211.03726) applied to 3D positions with absolute metric thresholds and averaged PER TRACK
    (not pooled per video as in TAP-Vid; not the depth-adaptive thresholds of the official TAPVid-3D metric,
    arXiv:2407.05921). Step by step:

    1. GT visibility is restricted to t >= query_t (frames before the query count as occluded).
    2. Tracks with fewer than 2 GT-visible frames (the query frame counts) are dropped.
    3. Scored entries of a track = GT-visible frames with t > query_t ("first" query mode: the query frame and
       everything before it are not evaluated).
    4. Per track and threshold d (metres, strict ``<``):
           pts_within_d = #(scored & err < d) / #scored
           jaccard_d    = TP / (#scored + FP),  TP = #(scored & err < d & pred visible),
                          FP = #(t > query_t & pred visible & (GT occluded | not err < d))
       delta_avg = mean_d pts_within_d, aj = mean_d jaccard_d (per track, then over tracks).
       Occlusion accuracy = share of frames t > query_t where (pred visible) == (GT visible).
       Survival = (f - query_t) / (T - query_t) with f the first GT-visible frame t >= query_t whose error exceeds
       ``survival_m`` (strict ``>``; f = T if there is none). A NaN error never counts as "within" and never as a
       survival failure (MVTracker's semantics; an infinite error is a failure).
    5. Splits by the GT path length L of the track (sum of the steps between consecutive GT-visible frames from the
       query on): any (all kept tracks), static (L < static_m), dynamic (L > dynamic_m).
    6. Means over the tracks of the split, in percent, computed in float64. ``evaluate_predictions`` accumulates in
       float32 (per-track values agree exactly; means differ by < 1e-4 percentage points) and rounds every value to
       2 decimals (pandas). ``np.round(x, 2)`` of this function's output gives the pre-release per-clip values, except
       at exact rounding ties (e.g. 0.625), where MVTracker's float32 noise decided the direction.

    Correspondence with ``evaluate_predictions`` column ``all_<split>``: ``{split}/delta_avg`` =
    average_pts_within_thresh, ``{split}/delta@d`` = pts_within_d, ``{split}/jaccard@d`` = jaccard_d, ``{split}/aj`` =
    average_jaccard, ``{split}/oa`` = occlusion_accuracy, ``{split}/survival`` = survival (MVTracker's per-track
    ate/mte/fde_visible are not reproduced). Pre-release error columns: Simul-* / Real-Robot ADE, MTE, FDE =
    ``ade``/``mte``/``fde`` with ``mask=None``; Real-General ADE, MTE = ``ade_all``/``mte_all``. The pre-release Simul-*
    runs had no static/dynamic split (``static_m=None, dynamic_m=None``; the "any" values do not depend on it).

2D: ``track2d_metrics``
    Port of ``compute_tapvid_metrics`` in google-deepmind/tapnet (``tapnet/tapvid/evaluation_datasets.py``,
    Apache-2.0): occlusion accuracy, ``pts_within_{x}``, ``jaccard_{x}``, ``average_jaccard`` and
    ``average_pts_within_thresh``, POOLED over all tracks and frames of one video (the benchmark number is the mean
    over videos). Coordinates are rescaled so the image is 256x256 before the pixel thresholds 1, 2, 4, 8, 16 apply.

Trajectory errors: ``point_errors``, ``ade``, ``mte``, ``fde`` (Euclidean distance in the input units).
Reference prediction: ``frozen_prediction`` (every point stays at its query-frame GT position).

Motion: ``moving_delta`` - delta on the MOVING entries only: GT-visible frames after the query frame where the GT
point is at least ``min_disp`` (default: the largest threshold) away from its query-frame position. The frozen-points
prediction scores exactly 0 there, so the value isolates how well the motion itself is reproduced (the per-track
"dynamic" split still counts the static frames of moving tracks, where a frozen prediction is exact).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "TAPVID_THRESHOLDS_PX", "threshold_key", "point_errors", "scored_mask", "ade", "mte", "fde", "frozen_prediction",
    "track3d_metrics", "track2d_metrics", "moving_delta",
]

TAPVID_THRESHOLDS_PX = (1, 2, 4, 8, 16)
_NAN = float("nan")


def threshold_key(thr) -> str:
    """Name of a threshold inside metric keys: ``f"{thr:g}"`` (0.01 -> '0.01', 0.10 -> '0.1', 16 -> '16')."""
    return f"{float(thr):g}"


def _positions(x, name: str, dim: Optional[int] = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3 or (dim is not None and a.shape[-1] != dim):
        want = f"(T, N, {dim})" if dim is not None else "(T, N, D)"
        raise ValueError(f"{name} must have shape {want}, got {a.shape}")
    return a


def _visibility(v, shape: tuple, name: str) -> np.ndarray:
    """(T, N) bool. None -> all visible; numeric scores -> occluded iff score < 0.5 (MVTracker's convention)."""
    if v is None:
        return np.ones(shape, dtype=bool)
    a = np.asarray(v)
    if a.shape != shape:
        hint = " (reduce per-view visibility first, e.g. visible.any(-1))" if a.ndim == len(shape) + 1 else ""
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}{hint}")
    return a.astype(bool) if a.dtype == bool else ~(a < 0.5)


def _query_frames(query_t, n_tracks: int, n_frames: int) -> np.ndarray:
    q = np.asarray(query_t)
    if q.ndim == 0:
        q = np.full(n_tracks, q)
    if q.shape != (n_tracks,):
        raise ValueError(f"query_t must have shape ({n_tracks},), got {q.shape}")
    if not np.issubdtype(q.dtype, np.integer):
        qf = q.astype(np.float64)
        if not np.all(np.isfinite(qf)) or np.any(qf != np.round(qf)):
            raise ValueError("query_t must hold integer frame indices")
        q = qf
    q = q.astype(np.int64)
    if n_tracks and (q.min() < 0 or q.max() >= n_frames):
        raise ValueError(f"query_t must lie in [0, {n_frames - 1}], got [{q.min()}, {q.max()}]")
    return q


def _thresholds(values, name: str) -> np.ndarray:
    t = np.asarray([float(v) for v in values], dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError(f"{name} must be a non-empty sequence of positive numbers, got {values!r}")
    if len({threshold_key(v) for v in t}) != t.size:
        raise ValueError(f"{name} has duplicate thresholds: {values!r}")
    return t


def point_errors(gt, pred) -> np.ndarray:
    """Euclidean error ``||pred - gt||`` per frame and track.

    gt, pred: (T, N, D) positions in the same units (metres for 3D, pixels for 2D).
    Returns (T, N) float64 in those units; non-finite where either input is non-finite.
    """
    g = _positions(gt, "gt")
    p = _positions(pred, "pred")
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} differ in shape")
    return np.linalg.norm(p - g, axis=-1)


def scored_mask(gt_vis, query_t, gt=None) -> np.ndarray:
    """(T, N) bool: the entries the track metrics score, i.e. GT-visible frames strictly after each track's query
    frame (``t > query_t[n]``). With ``gt`` (T, N, D) given, entries whose GT position is not finite are removed.

    gt_vis: (T, N) bool; query_t: (N,) int (or one int for all tracks).
    """
    v = np.asarray(gt_vis)
    if v.ndim != 2:
        raise ValueError(f"gt_vis must have shape (T, N), got {v.shape}")
    vis = _visibility(v, v.shape, "gt_vis")
    T, N = vis.shape
    q = _query_frames(query_t, N, T)
    out = vis & (np.arange(T)[:, None] > q[None, :])
    if gt is not None:
        g = _positions(gt, "gt")
        if g.shape[:2] != (T, N):
            raise ValueError(f"gt {g.shape} does not match gt_vis {vis.shape}")
        out &= np.isfinite(g).all(-1)
    return out


def _select(gt, pred, mask):
    err = point_errors(gt, pred)
    sel = np.isfinite(np.asarray(gt, dtype=np.float64)).all(-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != sel.shape:
            raise ValueError(f"mask must have shape {sel.shape}, got {m.shape}")
        sel = sel & m
    return err, sel


def _mean(e: np.ndarray) -> float:
    return float(np.mean(e)) if e.size else _NAN


def _median(e: np.ndarray) -> float:
    return float(np.median(e)) if e.size else _NAN


def _final_errors(err: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Error at the last selected frame of every track that has one."""
    has = sel.any(0)
    last = sel.shape[0] - 1 - np.argmax(sel[::-1], axis=0)
    cols = np.nonzero(has)[0]
    return err[last[cols], cols]


def ade(gt, pred, mask=None) -> float:
    """Average displacement error: mean of ``||pred - gt||`` over the selected entries, in the input units.

    gt, pred: (T, N, D). mask: (T, N) bool selecting the entries (e.g. ``scored_mask(...)``); entries with a
    non-finite GT position are always excluded. ``mask=None`` selects every entry with a finite GT position:
    all frames, the query frame and occluded frames included (the pre-release Simul-*/Real-Robot "ADE").
    Returns NaN when nothing is selected; a non-finite prediction inside the selection makes the result non-finite.
    """
    err, sel = _select(gt, pred, mask)
    return _mean(err[sel])


def mte(gt, pred, mask=None) -> float:
    """Median trajectory error: median of ``||pred - gt||`` over the selected entries (pooled over frames and tracks,
    ``np.median``), in the input units. Selection, NaN rules and ``mask=None`` exactly as in :func:`ade`."""
    err, sel = _select(gt, pred, mask)
    return _median(err[sel])


def fde(gt, pred, mask=None) -> float:
    """Final displacement error: mean over tracks of the error at the track's LAST selected frame, in the input
    units. gt, pred (T, N, D); mask (T, N) bool (entries with a non-finite GT position are excluded); tracks
    without a selected frame are skipped. ``mask=None`` uses the final frame T-1 of every track whose GT is finite
    there (the pre-release definition: visibility ignored). Returns NaN when no track qualifies."""
    err, sel = _select(gt, pred, mask)
    if mask is None:
        return _mean(err[-1][sel[-1]])
    return _mean(_final_errors(err, sel))


def frozen_prediction(gt, query_t) -> np.ndarray:
    """The "frozen points" reference prediction: every track stays at its GT position of its query frame.

    gt: (T, N, D) positions; query_t: (N,) int (or one int for all tracks).
    Returns (T, N, D) with ``pred[t, n] = gt[query_t[n], n]`` for every t, same dtype as ``gt`` (NaN if the GT is
    unknown at the query frame). Score it with ``pred_vis=None`` (always visible), as the pre-release reference row.
    """
    g = np.asarray(gt)
    if g.ndim != 3:
        raise ValueError(f"gt must have shape (T, N, D), got {g.shape}")
    T, N = g.shape[:2]
    q = _query_frames(query_t, N, T)
    return np.broadcast_to(g[q, np.arange(N)][None], g.shape).copy()


def _path_length(xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(N,) GT path length over the visible frames of each track (same arithmetic as MVTracker's loop)."""
    out = np.zeros(vis.shape[1], dtype=np.float64)
    for n in range(vis.shape[1]):
        p = xyz[vis[:, n], n]
        out[n] = np.linalg.norm(p[1:] - p[:-1], axis=-1).sum()
    return out


def _track3d_per_track(gt, gt_vis, pred, pred_vis, query_t, thresholds_m, survival_m, static_m, dynamic_m) -> dict:
    """Per-track values behind :func:`track3d_metrics` (fractions, not percent).

    Returns a dict with "thresholds" (K,), "kept" (Nk,) indices of the tracks with >= 2 visible frames, per kept
    track "pts_within" / "jaccard" (Nk, K), "occlusion_accuracy" / "survival" / "path_length" (Nk,), boolean split
    masks over the kept tracks in "splits", and (T, N) arrays "err" (metres) and "scored".
    """
    g_in = _positions(gt, "gt")
    p = _positions(pred, "pred")
    if p.shape != g_in.shape:
        raise ValueError(f"pred {p.shape} and gt {g_in.shape} differ in shape")
    T, N = g_in.shape[:2]
    q = _query_frames(query_t, N, T)
    thr = _thresholds(thresholds_m, "thresholds_m")
    survival_m = float(survival_m)

    finite = np.isfinite(g_in).all(-1)
    g = np.where(finite[..., None], g_in, 0.0)
    tt = np.arange(T)[:, None]
    vis = _visibility(gt_vis, (T, N), "gt_vis") & finite & (tt >= q[None, :])
    pvis = _visibility(pred_vis, (T, N), "pred_vis")
    ev = tt > q[None, :]
    err = np.linalg.norm(p - g, axis=-1)
    scored = vis & ev

    k = np.nonzero(vis.sum(0) >= 2)[0]
    vk, pk, ek, dk, qk, sk = vis[:, k], pvis[:, k], ev[:, k], err[:, k], q[k], scored[:, k]
    K = thr.size
    if k.size:
        n_sc = sk.sum(0).astype(np.float64)
        oa = ((pk == vk) & ek).sum(0) / ek.sum(0)
        within = dk[..., None] < thr
        pw = (within & sk[..., None]).sum(0) / n_sc[:, None]
        tp = (within & (pk & sk)[..., None]).sum(0)
        fp = (((~within) & pk[..., None]) | (pk & ~vk)[..., None]) & ek[..., None]
        jac = tp / (n_sc[:, None] + fp.sum(0))
        failed = (dk > survival_m) & vk
        first_fail = np.where(failed.any(0), np.argmax(failed, axis=0), T)
        surv = (first_fail - qk) / (T - qk)
        moved = _path_length(g[:, k], vk)
    else:
        pw = jac = np.zeros((0, K))
        oa = surv = moved = np.zeros(0)
    splits = {"any": np.ones(k.size, dtype=bool)}
    if static_m is not None:
        splits["static"] = moved < float(static_m)
    if dynamic_m is not None:
        splits["dynamic"] = moved > float(dynamic_m)
    return {"thresholds": thr, "kept": k, "pts_within": pw, "jaccard": jac, "occlusion_accuracy": oa,
            "survival": surv, "path_length": moved, "splits": splits, "err": err, "scored": scored}


def track3d_metrics(gt, gt_vis, pred, pred_vis, query_t, thresholds_m: Sequence[float], survival_m: float,
                    static_m: Optional[float] = 0.01, dynamic_m: Optional[float] = 0.10) -> dict:
    """3D point-tracking metrics of one clip, identical to MVTracker's ``evaluate_predictions`` (see module doc).

    Args:
        gt:        (T, N, 3) float, GT world positions in metres (NaN = unknown; treated as not visible).
        gt_vis:    (T, N) bool, GT visibility (e.g. visible in at least one view). Numeric: occluded iff < 0.5.
        pred:      (T, N, 3) float, predicted positions in metres.
        pred_vis:  (T, N) bool predicted visibility, or None (= always visible, e.g. reference rows).
                   Numeric scores: occluded iff score < 0.5.
        query_t:   (N,) int query frame of every track (one int = same for all).
        thresholds_m: distance ladder in metres, e.g. (0.01, 0.02, 0.05, 0.10, 0.20). Strict ``<``.
        survival_m: a track fails at its first GT-visible frame (t >= query_t) with error > survival_m.
        static_m / dynamic_m: path-length limits (metres) of the static (L < static_m) and dynamic (L > dynamic_m)
                   splits; None disables that split (its keys are then absent).

    Returns:
        dict of Python floats/ints. For each split in ("any", "static", "dynamic"):
            "{split}/delta_avg"        percent, mean over the ladder of the share of scored frames within d
            "{split}/delta@{d}"        percent, one per threshold (``d = threshold_key(thr)``, metres)
            "{split}/jaccard@{d}"      percent, one per threshold
            "{split}/aj"               percent, average Jaccard
            "{split}/oa"               percent, occlusion accuracy
            "{split}/survival"         percent
            "{split}/ade", "{split}/mte"   metres, mean / median error over the split's scored entries (pooled)
            "{split}/n_tracks"         number of tracks in the split (after dropping tracks with < 2 visible frames)
        Empty splits have n_tracks = 0 and NaN values. Over all scored entries of all tracks (no 2-frame rule):
            "ade_all", "mte_all"       metres, mean / median error
            "fde"                      metres, mean over tracks of the error at their last scored frame
            "n_scored"                 number of scored (frame, track) entries
        "Scored" = GT-visible frames with t > query_t. ADE/MTE/FDE are non-finite if a scored prediction is.
    """
    r = _track3d_per_track(gt, gt_vis, pred, pred_vis, query_t, thresholds_m, survival_m, static_m, dynamic_m)
    keys = [threshold_key(v) for v in r["thresholds"]]
    pw, jac, oa, surv = r["pts_within"], r["jaccard"], r["occlusion_accuracy"], r["survival"]
    err, scored = r["err"], r["scored"]
    dk, sk = err[:, r["kept"]], scored[:, r["kept"]]

    out = {}
    for name, m in r["splits"].items():
        n = int(m.sum())
        if n == 0:
            out[f"{name}/delta_avg"] = _NAN
            out.update({f"{name}/delta@{key}": _NAN for key in keys})
            out.update({f"{name}/jaccard@{key}": _NAN for key in keys})
            out.update({f"{name}/{x}": _NAN for x in ("aj", "oa", "survival", "ade", "mte")})
        else:
            out[f"{name}/delta_avg"] = 100.0 * float(np.mean(pw[m].mean(-1)))
            for i, key in enumerate(keys):
                out[f"{name}/delta@{key}"] = 100.0 * float(np.mean(pw[m, i]))
            for i, key in enumerate(keys):
                out[f"{name}/jaccard@{key}"] = 100.0 * float(np.mean(jac[m, i]))
            out[f"{name}/aj"] = 100.0 * float(np.mean(jac[m].mean(-1)))
            out[f"{name}/oa"] = 100.0 * float(np.mean(oa[m]))
            out[f"{name}/survival"] = 100.0 * float(np.mean(surv[m]))
            e = dk[:, m][sk[:, m]]
            out[f"{name}/ade"] = _mean(e)
            out[f"{name}/mte"] = _median(e)
        out[f"{name}/n_tracks"] = n

    e_all = err[scored]
    out["ade_all"] = _mean(e_all)
    out["mte_all"] = _median(e_all)
    out["fde"] = _mean(_final_errors(err, scored))
    out["n_scored"] = int(scored.sum())
    return out


def track2d_metrics(gt_uv, gt_vis, pred_uv, pred_vis, query_t, image_size, thresholds_px=TAPVID_THRESHOLDS_PX,
                    normalize_to=(256, 256), query_mode: str = "first") -> dict:
    """TAP-Vid metrics of one video (one view), ported from tapnet's ``compute_tapvid_metrics``.

    Reference: C. Doersch et al., "TAP-Vid: A Benchmark for Tracking Any Point in a Video", NeurIPS 2022 Datasets
    and Benchmarks (arXiv:2211.03726); code: github.com/google-deepmind/tapnet, tapnet/tapvid/evaluation_datasets.py
    (``compute_tapvid_metrics``, Apache-2.0).

    Args:
        gt_uv:    (T, N, 2) float GT pixel coordinates (x = u right, y = v down) on an image of ``image_size``.
                  Non-finite entries are treated as occluded.
        gt_vis:   (T, N) bool GT visibility (numeric: occluded iff < 0.5).
        pred_uv:  (T, N, 2) float predicted pixel coordinates on the same image.
        pred_vis: (T, N) bool predicted visibility, or None (= always visible). Numeric: occluded iff < 0.5.
        query_t:  (N,) int query frame of every track (TAP-Vid samples queries where the point is visible).
        image_size: (W, H) of the image the coordinates refer to.
        thresholds_px: pixel thresholds applied after rescaling (TAP-Vid: 1, 2, 4, 8, 16). Strict ``<``.
        normalize_to: (W', H') the coordinates are rescaled to before thresholding: x *= W'/W, y *= H'/H
                  (TAP-Vid evaluates on 256x256 frames). None = keep the native pixels.
        query_mode: "first" (frames after the query frame are evaluated; earlier frames are assumed occluded and
                  ignored) or "strided" (every frame except the query frame is evaluated).

    Pooled over all tracks and evaluated frames of the video (the per-video numbers of tapnet; a benchmark score is
    the plain mean over videos):
        occlusion_accuracy = #(eval & pred_vis == gt_vis) / #eval
        pts_within_x       = #(eval & gt_vis & dist < x) / #(eval & gt_vis)
        jaccard_x          = TP / (#(eval & gt_vis) + FP),  TP = #(eval & gt_vis & pred_vis & dist < x),
                             FP = #(eval & pred_vis & (~gt_vis | dist >= x))
        average_jaccard, average_pts_within_thresh = means over the thresholds.

    Returns:
        dict with "occlusion_accuracy", "pts_within_{x}", "jaccard_{x}" (``x = threshold_key(thr)``),
        "average_jaccard", "average_pts_within_thresh" in PERCENT (tapnet returns fractions; the paper reports x100),
        plus "n_tracks" (N) and "n_visible" (#(eval & gt_vis), the denominator of pts_within). Metrics with an
        empty denominator are NaN.
    """
    g_in = _positions(gt_uv, "gt_uv", 2)
    p = _positions(pred_uv, "pred_uv", 2)
    if p.shape != g_in.shape:
        raise ValueError(f"pred_uv {p.shape} and gt_uv {g_in.shape} differ in shape")
    T, N = g_in.shape[:2]
    q = _query_frames(query_t, N, T)
    thr = _thresholds(thresholds_px, "thresholds_px")
    W, H = (float(v) for v in image_size)
    if W <= 0 or H <= 0:
        raise ValueError(f"image_size must be positive (W, H), got {image_size!r}")
    scale = np.ones(2) if normalize_to is None else np.array([float(normalize_to[0]) / W, float(normalize_to[1]) / H])

    finite = np.isfinite(g_in).all(-1)
    vis = _visibility(gt_vis, (T, N), "gt_vis") & finite
    pvis = _visibility(pred_vis, (T, N), "pred_vis")
    tt = np.arange(T)[:, None]
    if query_mode == "first":
        ev = tt > q[None, :]
    elif query_mode == "strided":
        ev = tt != q[None, :]
    else:
        raise ValueError(f"query_mode must be 'first' or 'strided', got {query_mode!r}")

    g = np.where(finite[..., None], g_in, 0.0) * scale
    sq = np.sum(np.square(p * scale - g), axis=-1)

    def ratio(a, b):
        return 100.0 * float(a) / float(b) if b > 0 else _NAN

    n_eval = int(ev.sum())
    n_vis = int((vis & ev).sum())
    out = {"occlusion_accuracy": ratio(((pvis == vis) & ev).sum(), n_eval)}
    pw, jac = [], []
    for t_px in thr:
        within = sq < np.square(t_px)
        key = threshold_key(t_px)
        pw.append(ratio((within & vis & ev).sum(), n_vis))
        tp = (within & vis & pvis & ev).sum()
        fp = ((((~vis) & pvis) | ((~within) & pvis)) & ev).sum()
        jac.append(ratio(tp, n_vis + fp))
        out[f"pts_within_{key}"] = pw[-1]
        out[f"jaccard_{key}"] = jac[-1]
    out["average_jaccard"] = float(np.mean(jac))
    out["average_pts_within_thresh"] = float(np.mean(pw))
    out["n_tracks"] = int(N)
    out["n_visible"] = n_vis
    return out


def moving_delta(gt, gt_vis, pred, query_t, thresholds_m: Sequence[float], min_disp: Optional[float] = None) -> dict:
    """delta on the moving entries (module docstring), POOLED over the entries of all tracks.

    gt, pred (T, N, 3) metres; gt_vis (T, N) bool (non-finite GT = not visible); query_t (N,) int or one int;
    min_disp: GT displacement from the query-frame position (m) that makes an entry "moving"; None = the largest
    threshold, so that the frozen prediction (error = displacement) is outside every threshold and scores 0.
    Entries: t > query_t, GT visible, GT known at the query frame and displacement >= min_disp. Non-finite predictions
    are failures. Returns {'moving/delta_avg', 'moving/delta@<d>' (percent; NaN without entries), 'moving/n_entries',
    'moving/n_tracks'}."""
    g = _positions(gt, "gt", 3)
    p = _positions(pred, "pred", 3)
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} differ in shape")
    T, N = g.shape[:2]
    vis = _visibility(gt_vis, (T, N), "gt_vis") & np.isfinite(g).all(-1)
    q = _query_frames(query_t, N, T)
    thr = _thresholds(thresholds_m, "thresholds_m")
    md = float(thr.max()) if min_disp is None else float(min_disp)
    g0 = g[q, np.arange(N)]
    disp = np.linalg.norm(g - g0[None], axis=-1)
    t = np.arange(T)[:, None]
    with np.errstate(invalid="ignore"):
        sel = vis & (t > q[None]) & np.isfinite(disp) & (disp >= md)
    err = np.linalg.norm(p - g, axis=-1)
    err = np.where(np.isfinite(err), err, np.inf)
    out = {"moving/n_entries": int(sel.sum()), "moving/n_tracks": int(sel.any(0).sum())}
    if not sel.any():
        out.update({"moving/delta_avg": _NAN, **{f"moving/delta@{threshold_key(d)}": _NAN for d in thr}})
        return out
    e = err[sel]
    vals = [float(np.mean(e < d) * 100.0) for d in thr]
    out.update({f"moving/delta@{threshold_key(d)}": v for d, v in zip(thr, vals)})
    out["moving/delta_avg"] = float(np.mean(vals))
    return out
