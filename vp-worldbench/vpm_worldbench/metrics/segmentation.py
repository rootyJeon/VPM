"""Video segmentation metrics (task `segmentation`): DAVIS region similarity J and boundary F-measure F, J&F, and
sparse point-label precision / recall / IoU.

    j = jaccard(pred, gt)                     # (T,) IoU per frame
    f = boundary_f(pred, gt)                  # (T,) boundary F-measure per frame
    s = j_and_f(pred, gt, frames=range(1, T)) # {"J", "F", "J&F", recall / decay, ...} in percent
    p = point_label_metrics(pred, uv, inside) # masks scored at labelled points (e.g. GT track points)

Conventions
- Masks are (T, H, W) videos or (H, W) frames on the same grid. bool is used as is, integer masks as `!= 0`, float
  masks as `> 0.5` (probabilities). Label maps (int, 0 = background) go to `j_and_f_objects`.
- `void` (optional, bool, shape of gt or (H, W) for every frame) marks pixels that are ignored (DAVIS "void").
- Per-frame helpers (`jaccard`, `boundary_f`, `davis_statistics`) return fractions in [0, 1] like the DAVIS code;
  summary dicts (`j_and_f`, `j_and_f_objects`, `point_label_metrics`) return percent (0-100).
- No evidence, no score: a frame without any non-void pixel (all void, or H * W = 0) has J = F = NaN and is left out
  of the sequence statistics, and point-label ratios are NaN when no labelled point is scored. Nothing is ever
  scored as perfect for lack of data.

Definitions (identical to the official DAVIS 2017 evaluation code, davis2017-evaluation `metrics.py` / `utils.py`;
Perazzi et al., CVPR 2016; Pont-Tuset et al., arXiv:1704.00675, on every frame that has a non-void pixel; DAVIS
scores a frame without one as J = F = 1, a case its data never contains):
- J = |P & G| / |P | G| over non-void pixels; 1 when the union is empty (object correctly absent).
- F: one-pixel boundary maps of P and G (`seg2bmap`, the Berkeley boundary benchmark's half-pixel-shifted maps),
  each dilated by a disk of radius r = ceil(bound_th * sqrt(H^2 + W^2)) (bound_th = 0.008; bound_th >= 1 is an
  absolute radius in pixels); precision = share of predicted boundary pixels within the dilated GT boundary,
  recall = share of GT boundary pixels within the dilated predicted boundary, F = 2PR / (P + R) (0 if P + R = 0);
  empty boundaries: none predicted / some GT -> P = 1, R = 0; some predicted / none GT -> P = 0, R = 1; both empty
  -> P = R = 1.
- Per sequence (`davis_statistics`): mean (nanmean over frames), recall (share of frames with value > 0.5) and decay
  (mean of the first quarter minus mean of the last quarter of the frames, DAVIS binning). DAVIS's code casts the
  bin edges to uint8, which breaks sequences longer than 255 frames; int64 is used here (identical below 256).
- J&F = (J + F) / 2. For several objects (`j_and_f_objects`) every statistic is averaged over objects, as DAVIS
  averages its per-object rows. The semi-supervised DAVIS protocol scores frames 1 .. T-2 (the prompt frame 0 and the
  last frame are excluded); pass `frames` to choose the scored frames (e.g. exclude the prompt frame t_q).

Point labels: the predicted mask is read at the pixel of each labelled point (OpenCV pixel centres: pixel
floor(u + 0.5), floor(v + 0.5); points outside the grid, not `valid`, or with a NaN label are not scored) and
compared with its label (True = the point belongs to the object): precision = TP / (TP + FP), recall = TP / (TP + FN),
IoU = TP / (TP + FP + FN) (a point-sampled estimate of the mask IoU), F1, accuracy; counts are pooled over all frames.
Undefined ratios are NaN: precision without predicted positives, recall without labelled positives, everything when
no point is scored. IoU is 100 when points are scored but none is positive in either (all true negatives: the
object is correctly absent, DAVIS's empty-union rule), and NaN when no point is scored.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import numpy as np

__all__ = ["jaccard", "boundary_f", "seg2bmap", "boundary_radius", "davis_statistics", "j_and_f", "j_and_f_objects",
           "sample_mask", "point_label_metrics", "BOUND_TH"]

BOUND_TH = 0.008
_NAN = float("nan")


def _as_mask(x, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    if np.issubdtype(a.dtype, np.floating):
        with np.errstate(invalid="ignore"):
            return a > 0.5
    if np.issubdtype(a.dtype, np.integer):
        return a != 0
    raise TypeError(f"{name}: unsupported mask dtype {a.dtype}")


def _pair(pred, gt, void):
    P, G = _as_mask(pred, "pred"), _as_mask(gt, "gt")
    if P.shape != G.shape:
        raise ValueError(f"pred {P.shape} and gt {G.shape} must have the same shape")
    if P.ndim not in (2, 3):
        raise ValueError(f"masks must be (H, W) or (T, H, W), got {P.shape}")
    if void is None:
        V = np.zeros(G.shape[-2:], bool)
    else:
        V = _as_mask(void, "void")
        if V.shape not in (G.shape, G.shape[-2:]):
            raise ValueError(f"void {V.shape} must be {G.shape} or {G.shape[-2:]}")
    return P, G, V


def _frame_void(V: np.ndarray, t: int) -> np.ndarray:
    return V if V.ndim == 2 else V[t]


def _scorable(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """bool per frame ((T,) or scalar): the frame has at least one non-void pixel."""
    n_pix = P.shape[-2] * P.shape[-1]
    n_void = np.sum(np.broadcast_to(V, P.shape), axis=(-2, -1))
    return n_void < n_pix


def jaccard(pred, gt, void=None):
    """Region similarity J (IoU) per frame: float for (H, W) inputs, (T,) float64 for (T, H, W). Fraction in [0, 1];
    1 when prediction and GT are both empty over the non-void pixels (DAVIS); NaN for a frame without any non-void
    pixel (no evidence)."""
    P, G, V = _pair(pred, gt, void)
    keep = ~V
    inter = np.sum(P & G & keep, axis=(-2, -1))
    union = np.sum((P | G) & keep, axis=(-2, -1))
    j = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    j = np.where(_scorable(P, V), j, np.nan)
    return float(j) if P.ndim == 2 else j.astype(np.float64)


def seg2bmap(seg) -> np.ndarray:
    """One-pixel boundary map (H, W) bool of a binary mask, offset by half a pixel towards the origin (Berkeley /
    DAVIS `_seg2bmap`, same-size case): a pixel is a boundary pixel if it differs from its right, lower or lower-right
    neighbour; the last row / column compare with their right / lower neighbour only; the corner pixel is never one.
    A zero-size mask gives an empty map."""
    seg = _as_mask(seg, "seg")
    if seg.ndim != 2:
        raise ValueError(f"seg2bmap expects (H, W), got {seg.shape}")
    if seg.size == 0:
        return np.zeros(seg.shape, bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def boundary_radius(shape, bound_th: float = BOUND_TH) -> int:
    """Boundary tolerance in pixels for an (H, W) grid: bound_th if bound_th >= 1 (taken as an integer radius) else
    ceil(bound_th * sqrt(H^2 + W^2)): 8 px for DAVIS 480x854, 6 px for 480x480, 7 px for 480x640,
    8 px for 480x848, 11 px for 768x1024."""
    if bound_th >= 1:
        return int(bound_th)
    H, W = int(shape[-2]), int(shape[-1])
    return int(math.ceil(bound_th * math.sqrt(float(H * H + W * W))))


def _disk(r: int) -> np.ndarray:
    """(2r+1, 2r+1) uint8 disk x^2 + y^2 <= r^2 (= skimage.morphology.disk(r))."""
    L = np.arange(-r, r + 1)
    X, Y = np.meshgrid(L, L)
    return ((X ** 2 + Y ** 2) <= r ** 2).astype(np.uint8)


def _dilate(b: np.ndarray, r: int, backend: str = "auto") -> np.ndarray:
    if r <= 0:
        return b.copy()
    k = _disk(r)
    if backend in ("auto", "cv2"):
        try:
            import cv2
            return cv2.dilate(b.astype(np.uint8), k).astype(bool)
        except ImportError:
            if backend == "cv2":
                raise ImportError("cv2 is required: pip install opencv-python-headless (a core dependency)")
    from scipy.ndimage import binary_dilation
    return binary_dilation(b, structure=k.astype(bool))


def _f_measure(fg: np.ndarray, gt: np.ndarray, void: np.ndarray, r: int) -> float:
    """DAVIS f_measure of one frame with the boundary tolerance r (pixels); NaN without any non-void pixel."""
    keep = ~void
    if not keep.any():
        return _NAN
    fg_b = seg2bmap(fg & keep)
    gt_b = seg2bmap(gt & keep)
    fg_dil = _dilate(fg_b, r)
    gt_dil = _dilate(gt_b, r)
    n_fg, n_gt = int(fg_b.sum()), int(gt_b.sum())
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float(np.sum(fg_b & gt_dil)) / n_fg
        recall = float(np.sum(gt_b & fg_dil)) / n_gt
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def boundary_f(pred, gt, void=None, bound_th: float = BOUND_TH):
    """Boundary F-measure per frame (DAVIS): float for (H, W), (T,) float64 for (T, H, W); fraction in [0, 1]; NaN for
    a frame without any non-void pixel."""
    P, G, V = _pair(pred, gt, void)
    r = boundary_radius(P.shape, bound_th)
    if P.ndim == 2:
        return _f_measure(P, G, V, r)
    return np.array([_f_measure(P[t], G[t], _frame_void(V, t), r) for t in range(len(P))], np.float64)


def davis_statistics(per_frame) -> dict:
    """{"mean", "recall", "decay"} of per-frame values (fractions), as DAVIS `db_statistics`: nanmean; share of frames
    with value > 0.5 (NaN frames count as misses); first-quarter mean minus last-quarter mean (4 bins with edges
    round(linspace(1, n, 5) + 1e-10) - 1, each bin including its right edge)."""
    v = np.asarray(per_frame, np.float64).ravel()
    if v.size == 0:
        return {"mean": _NAN, "recall": _NAN, "decay": _NAN}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = float(np.nanmean(v))
        with np.errstate(invalid="ignore"):
            recall = float(np.mean(v > 0.5))
        ids = (np.round(np.linspace(1, len(v), 5) + 1e-10) - 1).astype(np.int64)
        bins = [v[ids[i]:ids[i + 1] + 1] for i in range(4)]
        decay = float(np.nanmean(bins[0]) - np.nanmean(bins[3]))
    return {"mean": mean, "recall": recall, "decay": decay}


def _frame_index(frames, n: int) -> np.ndarray:
    """Scored frame indices from `frames` (None = all n, a bool mask of length n, or indices in [0, n))."""
    if frames is None:
        return np.arange(n)
    f = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames)
    if f.dtype == bool:
        if f.shape != (n,):
            raise ValueError(f"a bool frame mask must have shape ({n},), got {f.shape}")
        return np.nonzero(f)[0]
    if f.size and not np.issubdtype(f.dtype, np.integer):
        raise TypeError(f"frames must be integer indices or a bool mask, got dtype {f.dtype}")
    f = f.astype(np.int64).ravel()
    if f.size and (f.min() < 0 or f.max() >= n):
        raise ValueError(f"frame indices must be in [0, {n}) (negative indices are not accepted), got "
                         f"{f[(f < 0) | (f >= n)][:5].tolist()}")
    return f


def _select_frames(P, G, V, idx: np.ndarray):
    if P.ndim == 2:
        P, G = P[None], G[None]
    return P[idx], G[idx], (V if V.ndim == 2 else V[idx])


def j_and_f(pred, gt, void=None, bound_th: float = BOUND_TH, frames: Optional[Sequence[int]] = None,
            per_frame: bool = False) -> dict:
    """J&F of one object over a video (sequence statistics, optionally also per frame).

    Args:
        pred, gt: binary masks (T, H, W) or (H, W).
        void: optional ignore mask ((T, H, W) or (H, W)).
        bound_th: boundary tolerance (see `boundary_radius`).
        frames: indices in [0, T) (or a (T,) bool mask) of the frames to score; None = all.
        per_frame: also return the per-frame values.

    Frames without any non-void pixel carry no evidence: they are left out of the statistics (and counted in
    "n_frames_skipped"). Raises ValueError when no frame is left to score.

    Returns:
        {"J", "J_recall", "J_decay", "F", "F_recall", "F_decay", "J&F" [percent], "n_frames" (scored frames),
        "n_frames_skipped"}; with per_frame: "per_frame": {"frame": the selected frame indices, "J", "F", "J&F": lists
        [percent], NaN for skipped frames}.
    """
    P, G, V = _pair(pred, gt, void)
    idx = _frame_index(frames, 1 if P.ndim == 2 else len(P))
    P, G, V = _select_frames(P, G, V, idx)
    if len(P) == 0:
        raise ValueError("no frames to score")
    J = jaccard(P, G, V)
    F = boundary_f(P, G, V, bound_th)
    scored = ~np.isnan(J)
    if not scored.any():
        raise ValueError("no frame to score: every selected frame is void (or has no pixels)")
    sj, sf = davis_statistics(J[scored]), davis_statistics(F[scored])
    out = {"J": 100.0 * sj["mean"], "J_recall": 100.0 * sj["recall"], "J_decay": 100.0 * sj["decay"],
           "F": 100.0 * sf["mean"], "F_recall": 100.0 * sf["recall"], "F_decay": 100.0 * sf["decay"]}
    out["J&F"] = (out["J"] + out["F"]) / 2.0
    out["n_frames"] = int(scored.sum())
    out["n_frames_skipped"] = int((~scored).sum())
    if per_frame:
        out["per_frame"] = {"frame": [int(i) for i in idx], "J": (100.0 * J).tolist(), "F": (100.0 * F).tolist(),
                            "J&F": (50.0 * (J + F)).tolist()}
    return out


def j_and_f_objects(pred_labels, gt_labels, void=None, object_ids=None, bound_th: float = BOUND_TH,
                    frames: Optional[Sequence[int]] = None) -> dict:
    """Multi-object J&F (DAVIS): label maps (T, H, W) int (0 = background); every object id is scored as a binary
    mask (an object absent from the prediction scores an empty mask) and the statistics are averaged over objects.

    Args: object_ids: ids to score (default: the non-zero ids present in gt_labels); void, bound_th, frames as
        `j_and_f`.
    Returns: {"J", "J_recall", "J_decay", "F", "F_recall", "F_decay", "J&F" [percent], "n_objects",
              "per_object": {id: j_and_f dict}}.
    """
    Pl, Gl = np.asarray(pred_labels), np.asarray(gt_labels)
    if Pl.shape != Gl.shape:
        raise ValueError(f"pred_labels {Pl.shape} and gt_labels {Gl.shape} must have the same shape")
    ids = sorted(int(i) for i in np.unique(Gl) if i != 0) if object_ids is None else [int(i) for i in object_ids]
    if not ids:
        raise ValueError("no object ids to score")
    per = {i: j_and_f(Pl == i, Gl == i, void, bound_th, frames) for i in ids}
    keys = ("J", "J_recall", "J_decay", "F", "F_recall", "F_decay")
    out = {k: float(np.mean([per[i][k] for i in ids])) for k in keys}
    out["J&F"] = (out["J"] + out["F"]) / 2.0
    out["n_objects"] = len(ids)
    out["per_object"] = per
    return out


def sample_mask(mask, uv):
    """Read a binary mask (H, W) at pixel positions uv (..., 2) (OpenCV centres: pixel floor(u + 0.5), floor(v + 0.5)).
    Returns (value (...,) bool, inside (...,) bool); value is False where the point is outside the grid / non-finite."""
    M = _as_mask(mask, "mask")
    if M.ndim != 2:
        raise ValueError(f"mask must be (H, W), got {M.shape}")
    H, W = M.shape
    uv = np.asarray(uv, np.float64)
    u, v = uv[..., 0], uv[..., 1]
    with np.errstate(invalid="ignore"):
        ins = np.isfinite(u) & np.isfinite(v) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    val = np.zeros(u.shape, bool)
    iu = np.clip(np.floor(u[ins] + 0.5).astype(np.int64), 0, W - 1)
    iv = np.clip(np.floor(v[ins] + 0.5).astype(np.int64), 0, H - 1)
    val[ins] = M[iv, iu]
    return val, ins


def _ratio(num: int, den: int) -> float:
    return 100.0 * num / den if den > 0 else _NAN


def _point_iou(tp: int, fp: int, fn: int, tn: int) -> float:
    """Point IoU in percent: NaN without any scored point; 100 when every scored point is a true negative."""
    if tp + fp + fn > 0:
        return 100.0 * tp / (tp + fp + fn)
    return 100.0 if tn > 0 else _NAN


def _point_labels(labels):
    """(labels bool, labelled bool): bool as is; numeric labels must be 0 / 1, NaN = no label at that point."""
    a = np.asarray(labels)
    if a.dtype == bool:
        return a, np.ones(a.shape, bool)
    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        raise TypeError(f"labels must be bool or numeric 0 / 1 (NaN = unlabelled), got dtype {a.dtype}")
    has = ~np.isnan(a) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape, bool)
    bad = has & (a != 0) & (a != 1)
    if bad.any():
        raise ValueError(f"numeric labels must be 0 or 1 (NaN = unlabelled), got {np.unique(a[bad])[:5].tolist()}")
    return has & (a == 1), has


def point_label_metrics(pred_mask, uv, labels, valid=None, per_frame: bool = False) -> dict:
    """Score a predicted mask against labelled points.

    Args:
        pred_mask: (T, H, W) or (H, W) predicted masks.
        uv: (T, N, 2) or (N, 2) pixel coordinates of the points on the mask grid (e.g. `Sample.tracks2d`).
        labels: (T, N) or (N,): True / 1 = the point belongs to the object at that frame, False / 0 = it does not;
            bool, or numeric with NaN = no label (that point is not scored, like valid=False). Other values raise.
        valid: optional (T, N) or (N,) bool: points that carry a label at that frame (e.g. GT-visible in this view).
        per_frame: also return per-frame counts and IoU.

    Returns:
        {"precision", "recall", "iou", "f1", "accuracy" [percent], "tp", "fp", "fn", "tn", "n_points" (scored),
         "n_unscored" (valid, labelled points outside the grid or with non-finite uv)}; with per_frame: "per_frame":
        {"tp", "fp", "fn", "tn", "iou"} lists of length T. Ratios follow the module docstring: NaN when undefined,
        all NaN when n_points == 0 (per frame: iou NaN for a frame without a scored point).
    """
    M = _as_mask(pred_mask, "pred_mask")
    uv = np.asarray(uv, np.float64)
    lab, labelled = _point_labels(labels)
    if valid is not None:
        valid = np.asarray(valid)
        if np.issubdtype(valid.dtype, np.integer) and np.isin(valid, (0, 1)).all():
            valid = valid != 0
        elif valid.dtype != bool:
            raise TypeError(f"valid must be bool (or integer 0 / 1), got dtype {valid.dtype}")
    if M.ndim == 2:
        M, uv, lab, labelled = M[None], uv[None], lab[None], labelled[None]
        valid = None if valid is None else valid[None]
    if M.ndim != 3 or uv.ndim != 3 or uv.shape[-1] != 2:
        raise ValueError(f"expected masks (T, H, W) and uv (T, N, 2); got {M.shape} and {uv.shape}")
    T = M.shape[0]
    if uv.shape[0] != T or lab.shape != uv.shape[:2]:
        raise ValueError(f"uv {uv.shape} / labels {lab.shape} do not match {T} frames")
    if valid is not None and valid.shape != lab.shape:
        raise ValueError(f"valid {valid.shape} must match labels {lab.shape}")
    val = labelled if valid is None else valid & labelled
    rows = []
    n_out = 0
    for t in range(T):
        pv, ins = sample_mask(M[t], uv[t])
        sc = ins & val[t]
        n_out += int((val[t] & ~ins).sum())
        p, g = pv[sc], lab[t][sc]
        rows.append((int(np.sum(p & g)), int(np.sum(p & ~g)), int(np.sum(~p & g)), int(np.sum(~p & ~g))))
    tp, fp, fn, tn = (int(sum(r[k] for r in rows)) for k in range(4))
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    if np.isnan(prec) or np.isnan(rec):
        f1 = _NAN
    else:
        f1 = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    n = tp + fp + fn + tn
    out = {"precision": prec, "recall": rec, "iou": _point_iou(tp, fp, fn, tn), "f1": f1,
           "accuracy": _ratio(tp + tn, n), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_points": n,
           "n_unscored": n_out}
    if per_frame:
        out["per_frame"] = {
            "tp": [r[0] for r in rows], "fp": [r[1] for r in rows], "fn": [r[2] for r in rows],
            "tn": [r[3] for r in rows], "iou": [_point_iou(*r) for r in rows]}
    return out
