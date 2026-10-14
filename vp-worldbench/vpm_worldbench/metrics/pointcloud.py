"""Point-cloud metrics: accuracy, completeness, Chamfer distance and F-score between a predicted and a GT point set
(task `recon4d`: per-frame point maps / fused clouds against GT depth lifted to 3D or the Real-General
reconstruction).

    m = pointcloud_metrics(pred_xyz, gt_xyz, thresholds=(0.05, 0.10))
    m = pointcloud_metrics(pred_map, gt_map, align="sim3")      # point maps in pixel correspondence: align first

Conventions
- Points are world coordinates in metres, any (..., 3) array (flattened); rows with a non-finite coordinate are
  ignored (so NaN marks "no point", e.g. invalid GT depth in a point map).
- Distances are Euclidean (not squared), nearest neighbours by a KD-tree (scipy.spatial.cKDTree, exact).
- Summary dicts give shares (precision, recall, F-score) in percent (0-100) and distances in metres.

Definitions (DTU, Aanaes et al., IJCV 2016; Tanks and Temples, Knapitsch et al., SIGGRAPH 2017; identical to
Depth-Anything-3's `evaluate_3d_reconstruction` acc / comp / overall / precision / recall / fscore), with
d(x, S) = min_{y in S} |x - y|:

    accuracy     acc_i  = d(pred_i, GT)       acc_mean / acc_median over predicted points
    completeness comp_j = d(gt_j, Pred)       comp_mean / comp_median over GT points
    chamfer_mean   = (acc_mean + comp_mean) / 2        (DTU "overall")
    chamfer_median = (acc_median + comp_median) / 2
    chamfer_sum    = acc_mean + comp_mean              (the "CD = mean_a min_b + mean_b min_a" convention)
    precision@tau = 100 * mean(acc_i < tau)   recall@tau = 100 * mean(comp_j < tau)
    fscore@tau    = 2 P R / (P + R)  (0 when P + R = 0)

An empty prediction or GT, or a prediction that cannot be aligned, gives inf distances and 0 precision / recall /
F-score (a missing reconstruction is a failure, not a skipped sample). Alignment (`align`) needs point-to-point
correspondences (pred[i] <-> gt[i], e.g. point maps on the same pixel grid): Umeyama Sim(3) / SE(3), or a
least-squares scale about the origin (camera-frame point maps); it is estimated on rows finite in both, then applied
to every predicted point. For clouds without correspondences, align them beforehand (e.g. with the camera alignment
of `metrics.pose`).
`voxel_downsample` follows Open3D's `voxel_down_sample` (grid anchored at min_bound - voxel/2, mean of the points of
each voxel), returned in a deterministic order.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import umeyama

__all__ = ["as_points", "voxel_downsample", "nn_distance", "fit_similarity", "apply_similarity", "subsample",
           "pointcloud_metrics", "ALIGN_MODES"]

ALIGN_MODES = ("none", "scale", "se3", "sim3")
_INF = float("inf")


def _key(thr) -> str:
    return f"{float(thr):g}"


def _kdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError as e:
        raise ImportError("point-cloud metrics need scipy: pip install scipy (a core dependency of vpm-worldbench)") \
            from e
    return cKDTree


def as_points(x, finite: bool = True) -> np.ndarray:
    """(..., 3) -> (N, 3) float64; with finite=True rows containing NaN / inf are dropped."""
    a = np.asarray(x, np.float64)
    if a.ndim == 0 or a.shape[-1] != 3:
        raise ValueError(f"points must be (..., 3), got {a.shape}")
    a = a.reshape(-1, 3)
    return a[np.isfinite(a).all(1)] if finite else a


def _row_mask(mask, lead_shape) -> np.ndarray:
    """Flattened bool row mask; `mask` (bool, or integer read as != 0) must have exactly the leading shape of the
    points (no reshaping of a transposed (W, H) mask onto (H, W) point maps, no broadcasting)."""
    m = np.asarray(mask)
    if np.issubdtype(m.dtype, np.integer):
        m = m != 0
    elif m.dtype != bool:
        raise TypeError(f"mask must be bool (or integer), got dtype {m.dtype}")
    if m.shape != tuple(lead_shape):
        raise ValueError(f"mask {m.shape} must have the leading shape of the points {tuple(lead_shape)}")
    return m.reshape(-1)


def voxel_downsample(points, voxel_size: float, reduce: str = "mean", origin=None) -> np.ndarray:
    """Keep one point per occupied voxel.

    Args:
        points: (..., 3) metres (non-finite rows dropped).
        voxel_size: voxel edge in metres (<= 0: no downsampling, finite points returned).
        reduce: "mean" (centroid of the voxel's points, as Open3D), "first" (the voxel's first point in input order)
            or "center" (the voxel centre).
        origin: grid origin (3,); None = Open3D's choice min(points) - voxel_size / 2.

    Returns:
        (M, 3) float64, voxels in lexicographic order of their integer (x, y, z) index (deterministic).
    """
    if reduce not in ("mean", "first", "center"):
        raise ValueError(f"reduce={reduce!r} not in ('mean', 'first', 'center')")
    p = as_points(points)
    if voxel_size is None or voxel_size <= 0 or len(p) == 0:
        return p
    o = p.min(0) - voxel_size / 2.0 if origin is None else np.asarray(origin, np.float64).reshape(3)
    k = np.floor((p - o) / voxel_size).astype(np.int64)
    kmin = k.min(0)
    k -= kmin
    dims = k.max(0) + 1
    if float(np.prod(dims.astype(np.float64))) < 2.0 ** 62:
        lin = np.ravel_multi_index(tuple(k.T), tuple(int(d) for d in dims))
        uniq, inv = np.unique(lin, return_inverse=True)
        cells = np.stack(np.unravel_index(uniq, tuple(int(d) for d in dims)), -1)
    else:
        cells, inv = np.unique(k, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    m = len(cells)
    if reduce == "mean":
        cnt = np.bincount(inv, minlength=m).astype(np.float64)
        return np.stack([np.bincount(inv, weights=p[:, d], minlength=m) for d in range(3)], -1) / cnt[:, None]
    if reduce == "first":
        first = np.full(m, len(p), np.int64)
        np.minimum.at(first, inv, np.arange(len(p)))
        return p[first]
    return o + (cells + kmin + 0.5) * voxel_size


def nn_distance(src, dst, workers: int = -1) -> np.ndarray:
    """For every point of src (..., 3) (flattened to N rows), the Euclidean distance to its nearest neighbour in dst
    (..., 3), metres. Returns (N,) float64: NaN for non-finite src rows, inf when dst has no finite row (non-finite
    dst rows are ignored). Exact KD-tree search (results do not depend on `workers`)."""
    src, dst = as_points(src, finite=False), as_points(dst)
    out = np.full(len(src), np.nan)
    ok = np.isfinite(src).all(1)
    if not ok.any():
        return out
    if len(dst) == 0:
        out[ok] = _INF
        return out
    d, _ = _kdtree()(dst).query(src[ok], k=1, workers=workers)
    out[ok] = d
    return out


def fit_similarity(src, dst, mode: str = "sim3", mask=None):
    """Transform (s, R, t) with dst ~ s R src + t from corresponding points src[i] <-> dst[i] ((..., 3), same shape).

    mode: "sim3" / "se3" (Umeyama, with / without scale; >= 3 points), "scale" (s = sum <src, dst> / sum |src|^2,
    R = I, t = 0; >= 1 point) or "none" (identity). Rows non-finite in either array, or False in `mask` (bool of
    exactly the leading shape src.shape[:-1]), are not used."""
    if mode not in ALIGN_MODES:
        raise ValueError(f"mode={mode!r} not in {ALIGN_MODES}")
    if mode == "none":
        return 1.0, np.eye(3), np.zeros(3)
    a, b = as_points(src, finite=False), as_points(dst, finite=False)
    if a.shape != b.shape:
        raise ValueError(f"alignment needs corresponding points: src {np.shape(src)} vs dst {np.shape(dst)}")
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if mask is not None:
        ok &= _row_mask(mask, np.shape(src)[:-1])
    a, b = a[ok], b[ok]
    if mode == "scale":
        den = float(np.sum(a * a))
        if len(a) == 0 or den <= 0:
            raise ValueError("scale alignment needs at least one non-zero corresponding point")
        return float(np.sum(a * b)) / den, np.eye(3), np.zeros(3)
    if len(a) < 3:
        raise ValueError(f"{mode} alignment needs >= 3 corresponding points, got {len(a)}")
    s, R, t = umeyama(a, b, with_scale=(mode == "sim3"))
    return float(s), R, t


def apply_similarity(points, s: float, R, t) -> np.ndarray:
    """s R x + t for points (..., 3) (shape kept, NaN rows stay NaN)."""
    p = np.asarray(points, np.float64)
    return s * p @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)


def subsample(points, max_points: Optional[int], seed: int = 0) -> np.ndarray:
    """At most max_points rows of (N, 3), drawn without replacement with np.random.default_rng(seed), input order
    kept (deterministic). None or N <= max_points: unchanged."""
    p = np.asarray(points)
    if max_points is None or len(p) <= max_points:
        return p
    idx = np.sort(np.random.default_rng(seed).choice(len(p), int(max_points), replace=False))
    return p[idx]


def pointcloud_metrics(pred, gt, thresholds: Sequence[float] = (0.05,), *, align: str = "none", mask=None,
                       voxel_size: Optional[float] = None, max_points: Optional[int] = None, seed: int = 0,
                       workers: int = -1) -> dict:
    """Accuracy / completeness / Chamfer / F-score of a predicted point set against GT.

    Args:
        pred, gt: (..., 3) world points in metres; NaN rows ignored. Leading shapes may differ unless `align` is not
            "none" or `mask` is given (then pred[i] <-> gt[i] correspond and shapes must match).
        thresholds: F-score distances tau in metres.
        align: "none", "scale", "se3" or "sim3" (see `fit_similarity`; estimated on corresponding rows that are
            finite in both and in `mask`, then applied to all predicted points).
        mask: optional bool of exactly the leading shape pred.shape[:-1]: rows used from both pred and gt (e.g.
            GT-valid pixels of point maps on the same grid).
        voxel_size: optional voxel downsampling of both sets (metres) after alignment (`voxel_downsample`).
        max_points: optional cap on each set after downsampling (`subsample` with `seed`).
        workers: KD-tree query threads (-1 = all; the result does not depend on it).

    Returns:
        {"acc_mean", "acc_median", "comp_mean", "comp_median", "chamfer_mean", "chamfer_median", "chamfer_sum" [m],
         "precision@<tau>", "recall@<tau>", "fscore@<tau>" [percent], "n_pred", "n_gt"} plus, when aligned,
        "align": {"mode", "failed", "scale", "R" (3x3 list), "t" (list)}. A prediction that cannot be aligned (fewer
        corresponding finite rows than the mode needs) is scored as a failure (inf distances, 0 percent) with
        "align": {"mode", "failed": True, "error"}.
    """
    if align not in ALIGN_MODES:
        raise ValueError(f"align={align!r} not in {ALIGN_MODES}")
    P = np.asarray(pred, np.float64)
    G = np.asarray(gt, np.float64)
    if P.shape[-1] != 3 or G.shape[-1] != 3:
        raise ValueError(f"pred {P.shape} and gt {G.shape} must be (..., 3)")
    if (align != "none" or mask is not None) and P.shape != G.shape:
        raise ValueError(f"align / mask need corresponding points: pred {P.shape} vs gt {G.shape}")
    m = None if mask is None else _row_mask(mask, P.shape[:-1])
    P, G = P.reshape(-1, 3), G.reshape(-1, 3)
    if m is not None:
        P, G = P[m], G[m]
    out = {}
    failed = False
    if align != "none":
        try:
            s, R, t = fit_similarity(P, G, align)
        except ValueError as e:
            failed = True
            out["align"] = {"mode": align, "failed": True, "error": str(e)}
        else:
            P = apply_similarity(P, s, R, t)
            out["align"] = {"mode": align, "failed": False, "scale": float(s), "R": np.asarray(R).tolist(),
                            "t": np.asarray(t).tolist()}
    P, G = as_points(P), as_points(G)
    if voxel_size and not failed:
        P, G = voxel_downsample(P, voxel_size), voxel_downsample(G, voxel_size)
    P, G = subsample(P, max_points, seed), subsample(G, max_points, seed)
    res = {"n_pred": int(len(P)), "n_gt": int(len(G))}
    if failed or len(P) == 0 or len(G) == 0:
        res.update(acc_mean=_INF, acc_median=_INF, comp_mean=_INF, comp_median=_INF, chamfer_mean=_INF,
                   chamfer_median=_INF, chamfer_sum=_INF)
        for tau in thresholds:
            res.update({f"precision@{_key(tau)}": 0.0, f"recall@{_key(tau)}": 0.0, f"fscore@{_key(tau)}": 0.0})
        res.update(out)
        return res
    acc = nn_distance(P, G, workers)
    comp = nn_distance(G, P, workers)
    res.update(acc_mean=float(acc.mean()), acc_median=float(np.median(acc)), comp_mean=float(comp.mean()),
               comp_median=float(np.median(comp)))
    res["chamfer_mean"] = (res["acc_mean"] + res["comp_mean"]) / 2.0
    res["chamfer_median"] = (res["acc_median"] + res["comp_median"]) / 2.0
    res["chamfer_sum"] = res["acc_mean"] + res["comp_mean"]
    for tau in thresholds:
        prec = 100.0 * float(np.mean(acc < tau))
        rec = 100.0 * float(np.mean(comp < tau))
        res[f"precision@{_key(tau)}"] = prec
        res[f"recall@{_key(tau)}"] = rec
        res[f"fscore@{_key(tau)}"] = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    res.update(out)
    return res
