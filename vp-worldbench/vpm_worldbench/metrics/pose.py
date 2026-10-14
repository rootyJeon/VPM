"""Camera pose metrics for trajectories and multi-view rigs (task `recon4d`, camera pose estimation).

    m = pose_metrics(pred_w2c, gt_w2c)          # ATE / RPE after Sim(3) alignment + RRA / RTA / AUC over camera pairs
    e = relative_pose_errors(pred_w2c, gt_w2c)  # per-pair rotation / translation-direction errors (degrees)

Conventions
- Poses are world-to-camera 4x4 matrices `w2c` (OpenCV camera axes: x right, y down, z forward), as stored in
  cameras.npz; (..., 3, 4) is accepted too. Any leading shape is flattened in C order into N poses, e.g. (V, T, 4, 4)
  = V views x T frames (pose index v * T + t). `pred` and `gt` must have the same shape. The predicted world frame and
  scale are arbitrary: every metric below is either invariant to them or is computed after an explicit alignment.
- Poses must be rigid: the 3x3 block of every finite pose must be a rotation (max |R^T R - I| <= ROT_TOL and
  det R > 0), else ValueError: a scaled, sheared or mirrored matrix would otherwise get meaningless (even perfect)
  scores. Project such predictions onto SO(3) first. The bottom row of a 4x4 input is ignored ([0, 0, 0, 1] is used).
- Camera centre c = -R^T t (metres in the GT world). Rotation errors are geodesic angles in degrees,
  angle(R) = atan2(|vee(R - R^T)| / 2, (tr R - 1) / 2) (accurate for small and large angles).
- A pose with a non-finite entry in its [R | t] block is *missing*: in pair metrics it counts as a failure
  (error = inf); in ATE / RPE it is excluded and counted.
- Summary dicts give shares (RRA, RTA, AUC) in percent (0-100); errors in metres / degrees.

Metrics
- Absolute trajectory error (ATE; Sturm et al., "A benchmark for the evaluation of RGB-D SLAM systems", IROS 2012):
  predicted camera centres are aligned to the GT centres with Umeyama (Sim(3) for scale-free predictions, SE(3) for
  metric ones, or no alignment); ATE = RMSE (also mean / median / max) of the centre distances after alignment. The
  orientation error of the aligned cameras is reported as ARE (absolute rotation error). Alignment needs >= 3
  non-collinear centres to fix the rotation; "degenerate" flags inputs where it is not unique (e.g. a 2-view rig, or
  a static or straight-line trajectory): ATE is still well defined, ARE is not.
- Relative pose error (RPE; same reference) along each trajectory with frame step `delta`:
  E_i = (Q_i^-1 Q_{i+delta})^-1 (P_i^-1 P_{i+delta}) with Q / P the GT / predicted camera-to-world poses (the
  predicted translations multiplied by the Sim(3) scale when align="sim3", as `evo -as` does); RPE_trans = |t(E_i)|
  [m], RPE_rot = angle(R(E_i)) [deg]. With align="sim3" and delta=1, `ate_rmse`, `rpe_trans_rmse` and
  `rpe_rot_rmse_deg` are the ATE / RPE-trans / RPE-rot triple of video-reconstruction papers (MonST3R's
  `eval_metrics`: evo APE / RPE with align=True, correct_scale=True, delta=1 frame, all pairs, RMSE).
- Pairwise relative pose accuracy (the camera protocol of PoseDiffusion, Wang et al., ICCV 2023, also reported by
  DUSt3R and VGGT on CO3Dv2 / RealEstate10K): for every camera pair i < j the relative pose of the prediction is
  compared with that of the GT: rotation error = geodesic angle between the two relative rotations, translation
  error = angle between the two relative translation directions (sign-invariant by default, as in the reference
  code: min(a, 180 - a)). RRA@tau / RTA@tau = percent of pairs with error < tau degrees; AUC@tau (a.k.a. mAA(tau))
  = mean over the integer thresholds 1..tau of the percent of pairs with max(rotation error, translation error)
  below the threshold (the histogram-and-cumsum computation of the reference code, whose last bin is closed: the
  tau-th term counts errors <= tau; tau must be an integer). Pairs whose GT camera centres are closer than
  `min_baseline` (e.g. the same
  static camera at two times) have no translation direction: their translation error is NaN, they are left out of
  RTA and enter AUC through the rotation error alone. For a (V, T) rig, all V T (V T - 1) / 2 pairs are scored by
  default; `rig_pairs` selects the cross-view pairs of each frame or the temporal pairs of each view instead (with
  static cameras most temporal pairs are trivial). The relative pose of a pair follows `convention`:
      "relative"      T_ij = w2c_j inv(w2c_i) (camera i -> camera j): PoseDiffusion's `camera_to_rel_deg` (default).
      "vggt"          T = w2c_i inv(w2c_j) (camera j -> camera i): VGGT's CO3D evaluation (`evaluation` branch,
                      `test_co3d.py` since its 2025-07-08 fix). Same rotation errors as "relative"; the translation
                      direction is taken in the other camera of the pair, so translation errors differ when the
                      relative rotation is wrong.
      "first_camera"  both pose sets are first expressed relative to camera 0 (w2c_k inv(w2c_0)), then
                      T = inv(w2c'_i) w2c'_j: Depth-Anything-3's `bench.utils.compute_pose` and VGGT's evaluation
                      before that fix. Depends on which camera comes first; use it only to reproduce those numbers.
  All three are invariant to the world frame and scale of the prediction (no alignment is needed), and all errors
  vanish when the prediction equals the GT up to a similarity; only compare numbers of the same convention. Angles
  are exact here; PoseDiffusion's pytorch3d `so3_relative_angle(eps=1e-4)` extrapolates acos linearly below ~0.8 deg
  (an exact 0 reads ~0.4 deg), which cannot change RRA / RTA / AUC at thresholds of 1 deg or more.

Pure numpy; deterministic.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import camera_center, umeyama

__all__ = ["as_w2c", "invert_rigid", "rotation_error_deg", "direction_error_deg", "all_pairs", "rig_pairs",
           "relative_pose_errors", "rra_rta", "pose_auc", "relative_pose_metrics", "align_cameras", "transform_w2c",
           "ate", "rpe", "pose_metrics", "RA_THRESHOLDS", "AUC_THRESHOLDS", "ALIGN_MODES", "CONVENTIONS", "ROT_TOL"]

RA_THRESHOLDS = (5, 15, 30)
AUC_THRESHOLDS = (3, 5, 15, 30)
ALIGN_MODES = ("sim3", "se3", "none")
CONVENTIONS = ("relative", "vggt", "first_camera")
ROT_TOL = 1e-3
_PAIR_CHUNK = 1 << 17
_NAN = float("nan")


def _key(thr) -> str:
    return f"{float(thr):g}"


def _check_rotations(a: np.ndarray) -> None:
    """ValueError if the 3x3 block of a finite pose of a (N, 4, 4) is not a rotation (see the module docstring)."""
    fin = np.isfinite(a[:, :3, :]).all(axis=(1, 2))
    if not fin.any():
        return
    R = a[fin, :3, :3]
    err = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max(axis=(-2, -1))
    det = np.einsum("...i,...i->...", R[:, 0], np.cross(R[:, 1], R[:, 2]))
    bad = (err > ROT_TOL) | ~(det > 0)
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"pose {int(np.flatnonzero(fin)[k])} is not rigid: its 3x3 block is not a rotation "
                         f"(max |R^T R - I| = {err[k]:.3g}, det R = {det[k]:.4g}; {int(bad.sum())} such poses). Pass "
                         "world-to-camera poses [R | t] with R in SO(3) (e.g. project R to U V^T of its SVD).")


def as_w2c(w2c, check: bool = True) -> np.ndarray:
    """(..., 4, 4) or (..., 3, 4) world-to-camera poses -> (N, 4, 4) float64 copy (leading dims flattened in C
    order), bottom row set to [0, 0, 0, 1] (the input's is ignored).

    check: validate that the 3x3 block of every finite pose is a rotation (max |R^T R - I| <= ROT_TOL, det R > 0) and
    raise ValueError otherwise. Poses with a non-finite entry in [R | t] (missing) are not checked."""
    a = np.array(w2c, dtype=np.float64)
    if a.ndim < 2 or a.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"expected poses (..., 4, 4) or (..., 3, 4), got {a.shape}")
    if a.shape[-2:] == (3, 4):
        a = np.concatenate([a, np.zeros(a.shape[:-2] + (1, 4))], -2)
    a = a.reshape(-1, 4, 4)
    a[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    if check:
        _check_rotations(a)
    return a


def invert_rigid(T) -> np.ndarray:
    """Closed-form inverse of rigid 4x4 transforms (..., 4, 4): [R^T, -R^T t]."""
    T = np.asarray(T, np.float64)
    R, t = T[..., :3, :3], T[..., :3, 3]
    out = np.zeros(T.shape)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def _angle_rad(R: np.ndarray) -> np.ndarray:
    """Rotation angle of (..., 3, 3) rotation matrices, radians in [0, pi]."""
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return np.arctan2(np.linalg.norm(w, axis=-1) / 2.0, c)


def rotation_error_deg(R1, R2) -> np.ndarray:
    """Geodesic angle (degrees) between rotation matrices R1, R2 (..., 3, 3): angle(R1^T R2)."""
    R1, R2 = np.asarray(R1, np.float64), np.asarray(R2, np.float64)
    return np.degrees(_angle_rad(np.einsum("...ji,...jk->...ik", R1, R2)))


def direction_error_deg(a, b, sign_invariant: bool = False) -> np.ndarray:
    """Angle (degrees) between vectors a, b (..., 3), in [0, 180] (or [0, 90] if sign_invariant: min(x, 180 - x)).
    A zero-length vector has no direction: its angle to anything is defined as 90 degrees."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, -1)))
    zero = (np.linalg.norm(a, axis=-1) == 0) | (np.linalg.norm(b, axis=-1) == 0)
    ang = np.where(zero, 90.0, ang)
    if sign_invariant:
        ang = np.minimum(ang, 180.0 - ang)
    return ang


def all_pairs(n: int) -> np.ndarray:
    """(n (n - 1) / 2, 2) int64 index pairs (i, j), i < j, in lexicographic order."""
    i, j = np.triu_indices(int(n), k=1)
    return np.stack([i, j], -1).astype(np.int64)


def rig_pairs(n_views: int, n_frames: int, mode: str = "same_frame") -> np.ndarray:
    """Pair subsets of a (V, T) rig for `relative_pose_errors(..., pairs=...)`, as (P, 2) int64 indices into the
    flattened poses (index v * T + t, the C order of (V, T, 4, 4)), each pair (i, j) with i < j.

    mode: "same_frame" (every view pair at every frame: the multi-view rig; T * V (V - 1) / 2 pairs), "same_view"
    (every frame pair of every view: the trajectories; V * T (T - 1) / 2 pairs) or "all" (= all_pairs(V * T))."""
    V, T = int(n_views), int(n_frames)
    if mode == "all":
        return all_pairs(V * T)
    if mode == "same_frame":
        a, b = np.triu_indices(V, k=1)
        t = np.arange(T)[:, None]
        return np.stack([(a[None] * T + t).ravel(), (b[None] * T + t).ravel()], -1).astype(np.int64)
    if mode == "same_view":
        a, b = np.triu_indices(T, k=1)
        v = np.arange(V)[:, None] * T
        return np.stack([(v + a[None]).ravel(), (v + b[None]).ravel()], -1).astype(np.int64)
    raise ValueError(f"mode={mode!r} not in ('same_frame', 'same_view', 'all')")


def _finite_poses(P: np.ndarray) -> np.ndarray:
    return np.isfinite(P).all(axis=(-2, -1))


def _valid_poses(valid, n: int) -> np.ndarray:
    """(n,) bool from a `valid` argument with one entry per pose (any shape of n entries, e.g. (V, T); no
    broadcasting); bool or integer 0 / 1."""
    v = np.asarray(valid)
    if v.dtype != bool:
        if not (np.issubdtype(v.dtype, np.integer) and np.isin(v, (0, 1)).all()):
            raise TypeError(f"valid must be bool (or integer 0 / 1), got dtype {v.dtype}")
        v = v != 0
    if v.size != n:
        raise ValueError(f"valid has {v.size} entries (shape {v.shape}); expected one per pose ({n})")
    return v.reshape(n)


def _pair_shapes(pred_w2c, gt_w2c):
    P, G = as_w2c(pred_w2c), as_w2c(gt_w2c)
    if P.shape != G.shape:
        raise ValueError(f"pred poses {P.shape} and gt poses {G.shape} differ (flattened)")
    return P, G


def relative_pose_errors(pred_w2c, gt_w2c, pairs=None, *, sign_invariant: bool = True, min_baseline: float = 1e-6,
                         convention: str = "relative") -> dict:
    """Per-pair relative rotation / translation-direction errors.

    Args:
        pred_w2c, gt_w2c: (..., 4, 4) world-to-camera poses (flattened to N poses).
        pairs: (P, 2) int index pairs into the N poses; None = all pairs i < j.
        sign_invariant: translation directions compared up to sign (reference-code default).
        min_baseline: GT pairs whose camera centres are closer than min_baseline (GT units, metres) have an
            undefined translation direction.
        convention: "relative" (T_ij = w2c_j inv(w2c_i), PoseDiffusion), "vggt" (w2c_i inv(w2c_j), VGGT's CO3D
            evaluation) or "first_camera" (poses re-expressed relative to camera 0, then inv(w2c_i) w2c_j;
            Depth-Anything-3); see the module docstring.

    Returns:
        {"pairs": (P, 2), "rot_err_deg": (P,), "trans_err_deg": (P,), "gt_baseline": (P,) [m] (GT camera-centre
        distance)}. Errors are inf when a predicted pose of the pair is missing (non-finite; with "first_camera" also
        when predicted pose 0 is missing), NaN when the GT of the pair is missing (pair not scored); trans_err_deg is
        also NaN when the GT baseline is below min_baseline.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention={convention!r} not in {CONVENTIONS}")
    P, G = _pair_shapes(pred_w2c, gt_w2c)
    n = len(G)
    pairs = all_pairs(n) if pairs is None else np.asarray(pairs, np.int64).reshape(-1, 2)
    if pairs.size and (pairs.min() < 0 or pairs.max() >= n):
        raise IndexError(f"pair indices must be in [0, {n})")
    i, j = pairs[:, 0], pairs[:, 1]
    rot, trans = np.empty(len(pairs)), np.empty(len(pairs))
    with np.errstate(invalid="ignore", over="ignore"):
        A, B = G, P
        if convention == "first_camera":
            A, B = G @ invert_rigid(G[0])[None], P @ invert_rigid(P[0])[None]
        Ai, Bi = invert_rigid(A), invert_rigid(B)
        for a in range(0, len(pairs), _PAIR_CHUNK):
            ii, jj = i[a:a + _PAIR_CHUNK], j[a:a + _PAIR_CHUNK]
            if convention == "relative":
                rel_g, rel_p = A[jj] @ Ai[ii], B[jj] @ Bi[ii]
            elif convention == "vggt":
                rel_g, rel_p = A[ii] @ Ai[jj], B[ii] @ Bi[jj]
            else:
                rel_g, rel_p = Ai[ii] @ A[jj], Bi[ii] @ B[jj]
            rot[a:a + len(ii)] = rotation_error_deg(rel_g[:, :3, :3], rel_p[:, :3, :3])
            trans[a:a + len(ii)] = direction_error_deg(rel_g[:, :3, 3], rel_p[:, :3, 3], sign_invariant)
        centres = camera_center(G)
        base = np.linalg.norm(centres[i] - centres[j], axis=-1)
    gt_ok, pred_ok = _finite_poses(G), _finite_poses(P)
    if convention == "first_camera":
        gt_ok, pred_ok = gt_ok & gt_ok[0], pred_ok & pred_ok[0]
    pair_gt = gt_ok[i] & gt_ok[j]
    pair_pred = pred_ok[i] & pred_ok[j]
    with np.errstate(invalid="ignore"):
        gt_direction = base >= min_baseline
    rot = np.where(pair_pred, rot, np.inf)
    trans = np.where(pair_pred, trans, np.inf)
    trans = np.where(gt_direction, trans, np.nan)
    rot = np.where(pair_gt, rot, np.nan)
    trans = np.where(pair_gt, trans, np.nan)
    base = np.where(pair_gt, base, np.nan)
    return {"pairs": pairs, "rot_err_deg": rot, "trans_err_deg": trans, "gt_baseline": base}


def rra_rta(rot_err_deg, trans_err_deg, thresholds: Sequence[float] = RA_THRESHOLDS) -> dict:
    """Relative rotation / translation accuracy: {"rra@<tau>", "rta@<tau>"} = percent of scored pairs with error
    < tau degrees. NaN errors (unscored pairs) are left out of the denominator; inf (missing prediction) counts as a
    failure."""
    r = np.asarray(rot_err_deg, np.float64).ravel()
    t = np.asarray(trans_err_deg, np.float64).ravel()
    r, t = r[~np.isnan(r)], t[~np.isnan(t)]
    out = {}
    for tau in thresholds:
        out[f"rra@{_key(tau)}"] = 100.0 * float(np.mean(r < tau)) if r.size else _NAN
        out[f"rta@{_key(tau)}"] = 100.0 * float(np.mean(t < tau)) if t.size else _NAN
    return out


def pose_auc(rot_err_deg, trans_err_deg, max_threshold: int = 30) -> float:
    """AUC@max_threshold (mAA) of the pairwise pose errors, percent.

    e = max(rotation error, translation error) per pair (np.fmax: a NaN translation error -> rotation error alone;
    pairs with both NaN are not scored); AUC = mean over k = 1..tau of the share of pairs with e < k, except the last
    term (k = tau) which uses e <= tau, exactly as `np.histogram(e, bins=np.arange(tau + 1))` + cumsum in the
    PoseDiffusion / VGGT code. max_threshold must be an integer >= 1 (degrees; the AUC uses 1-degree bins)."""
    tau = int(max_threshold)
    if tau != max_threshold or tau < 1:
        raise ValueError(f"max_threshold must be an integer >= 1 (degrees; 1-degree bins), got {max_threshold!r}")
    e = np.fmax(np.asarray(rot_err_deg, np.float64).ravel(), np.asarray(trans_err_deg, np.float64).ravel())
    e = e[~np.isnan(e)]
    if e.size == 0:
        return _NAN
    acc = [np.mean(e < k) for k in range(1, tau)] + [np.mean(e <= tau)]
    return 100.0 * float(np.mean(acc))


def relative_pose_metrics(pred_w2c, gt_w2c, *, pairs=None, thresholds: Sequence[float] = RA_THRESHOLDS,
                          auc_thresholds: Sequence[int] = AUC_THRESHOLDS, sign_invariant: bool = True,
                          min_baseline: float = 1e-6, convention: str = "relative") -> dict:
    """RRA / RTA / AUC over camera pairs (see the module docstring; arguments as `relative_pose_errors`, thresholds
    in degrees).

    Returns: {"rra@<tau>", "rta@<tau>", "auc@<tau>" [percent], "rre_mean_deg", "rre_median_deg", "rte_mean_deg",
    "rte_median_deg" (over scored pairs; inf if a prediction is missing), "n_pairs" (pairs with a rotation error),
    "n_pairs_translation" (pairs with a translation error)}."""
    e = relative_pose_errors(pred_w2c, gt_w2c, pairs, sign_invariant=sign_invariant, min_baseline=min_baseline,
                             convention=convention)
    r, t = e["rot_err_deg"], e["trans_err_deg"]
    out = rra_rta(r, t, thresholds)
    for tau in auc_thresholds:
        out[f"auc@{_key(tau)}"] = pose_auc(r, t, tau)
    rv, tv = r[~np.isnan(r)], t[~np.isnan(t)]
    out["rre_mean_deg"] = float(np.mean(rv)) if rv.size else _NAN
    out["rre_median_deg"] = float(np.median(rv)) if rv.size else _NAN
    out["rte_mean_deg"] = float(np.mean(tv)) if tv.size else _NAN
    out["rte_median_deg"] = float(np.median(tv)) if tv.size else _NAN
    out["n_pairs"] = int(rv.size)
    out["n_pairs_translation"] = int(tv.size)
    return out


def _check_align(mode: str) -> None:
    if mode not in ALIGN_MODES:
        raise ValueError(f"align={mode!r} not in {ALIGN_MODES}")


def align_cameras(pred_w2c, gt_w2c, mode: str = "sim3", valid=None):
    """Similarity (s, R, t) with X_gt ~ s R X_pred + t, fitted (Umeyama) on the camera centres of the poses that are
    finite in both sets (and `valid`, one bool per pose, if given). mode: "sim3", "se3" (s = 1) or "none"
    (identity)."""
    _check_align(mode)
    P, G = _pair_shapes(pred_w2c, gt_w2c)
    if mode == "none":
        return 1.0, np.eye(3), np.zeros(3)
    ok = _finite_poses(P) & _finite_poses(G)
    if valid is not None:
        ok &= _valid_poses(valid, len(P))
    if not ok.any():
        raise ValueError("no pose is finite in both pred and gt")
    s, R, t = umeyama(camera_center(P[ok]), camera_center(G[ok]), with_scale=(mode == "sim3"))
    return float(s), R, t


def transform_w2c(w2c, s: float, R, t) -> np.ndarray:
    """Express cameras (..., 4, 4) w2c in the world X' = s R X + t (camera coordinates scaled by s, so the result
    is in the units of the target world). Centre c' = s R c + t; orientation (c2w) R R_c2w."""
    w2c = np.asarray(w2c, np.float64)
    R, t = np.asarray(R, np.float64), np.asarray(t, np.float64)
    out = np.array(w2c, copy=True)
    Rc, tc = w2c[..., :3, :3], w2c[..., :3, 3]
    Rn = Rc @ R.T
    out[..., :3, :3] = Rn
    out[..., :3, 3] = s * tc - np.einsum("...ij,j->...i", Rn, t)
    return out


def _rank(X: np.ndarray) -> int:
    if len(X) < 2:
        return 0
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return int(np.sum(sv > max(1e-9, 1e-6 * sv[0])))


def ate(pred_w2c, gt_w2c, align: str = "sim3", valid=None) -> dict:
    """Absolute trajectory error of the camera centres after alignment.

    Args:
        pred_w2c, gt_w2c: (..., 4, 4) world-to-camera poses (flattened to N; one alignment for all of them, e.g. a
            whole multi-view rig over time).
        align: "sim3" (scale-free prediction), "se3" (metric prediction) or "none".
        valid: optional bool with one entry per pose ((N,) or the leading shape, e.g. (V, T)): poses to use.

    Returns:
        {"ate_rmse", "ate_mean", "ate_median", "ate_max" [m], "are_rmse_deg", "are_mean_deg", "are_median_deg"
         (orientation error of the aligned cameras), "scale" (s of the alignment), "n_poses" (used), "n_missing"
         (GT poses without a finite prediction), "degenerate" (alignment rotation not unique: < 3 non-collinear
         centres), "align"}.
    """
    _check_align(align)
    P, G = _pair_shapes(pred_w2c, gt_w2c)
    gt_ok = _finite_poses(G)
    ok = gt_ok & _finite_poses(P)
    if valid is not None:
        v = _valid_poses(valid, len(P))
        gt_ok &= v
        ok &= v
    out = {"align": align, "n_poses": int(ok.sum()), "n_missing": int((gt_ok & ~ok).sum())}
    if not ok.any():
        out.update({k: _NAN for k in ("ate_rmse", "ate_mean", "ate_median", "ate_max", "are_rmse_deg", "are_mean_deg",
                                      "are_median_deg", "scale")}, degenerate=True)
        return out
    P, G = P[ok], G[ok]
    s, R, t = align_cameras(P, G, align)
    cp, cg = camera_center(P), camera_center(G)
    err = np.linalg.norm(s * cp @ R.T + t - cg, axis=-1)
    rot = rotation_error_deg(G[:, :3, :3], transform_w2c(P, s, R, t)[:, :3, :3])
    out.update(ate_rmse=float(np.sqrt(np.mean(err ** 2))), ate_mean=float(err.mean()),
               ate_median=float(np.median(err)), ate_max=float(err.max()),
               are_rmse_deg=float(np.sqrt(np.mean(rot ** 2))), are_mean_deg=float(rot.mean()),
               are_median_deg=float(np.median(rot)), scale=float(s),
               degenerate=bool(align != "none" and min(_rank(cp), _rank(cg)) < 2))
    return out


def rpe(pred_w2c, gt_w2c, delta: int = 1, align: str = "sim3") -> dict:
    """Relative pose error between frames t and t + delta of each trajectory.

    Args:
        pred_w2c, gt_w2c: (T, 4, 4) one trajectory or (V, T, 4, 4) one trajectory per view (pairs never cross views);
            a single (4, 4) pose is a trajectory of T = 1 (no pair). (3, 4) blocks are accepted too.
        delta: frame step (>= 1).
        align: "sim3" multiplies the predicted translations by the Sim(3) scale of `align_cameras` (fitted on all
            poses finite in both); "se3" / "none" use the prediction as is (RPE does not depend on a rigid alignment).

    Returns:
        {"rpe_trans_rmse", "rpe_trans_mean", "rpe_trans_median" [m], "rpe_rot_rmse_deg", "rpe_rot_mean_deg",
         "rpe_rot_median_deg", "rpe_delta", "rpe_scale", "n_rpe_pairs" (pairs with finite poses)}. Errors are NaN
        when no pair has finite poses (rpe_scale is NaN when no pose is finite in both for align="sim3").
    """
    _check_align(align)
    if delta < 1:
        raise ValueError("delta must be >= 1")
    P = as_w2c(pred_w2c).reshape(np.shape(pred_w2c)[:-2] + (4, 4))
    G = as_w2c(gt_w2c).reshape(np.shape(gt_w2c)[:-2] + (4, 4))
    if P.shape != G.shape:
        raise ValueError(f"pred poses {P.shape} and gt poses {G.shape} differ")
    if P.ndim == 2:
        P, G = P[None, None], G[None, None]
    elif P.ndim == 3:
        P, G = P[None], G[None]
    if P.ndim != 4:
        raise ValueError(f"expected (T, 4, 4) or (V, T, 4, 4) poses, got {P.shape}")
    T = P.shape[1]
    s = 1.0
    if align == "sim3":
        both = _finite_poses(P) & _finite_poses(G)
        s = align_cameras(P, G, "sim3")[0] if both.any() else _NAN
    keys = ("rpe_trans_rmse", "rpe_trans_mean", "rpe_trans_median", "rpe_rot_rmse_deg", "rpe_rot_mean_deg",
            "rpe_rot_median_deg")
    out = {"rpe_delta": int(delta), "rpe_scale": float(s)}
    if T <= delta:
        out.update({k: _NAN for k in keys}, n_rpe_pairs=0)
        return out
    with np.errstate(invalid="ignore", over="ignore"):
        Pc, Gc = invert_rigid(P), invert_rigid(G)
        Pc[..., :3, 3] *= s
        i = np.arange(T - delta)
        rel_g = invert_rigid(Gc[:, i]) @ Gc[:, i + delta]
        rel_p = invert_rigid(Pc[:, i]) @ Pc[:, i + delta]
        E = (invert_rigid(rel_g) @ rel_p).reshape(-1, 4, 4)
    ok = _finite_poses(E)
    if not ok.any():
        out.update({k: _NAN for k in keys}, n_rpe_pairs=0)
        return out
    tr = np.linalg.norm(E[ok, :3, 3], axis=-1)
    rot = np.degrees(_angle_rad(E[ok, :3, :3]))
    out.update(rpe_trans_rmse=float(np.sqrt(np.mean(tr ** 2))), rpe_trans_mean=float(tr.mean()),
               rpe_trans_median=float(np.median(tr)), rpe_rot_rmse_deg=float(np.sqrt(np.mean(rot ** 2))),
               rpe_rot_mean_deg=float(rot.mean()), rpe_rot_median_deg=float(np.median(rot)), n_rpe_pairs=int(ok.sum()))
    return out


def pose_metrics(pred_w2c, gt_w2c, *, align: str = "sim3", thresholds: Sequence[float] = RA_THRESHOLDS,
                 auc_thresholds: Sequence[int] = AUC_THRESHOLDS, rpe_delta: Optional[int] = 1, pairs=None,
                 sign_invariant: bool = True, min_baseline: float = 1e-6, convention: str = "relative") -> dict:
    """All camera metrics of one sample: `ate` + `rpe` (with `align`) + `relative_pose_metrics` (alignment-free, on
    `pairs`; None = all pairs, see `rig_pairs`).

    pred_w2c / gt_w2c are (T, 4, 4) or (V, T, 4, 4); for RPE, axis -3 is time (rpe_delta=None skips RPE).
    Returns the union of the three dicts (keys are disjoint)."""
    out = ate(pred_w2c, gt_w2c, align)
    if rpe_delta is not None:
        out.update(rpe(pred_w2c, gt_w2c, rpe_delta, align))
    out.update(relative_pose_metrics(pred_w2c, gt_w2c, pairs=pairs, thresholds=thresholds,
                                     auc_thresholds=auc_thresholds, sign_invariant=sign_invariant,
                                     min_baseline=min_baseline, convention=convention))
    return out
