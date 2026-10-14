"""Task `depth`: video depth estimation, one view at a time, on a frame window.

Task
    For every view k of a sample, predict the depth of every frame of the window
    t = ref_frame + j * frame_step, j = 0..horizon (default: frames 16, 17, ..., 80 = 65 frames) from the video of that
    view alone (a video-depth model sees the whole window; a single-image model is run frame by frame).

Ground truth (planar camera z in metres, docs/DATA_FORMAT.md)
    dense    per-pixel depth from depth/depth.zip:
               Simul-Robot, Simul-General   exact renderer depth (synthetic GT);
               Real-General                 reconstruction pseudo-GT (orhsurf points rendered at 512x384,
                                            holes = no GT);
               Real-Robot                   none.
             Pixels outside the valid field of view of an undistorted video (Real-General's black undistortion
             border, `common.fov_mask`) are not scored.
    sparse   camera z of the GT 3D track points (tracks3d.npz) that are `default_gt` and visible in the view at that
             frame, projected with the GT camera. Every subset has it; it is the only depth GT of Real-Robot.
             Where dense GT exists, a point is kept only if the dense GT agrees with it, |D - z| <= max(a, r z)
             (`sparse_consistency`, default (a, r) = (2 cm, 5 %) on Simul-Robot, whose scenes are 0.3-1 m from the
             cameras, and (5 cm, 5 %) elsewhere), at its nearest pixel of the video grid (the point is the visible
             surface there: drops e.g. Simul-Robot silhouette points seen against a farther background) and at its
             nearest pixel of the evaluation grid (no other surface in that coarser pixel: drops object edges). The
             kept set depends only on the GT and the evaluation grid, so every method is scored on the same points.
             Points that project onto an invalid-FOV pixel are dropped. Caveats (sample.json
             tracks.visibility_rule): Real-Robot visibility models robot self-occlusion only (scene objects never
             occlude) and cannot be cross-checked; Real-General is pseudo-GT.

Inputs (`export_inputs`), per evaluable sample, in <out>/<Subset>/<id>/
    view{k}.mp4                        the window frames of view k (frames="mp4", default; H.264 crf 10 or lossless), or
    view{k}/frame_{t:04d}.png          one PNG per window frame (frames="png"), or nothing (frames="none": read the
                                       frames from the benchmark, `Sample.frames(k, window)`; paths in inputs.json)
    inputs.json                        window frame indices, fps, per view: file, size, intrinsics K at that size
                                       (optional input: a method that uses K should say so), source video path
    <out>/manifest.jsonl, protocol.json, skipped.json
    Frames are native resolution unless `size` = (W, H) is given (stretched with the 'resize' map, K adjusted).

Prediction format
    <pred_root>/<Subset>/<id>/view{k}.npz with ONE of
        'depth'       (n, H, W) float16/32 planar depth (metres for metric models; any positive unit otherwise);
                      non-finite or <= 0 = no prediction
        'disparity'   (n, H, W) inverse depth, possibly affine-invariant (DepthCrafter / Video Depth Anything style);
                      non-finite = no prediction
    and optionally 'frame_indices' (n,) int: the GT frame of each map. Without it, n must equal the window length
    (map j = window frame j) or the clip length T (map t = frame t; the window frames are taken). Any resolution H x W:
    the map is taken to cover the full frame of the video (the 'resize' pixel map of geometry.image_map, pixel centres
    at integers). Dense scoring resizes it to the evaluation grid (`resize_interp`): "auto" (default) = exact
    pixel-centre nearest neighbour when the map is at least as fine as the evaluation grid in both axes (the rule the GT
    is resampled with, so a perfect native-resolution prediction scores exactly), else bilinear; "bilinear" is
    missing-value aware (missing pixels keep their nearest-neighbour footprint and never mix into valid ones,
    `common.resize_map`); "nearest" always nearest. Sparse scoring samples the prediction on its own grid at the exact
    sub-pixel projection of each GT point (`sparse_interp`: nearest by default, or bilinear). Use
    np.savez_compressed. A missing file makes the sample "missing"; an unreadable or malformed one (keys, shape, frame
    count, dtype) "invalid" (common.STATUSES). Maps whose aspect is the TRANSPOSED aspect of the video ((n, W, H)
    saved instead of (n, H, W)) are scored as given but flagged in the per-sample "warnings" and in the summary.

Metrics (`vpm_worldbench.metrics.depth`; per view, then the mean over the views of the sample, then mean +- s.e. over
samples per subset)
    AbsRel, SqRel, RMSE, RMSE-log, log10, SILog, delta1-3 (percent) over the GT pixels / points THAT HAVE A PREDICTION,
    coverage (percent of GT pixels / points with a prediction) and delta1_all (percent of ALL scored GT pixels / points
    whose prediction exists and is within delta1: a missing prediction counts as a failure; read the other metrics
    together with coverage). A view with GT but no prediction at all has no AbsRel etc. (left out of the sample mean)
    but coverage 0 and delta1_all 0 (kept). Reported separately for the dense and the sparse GT, after each alignment
    mode:
        none          metric depth as is (metric-depth models)
        median        s = median(gt) / median(pred), depth space (Monodepth2-style; per sequence by default)
        scale         least-squares scale, depth space
        scale_shift   least-squares scale and shift in DISPARITY space, one fit per sequence (= per view and window):
                      the video-depth default (DepthCrafter, Video Depth Anything); the headline number
    The fit uses the dense GT pixels (one scene-level fit per view and window) and the same transformation is applied
    to the sparse samples, so the sparse block measures the depth of the tracked (mostly moving) objects under the
    scene alignment. Real-Robot has no dense GT: its fit uses the sparse points themselves; the robot spans a narrow
    depth range there, so a constant prediction already reaches delta1 ~ 85-92 % after scale_shift (the `constant`
    baseline; the "none" mode and AbsRel are the informative numbers for Real-Robot). Dense metrics are computed per
    frame and averaged over the frames (`reduce="frames"`, the DepthCrafter / VDA code); sparse metrics pool the points
    of all frames of the view (`reduce_sparse="pixels"`: few points per frame). GT depth is scored in
    (min_depth, max_depth) with max_depth = the subset's far clip (Simul-* 80 m, Real-* 20 m).

Evaluation grid (dense)
    The generation eval grid of the subset (presets: Simul-* 480x480, Real-Robot 848x480, Real-General 640x480): the GT
    depth is `Sample.depth(view, indices, size)` (exact pixel-centre nearest-neighbour; the Real-General 512x384 render
    is mapped onto the 1024x768 video grid first, an exact 2x replication). `eval_size` overrides it per subset. With
    resize_interp "auto" a perfect prediction at the video resolution scores exactly (forcing "bilinear" costs it
    AbsRel 0.004-0.024 from blending across depth edges).

Floors of the GT itself (oracle, default protocol): dense exact everywhere; sparse AbsRel ~ 0.013 on Simul-Robot
(track points vs the rendered surface within one 480 px pixel at 0.3-1 m), ~ 0.005 on Simul-General and
Real-General (the consistency test keeps 90 / 96 / 99 % of the visible points).

Default protocol (`DepthProtocol`): ref_frame 16, horizon 64, frame_step 1 (frames 16..80), all views, the four
alignment modes, per-sequence fits. A sample is evaluable iff the window lies inside the clip. Per-subset fields
(eval_size, max_depth, sparse_consistency) accept {subset or alias: value}.

Baselines (`BASELINES`, sanity checks)
    oracle     the GT itself on the eval grid (dense GT where available, else the visible GT track points splatted with
               a z-buffer); exact dense metrics, sparse errors within the consistency tolerance
    constant   1 m everywhere: what alignment alone achieves (median alignment = the GT median)

Usage
    from vpm_worldbench import Benchmark
    from vpm_worldbench.tasks import depth
    bench = Benchmark("/data/VP-WorldBench")
    depth.export_inputs(bench, "inputs/depth")                  # frames for your model
    depth.run_baseline(bench, "preds/depth_oracle", kind="oracle")   # or write <pred>/<Subset>/<id>/view{k}.npz
    depth.evaluate(bench, "preds/depth_oracle", "results/depth_oracle", jobs=8)
    vpmwb depth evaluate --root /data/VP-WorldBench --pred preds/x --out results/x --jobs 8
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import SUBSETS
from ..geometry import image_map
from ..metrics import depth as dm
from ..presets import SUBSET_DEFAULTS
from .common import (InvalidPrediction, MissingPrediction, TaskProtocol, as_size, baseline_args, evaluate_task,
                     export_view_frames, fov_mask, frame_index_array, gt_depth, read_gt_depth, read_npz, resize_map,
                     run_baseline_task, run_jobs, sample_map, select_samples, summarize_records, task_cli,
                     track_observations, transposed_hint, write_json, write_manifest)

__all__ = [
    "DepthProtocol", "export_inputs", "evaluate", "evaluate_sample", "evaluate_view", "summarize", "load_prediction",
    "oracle_maps", "write_oracle", "write_constant", "run_baseline", "BASELINES", "ALIGN_DEFAULT", "ALIGN_SPACE",
    "DEPTH_EVAL_SIZE", "DEPTH_MAX_M", "SPARSE_TOL", "main",
]

TASK = "depth"
ALIGN_DEFAULT = ("none", "median", "scale", "scale_shift")
ALIGN_SPACE = {"none": "depth", "median": "depth", "scale": "depth", "scale_shift": "disparity"}
HEADLINE_MODE = "scale_shift"

DEPTH_MAX_M = {s: float(SUBSET_DEFAULTS[s]["far_clip_m"]) for s in SUBSETS}
DEPTH_EVAL_SIZE = {s: tuple(SUBSET_DEFAULTS[s]["eval_size"]) for s in SUBSETS}
SPARSE_TOL = {"Simul-Robot": (0.02, 0.05), "Simul-General": (0.05, 0.05), "Real-Robot": (0.05, 0.05),
              "Real-General": (0.05, 0.05)}


@dataclass
class DepthProtocol(TaskProtocol):
    """Protocol of the depth task (module docstring). Per-subset dicts ({subset or alias: value}; missing subsets use
    the defaults): eval_size (W, H) [DEPTH_EVAL_SIZE], max_depth metres [DEPTH_MAX_M], sparse_consistency (abs m, rel)
    or None = keep every visible point [SPARSE_TOL]; sparse_consistency may also be one (abs, rel) pair for every
    subset, or None (no test anywhere). align_space {mode: "depth" | "disparity"} overrides ALIGN_SPACE."""
    SUBSET_FIELDS = ("eval_size", "max_depth", "sparse_consistency")

    name: str = "depth-v1"
    eval_size: dict = field(default_factory=dict)
    align: tuple = ALIGN_DEFAULT
    align_per: str = "sequence"
    align_space: dict = field(default_factory=dict)
    pred_min: Optional[float] = None
    min_depth: float = dm.DEFAULT_MIN_DEPTH
    max_depth: dict = field(default_factory=dict)
    reduce: str = "frames"
    reduce_sparse: str = "pixels"
    resize_interp: str = "auto"
    sparse_interp: str = "nearest"
    sparse_visibility: str = "visible"
    sparse_consistency: object = field(default_factory=dict)
    use_fov_mask: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.align = tuple(self.align)
        for m in self.align:
            if m not in dm.ALIGN_MODES:
                raise ValueError(f"alignment mode {m!r} not in {dm.ALIGN_MODES}")
        if self.align_per not in dm.ALIGN_PER:
            raise ValueError(f"align_per must be one of {dm.ALIGN_PER}")
        if self.reduce not in dm.REDUCE_MODES or self.reduce_sparse not in dm.REDUCE_MODES:
            raise ValueError(f"reduce / reduce_sparse must be one of {dm.REDUCE_MODES}")
        if self.resize_interp not in ("auto", "bilinear", "nearest"):
            raise ValueError("resize_interp must be 'auto', 'bilinear' or 'nearest'")
        if self.sparse_interp not in ("bilinear", "nearest"):
            raise ValueError("sparse_interp must be 'bilinear' or 'nearest'")
        if self.sparse_visibility not in ("visible", "in_frustum"):
            raise ValueError("sparse_visibility must be 'visible' or 'in_frustum'")

    def eval_size_for(self, subset: str) -> tuple:
        return as_size(self.per_subset("eval_size", subset, DEPTH_EVAL_SIZE))

    def max_depth_for(self, subset: str) -> Optional[float]:
        v = self.per_subset("max_depth", subset, DEPTH_MAX_M)
        return None if v is None else float(v)

    def tolerance_for(self, subset: str) -> Optional[tuple]:
        c = self.sparse_consistency
        v = self.per_subset("sparse_consistency", subset, SPARSE_TOL) if isinstance(c, dict) else c
        return None if v is None else (float(v[0]), float(v[1]))

    def space_for(self, mode: str) -> str:
        return self.align_space.get(mode, ALIGN_SPACE[mode])

    def resize_interp_for(self, pred_size, eval_size) -> str:
        """The dense resize of a (W, H) prediction onto the (w, h) evaluation grid ("auto" resolved)."""
        if self.resize_interp != "auto":
            return self.resize_interp
        return "nearest" if pred_size[0] >= eval_size[0] and pred_size[1] >= eval_size[1] else "bilinear"


def prediction_path(pred_root, sample, view: int) -> Path:
    return Path(pred_root) / sample.subset / sample.id / f"view{view}.npz"


def load_prediction(pred_root, sample, view: int, protocol: DepthProtocol):
    """(maps (n, H, W) float32 on the window frames with NaN = missing, pred_space "depth" | "disparity").
    Raises MissingPrediction when the file is missing and InvalidPrediction (a ValueError) when it is unreadable or
    malformed."""
    p = prediction_path(pred_root, sample, view)
    if not p.exists():
        raise MissingPrediction(f"missing prediction {p}")
    files, z = read_npz(p, ("depth", "disparity", "frame_indices"))
    keys = [k for k in ("depth", "disparity") if k in z]
    if len(keys) != 1:
        raise InvalidPrediction(f"{p}: needs exactly one of 'depth' / 'disparity', has {files}")
    key = keys[0]
    x = z[key]
    if x.dtype == bool or not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        raise InvalidPrediction(f"{p}: '{key}' must hold real numbers, got dtype {x.dtype}")
    fi = None if "frame_indices" not in z else frame_index_array(z["frame_indices"], p)
    if x.ndim != 3:
        raise InvalidPrediction(f"{p}: '{key}' must be (n, H, W), got {x.shape}")
    win = protocol.window_indices
    if fi is not None:
        if len(fi) != len(x):
            raise InvalidPrediction(f"{p}: frame_indices has {len(fi)} entries for {len(x)} maps")
        pos = {int(t): i for i, t in enumerate(fi)}
        miss = [t for t in win if t not in pos]
        if miss:
            raise InvalidPrediction(f"{p}: window frames {miss[:5]} missing from frame_indices")
        x = x[[pos[t] for t in win]]
    elif len(x) == len(win):
        pass
    elif len(x) == sample.n_frames:
        x = x[win]
    else:
        raise InvalidPrediction(f"{p}: {len(x)} maps; expected the window ({len(win)}) or the clip "
                                f"({sample.n_frames}) or a 'frame_indices' array")
    x = x.astype(np.float32)
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(x) if key == "disparity" else ~(np.isfinite(x) & (x > 0))
    x[bad] = np.nan
    return x, key


def score_depth(pred, gt, mask, min_depth: float, max_depth, reduce: str) -> dict:
    """`metrics.depth.depth_metrics` of an aligned (metric) prediction plus "delta1_all": percent of the scored GT
    pixels whose prediction exists AND is within delta1 (a missing prediction counts as a failure), reduced like the
    other metrics (reduce="frames": mean over the frames with scored GT pixels; "pixels": pooled). One pass over the
    pixels (`depth_metrics_per_frame`) for reduce="frames"."""
    per = dm.depth_metrics_per_frame(pred, gt, mask, min_depth=min_depth, max_depth=max_depth)
    nv, ns = per["n_valid"].astype(np.float64), per["n_scored"].astype(np.float64)
    good = np.where(ns > 0, np.nan_to_num(per["delta1"]) / 100.0 * ns, 0.0)
    if reduce == "frames":
        has = ns > 0
        out = {k: (float(np.mean(per[k][has])) if has.any() else float("nan")) for k in dm.METRIC_KEYS}
        n_valid, n_scored = int(nv.sum()), int(ns.sum())
        out.update(n_valid=n_valid, n_scored=n_scored,
                   coverage=100.0 * n_scored / n_valid if n_valid else float("nan"), n_frames=int(has.sum()),
                   reduce=reduce)
        fr = nv > 0
        out["delta1_all"] = float(np.mean(100.0 * good[fr] / nv[fr])) if fr.any() else float("nan")
    else:
        out = dm.depth_metrics(pred, gt, mask, min_depth=min_depth, max_depth=max_depth, reduce=reduce)
        out["delta1_all"] = 100.0 * float(good.sum()) / nv.sum() if nv.sum() else float("nan")
    return out


def _view_metrics(pr, gt, mask, P, G, protocol: DepthProtocol, pred_space: str, max_depth) -> tuple:
    """Dense and sparse metrics of one view for every alignment mode.

    pr / gt (n, h, w) dense prediction / GT on the eval grid and mask (n, h, w) or None, or pr = gt = None without
    dense GT; P / G (n, N) sparse prediction / GT (NaN = not scored). With dense GT, ONE fit per mode on the dense
    pixels (scene-level alignment): the sparse samples are appended to every frame as extra pixels excluded from the
    fit (`metrics.depth.align` fit mask), so they get exactly the dense transformation. Without dense GT the sparse
    points are fitted themselves. Returns ({mode: dense metrics} or None, {mode: sparse metrics})."""
    n, N = P.shape
    if gt is not None:
        h, w = gt.shape[1:]
        Pc = np.concatenate([pr.reshape(n, h * w).astype(np.float64), P], 1)[:, None]
        Gc = np.concatenate([gt.reshape(n, h * w).astype(np.float64), np.zeros((n, N))], 1)[:, None]
        fit = np.ones((n, h * w), bool) if mask is None else np.asarray(mask).reshape(n, h * w)
        fit = np.concatenate([fit, np.zeros((n, N), bool)], 1)[:, None]
    else:
        Pc, Gc, fit = P[..., None], G[..., None], None
    dense, sparse = ({} if gt is not None else None), {}
    for mode in protocol.align:
        al, prm = dm.align(Pc, Gc, fit, mode=mode, per=protocol.align_per, space=protocol.space_for(mode),
                           pred_space=pred_space, min_depth=protocol.min_depth, max_depth=max_depth,
                           pred_min=protocol.pred_min, return_params=True)
        prm = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in prm.items()}
        if gt is not None:
            al = al[:, 0]
            d = score_depth(al[:, :h * w].reshape(n, h, w), gt, mask, protocol.min_depth, max_depth, protocol.reduce)
            sp = score_depth(al[:, h * w:, None], G[..., None], None, protocol.min_depth, max_depth,
                             protocol.reduce_sparse)
            d["align"] = dict(prm, fitted_on="self")
            sp["align"] = dict(prm, fitted_on="dense")
            dense[mode] = d
        else:
            sp = score_depth(al, Gc, None, protocol.min_depth, max_depth, protocol.reduce_sparse)
            sp["align"] = dict(prm, fitted_on="self")
        sparse[mode] = sp
    return dense, sparse


def evaluate_view(sample, view: int, pred: np.ndarray, pred_space: str, protocol: DepthProtocol) -> dict:
    """Dense + sparse metrics of one view. pred = (n, H, W) maps on the window (NaN = missing)."""
    win = protocol.window_indices
    max_depth = protocol.max_depth_for(sample.subset)
    native = sample.image_size(view)
    fov_native = fov_mask(sample, view) if protocol.use_fov_mask else None
    res = {"pred_size": [int(pred.shape[2]), int(pred.shape[1])]}
    hint = transposed_hint((pred.shape[2], pred.shape[1]), native)
    if hint:
        res["warning"] = f"view{view}: {hint}"
    obs = track_observations(sample, view, win, protocol.sparse_visibility, fov=fov_native)
    size = protocol.eval_size_for(sample.subset)
    gt, keep = read_gt_depth(sample, view, win, size, obs, protocol.tolerance_for(sample.subset))
    imap = image_map(native, (pred.shape[2], pred.shape[1]), "resize")
    P = np.full(obs.z.shape, np.nan)
    G = np.where(keep, obs.z, np.nan)
    for j in range(len(win)):
        m = keep[j]
        if m.any():
            P[j, m] = sample_map(pred[j], imap.apply_uv(obs.uv[j, m]), protocol.sparse_interp)
    pr, mask = None, None
    if gt is not None:
        fov = fov_mask(sample, view, size) if protocol.use_fov_mask else None
        if fov is not None:
            mask = np.broadcast_to(fov, gt.shape)
        interp = protocol.resize_interp_for((pred.shape[2], pred.shape[1]), size)
        pr = resize_map(pred, size, interp)
        res["eval_size"], res["resize_interp"] = list(size), interp
    res["dense"], res["sparse"] = _view_metrics(pr, gt, mask, P, G, protocol, pred_space, max_depth)
    res["n_sparse_points"] = int(keep.sum())
    res["n_sparse_inconsistent"] = int(obs.mask.sum() - keep.sum())
    return res


_FLAT_KEYS = dm.METRIC_KEYS + ("coverage", "delta1_all")


def evaluate_sample(sample, pred_root, protocol: DepthProtocol) -> dict:
    """Metrics of one sample (all views). Returns {"status": "ok", "views", "dense_gt", "prediction", "per_view",
    "metrics"} where "metrics" holds "<dense|sparse>/<mode>/<metric>" = mean over the views that have a value (a view
    without any prediction has coverage 0 and delta1_all 0 but no AbsRel etc.), "<dense|sparse>/coverage" (that of
    the first alignment mode), "sparse/n_points" and "sparse/n_inconsistent"; plus "warnings" (list) when a view's
    maps have the transposed aspect of its video (`common.transposed_hint`)."""
    views = protocol.views_for(sample)
    per_view, kinds = {}, set()
    for k in views:
        pred, key = load_prediction(pred_root, sample, k, protocol)
        kinds.add(key)
        per_view[f"view{k}"] = evaluate_view(sample, k, pred, key, protocol)
    flat = {}
    for block in ("dense", "sparse"):
        for mode in protocol.align:
            for key in _FLAT_KEYS:
                vals = [pv[block][mode][key] for pv in per_view.values() if pv.get(block)]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                if vals:
                    flat[f"{block}/{mode}/{key}"] = float(np.mean(vals))
        if f"{block}/{protocol.align[0]}/coverage" in flat:
            flat[f"{block}/coverage"] = flat[f"{block}/{protocol.align[0]}/coverage"]
    flat["sparse/n_points"] = int(sum(pv["n_sparse_points"] for pv in per_view.values()))
    flat["sparse/n_inconsistent"] = int(sum(pv["n_sparse_inconsistent"] for pv in per_view.values()))
    dense_kind = sample.meta["depth"].get("kind") if sample.has_depth else None
    rec = {"status": "ok", "views": views, "dense_gt": dense_kind, "prediction": {"keys": sorted(kinds)},
           "max_depth": protocol.max_depth_for(sample.subset),
           "sparse_tolerance": protocol.tolerance_for(sample.subset), "per_view": per_view, "metrics": flat}
    warn = [pv["warning"] for pv in per_view.values() if pv.get("warning")]
    if warn:
        rec["warnings"] = warn
    return rec


def evaluate(bench, pred_root, out_dir, protocol: Optional[DepthProtocol] = None, subsets=None, ids=None,
             jobs: int = 1, overwrite: bool = False, strict: bool = False, progress: bool = True) -> dict:
    """Score depth predictions in <pred_root>/<Subset>/<id>/view{k}.npz; writes <out_dir>/<Subset>/<id>.json per
    sample (status ok | skipped | missing | invalid | error), <out_dir>/run.json, protocol.json and
    summary.json|md. Finished samples of the same run are reused while their prediction files are unchanged; another
    run (protocol or prediction root) in the same out_dir raises `common.RunDirError` unless overwrite=True
    (--overwrite; `common.evaluate_task`). Returns the summary dict."""
    protocol = protocol or DepthProtocol()
    return evaluate_task(TASK, evaluate_sample, protocol, DepthProtocol, bench, pred_root, out_dir, subsets, ids, jobs,
                         overwrite, strict, progress, summarize_fn=summarize)


_LABEL = {"abs_rel": ("AbsRel", 4), "delta1": ("delta1 [%]", 2), "delta1_all": ("delta1, missing = fail [%]", 2),
          "rmse": ("RMSE [m]", 3), "coverage": ("coverage [%]", 1)}


def summarize(out_dir) -> dict:
    """(Re)build <out_dir>/summary.json and summary.md from the per-sample JSONs."""
    head = []
    for block in ("dense", "sparse"):
        head.append((f"{block}/coverage", f"{block} / coverage [%]", 1))
        for mode in ALIGN_DEFAULT:
            for key in ("abs_rel", "delta1", "delta1_all", "rmse"):
                label, digits = _LABEL[key]
                head.append((f"{block}/{mode}/{key}", f"{block} / {mode} / {label}", digits))
    head.append(("sparse/n_points", "sparse points (per sample)", 0))
    head.append(("sparse/n_inconsistent", "sparse points dropped by the dense consistency test", 0))
    notes = ["dense GT: Simul-* exact renders, Real-General reconstruction pseudo-GT, Real-Robot none",
             "sparse GT: camera z of the visible default_gt GT track points (every subset), kept where the dense GT "
             "agrees",
             f"headline: {HEADLINE_MODE} (scale+shift in disparity, one fit per view and window); 'none' for "
             "metric-depth models",
             "AbsRel / delta / RMSE cover the GT pixels that have a prediction: read them with coverage; 'delta1, "
             "missing = fail' counts a missing prediction as a failure"]
    sub_notes = {"Real-Robot": ["sparse GT only, on the robot (narrow depth range): after the self-fitted "
                                "scale_shift a constant prediction already reaches delta1 ~ 85-92 % (`constant` "
                                "baseline); compare the 'none' mode or AbsRel"]}
    return summarize_records(out_dir, TASK, "VP-WorldBench - depth (video depth estimation)", head, notes, sub_notes)


def export_sample(sample, protocol: DepthProtocol, out_dir, frames: str = "mp4", size=None, crf: int = 10,
                  lossless: bool = False) -> Optional[dict]:
    ok, _ = protocol.evaluable(sample)
    if not ok:
        return None
    d = Path(out_dir) / sample.subset / sample.id
    d.mkdir(parents=True, exist_ok=True)
    win = protocol.window_indices
    views = protocol.views_for(sample)
    vrec = {f"view{k}": export_view_frames(sample, k, win, d, frames, size, "resize", crf, lossless) for k in views}
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id,
           "views": [f"view{k}" for k in views], "fps": sample.fps, "window_fps": sample.fps / protocol.frame_step,
           "frame_indices": win, "n_frames": len(win),
           "frames": vrec, "dir": f"{sample.subset}/{sample.id}",
           "prediction": {"files": {f"view{k}": f"{sample.subset}/{sample.id}/view{k}.npz" for k in views},
                          "key": "depth | disparity", "shape": f"({len(win)}, H, W)"}}
    write_json(d / "inputs.json", rec)
    return rec


def _export_one(root, subset, sid, protocol_dict, out_dir, frames, size, crf, lossless):
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = DepthProtocol.from_dict(protocol_dict)
    rec = export_sample(s, p, out_dir, frames, size, crf, lossless)
    return rec if rec is not None else {"uid": s.uid, "skipped": p.evaluable(s)[1]}


def export_inputs(bench, out_dir, protocol: Optional[DepthProtocol] = None, subsets=None, ids=None,
                  frames: str = "mp4", size=None, crf: int = 10, lossless: bool = False, jobs: int = 1,
                  progress: bool = True) -> dict:
    """Write the model inputs of every evaluable sample (module docstring) plus manifest.jsonl, protocol.json and
    skipped.json in out_dir. Returns {"exported": n, "skipped": m}."""
    protocol = protocol or DepthProtocol()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [(str(bench.root), s.subset, s.id, protocol.to_dict(), str(out_dir), frames, as_size(size), crf, lossless)
            for s in select_samples(bench, subsets, ids, protocol.clip_filter)]
    res = run_jobs(_export_one, args, jobs, desc="export depth", progress=progress)
    recs = [r for r in res if "skipped" not in r]
    skipped = [r for r in res if "skipped" in r]
    write_manifest(out_dir, recs, skipped, protocol, ("uid", "subset", "id", "views", "fps", "window_fps",
                                                      "frame_indices", "dir"))
    return {"exported": len(recs), "skipped": len(skipped)}


def oracle_maps(sample, view: int, protocol: DepthProtocol) -> np.ndarray:
    """GT depth maps (n, h, w) float32 on the eval grid (0 = no GT): the dense GT where the sample has depth, else the
    visible default_gt track points splatted at their nearest eval pixel with a z-buffer (Real-Robot)."""
    win = protocol.window_indices
    size = protocol.eval_size_for(sample.subset)
    if sample.has_depth:
        return gt_depth(sample, view, win, size)
    w, h = size
    out = np.full((len(win), h, w), np.inf, np.float32)
    obs = track_observations(sample, view, win, protocol.sparse_visibility)
    imap = image_map(sample.image_size(view), size, "resize")
    for j in range(len(win)):
        m = obs.mask[j]
        if not m.any():
            continue
        uv = imap.apply_uv(obs.uv[j, m])
        iu = np.clip(np.floor(uv[:, 0] + 0.5).astype(np.int64), 0, w - 1)
        iv = np.clip(np.floor(uv[:, 1] + 0.5).astype(np.int64), 0, h - 1)
        np.minimum.at(out[j], (iv, iu), obs.z[j, m].astype(np.float32))
    out[~np.isfinite(out)] = 0.0
    return out


def write_oracle(sample, pred_root, protocol: DepthProtocol) -> None:
    for k in protocol.views_for(sample):
        p = prediction_path(pred_root, sample, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, depth=oracle_maps(sample, k, protocol))


def write_constant(sample, pred_root, protocol: DepthProtocol, value: float = 1.0, size=(8, 8)) -> None:
    """Constant depth `value` (metres) on a tiny (n, 8, 8) grid (any resolution is accepted)."""
    n = protocol.horizon + 1
    for k in protocol.views_for(sample):
        p = prediction_path(pred_root, sample, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, depth=np.full((n, size[1], size[0]), value, np.float32))


BASELINES = {"oracle": write_oracle, "constant": write_constant}


def run_baseline(bench, pred_root=None, kind: str = "oracle", protocol: Optional[DepthProtocol] = None, subsets=None,
                 ids=None, jobs: int = 1, progress: bool = True, **opts) -> dict:
    """Write the predictions of baseline `kind` ("oracle" | "constant") for every evaluable sample; `opts` go to the
    baseline (e.g. value=2.0 for "constant"). The call form of every task, run_baseline(bench, pred_root,
    kind="constant", ...); the earlier run_baseline("constant", bench, pred_root, ...) is still accepted."""
    bench, pred_root, kind = baseline_args(bench, pred_root, kind, opts)
    return run_baseline_task(kind, BASELINES, protocol or DepthProtocol(), DepthProtocol, bench, pred_root, subsets,
                             ids, jobs, progress, **opts)


def main(argv=None) -> int:
    """`vpmwb depth {export,baseline,evaluate,summarize}` (`common.task_cli`)."""
    return task_cli(argv, TASK, DepthProtocol, export_fn=export_inputs, evaluate_fn=evaluate, summarize_fn=summarize,
                    baseline_fn=run_baseline, baselines=BASELINES, default_baseline="oracle",
                    description="video depth estimation per view (docs/TASKS.md)")


if __name__ == "__main__":
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
