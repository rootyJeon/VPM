"""Task `recon4d`: multi-view 4D reconstruction and camera estimation from the multi-view video.

Task
    From the V synchronised videos of a sample on the window t = ref_frame + j * frame_step, j = 0..horizon (default
    frames 16, 20, ..., 80: 17 frames spanning the generation window), reconstruct the dynamic scene in ONE world frame
    that is consistent over views and time: per window frame and view a world point map (the 3D point seen at every
    pixel), and/or the cameras (world-to-camera pose of every view at every window frame). The prediction's world is
    arbitrary (any similarity of the GT world): every metric is alignment-free or computed after a fitted similarity.

Inputs (`export_inputs`), per evaluable sample, in <out>/<Subset>/<id>/
    view{k}.mp4 | view{k}/frame_{t:04d}.png | nothing   window frames of every view (as `tasks.depth`; frames="mp4"
                                                         default, "png", "none")
    inputs.json      frame indices, fps, per view the exported size and, by default, the intrinsics K at that size
                     (`provide_intrinsics=True`: calibrated setting; False = uncalibrated, K not exported); with
                     `provide_cameras=True` also the GT w2c (V, n, 4, 4) (known-pose setting: camera metrics are then
                     not reported, the task becomes multi-view depth / point-map estimation)
    <out>/manifest.jsonl, protocol.json, skipped.json

Prediction format: <pred_root>/<Subset>/<id>/recon4d.npz (np.savez_compressed) with
    'points'  (n, V, H, W, 3) world point maps, float32/64, NaN = no point;            - and / or -
    'depth'   (n, V, H, W) planar depth (non-finite or <= 0 = no point), used when 'points' is absent: unprojected with
              the predicted 'K' / 'w2c' when given, otherwise with the GT intrinsics (on the prediction grid) / the
              GT cameras;
    'w2c'     optional (V, n, 4, 4), or (V, 4, 4) for a static rig: world-to-camera poses (OpenCV axes) in the
              prediction's world. Enables the camera metrics. A prediction may hold only 'w2c' (camera estimation);
    'K'       optional (V, 3, 3) or (V, n, 3, 3): intrinsics on the prediction grid (focal error reported);
    'frame_indices' optional (n,) GT frame of every map (else n = window length, or the clip length T).
    n = window frames, V = evaluated views in order (all views by default), H x W = any resolution. Every map covers the
    full frame of its view (the 'resize' pixel map of geometry.image_map, pixel centres at integers). A missing file
    makes the sample "missing"; an unreadable or malformed one "invalid" (common.STATUSES). Maps whose aspect is the
    TRANSPOSED aspect of a view's video (W and H swapped) are scored as given but flagged in the per-sample "warnings"
    and in the summary.

Ground truth
    cameras  cameras.npz (every subset; exact for Simul-*, calibrated for Real-*; Real-Robot's ego extrinsic is a manual
             fit, flag ego_extrinsic_manual_fit).
    points   GT depth unprojected with the GT cameras on the evaluation grid (exact pixel-centre nearest resampling):
             Simul-* exact renders, Real-General reconstruction pseudo-GT (holes = no GT); Real-Robot none. Scored
             pixels: GT depth in (min_depth, max_depth) (Simul-Robot 3 m = the manipulation workspace, Simul-General
             30 m, Real-General 8 m) inside the valid field of view.
    tracks   GT 3D track positions (every subset; the only geometric GT of Real-Robot besides the cameras): the
             observations default_gt & visible in a view, kept only if consistent with the dense GT depth where it
             exists (the `tasks.depth` sparse rule |D - z| <= max(a, r z), `track_consistency`, default (2 cm, 5 %) on
             Simul-Robot and (5 cm, 5 %) elsewhere).

Alignment of points (`align`, default "sim3"; "se3", "scale" or "none" for metric predictions in the GT world): one
Umeyama similarity per sample (`align_per="frame"`: one per window frame) fitted on corresponding pixels, i.e. the
predicted point and the GT point of the same evaluation-grid pixel with valid GT depth (Simul-*, Real-General) or, for
samples without GT depth (Real-Robot) or with align_on="tracks", the predicted point at the GT track's pixel and the
GT track position. The fit is made robust (`robust_similarity`, `align_robust=True`): predicted points farther than
10x the median distance from their median are not used, and the least-squares fit is refined 3 times on the 90 % of
correspondences with the smallest residuals (deterministic; the metrics still use every point). At most
`max_align_points` correspondences (seeded subsample). A prediction that cannot be aligned (fewer than 3 finite
correspondences) scores as a failure: points acc / comp / chamfer inf, precision / recall / F-score 0, coverage 0;
depth abs_rel / rmse inf, delta1 0, coverage 0; tracks delta 0, errors inf; and "align/failed_pct" = 100.

Metrics (per sample; summary = mean +- s.e. over samples per subset; inf failures are counted, not averaged)
    camera   (predicted 'w2c', cameras not provided)  metrics.pose:
             rra@5/15/30, rta@5/15/30 [%], auc@3/5/15/30 [%], rre/rte mean [deg] over the view pairs of every window
             frame (`rig_pairs(.., "same_frame")`, PoseDiffusion convention; alignment-free);
             rpe_trans_rmse [m] / rpe_rot_rmse_deg between consecutive window frames after the Sim(3) scale;
             ate_rmse / ate_mean / ate_median [m] of the camera centres after a Sim(3) fitted on them - summarised
             for every sample whose GT camera centres move in the window (Simul-General; the 27 Simul-Robot samples
             with moving cameras pan / tilt in place), whatever the prediction (a collapsed or collinear predicted
             trajectory gets its large ATE: with coincident predicted centres the best similarity maps them all to the
             GT centroid);
             are_mean_deg (orientation error after that alignment) - summarised for samples whose GT cameras move or
             rotate AND whose GT centres fix the alignment rotation (>= 3 non-collinear centres; a static 2-view rig,
             Real-Robot, does not); when the PREDICTED centres do not fix it (coincident or collinear) the rotation of
             the alignment is undefined and ARE scores the maximum, 180 deg (flag pred_degenerate);
             coverage [%] = poses with a finite prediction; focal_rel_err [%] when K is predicted. The per-sample files
             keep every value with the flags centres_move / cameras_move / gt_degenerate / pred_degenerate and the GT
             travel / rotation.
    points   (GT depth)  per window frame the V views' points are fused and compared with metrics.pointcloud (no
             voxelisation; clouds above `max_points` = 100k points are subsampled with a fixed seed, which bounds the
             cost of exact nearest-neighbour queries of far outliers): acc / comp mean and median,
             chamfer_mean = (acc + comp) / 2 [m],
             precision / recall / fscore@tau [%] (Simul-Robot 1/2/5 cm, Simul-General 10/25/50 cm, Real-General
             2/5/10 cm), coverage = % of GT pixels with a predicted point; means over the window frames.
    depth    (GT depth)  camera z of the aligned points in the GT cameras vs the GT depth (metrics.depth, no further
             alignment): abs_rel, delta1 [%], rmse [m], coverage - the multi-view scale consistency of the
             reconstruction.
    tracks   predicted point at the track's pixel (`track_interp`: nearest pixel of the prediction grid by default;
             "bilinear" interpolates the point map, NaN if a neighbour has no point, which blends foreground and
             background at depth edges and loses sparse maps: the oracle drops to 86 % on Simul-General and ~1 % on
             Real-Robot's splatted points) vs the GT position, pooled over the window frames and views: err_mean /
             err_median [m], delta@tau and delta_avg [%] over the subset's delta ladder (presets: Simul-Robot 1-16 cm,
             Simul-General 10-160 cm, Real-Robot 1-20 cm, Real-General 5-40 cm; an observation without a prediction
             counts as a miss), coverage; the same for the dynamic points (GT path over the window frames
             ref_frame..last_frame >= 10 cm) and the static points (< 1 cm). The path sums the steps between
             consecutive valid GT positions of the track, bridging gaps (`common.track_path_length`); a track with
             fewer than 2 valid positions in the window is in neither split.

Default protocol (`Recon4DProtocol`): ref_frame 16, horizon 16, frame_step 4 (17 frames), all views, evaluation grids
Simul-* 256x256, Real-Robot 320x180, Real-General 256x192, provide_intrinsics=True, provide_cameras=False, sim3,
track_interp nearest. Per-subset fields (eval_size, max_depth, fscore_m, track_ladder_m, voxel_size,
track_consistency) accept {subset or alias: value}.

Resolution floors: a perfect point map at the evaluation grid (the oracle) does not reach 100 % on the track metric
because a GT point is compared with the point of the pixel it falls in (up to half a pixel away, and up to the
consistency tolerance off the surface). Default protocol, 6 samples per subset: tracks delta_avg 90.6 % Simul-Robot
(1-2 cm rungs at 256 px), 96.5 % Simul-General, 99.5 % Real-Robot, 99.6 % Real-General. Higher-resolution predictions
sample closer to the GT point.

Baselines (`BASELINES`, sanity checks)
    oracle   GT point maps (GT depth unprojected on the eval grid; GT track points splatted with a z-buffer where there
             is no GT depth, i.e. everywhere on Real-Robot) + GT cameras + GT K: exact camera / point / depth metrics;
             the track metric shows the resolution floor above
    frozen   the oracle of the first window frame repeated for every frame (static-scene assumption): exact where the
             scene does not change, wrong where it moves (dynamic tracks, moving cameras). A point map has no point
             identity: a point gliding over an unchanged surface is not detected by the track metric

Usage
    from vpm_worldbench.tasks import recon4d
    recon4d.export_inputs(bench, "inputs/recon4d")
    recon4d.evaluate(bench, "preds/my_model", "results/recon4d/my_model", jobs=8)
    recon4d.run_baseline(bench, "preds/frozen", kind="frozen")
    vpmwb recon4d baseline --kind oracle --root /data/VP-WorldBench --pred preds/oracle
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import SUBSETS
from ..geometry import camera_center, image_map
from ..metrics import depth as dm
from ..metrics import pointcloud as pcm
from ..metrics import pose as pm
from ..presets import DYNAMIC_MIN_M, STATIC_MAX_M, SUBSET_DEFAULTS
from .common import (InvalidPrediction, MissingPrediction, TaskProtocol, as_size, baseline_args, evaluate_task,
                     export_view_frames, fov_mask, frame_index_array, nearest_index, read_gt_depth, read_npz,
                     run_baseline_task, run_jobs, sample_map, select_samples, summarize_records, task_cli,
                     track_observations, track_path_length, transposed_hint, write_json, write_manifest)
from .depth import SPARSE_TOL

__all__ = ["Recon4DProtocol", "export_inputs", "evaluate", "evaluate_sample", "summarize", "load_prediction",
           "ReconPrediction", "robust_similarity", "camera_metrics", "write_oracle", "write_frozen", "run_baseline",
           "BASELINES", "ARE_FAIL_DEG", "main"]

TASK = "recon4d"
PRED_FILE = "recon4d.npz"
RECON_EVAL_SIZE = {"Simul-Robot": (256, 256), "Simul-General": (256, 256), "Real-Robot": (320, 180),
                   "Real-General": (256, 192)}
RECON_MAX_DEPTH = {"Simul-Robot": 3.0, "Simul-General": 30.0, "Real-Robot": None, "Real-General": 8.0}
RECON_FSCORE_M = {"Simul-Robot": (0.01, 0.02, 0.05), "Simul-General": (0.10, 0.25, 0.50), "Real-Robot": (),
                  "Real-General": (0.02, 0.05, 0.10)}
TRACK_LADDER_M = {s: tuple(SUBSET_DEFAULTS[s]["delta_ladder_m"]) for s in SUBSETS}
ALIGN_MODES = pcm.ALIGN_MODES
ARE_FAIL_DEG = 180.0


@dataclass
class Recon4DProtocol(TaskProtocol):
    """Protocol of the recon4d task (module docstring). Per-subset dicts ({subset or alias: value}; missing subsets use
    the defaults): eval_size (W, H), max_depth metres or None, fscore_m thresholds, track_ladder_m thresholds,
    voxel_size metres (or one number for all), track_consistency (abs m, rel) or None (or one pair / None for all)."""
    SUBSET_FIELDS = ("eval_size", "max_depth", "fscore_m", "track_ladder_m", "voxel_size", "track_consistency")

    name: str = "recon4d-v1"
    horizon: int = 16
    frame_step: int = 4
    eval_size: dict = field(default_factory=dict)
    provide_intrinsics: bool = True
    provide_cameras: bool = False
    align: str = "sim3"
    align_on: str = "auto"
    align_per: str = "sample"
    max_align_points: int = 1_000_000
    align_robust: bool = True
    min_depth: float = dm.DEFAULT_MIN_DEPTH
    max_depth: dict = field(default_factory=dict)
    fscore_m: dict = field(default_factory=dict)
    track_ladder_m: dict = field(default_factory=dict)
    track_visibility: str = "visible"
    track_consistency: object = field(default_factory=dict)
    track_interp: str = "nearest"
    dynamic_min_m: float = DYNAMIC_MIN_M
    static_max_m: float = STATIC_MAX_M
    camera_convention: str = "relative"
    moving_min_m: float = 0.01
    moving_min_deg: float = 0.5
    voxel_size: object = field(default_factory=dict)
    max_points: Optional[int] = 100_000
    use_fov_mask: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.align not in ALIGN_MODES:
            raise ValueError(f"align must be one of {ALIGN_MODES}")
        if self.align_on not in ("auto", "depth", "tracks"):
            raise ValueError("align_on must be 'auto', 'depth' or 'tracks'")
        if self.align_per not in ("sample", "frame"):
            raise ValueError("align_per must be 'sample' or 'frame'")
        if self.track_visibility not in ("visible", "in_frustum"):
            raise ValueError("track_visibility must be 'visible' or 'in_frustum'")
        if self.track_interp not in ("nearest", "bilinear"):
            raise ValueError("track_interp must be 'nearest' or 'bilinear'")
        if self.camera_convention not in pm.CONVENTIONS:
            raise ValueError(f"camera_convention must be one of {pm.CONVENTIONS}")

    def eval_size_for(self, subset):
        return as_size(self.per_subset("eval_size", subset, RECON_EVAL_SIZE))

    def max_depth_for(self, subset):
        v = self.per_subset("max_depth", subset, RECON_MAX_DEPTH)
        return None if v is None else float(v)

    def fscore_for(self, subset):
        return tuple(float(x) for x in self.per_subset("fscore_m", subset, RECON_FSCORE_M))

    def ladder_for(self, subset):
        return tuple(float(x) for x in self.per_subset("track_ladder_m", subset, TRACK_LADDER_M))

    def voxel_for(self, subset):
        v = self.per_subset("voxel_size", subset, {s: None for s in SUBSETS}) if isinstance(self.voxel_size, dict) \
            else self.voxel_size
        return None if v is None else float(v)

    def tolerance_for(self, subset) -> Optional[tuple]:
        c = self.track_consistency
        v = self.per_subset("track_consistency", subset, SPARSE_TOL) if isinstance(c, dict) else c
        return None if v is None else (float(v[0]), float(v[1]))


def prediction_path(pred_root, sample) -> Path:
    return Path(pred_root) / sample.subset / sample.id / PRED_FILE


class ReconPrediction:
    """A loaded recon4d prediction restricted to the window frames and the evaluated views.

    points (n, V, H, W, 3) or None, depth (n, V, H, W) or None, w2c (V, n, 4, 4) float64 or None, K (V, n, 3, 3) float64
    or None, size (W, H) of the maps (None for a cameras-only prediction)."""

    def __init__(self, path, sample, protocol: Recon4DProtocol):
        path = Path(path)
        if not path.exists():
            raise MissingPrediction(f"missing prediction {path}")
        win = protocol.window_indices
        views = protocol.views_for(sample)
        n, V = len(win), len(views)
        files, z = read_npz(path, ("points", "depth", "w2c", "K", "frame_indices"))
        if not set(z) & {"points", "depth", "w2c"}:
            raise InvalidPrediction(f"{path}: needs 'points', 'depth' and/or 'w2c', has {sorted(files)}")
        for k, a in z.items():
            if k != "frame_indices" and (a.dtype == bool or not (np.issubdtype(a.dtype, np.integer)
                                                                  or np.issubdtype(a.dtype, np.floating))):
                raise InvalidPrediction(f"{path}: '{k}' must hold real numbers, got dtype {a.dtype}")
        fi = frame_index_array(z["frame_indices"], path) if "frame_indices" in z else None
        self.points = z.get("points")
        self.depth = z.get("depth") if self.points is None else None
        w2c = np.asarray(z["w2c"], np.float64) if "w2c" in z else None
        K = np.asarray(z["K"], np.float64) if "K" in z else None
        sel = self._time_index(fi, win, path)
        T = sample.n_frames

        def pick(a, axis, name):
            """Window frames along `axis`: via frame_indices, or the window length, or the clip length."""
            L = a.shape[axis]
            if sel is not None:
                if L != len(fi):
                    raise InvalidPrediction(f"{path}: '{name}' has {L} frames, frame_indices {len(fi)}")
                return np.take(a, sel, axis=axis)
            if L == n:
                return a
            if L == T:
                return np.take(a, win, axis=axis)
            raise InvalidPrediction(f"{path}: '{name}' has {L} frames; expected the window ({n}), the clip ({T}) or "
                                    "a 'frame_indices' array")

        if self.points is not None:
            if self.points.ndim != 5 or self.points.shape[-1] != 3 or self.points.shape[1] != V:
                raise InvalidPrediction(f"{path}: 'points' must be (n, {V}, H, W, 3), got {self.points.shape}")
            self.points = pick(self.points, 0, "points")
        if self.depth is not None:
            if self.depth.ndim != 4 or self.depth.shape[1] != V:
                raise InvalidPrediction(f"{path}: 'depth' must be (n, {V}, H, W), got {self.depth.shape}")
            self.depth = pick(self.depth, 0, "depth")
        maps = self.points if self.points is not None else self.depth
        self.size = None if maps is None else (int(maps.shape[3]), int(maps.shape[2]))
        if w2c is not None:
            if w2c.shape == (V, 4, 4):
                w2c = np.repeat(w2c[:, None], n, 1)
            elif w2c.ndim == 4 and w2c.shape[0] == V and w2c.shape[2:] == (4, 4):
                w2c = pick(w2c, 1, "w2c")
            else:
                raise InvalidPrediction(f"{path}: 'w2c' must be ({V}, n, 4, 4) or ({V}, 4, 4), got {w2c.shape}")
        if K is not None:
            if K.shape == (V, 3, 3):
                K = np.repeat(K[:, None], n, 1)
            elif K.ndim == 4 and K.shape[0] == V and K.shape[2:] == (3, 3):
                K = pick(K, 1, "K")
            else:
                raise InvalidPrediction(f"{path}: 'K' must be ({V}, 3, 3) or ({V}, n, 3, 3), got {K.shape}")
        self.w2c, self.K = w2c, K
        self._unproj_K = self._unproj_w2c = None
        if self.depth is not None:
            self._unproj_K = K if K is not None else np.repeat(
                np.stack([sample.K(v, self.size) for v in views])[:, None], n, 1)
            self._unproj_w2c = w2c if w2c is not None else np.asarray(sample.cameras.w2c[views][:, win], np.float64)
            self._unproj_c2w = pm.invert_rigid(self._unproj_w2c)

    @staticmethod
    def _time_index(fi, win, path):
        if fi is None:
            return None
        pos = {int(t): i for i, t in enumerate(fi)}
        miss = [t for t in win if t not in pos]
        if miss:
            raise InvalidPrediction(f"{path}: window frames {miss[:5]} missing from frame_indices")
        return np.array([pos[t] for t in win])

    @property
    def has_points(self) -> bool:
        return self.points is not None or self.depth is not None

    def _unproject(self, j: int, v: int, d: np.ndarray, x_px, y_px) -> np.ndarray:
        """World points of depth d (NaN / <= 0 = none) at prediction-grid pixel positions (x_px, y_px) (floats)."""
        K = self._unproj_K[v, j]
        with np.errstate(invalid="ignore"):
            d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
        x_px, y_px = np.broadcast_arrays(np.asarray(x_px, np.float64), np.asarray(y_px, np.float64))
        y = (y_px - K[1, 2]) / K[1, 1]
        x = (x_px - K[0, 2] - K[0, 1] * y) / K[0, 0]
        Xc = np.stack([x * d, y * d, d], -1)
        c2w = self._unproj_c2w[v, j]
        return Xc @ c2w[:3, :3].T + c2w[:3, 3]

    def points_at(self, j: int, v: int, iy, ix) -> np.ndarray:
        """Predicted world points (float64, NaN = none) of window frame j, view position v at prediction-grid pixels
        (iy, ix) (broadcastable integer arrays)."""
        if self.points is not None:
            return self.points[j, v][iy, ix].astype(np.float64)
        return self._unproject(j, v, self.depth[j, v][iy, ix].astype(np.float64), ix, iy)

    def points_at_uv(self, j: int, v: int, uv: np.ndarray, interp: str = "nearest") -> np.ndarray:
        """Predicted world points (M, 3) at sub-pixel positions uv (M, 2) of the prediction grid: the nearest pixel, or
        bilinear interpolation of the point map (of the depth map, then unprojected at uv) that is NaN when any
        neighbour with non-zero weight has no point (`common.sample_map`)."""
        W, H = self.size
        if interp == "nearest":
            ix = np.clip(np.floor(uv[:, 0] + 0.5).astype(np.int64), 0, W - 1)
            iy = np.clip(np.floor(uv[:, 1] + 0.5).astype(np.int64), 0, H - 1)
            return self.points_at(j, v, iy, ix)
        if self.points is not None:
            return sample_map(self.points[j, v], uv, "bilinear")
        dmap = self.depth[j, v].astype(np.float64)
        with np.errstate(invalid="ignore"):
            dmap = np.where(np.isfinite(dmap) & (dmap > 0), dmap, np.nan)
        return self._unproject(j, v, sample_map(dmap, uv, "bilinear"), uv[:, 0], uv[:, 1])


def load_prediction(pred_root, sample, protocol: Recon4DProtocol) -> ReconPrediction:
    return ReconPrediction(prediction_path(pred_root, sample), sample, protocol)


class _GT:
    """Per-sample GT on the evaluation grid: depth (n, V, h, w), valid mask, world points (n, V, h, w, 3) (NaN where not
    scored), K on the grid (V, 3, 3), w2c (V, n, 4, 4), and the track observations (consistent mask) per view."""

    def __init__(self, sample, protocol: Recon4DProtocol):
        self.win = protocol.window_indices
        self.views = protocol.views_for(sample)
        self.size = protocol.eval_size_for(sample.subset)
        w, h = self.size
        n, V = len(self.win), len(self.views)
        self.max_depth = protocol.max_depth_for(sample.subset)
        self.w2c = np.asarray(sample.cameras.w2c[self.views][:, self.win], np.float64)
        self.K = np.stack([sample.K(v, self.size) for v in self.views])
        self.obs, self.track_mask = [], []
        self.has_depth = sample.has_depth
        self.depth = np.zeros((n, V, h, w), np.float32) if sample.has_depth else None
        self.valid = np.zeros((n, V, h, w), bool) if sample.has_depth else None
        tol = protocol.tolerance_for(sample.subset)
        for i, v in enumerate(self.views):
            fov_n = fov_mask(sample, v) if protocol.use_fov_mask else None
            obs = track_observations(sample, v, self.win, protocol.track_visibility, fov=fov_n)
            d, keep = read_gt_depth(sample, v, self.win, self.size, obs, tol)
            self.obs.append(obs)
            self.track_mask.append(keep)
            if d is not None:
                self.depth[:, i] = d
                ok = (d > protocol.min_depth) & np.isfinite(d)
                if self.max_depth is not None:
                    ok &= d < self.max_depth
                fov = fov_mask(sample, v, self.size) if protocol.use_fov_mask else None
                if fov is not None:
                    ok &= fov[None]
                self.valid[:, i] = ok
        self.points = None
        if self.has_depth:
            u, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            self.points = np.full((n, V, h, w, 3), np.nan)
            for i in range(V):
                K = self.K[i]
                y = (vv - K[1, 2]) / K[1, 1]
                x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
                for j in range(n):
                    d = np.where(self.valid[j, i], self.depth[j, i].astype(np.float64), np.nan)
                    c2w = pm.invert_rigid(self.w2c[i, j])
                    Xc = np.stack([x * d, y * d, d], -1)
                    self.points[j, i] = Xc @ c2w[:3, :3].T + c2w[:3, 3]


def _pred_grid_points(pred: ReconPrediction, gt: _GT) -> np.ndarray:
    """Predicted points resampled (exact nearest) onto the evaluation grid: (n, V, h, w, 3) float64."""
    w, h = gt.size
    W, H = pred.size
    iy, ix = nearest_index(H, h)[:, None], nearest_index(W, w)[None, :]
    n, V = len(gt.win), len(gt.views)
    out = np.empty((n, V, h, w, 3))
    for j in range(n):
        for i in range(V):
            out[j, i] = pred.points_at(j, i, iy, ix)
    return out


def _track_pairs(sample, pred: ReconPrediction, gt: _GT, interp: str):
    """Predicted points at the GT track pixels (prediction grid, `interp`) and the GT positions, per observation:
    (pred (M, 3) NaN where no prediction, gt (M, 3), frame j (M,), point index (M,), view position (M,))."""
    W, H = pred.size
    P, G, J, N, Vp = [], [], [], [], []
    for i, v in enumerate(gt.views):
        obs, keep = gt.obs[i], gt.track_mask[i]
        imap = image_map(sample.image_size(v), (W, H), "resize")
        for j in range(len(gt.win)):
            m = keep[j]
            if not m.any():
                continue
            P.append(pred.points_at_uv(j, i, imap.apply_uv(obs.uv[j, m]), interp))
            G.append(obs.xyz[j, m])
            idx = np.flatnonzero(m)
            J.append(np.full(len(idx), j))
            N.append(idx)
            Vp.append(np.full(len(idx), i))
    if not P:
        z = np.zeros((0, 3))
        e = np.zeros(0, np.int64)
        return z, z, e, e, e
    return np.concatenate(P), np.concatenate(G), np.concatenate(J), np.concatenate(N), np.concatenate(Vp)


def _camera_motion(gt_w2c: np.ndarray) -> tuple:
    """(max centre travel [m], max rotation [deg]) of the GT cameras (V, n, 4, 4) relative to the first window frame."""
    c = camera_center(gt_w2c)
    travel = float(np.linalg.norm(c - c[:, :1], axis=-1).max())
    rot = float(pm.rotation_error_deg(gt_w2c[:, :1, :3, :3], gt_w2c[:, :, :3, :3]).max())
    return travel, rot


def camera_metrics(sample, pred: ReconPrediction, gt: _GT, protocol: Recon4DProtocol) -> dict:
    """metrics.pose.pose_metrics of the predicted cameras + the GT motion / degeneracy flags (module docstring).
    gt_degenerate: the GT camera centres do not fix the rotation of a similarity (< 3 non-collinear centres);
    pred_degenerate: the GT centres do, the predicted centres (coincident or collinear) do not."""
    V, n = gt.w2c.shape[:2]
    pairs = pm.rig_pairs(V, n, "same_frame")
    out = pm.pose_metrics(pred.w2c, gt.w2c, align="sim3", pairs=pairs, rpe_delta=1 if n > 1 else None,
                          convention=protocol.camera_convention)
    travel, rot = _camera_motion(gt.w2c)
    gt_deg = bool(pm.ate(gt.w2c, gt.w2c, align="sim3")["degenerate"])
    out.update(gt_travel_m=travel, gt_rotation_deg=rot, centres_move=bool(travel > protocol.moving_min_m),
               cameras_move=bool(travel > protocol.moving_min_m or rot > protocol.moving_min_deg), n_views=int(V),
               gt_degenerate=gt_deg, pred_degenerate=bool(out["degenerate"]) and not gt_deg,
               coverage=100.0 * float(np.isfinite(pred.w2c).all((-2, -1)).mean()))
    if pred.K is not None and pred.size is not None:
        Kg = np.stack([sample.K(v, pred.size) for v in gt.views])[:, None]
        err = np.abs(pred.K[..., [0, 1], [0, 1]] / Kg[..., [0, 1], [0, 1]] - 1.0)
        out["focal_rel_err"] = 100.0 * float(np.mean(err))
    return out


def robust_similarity(P: np.ndarray, G: np.ndarray, mode: str = "sim3", max_n: int = 1_000_000, robust: bool = True,
                      gate: float = 10.0, trim: float = 0.9, iters: int = 3, seed: int = 0):
    """Similarity (s, R, t) with G ~ s R P + t from correspondences P[i] <-> G[i] ((M, 3); rows non-finite in either
    are ignored), `metrics.pointcloud.fit_similarity` (Umeyama) inside a deterministic robust loop:
      1. at most max_n correspondences (seeded subsample);
      2. gate (scale-free): drop predicted points farther than gate x the median distance from their coordinate-wise
         median (e.g. sky / far background predicted at a pixel whose GT is a nearby surface);
      3. least-squares fit, then `iters` refits on the `trim` fraction of correspondences with the smallest residuals.
    robust=False is the plain least-squares fit (steps 1 and 3 without trimming). Returns (s, R, t, n_used); raises
    ValueError when there are too few correspondences for `mode`."""
    ok = np.isfinite(P).all(-1) & np.isfinite(G).all(-1)
    P, G = P[ok], G[ok]
    if len(P) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(P), max_n, replace=False))
        P, G = P[idx], G[idx]
    if robust and mode != "none" and len(P) >= 3:
        d = np.linalg.norm(P - np.median(P, 0), axis=1)
        md = float(np.median(d))
        if md > 0:
            keep = d <= gate * md
            P, G = P[keep], G[keep]
    s, R, t = pcm.fit_similarity(P, G, mode)
    n_used = len(P)
    if robust and mode != "none":
        for _ in range(int(iters)):
            r = np.linalg.norm(pcm.apply_similarity(P, s, R, t) - G, axis=1)
            keep = r <= np.quantile(r, trim)
            if keep.sum() < 3 or keep.all():
                break
            s, R, t = pcm.fit_similarity(P[keep], G[keep], mode)
            n_used = int(keep.sum())
    return float(s), R, t, int(n_used)


def _fail_points(thresholds) -> dict:
    out = {k: float("inf") for k in ("acc_mean", "acc_median", "comp_mean", "comp_median", "chamfer_mean",
                                     "chamfer_median", "chamfer_sum")}
    for tau in thresholds:
        out.update({f"precision@{tau:g}": 0.0, f"recall@{tau:g}": 0.0, f"fscore@{tau:g}": 0.0})
    out["coverage"] = 0.0
    return out


_FAIL_DEPTH = {"abs_rel": float("inf"), "rmse": float("inf"), "delta1": 0.0, "coverage": 0.0}


def _track_block(err: np.ndarray, ladder) -> dict:
    """err (M,) metres with NaN = no prediction (counts as a miss; errors are inf when no observation has one)."""
    M = len(err)
    fin = err[np.isfinite(err)]
    none = float("nan") if not M else float("inf")
    out = {"n_obs": int(M), "coverage": 100.0 * len(fin) / M if M else float("nan"),
           "err_mean": float(fin.mean()) if len(fin) else none,
           "err_median": float(np.median(fin)) if len(fin) else none}
    deltas = []
    for tau in ladder:
        d = 100.0 * float(np.sum(fin < tau)) / M if M else float("nan")
        out[f"delta@{tau:g}"] = d
        deltas.append(d)
    out["delta_avg"] = float(np.mean(deltas)) if deltas and M else float("nan")
    return out


def _camera_flat(cam: dict) -> dict:
    """The summarised camera metrics of one sample (module docstring: ATE / ARE gating)."""
    flat = {}
    for k in [k for k in cam if k.startswith(("rra@", "rta@", "auc@"))] + [
            "rre_mean_deg", "rte_mean_deg", "rpe_trans_rmse", "rpe_rot_rmse_deg", "focal_rel_err", "coverage"]:
        if k in cam:
            flat[f"camera/{k}"] = cam[k]
    if cam["centres_move"]:
        for k in ("ate_rmse", "ate_mean", "ate_median"):
            flat[f"camera/{k}"] = cam[k]
    if cam["cameras_move"] and not cam["gt_degenerate"]:
        flat["camera/are_mean_deg"] = ARE_FAIL_DEG if cam["pred_degenerate"] else cam["are_mean_deg"]
    return flat


def evaluate_sample(sample, pred_root, protocol: Recon4DProtocol) -> dict:
    """All recon4d metrics of one sample (module docstring). Returns {"status": "ok", "views", "prediction",
    "camera", "align", "points", "depth", "tracks", "metrics"}; "metrics" is the flat dict that is summarised; plus
    "warnings" (list) when the maps have the transposed aspect of a view's video (`common.transposed_hint`)."""
    pred = load_prediction(pred_root, sample, protocol)
    gt = _GT(sample, protocol)
    rec = {"status": "ok", "views": gt.views, "eval_size": list(gt.size),
           "prediction": {"points": pred.points is not None, "depth": pred.depth is not None,
                          "w2c": pred.w2c is not None, "K": pred.K is not None,
                          "size": None if pred.size is None else list(pred.size)},
           "camera": None, "align": None, "points": None, "depth": None, "tracks": None}
    if pred.size is not None:
        warn = [f"view{v}: {h}" for v in gt.views for h in [transposed_hint(pred.size, sample.image_size(v))] if h]
        if warn:
            rec["warnings"] = warn
    flat = {}
    if pred.w2c is not None and not protocol.provide_cameras:
        cam = camera_metrics(sample, pred, gt, protocol)
        rec["camera"] = cam
        flat.update(_camera_flat(cam))
    if not pred.has_points:
        rec["metrics"] = flat
        return rec
    n = len(gt.win)
    on = protocol.align_on
    if on == "auto":
        on = "depth" if gt.has_depth else "tracks"
    if on == "depth" and not gt.has_depth:
        raise ValueError(f"{sample.uid} has no GT depth: use align_on='tracks'")
    Pg = _pred_grid_points(pred, gt) if gt.has_depth else None
    tP, tG, tJ, tN, tV = _track_pairs(sample, pred, gt, protocol.track_interp)
    if on == "depth":
        src = lambda j: (Pg[j][gt.valid[j]], gt.points[j][gt.valid[j]])
    else:
        src = lambda j: (tP[tJ == j], tG[tJ == j])
    frames = [list(range(n))] if protocol.align_per == "sample" else [[j] for j in range(n)]
    transforms = [None] * n
    fits = []
    for group in frames:
        P = np.concatenate([src(j)[0] for j in group]) if group else np.zeros((0, 3))
        G = np.concatenate([src(j)[1] for j in group]) if group else np.zeros((0, 3))
        try:
            s, R, t, m = robust_similarity(P, G, protocol.align, protocol.max_align_points, protocol.align_robust)
            fits.append({"frames": group if protocol.align_per == "frame" else "all", "scale": s, "n": m,
                         "failed": False})
            for j in group:
                transforms[j] = (s, R, t)
        except ValueError as e:
            fits.append({"frames": group if protocol.align_per == "frame" else "all", "failed": True, "error": str(e)})
    rec["align"] = {"mode": protocol.align, "on": on, "per": protocol.align_per, "fits": fits}
    flat["align/failed_pct"] = 100.0 * float(np.mean([f["failed"] for f in fits]))

    def aligned(X, j):
        tr = transforms[j]
        if tr is None:
            return np.full(X.shape, np.nan)
        return pcm.apply_similarity(X, *tr)

    if gt.has_depth:
        thr = protocol.fscore_for(sample.subset)
        per_frame, z_pred, z_gt, z_mask = [], [], [], []
        for j in range(n):
            if not gt.valid[j].any():
                continue
            Pa = aligned(Pg[j], j)
            if transforms[j] is None:
                per_frame.append(_fail_points(thr))
            else:
                m = pcm.pointcloud_metrics(Pa, gt.points[j], thr, align="none", mask=gt.valid[j],
                                           voxel_size=protocol.voxel_for(sample.subset),
                                           max_points=protocol.max_points, workers=1)
                m["coverage"] = 100.0 * float(np.isfinite(Pa[gt.valid[j]]).all(-1).mean())
                per_frame.append(m)
            for i in range(len(gt.views)):
                w2c = gt.w2c[i, j]
                z_pred.append(Pa[i] @ w2c[2, :3] + w2c[2, 3])
                z_gt.append(gt.depth[j, i])
                z_mask.append(gt.valid[j, i])
        keys = [k for k in (per_frame[0] if per_frame else {})
                if isinstance(per_frame[0][k], (int, float)) and all(k in f for f in per_frame)]
        pts = {k: float(np.mean([f[k] for f in per_frame])) for k in keys if k not in ("n_pred", "n_gt")}
        pts["n_frames"] = len(per_frame)
        rec["points"] = pts
        for k, v in pts.items():
            if k != "n_frames" and k != "chamfer_sum":
                flat[f"points/{k}"] = v
        if z_pred:
            dmet = dm.depth_metrics(np.stack(z_pred), np.stack(z_gt), np.stack(z_mask), min_depth=protocol.min_depth,
                                    max_depth=gt.max_depth, reduce="frames")
            if not dmet["n_scored"]:
                dmet.update(_FAIL_DEPTH, failed=True)
            rec["depth"] = dmet
            for k in ("abs_rel", "delta1", "rmse", "coverage"):
                flat[f"depth/{k}"] = dmet[k]
    if len(tP):
        Pa = np.full(tP.shape, np.nan)
        for j in range(n):
            sel = tJ == j
            if sel.any() and transforms[j] is not None:
                Pa[sel] = aligned(tP[sel], j)
        err = np.linalg.norm(Pa - tG, axis=-1)
        tr = sample.tracks
        a, b = protocol.ref_frame, protocol.last_frame + 1
        path, n_valid = track_path_length(tr.xyz[a:b], tr.valid[a:b])
        moving = n_valid[tN] >= 2
        ladder = protocol.ladder_for(sample.subset)
        trk = {"all": _track_block(err, ladder),
               "dynamic": _track_block(err[moving & (path[tN] >= protocol.dynamic_min_m)], ladder),
               "static": _track_block(err[moving & (path[tN] < protocol.static_max_m)], ladder)}
        rec["tracks"] = trk
        for split, prefix in (("all", "tracks"), ("dynamic", "tracks_dynamic"), ("static", "tracks_static")):
            b_ = trk[split]
            if b_["n_obs"]:
                for k, v in b_.items():
                    flat[f"{prefix}/{k}"] = v
    rec["metrics"] = flat
    return rec


def evaluate(bench, pred_root, out_dir, protocol: Optional[Recon4DProtocol] = None, subsets=None, ids=None,
             jobs: int = 1, overwrite: bool = False, strict: bool = False, progress: bool = True) -> dict:
    """Score <pred_root>/<Subset>/<id>/recon4d.npz; writes <out_dir>/<Subset>/<id>.json per sample (status ok |
    skipped | missing | invalid | error), run.json, protocol.json and summary.json|md (mean +- s.e. over samples per
    subset). Finished samples of the same run are reused while their prediction files are unchanged; another run in
    the same out_dir raises `common.RunDirError` unless overwrite=True (--overwrite). Returns the summary dict."""
    protocol = protocol or Recon4DProtocol()
    return evaluate_task(TASK, evaluate_sample, protocol, Recon4DProtocol, bench, pred_root, out_dir, subsets, ids,
                         jobs, overwrite, strict, progress, summarize_fn=summarize)


def summarize(out_dir) -> dict:
    """(Re)build <out_dir>/summary.json and summary.md from the per-sample JSONs."""
    head = [("camera/rra@5", "camera RRA@5 [%]", 2), ("camera/rta@5", "camera RTA@5 [%]", 2),
            ("camera/rra@15", "camera RRA@15 [%]", 2), ("camera/rta@15", "camera RTA@15 [%]", 2),
            ("camera/auc@30", "camera AUC@30 [%]", 2), ("camera/rre_mean_deg", "camera rel. rot. err [deg]", 3),
            ("camera/rte_mean_deg", "camera rel. transl. dir. err [deg]", 3),
            ("camera/ate_rmse", "camera ATE RMSE [m] (moving centres)", 4),
            ("camera/are_mean_deg", "camera ARE [deg] (moving / rotating cameras)", 3),
            ("camera/rpe_trans_rmse", "camera RPE trans [m]", 4),
            ("camera/rpe_rot_rmse_deg", "camera RPE rot [deg]", 3),
            ("camera/focal_rel_err", "focal error [%]", 2), ("camera/coverage", "camera coverage [%]", 1),
            ("align/failed_pct", "point alignment failures [%]", 1),
            ("points/acc_mean", "points accuracy [m]", 4), ("points/comp_mean", "points completeness [m]", 4),
            ("points/chamfer_mean", "points Chamfer [m]", 4)]
    for sub_thr in sorted({t for v in RECON_FSCORE_M.values() for t in v}):
        head.append((f"points/fscore@{sub_thr:g}", f"points F-score@{sub_thr:g} m [%]", 2))
    head += [("points/coverage", "points coverage [%]", 1), ("depth/abs_rel", "depth AbsRel (aligned)", 4),
             ("depth/delta1", "depth delta1 [%]", 2), ("tracks/err_median", "tracks median error [m]", 4),
             ("tracks/delta_avg", "tracks delta_avg [%]", 2),
             ("tracks_dynamic/delta_avg", "dynamic tracks delta_avg [%]", 2),
             ("tracks_dynamic/err_median", "dynamic tracks median error [m]", 4),
             ("tracks_static/delta_avg", "static tracks delta_avg [%]", 2),
             ("tracks/coverage", "tracks coverage [%]", 1)]
    notes = ["points / depth GT: Simul-* exact renders, Real-General reconstruction pseudo-GT, Real-Robot none "
             "(tracks + cameras only)",
             "ATE over every sample whose GT camera centres move in the window, whatever the prediction; ARE over "
             f"samples whose GT cameras move or rotate and whose GT centres fix the alignment rotation; a prediction "
             f"whose centres do not (coincident / collinear) scores ARE = {ARE_FAIL_DEG:g} deg",
             "points aligned with one Sim(3) per sample (robust Umeyama on corresponding pixels; tracks on Real-Robot);"
             " a failed alignment scores inf distances and 0 %",
             "tracks: predicted point at the GT point's pixel; dynamic = GT path over the window >= 10 cm, static "
             "< 1 cm (gaps in the GT track bridged)",
             "resolution floor: the GT point map at the evaluation grid (oracle) scores tracks delta_avg ~ 90.6 % on "
             "Simul-Robot (1-2 cm rungs at 256 px), 96.5 % on Simul-General, 99.5 % on Real-*"]
    return summarize_records(out_dir, TASK, "VP-WorldBench - recon4d (multi-view 4D reconstruction)", head, notes)


def export_sample(sample, protocol: Recon4DProtocol, out_dir, frames: str = "mp4", size=None, crf: int = 10,
                  lossless: bool = False) -> Optional[dict]:
    ok, _ = protocol.evaluable(sample)
    if not ok:
        return None
    d = Path(out_dir) / sample.subset / sample.id
    d.mkdir(parents=True, exist_ok=True)
    win = protocol.window_indices
    views = protocol.views_for(sample)
    vrec = {}
    for k in views:
        r = export_view_frames(sample, k, win, d, frames, size, "resize", crf, lossless)
        if not protocol.provide_intrinsics:
            r.pop("K")
        if protocol.provide_cameras:
            r["w2c"] = sample.w2c(k, win).tolist()
        vrec[f"view{k}"] = r
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id,
           "views": [f"view{k}" for k in views], "fps": sample.fps, "window_fps": sample.fps / protocol.frame_step,
           "frame_indices": win, "n_frames": len(win), "provide_intrinsics": protocol.provide_intrinsics,
           "provide_cameras": protocol.provide_cameras, "frames": vrec, "dir": f"{sample.subset}/{sample.id}",
           "prediction": {"file": f"{sample.subset}/{sample.id}/{PRED_FILE}",
                          "points": f"({len(win)}, {len(views)}, H, W, 3)",
                          "depth": f"({len(win)}, {len(views)}, H, W)",
                          "w2c": f"({len(views)}, {len(win)}, 4, 4)", "K": f"({len(views)}, 3, 3)"}}
    write_json(d / "inputs.json", rec)
    return rec


def _export_one(root, subset, sid, protocol_dict, out_dir, frames, size, crf, lossless):
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = Recon4DProtocol.from_dict(protocol_dict)
    rec = export_sample(s, p, out_dir, frames, size, crf, lossless)
    return rec if rec is not None else {"uid": s.uid, "skipped": p.evaluable(s)[1]}


def export_inputs(bench, out_dir, protocol: Optional[Recon4DProtocol] = None, subsets=None, ids=None,
                  frames: str = "mp4", size=None, crf: int = 10, lossless: bool = False, jobs: int = 1,
                  progress: bool = True) -> dict:
    """Write the model inputs of every evaluable sample (module docstring) plus manifest.jsonl, protocol.json and
    skipped.json. Returns {"exported": n, "skipped": m}."""
    protocol = protocol or Recon4DProtocol()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [(str(bench.root), s.subset, s.id, protocol.to_dict(), str(out_dir), frames, as_size(size), crf, lossless)
            for s in select_samples(bench, subsets, ids, protocol.clip_filter)]
    res = run_jobs(_export_one, args, jobs, desc="export recon4d", progress=progress)
    recs = [r for r in res if "skipped" not in r]
    skipped = [r for r in res if "skipped" in r]
    write_manifest(out_dir, recs, skipped, protocol, ("uid", "subset", "id", "views", "fps", "window_fps",
                                                      "frame_indices", "dir"))
    return {"exported": len(recs), "skipped": len(skipped)}


def oracle_arrays(sample, protocol: Recon4DProtocol) -> dict:
    """GT reconstruction on the eval grid: points (n, V, h, w, 3) float32 (GT depth unprojected; GT track points
    splatted with a z-buffer at pixels without GT depth), w2c (V, n, 4, 4), K (V, 3, 3)."""
    win = protocol.window_indices
    views = protocol.views_for(sample)
    size = protocol.eval_size_for(sample.subset)
    w, h = size
    n, V = len(win), len(views)
    pts = np.full((n, V, h, w, 3), np.nan, np.float32)
    u, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    for i, v in enumerate(views):
        K = sample.K(v, size)
        d_all = read_gt_depth(sample, v, win, size)[0]
        obs = track_observations(sample, v, win, protocol.track_visibility)
        imap = image_map(sample.image_size(v), size, "resize")
        for j, t in enumerate(win):
            c2w = pm.invert_rigid(sample.cameras.w2c[v, t])
            if d_all is not None:
                d = d_all[j].astype(np.float64)
                d = np.where(d > 0, d, np.nan)
                y = (vv - K[1, 2]) / K[1, 1]
                x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
                Xc = np.stack([x * d, y * d, d], -1)
                pts[j, i] = (Xc @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
            m = obs.mask[j]
            if m.any():
                uv = imap.apply_uv(obs.uv[j, m])
                ix = np.clip(np.floor(uv[:, 0] + 0.5).astype(np.int64), 0, w - 1)
                iy = np.clip(np.floor(uv[:, 1] + 0.5).astype(np.int64), 0, h - 1)
                zbuf = np.full((h, w), np.inf)
                np.minimum.at(zbuf, (iy, ix), obs.z[j, m])
                win_pt = np.isclose(zbuf[iy, ix], obs.z[j, m])
                free = ~np.isfinite(pts[j, i, iy, ix]).all(-1)
                sel = win_pt & free
                pts[j, i, iy[sel], ix[sel]] = obs.xyz[j, np.flatnonzero(m)[sel]].astype(np.float32)
    w2c = np.asarray(sample.cameras.w2c[views][:, win], np.float64)
    K = np.stack([sample.K(v, size) for v in views])
    return {"points": pts, "w2c": w2c, "K": K}


def write_oracle(sample, pred_root, protocol: Recon4DProtocol) -> None:
    p = prediction_path(pred_root, sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **oracle_arrays(sample, protocol))


def write_frozen(sample, pred_root, protocol: Recon4DProtocol) -> None:
    """The oracle of the first window frame repeated for every window frame (points and cameras)."""
    a = oracle_arrays(sample, protocol)
    n = a["points"].shape[0]
    p = prediction_path(pred_root, sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, points=np.repeat(a["points"][:1], n, 0), w2c=np.repeat(a["w2c"][:, :1], n, 1), K=a["K"])


BASELINES = {"oracle": write_oracle, "frozen": write_frozen}


def run_baseline(bench, pred_root=None, kind: str = "oracle", protocol: Optional[Recon4DProtocol] = None,
                 subsets=None, ids=None, jobs: int = 1, progress: bool = True, **opts) -> dict:
    """Write the predictions of baseline `kind` ("oracle" | "frozen") for every evaluable sample. The call form of
    every task, run_baseline(bench, pred_root, kind="frozen", ...); the earlier run_baseline("frozen", bench,
    pred_root, ...) is still accepted."""
    bench, pred_root, kind = baseline_args(bench, pred_root, kind, opts)
    return run_baseline_task(kind, BASELINES, protocol or Recon4DProtocol(), Recon4DProtocol, bench, pred_root,
                             subsets, ids, jobs, progress, **opts)


def main(argv=None) -> int:
    """`vpmwb recon4d {export,baseline,evaluate,summarize}` (`common.task_cli`)."""
    return task_cli(argv, TASK, Recon4DProtocol, export_fn=export_inputs, evaluate_fn=evaluate,
                    summarize_fn=summarize, baseline_fn=run_baseline, baselines=BASELINES, default_baseline="oracle",
                    description="multi-view 4D reconstruction and camera estimation (docs/TASKS.md)")


if __name__ == "__main__":
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
