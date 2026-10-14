"""Task ``track3d``: multi-view (or monocular) 3D point tracking / 3D motion estimation.

Task
----
Given the frames of a time window of one sample, its cameras and a set of query points at the query frame t_q,
predict the 3D position of every query point at every frame of the window (and, optionally, whether it is visible).

    mode "world"      (default) the frames of all views (or ``protocol.views``) are given; predictions are WORLD
                      coordinates (right-handed, metres, +z up; docs/DATA_FORMAT.md). This is multi-view 3D point
                      tracking (MVTracker setting); with ``views=[k]`` it is single-view tracking in the world frame.
    mode "monocular"  only view ``protocol.mono_view`` is given; predictions are coordinates in THAT view's camera
                      frame at each frame (OpenCV: x right, y down, z forward, metres) - the TAPVid-3D setting, scored
                      with absolute metric thresholds.

Protocol (`Track3DProtocol`, same conventions as `vpm_worldbench.protocol.Protocol`)
    window      positions j = 0..n-1 <-> GT frames ``t_q + j * frame_step`` with t_q = ``ref_frame`` (default 16) and
                n - 1 = ``horizon`` (default 64; None = up to the end of the clip). Position 0 is the query frame.
    queries     the GT points with ``default_gt`` at t_q and GT-visible in >= 1 used view at t_q (monocular: in the
                view), in ascending order of their index in tracks3d.npz (``query_index``); every query has query
                position 0. ``query_requires_visible=False`` drops the visibility requirement (legacy Simul-*).
    GT          positions where ``default_gt`` (NaN elsewhere = unknown); per-frame GT visibility =
                ``default_gt & visible[..., used views].any(-1)`` (monocular: that view's flag; ``gt_visibility=
                "in_frustum"`` uses the frustum flag instead, legacy Simul-General).
    A sample is evaluable iff the window lies inside the clip, the views exist and there is >= 1 query; others are
    skipped and counted. With the defaults (world mode) all 1200 samples are evaluable; in monocular mode a sample
    whose view sees no query at t_q is skipped (view 0: 7 Simul-General and 4 Real-General samples; view 1: 8 / 4).

Inputs (`sample_inputs` in Python, `export_inputs` on disk; never contains GT beyond the queries)
    frames      (V, n, H, W, 3) uint8 RGB of the window. On disk: ``frames="none"`` (default) references the
                benchmark's own ``videos/view{k}.mp4`` (decode frames ``window_indices``, 0-based), ``"png"`` writes
                ``frames/view{k}/frame_{t:04d}.png``, ``"mp4"`` writes ``frames/view{k}.mp4`` (lossless libx264rgb,
                n frames); png/mp4 honour ``size=(W, H)`` and ``resize_mode`` (geometry.image_map). Lossless
                window clips are large at the stored resolution (~200 MB per sample for mp4); prefer "none" or a
                reduced size.
    cameras     ``K`` (V, 3, 3) intrinsics at the exported size (pixel centres at integers), ``w2c`` (V, n, 4, 4)
                world-to-camera per window frame (OpenCV, metres); inputs.json ``cameras.view{k}``.
    queries     queries.npz: ``query_index`` (N,) int32 point index in tracks3d.npz, ``query_t`` (N,) int32 window
                position (0), ``query_frame`` (N,) int32 GT frame (t_q), ``xyz`` (N, 3) float64 world metres at t_q,
                ``uv`` (V, N, 2) float64 pixels on the exported grid of each used view (NaN where not GT-visible there
                or outside the grid), ``visible`` (V, N) bool, ``view_index`` (V,) dataset view of each row;
                monocular adds ``xyz_cam`` (N, 3) in the view's camera frame at t_q. ``lifter_queries`` in
                `sample_inputs` = (N, 4) [t, x, y, z], the `vpm_worldbench.lifters` query format.
    depth       optional (``protocol.depth="query"``): depth_query.npz ``depth`` (V, H, W) float32 planar metres of
                frame t_q on the exported grid (0 = invalid); Simul-* rendered GT, Real-General reconstruction
                (pseudo-GT), none for Real-Robot. Results record which depth was given.

Prediction file (one per sample): ``<pred_root>/<Subset>/<id>.npz``
    world mode      ``xyz``      (n, N, 3) float world metres, rows = window positions, columns = queries in query
                                 order (position 0 should repeat the query point).
    monocular mode  ``xyz_cam``  (n, N, 3) float in the camera frame of the protocol view at each window frame (a
                                 world ``xyz`` is accepted instead and mapped with the GT w2c).
    optional        ``vis`` (n, N) bool or float score (occluded iff score < 0.5; non-finite scores = occluded;
                    (n, N, V) per-view flags are reduced with any); missing = always visible (AJ / OA then measure
                    the visibility prior only). ``query_index`` (N,): if present it must be a permutation of the
                    expected query indices; columns are reordered accordingly.
    Non-finite positions are FAILURES: NaN / inf become +inf before scoring, so they are never within a threshold,
    end survival, count as false positives when predicted visible, and make ADE / FDE / EPE3D infinite (the
    share-based metrics stay finite, and so does MTE unless half the entries fail). ``write_prediction`` writes a
    valid file.
    Monocular methods that only use the 2D queries (uv) cannot know the metric scale; score them with
    ``mono_scale="median"`` (one global factor per sample, like the global median scaling of TAPVid-3D) and
    say so.

Metrics (per sample and per point group; then mean ± s.e. over samples per subset)
    3D tracking   `metrics.tracking.track3d_metrics` (MVTracker's evaluate_predictions; TAP-Vid metrics on 3D
                  positions with absolute thresholds, averaged per track): ``{split}/delta_avg``, ``delta@d``, ``aj``,
                  ``oa``, ``survival`` (percent), ``ade``/``mte`` (m) for split any / static (path < 1 cm) / dynamic
                  (path > 10 cm), plus ``ade_all``, ``mte_all``, ``fde``. Scored entries = GT-visible frames after the
                  query frame. Ladders / survival per subset (presets.SUBSET_DEFAULTS): Simul-Robot 1-16 cm (survival
                  0.5 m), Simul-General 10-160 cm (0.5 m), Real-Robot 1-20 cm (0.10 m), Real-General 5-40 cm (1.0 m).
    scene flow    ``flow/*`` from the same tracks: for window positions j (j >= query position) and j + s with the
                  point GT-visible at j and its GT known at j and j + s: f_gt = X(j+s) - X(j), f_pred = P(j+s) - P(j),
                  e = ||f_pred - f_gt|| (m), rel = e / (||f_gt|| + 1e-4). The stride s defaults to ~0.5 s
                  (``flow_dt_s``; s = round(0.5 fps / frame_step): 10 positions on Simul-Robot (20 fps), 8 elsewhere
                  (15 fps)); ``flow_stride=1`` gives consecutive frames.
                  ``flow/epe3d`` = mean e over all entries (``flow/n``). ``flow/moving/*`` restricts to the MOVING
                  entries, ||f_gt|| >= ``flow_min_motion_m`` (default 2 cm): ``epe3d``, ``acc_strict`` = % (e < 0.05
                  or rel < 0.05), ``acc_relax`` = % (e < 0.1 or rel < 0.1), ``outliers`` = % (e > 0.3 or rel > 0.1),
                  ``n`` - the FlowNet3D (Liu et al., CVPR 2019) / HPLFlowNet (Gu et al., CVPR 2019) / PointPWC-Net
                  (Wu et al., ECCV 2020) criteria (thresholds configurable; ``flow_min_motion_m=0`` scores every
                  entry, the classical definition). Why the defaults: at 15-20 fps most frame-to-frame motion is
                  below a millimetre (Simul-Robot median 0.1 mm, Real-Robot 0.9 mm), so with consecutive frames and
                  all entries Acc3D saturates at 100 % for every method and the relative criterion (rel > 10 %)
                  marks any error above ~10 um on a static point as an outlier; the frozen baseline then ranks above
                  far better but slightly jittery trackers. Headline flow metric: ``flow/moving/epe3d`` (with
                  ``flow/moving/outliers`` in summary.md). Caveats (measured on 4 samples per subset, GT + 1 cm
                  Gaussian noise per coordinate vs frozen): ``flow/epe3d`` over all entries still favours frozen on
                  the mostly static Simul-Robot / Real-Robot scenes (16.8 vs 22.5 mm, 14.0 vs 22.6 mm), and the
                  FlowNet3D absolute thresholds (5 / 10 cm) are loose for cm-scale robot motion (frozen reaches
                  Acc3DS 91 % / Acc3DR 100 % on Real-Robot moving entries); these stay in summary.json only.
    all-frames    ``allframes/ade``, ``allframes/mte``, ``allframes/fde``: errors over every window frame with a known
                  GT position, query frame and occluded frames included (the pre-release Simul-* / Real-Robot "ADE");
                  the pre-release Real-General "ADE" is ``ade_all``.
    groups        the main group (sample.json ``tracks.default_report_group``: Simul-Robot "object", Simul-General
                  "target", Real-* "all") plus every other group and "all": Simul-Robot object / robot / all,
                  Real-Robot all / arm / hand, Real-General all / dynamic_surface / human_body.

Ground truth per subset
    Simul-Robot    exact synthetic GT (Unreal transforms), renderer depth-test visibility (3 cm).
    Simul-General  exact synthetic GT (CARLA skinning), visibility = depth test against the rendered depth.
    Real-Robot     forward kinematics of the measured joints (exact up to the robot model), time-aligned to the
                   videos (-2.3 frames); visibility models robot self-occlusion only (scene objects never occlude);
                   camera calibration error ~2 px (flags ``calibration_step_exo``, ``ego_extrinsic_manual_fit``).
    Real-General   PSEUDO-GT: multi-view triangulated tracks (orhtrack, mm-level) + whole-body model tracks
                   (``default_gt`` keeps frames with p90 error <= 20 mm); visibility from the reconstruction z-buffer.

Baselines (`run_baseline` / `BASELINES`)
    oracle  the GT positions (where unknown: the query point) and the GT visibility -> 100 % / zero errors.
    frozen  every point stays at its query position in the WORLD (monocular: mapped through the given camera
            poses, i.e. the query's camera coordinates for static cameras), always visible - the pre-release
            "frozen points" reference row. A constant-velocity baseline needs history, which the inputs do not give.
    lifter  (`run_lifter_baseline`, GPU) a `vpm_worldbench.lifters` Lifter on the real frames - by default
            MVTracker + MoGe-2, the "real video" reference row of the generation delta - using only the task
            inputs: depth anchored on the given depth of t_q (``depth="query"``) or else on the query points.

Results: ``evaluate`` writes ``<out>/<Subset>/<id>.json`` per sample and calls ``summarize``, which writes
``<out>/summary.json`` and ``<out>/summary.md`` (mean ± s.e. over samples per subset; s.e. = std(ddof=1)/sqrt(n); every
group row states how many samples have that group). Failures are not averaged away: a non-finite predicted position
makes the sample's ADE / FDE / EPE3D infinite and so the subset mean of that metric (summary.json counts such samples
as ``n_inf``; undefined (NaN) values are left out). In the JSON files NaN is written as null and infinities as the
strings "inf" / "-inf".
    statuses              the vocabulary of every task (`common.STATUSES`): "ok" scored; "skipped" not evaluable
                          under the protocol (``reason``); "missing" no prediction file; "invalid" a prediction file
                          that cannot be read or has the wrong keys / shape / dtype / ``query_index`` (``reason``
                          names the file and the problem); "error" any other failure (``reason`` + ``traceback``).
                          Invalid and error samples are counted (``n_invalid`` / ``n_error``) and named in a warning,
                          the run goes on, and the next ``evaluate`` scores them again; ``strict=True``
                          (``--strict``) raises instead.
    one run per out dir   an out dir holds the results of ONE run = task + protocol (``protocol_hash``, the protocol
                          without its name) + resolved prediction root, recorded in ``<out>/run.json`` (next to
                          ``protocol.json``) and as ``run_id`` in every record. ``evaluate`` refuses an out dir of
                          another protocol or prediction root (`RunDirError`) unless ``overwrite``, which deletes the
                          records of the previous run. Within a run, a record is reused while the prediction files it
                          read are unchanged (``pred_sig``: path, size, mtime; ``overwrite`` re-scores the selected
                          samples). ``summarize`` reports only the records of the run in run.json (others are counted
                          as ``n_stale``) and writes its ``pred_root`` into summary.json. Out dirs of older versions
                          (no run.json) still summarize by protocol; ``evaluate`` adopts their records that match the
                          current prediction files and refuses the dir while records of other files remain (unless
                          they are re-scored in that call, or ``overwrite``).
    selection             ``subsets`` (names or aliases) and ``ids`` (ids or uids, "rr/<id>" accepted); requested ids
                          that match no selected sample are named in one warning.
Command-line overrides of protocol fields rename the protocol ("track3d-v1+mode=monocular,mono_view=1"); a protocol
file without ``name`` is named after the file. ``jobs`` > 1 (export / baseline / evaluate) runs worker processes
started with "spawn": in a script, call these functions under ``if __name__ == "__main__":``. Every function taking
``protocol`` also accepts None (default), a preset name ("track3d-v1", "legacy-prerelease") or a json / yaml file
(`get_protocol`; anything else raises ValueError).

    from vpm_worldbench import Benchmark
    from vpm_worldbench.tasks import track3d
    bench = Benchmark("/data/VP-WorldBench")
    p = track3d.Track3DProtocol()                                   # world mode, t_q = 16, 64 frames
    for s in bench.samples("Real-Robot"):
        inp = track3d.sample_inputs(s, p)                           # frames, K, w2c, queries
        xyz, vis = my_tracker(inp["frames"], inp["K"], inp["w2c"], inp["lifter_queries"])
        track3d.write_prediction("preds", s, xyz=xyz, vis=vis, query_index=inp["query_index"])
    track3d.evaluate(bench, "preds", "results/track3d", p)

    vpmwb track3d export   --root R --out inputs/  [--mode monocular --view 1] [--subsets rr rg] [--jobs 8]
    vpmwb track3d baseline --root R --pred preds/ --kind frozen          (oracle | frozen | lifter)
    vpmwb track3d evaluate --root R --pred preds/ --out results/ [--jobs 8] [--overwrite] [--strict]
    vpmwb track3d summarize --out results/
(``python -m vpm_worldbench.tasks.track3d ...`` is the same command line; ``--horizon`` <= 0 = to the end of the clip.)
Exit status (the convention of every task command, `common.task_cli`): 0 done; 1 the selection matches no sample, or
none of the evaluable samples is scored (every one missing / invalid / error; `common.run_status`); 2 usage error:
an unknown subset, protocol or lifter, an invalid protocol override or size, no --root folder or no summarize
--out folder (usage + one line), or an out dir of another run without --overwrite (`common.RunDirError`, one
line).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..constants import SUBSETS, canonical_subset
from ..geometry import inside, project
from ..metrics.tracking import ade, fde, mte, track3d_metrics
from ..presets import DYNAMIC_MIN_M, LEGACY_PRERELEASE, STATIC_MAX_M, SUBSET_DEFAULTS
from ..utils import canonical_uid
from .common import (FAILED_STATUSES, InvalidPrediction, RunDirError, cli_size, evaluation_exit_status,
                     file_protocol_fields, read_npz, run_pred_root, run_status, selection_exit_status)

TASK = "track3d"
PRESETS = ("track3d-v1", "legacy-prerelease")
RUN_FILE = "run.json"
MODES = ("world", "monocular")
FRAME_EXPORTS = ("none", "png", "mp4")
DEPTH_INPUTS = ("none", "query")
MONO_SCALES = ("none", "median")
FLOW_EPS = 1e-4
FLOW_MIN_MOTION_M = 0.02

_MD_COLUMNS = (("any/delta_avg", "δ_avg ↑", 1), ("any/aj", "AJ ↑", 1), ("any/oa", "OA ↑", 1),
               ("any/survival", "Surv. ↑", 1), ("any/ade", "ADE m ↓", 4), ("any/mte", "MTE m ↓", 4),
               ("dynamic/delta_avg", "δ_avg dyn ↑", 1), ("flow/moving/epe3d", "EPE3D mov m ↓", 4),
               ("flow/moving/outliers", "Outl. mov ↓", 1))
_PER_SUBSET = ("query_requires_visible", "gt_visibility", "ladder_m", "survival_m", "main_group", "flow_min_motion_m")


@dataclass
class Track3DProtocol:
    """Protocol of the track3d task. Fields that may differ per subset accept a dict {subset: value} (subset names or
    aliases such as "rr"; unknown subsets raise ValueError); subsets missing from the dict use the default
    (presets.SUBSET_DEFAULTS for ladders / survival).

    name                    label stored with the results (`load`: a file without ``name`` is named after the file)
    mode                    "world" | "monocular"
    ref_frame               t_q: query frame = window position 0 (default 16, the generation protocol's t_ref)
    horizon                 number of window positions after t_q (default 64); None = up to the end of the clip
    frame_step              GT frames between consecutive window positions
    views                   world mode: dataset view indices given to the model (None = all)
    mono_view               monocular mode: the view
    query_requires_visible  bool or {subset: bool}: queries must be GT-visible in >= 1 used view at t_q
    gt_visibility           "visible" (occlusion-aware) | "in_frustum", or {subset: ...}
    ladder_m                {subset: thresholds in metres}; default presets ``delta_ladder_m``
    survival_m              {subset: metres}; default presets ``delta_survival_m``
    main_group              {subset: group name or "all"}; default sample.json ``tracks.default_report_group``
    static_m / dynamic_m    path-length limits (m) of the static / dynamic splits (None disables the split)
    flow_stride             scene flow between window positions j and j + flow_stride; None (default) = derived
                            from ``flow_dt_s`` (`flow_stride_for`)
    flow_dt_s               flow time step in seconds when flow_stride is None (default 0.5)
    flow_min_motion_m       metres or {subset: metres}: flow/moving/* entries have ||f_gt|| >= this (default 0.02)
    flow_strict / flow_relax  (abs_m, rel): Acc3D counts e < abs_m or rel < rel
    flow_outlier            (abs_m, rel): outlier if e > abs_m or rel > rel
    mono_scale              monocular: "none" (metric) | "median" (one factor per sample, median ||x_gt|| /
                            median ||x_pred|| over the scored entries, applied before scoring)
    depth                   inputs: "none" | "query" (depth of t_q per view, where the subset has depth)
    clip_filter             attribute filters for Benchmark.samples, e.g. {"difficulty": "hard"}
    """
    name: str = "track3d-v1"
    mode: str = "world"
    ref_frame: int = 16
    horizon: Optional[int] = 64
    frame_step: int = 1
    views: Optional[list] = None
    mono_view: int = 0
    query_requires_visible: Any = True
    gt_visibility: Any = "visible"
    ladder_m: dict = field(default_factory=dict)
    survival_m: dict = field(default_factory=dict)
    main_group: dict = field(default_factory=dict)
    static_m: Optional[float] = STATIC_MAX_M
    dynamic_m: Optional[float] = DYNAMIC_MIN_M
    flow_stride: Optional[int] = None
    flow_dt_s: float = 0.5
    flow_min_motion_m: Any = FLOW_MIN_MOTION_M
    flow_strict: tuple = (0.05, 0.05)
    flow_relax: tuple = (0.10, 0.10)
    flow_outlier: tuple = (0.30, 0.10)
    mono_scale: str = "none"
    depth: str = "none"
    clip_filter: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.ref_frame < 0 or self.frame_step < 1 or (self.horizon is not None and self.horizon < 1):
            raise ValueError("need ref_frame >= 0, frame_step >= 1 and horizon >= 1 (or None)")
        if self.flow_stride is not None:
            self.flow_stride = int(self.flow_stride)
            if self.flow_stride < 1:
                raise ValueError("flow_stride must be >= 1 (or None = from flow_dt_s)")
        if not self.flow_dt_s > 0:
            raise ValueError("flow_dt_s must be > 0")
        for k in _PER_SUBSET:
            setattr(self, k, canonical_subset_keys(getattr(self, k), k))
        if self.mono_scale not in MONO_SCALES:
            raise ValueError(f"mono_scale must be one of {MONO_SCALES}")
        if self.depth not in DEPTH_INPUTS:
            raise ValueError(f"depth must be one of {DEPTH_INPUTS}")
        for k in ("flow_strict", "flow_relax", "flow_outlier"):
            v = tuple(float(x) for x in getattr(self, k))
            if len(v) != 2:
                raise ValueError(f"{k} must be (abs_m, rel)")
            setattr(self, k, v)
        if self.views is not None:
            self.views = [int(v) for v in self.views]
        for s in SUBSETS:
            if self.get("gt_visibility", s) not in ("visible", "in_frustum"):
                raise ValueError("gt_visibility must be 'visible' or 'in_frustum'")

    def get(self, key: str, subset: str):
        """Value of `key` for `subset` (dict fields: the subset entry, else the default)."""
        subset = canonical_subset(subset)
        v = getattr(self, key)
        if isinstance(v, dict):
            if subset in v:
                return v[subset]
            return _subset_default(key, subset)
        return v

    def ladder_for(self, subset: str) -> tuple:
        return tuple(float(x) for x in self.get("ladder_m", subset))

    def survival_for(self, subset: str) -> float:
        return float(self.get("survival_m", subset))

    def flow_stride_for(self, fps: float) -> int:
        """Scene-flow stride in window positions: ``flow_stride``, or round(flow_dt_s * fps / frame_step) (>= 1)."""
        if self.flow_stride is not None:
            return int(self.flow_stride)
        return max(1, int(round(self.flow_dt_s * float(fps) / self.frame_step)))

    def hash(self) -> str:
        return protocol_hash(self)

    def window(self, sample) -> np.ndarray:
        """(n,) int GT frame indices of the window positions (position 0 = t_q)."""
        if self.horizon is None:
            h = (sample.n_frames - 1 - self.ref_frame) // self.frame_step
        else:
            h = self.horizon
        return self.ref_frame + self.frame_step * np.arange(max(h, 0) + 1)

    def views_for(self, sample) -> list:
        if self.mode == "monocular":
            return [int(self.mono_view)]
        return list(range(sample.n_views)) if self.views is None else list(self.views)

    def evaluable(self, sample) -> tuple:
        """(ok, reason) from the window and the views alone (the query count is checked by `task_case`)."""
        if self.ref_frame > sample.n_frames - 1:
            return False, f"query frame {self.ref_frame} > last frame {sample.n_frames - 1}"
        win = self.window(sample)
        if len(win) < 2:
            return False, "window has no frame after the query frame"
        if win[-1] > sample.n_frames - 1:
            return False, f"window ends at {int(win[-1])} > last frame {sample.n_frames - 1}"
        views = self.views_for(sample)
        if not views or min(views) < 0 or max(views) >= sample.n_views:
            return False, f"views {views} not in 0..{sample.n_views - 1}"
        return True, ""

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                d[k] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Track3DProtocol":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown track3d protocol fields {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path) -> "Track3DProtocol":
        """A json / yaml file of protocol fields; without ``name`` the protocol is named after the file (its stem), so
        a changed protocol is never labelled as the preset."""
        return cls.from_dict(load_protocol_file(path))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def legacy_prerelease(cls, **overrides) -> "Track3DProtocol":
        """The query / GT-visibility rules of the pre-release delta numbers (presets.LEGACY_PRERELEASE): Simul-* queries
        without a visibility requirement, Simul-General frustum-only visibility. The pre-release Real-Robot GT
        (un-aligned FK tracks, 480x848 z-buffer visibility) is not rebuilt: Real-Robot uses the released tracks."""
        kw = dict(name="track3d-legacy-prerelease",
                  query_requires_visible={s: bool(LEGACY_PRERELEASE[s]["query_requires_visible"]) for s in SUBSETS},
                  gt_visibility={s: str(LEGACY_PRERELEASE[s]["gt_visibility"]) for s in SUBSETS})
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_protocol(cls, p, **overrides) -> "Track3DProtocol":
        """Track3D protocol with the window, views, ladders, survival radii, main groups, clip filter and the query /
        GT-visibility switches of a generation `Protocol` (including the legacy per-subset switches of the
        'legacy-prerelease' preset). Its queries and GT visibility equal those of the generation delta only on 'resize'
        / 'letterbox' eval grids: where the generation protocol centre-crops (resize_mode 'center_crop'), the delta
        also requires the points to project inside the cropped grid (``visibility_in_grid``), a rule Track3DProtocol
        has no counterpart for (track3d has no eval grid); a warning names those subsets."""
        from ..protocol import legacy_option
        crop = [s for s in SUBSETS if p.resize_mode_for(s) == "center_crop"]
        if crop:
            warnings.warn(f"Track3DProtocol.from_protocol: {p.name!r} centre-crops {crop} (resize_mode center_crop); "
                          "there the generation delta also requires queries / GT-visible points to lie inside the "
                          "cropped grid, which track3d does not apply, so its queries and visibility differ")
        kw = dict(name=f"track3d/{p.name}", ref_frame=p.ref_frame, horizon=p.horizon, frame_step=p.frame_step,
                  views=None if p.views is None else list(p.views),
                  query_requires_visible={s: bool(legacy_option(p, s, "query_requires_visible",
                                                                p.query_requires_visible)) for s in SUBSETS},
                  gt_visibility={s: str(legacy_option(p, s, "gt_visibility", p.gt_visibility)) for s in SUBSETS},
                  ladder_m=dict(p.delta_ladder_m), survival_m=dict(p.delta_survival_m),
                  main_group={k: v for k, v in p.delta_group.items()}, clip_filter=dict(p.clip_filter))
        kw.update(overrides)
        return cls(**kw)


def _subset_default(key: str, subset: str):
    if key == "ladder_m":
        return SUBSET_DEFAULTS[subset]["delta_ladder_m"]
    if key == "survival_m":
        return SUBSET_DEFAULTS[subset]["delta_survival_m"]
    if key == "main_group":
        return None
    if key == "query_requires_visible":
        return True
    if key == "gt_visibility":
        return "visible"
    if key == "flow_min_motion_m":
        return FLOW_MIN_MOTION_M
    raise KeyError(f"no per-subset default for {key!r}")


def canonical_subset_keys(value, what: str = "value"):
    """A {subset: v} dict with its keys canonicalised (aliases such as "rr" -> "Real-Robot"); other values as is.
    Raises ValueError for an unknown subset or two keys naming the same subset."""
    if not isinstance(value, dict):
        return value
    out = {}
    for k, v in value.items():
        try:
            c = canonical_subset(str(k))
        except KeyError:
            raise ValueError(f"{what}: unknown subset {k!r}; expected one of {SUBSETS} (or an alias)") from None
        if c in out:
            raise ValueError(f"{what}: subset {c!r} given twice")
        out[c] = v
    return out


def protocol_hash(protocol) -> str:
    """Short hash of a task protocol's content (its `to_dict` without the name): equal hashes = same scoring."""
    d = {k: v for k, v in protocol.to_dict().items() if k != "name"}
    return hashlib.sha1(json.dumps(jsonable(d), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def override_protocol(p, overrides: dict, cls=None):
    """`p` with the given fields replaced; when a value changes, the name gets a "+field=value,..." suffix so that
    results of a non-default run are not labelled as the preset."""
    cls = cls or type(p)
    d = p.to_dict()
    changed = {k: v for k, v in overrides.items() if jsonable(d.get(k)) != jsonable(v)}
    if not changed:
        return p
    d.update(changed)
    d["name"] = f"{p.name}+" + ",".join(f"{k}={_fmt_value(v)}" for k, v in changed.items())
    return cls.from_dict(d)


def _fmt_value(v) -> str:
    if v is None or isinstance(v, (str, bool, int, float)):
        return str(v)
    return json.dumps(jsonable(v)).replace(" ", "")


def get_protocol(spec=None) -> Track3DProtocol:
    """None / "" / "track3d-v1" -> default; "legacy-prerelease" -> `Track3DProtocol.legacy_prerelease()`;
    a Track3DProtocol is returned as is; otherwise a json / yaml file path (ValueError if it is not a file)."""
    if isinstance(spec, Track3DProtocol):
        return spec
    if spec in (None, "", "track3d-v1", "default"):
        return Track3DProtocol()
    if spec in ("legacy-prerelease", "track3d-legacy-prerelease"):
        return Track3DProtocol.legacy_prerelease()
    check_protocol_spec(spec, TASK, PRESETS)
    return Track3DProtocol.load(spec)


def check_protocol_spec(spec, task: str, presets) -> None:
    """ValueError naming the presets when `spec` (not a preset) is not an existing protocol file."""
    if not Path(str(spec)).is_file():
        raise ValueError(f"unknown {task} protocol {str(spec)!r}: not a preset ({', '.join(presets)}) or an existing "
                         "JSON / YAML file")


def load_protocol_file(path) -> dict:
    """Protocol fields of a json / yaml file; a file without ``name`` gets the file stem as its name (the rule of
    every task, `common.file_protocol_fields`)."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return file_protocol_fields(yaml.safe_load(text), path)
    return file_protocol_fields(json.loads(text), path)


def report_groups(sample, main: str) -> list:
    """[main] + the other groups + "all" (when the sample has several groups), without duplicates."""
    names = [str(g) for g in sample.tracks.group_names]
    out = [main]
    if len(names) > 1:
        out += [g for g in names if g != main]
        if main != "all":
            out.append("all")
    return out


def task_case(sample, protocol: Track3DProtocol | None = None) -> dict:
    """Everything the task needs for one sample (deterministic from sample + protocol).

    Returns {"ok": False, "reason": str} or {"ok": True, "window" (n,) GT frames, "views" [k], "query_index" (N,),
    "xyz_q" (N, 3) world at t_q, "gt" (n, N, 3) GT in the prediction frame (world, or camera of the mono view) with
    NaN where unknown, "gt_world" (n, N, 3), "gt_vis" (n, N) bool, "vis_q" (V, N) per used view at t_q,
    "main_group", "groups" {name: (N,) bool}}.
    """
    p = get_protocol(protocol)
    ok, why = p.evaluable(sample)
    if not ok:
        return {"ok": False, "reason": why}
    sub = sample.subset
    tr = sample.tracks
    win = p.window(sample)
    tq = int(p.ref_frame)
    views = p.views_for(sample)
    flag = tr.visible if p.get("gt_visibility", sub) == "visible" else tr.in_frustum
    dgt = tr.default_gt & np.isfinite(tr.xyz).all(-1)
    seen = flag[:, :, views] & dgt[:, :, None]
    q = dgt[tq].copy()
    if p.get("query_requires_visible", sub):
        q &= seen[tq].any(-1)
    qidx = np.nonzero(q)[0]
    if len(qidx) == 0:
        return {"ok": False, "reason": f"no query point at frame {tq}"}
    X = tr.xyz[win][:, qidx].astype(np.float64)
    known = dgt[win][:, qidx]
    X[~known] = np.nan
    gt_vis = seen[win][:, qidx].any(-1) & known
    gt = X
    if p.mode == "monocular":
        E = sample.cameras.w2c[views[0]][win]
        gt = np.einsum("tij,tnj->tni", E[:, :3, :3], X) + E[:, None, :3, 3]
    main = p.get("main_group", sub) or sample.meta.get("tracks", {}).get("default_report_group") or "all"
    names = [str(g) for g in tr.group_names]
    if main != "all" and main not in names:
        raise KeyError(f"{sample.uid}: main group {main!r} not in {names}")
    groups = {g: tr.group_mask(g)[qidx] for g in report_groups(sample, main)}
    return {"ok": True, "window": win, "views": views, "query_index": qidx, "xyz_q": X[0].copy(), "gt": gt,
            "gt_world": X, "gt_vis": gt_vis, "vis_q": seen[tq][qidx].T.copy(), "main_group": main, "groups": groups}


def _export_size(sample, view: int, size):
    return tuple(int(v) for v in (sample.image_size(view) if size is None else size))


def sample_inputs(sample, protocol: Track3DProtocol | None = None, size=None, resize_mode: str = "resize",
                  load_frames: bool = True, case: dict | None = None) -> dict:
    """The model inputs of one sample, in memory.

    size (W, H) or None (stored resolution); resize_mode as geometry.image_map. Returns a dict with
    "uid", "subset", "id", "mode", "views" (dataset view indices), "window_indices" (n,), "fps", "query_frame",
    "frames" (V, n, H, W, 3) uint8 (a list per view if sizes differ; None without load_frames), "image_size" [(W, H)],
    "K" (V, 3, 3) at that size, "w2c" (V, n, 4, 4), "query_index" (N,), "query_t" (N,) zeros, "queries_xyz" (N, 3)
    world, "queries_uv" (V, N, 2) (NaN where not visible / outside the grid), "queries_visible" (V, N),
    "lifter_queries" (N, 4) [t, x, y, z], "depth" (V, H, W) metres or None, and in monocular mode
    "queries_xyz_cam" (N, 3). Raises ValueError for a sample that is not evaluable.
    """
    p = get_protocol(protocol)
    c = case or task_case(sample, p)
    if not c["ok"]:
        raise ValueError(f"{sample.uid}: {c['reason']}")
    win, views, tq = c["window"], c["views"], int(p.ref_frame)
    N = len(c["query_index"])
    sizes = [_export_size(sample, v, size) for v in views]
    K = np.stack([sample.K(v, sizes[i], resize_mode) for i, v in enumerate(views)])
    w2c = np.stack([sample.w2c(v, win) for v in views])
    uv = np.full((len(views), N, 2), np.nan)
    vq = np.zeros((len(views), N), bool)
    for i, v in enumerate(views):
        p2, z = project(c["xyz_q"], K[i], sample.cameras.w2c[v, tq])
        ok = c["vis_q"][i] & (z > 0) & inside(p2, sizes[i])
        uv[i, ok] = p2[ok]
        vq[i] = ok
    out = {"uid": sample.uid, "subset": sample.subset, "id": sample.id, "mode": p.mode, "views": list(views),
           "window_indices": win.copy(), "fps": sample.fps / p.frame_step, "query_frame": tq,
           "image_size": [list(s) for s in sizes], "K": K, "w2c": w2c, "query_index": c["query_index"].astype(np.int64),
           "query_t": np.zeros(N, np.int64), "queries_xyz": c["xyz_q"].copy(), "queries_uv": uv,
           "queries_visible": vq, "lifter_queries": np.concatenate([np.zeros((N, 1)), c["xyz_q"]], 1), "depth": None,
           "frames": None}
    if p.mode == "monocular":
        E = sample.cameras.w2c[views[0], tq]
        out["queries_xyz_cam"] = c["xyz_q"] @ E[:3, :3].T + E[:3, 3]
    if p.depth == "query" and sample.has_depth:
        out["depth"] = np.stack([sample.depth(v, [tq], size=sizes[i], mode=resize_mode)[0]
                                 for i, v in enumerate(views)]).astype(np.float32)
    if load_frames:
        fr = [sample.frames(v, win, size=None if size is None else sizes[i], mode=resize_mode)
              for i, v in enumerate(views)]
        out["frames"] = np.stack(fr) if len({f.shape for f in fr}) == 1 else fr
    return out


def _manifest_entry(rec: dict) -> dict:
    return {"uid": rec["uid"], "subset": rec["subset"], "id": rec["id"], "n_window": rec["n_window"],
            "n_queries": rec["n_queries"], "views": rec["views"], "dir": f"{rec['subset']}/{rec['id']}"}


def _export_one(args) -> dict:
    root, subset, sid, out_dir, pdict, frames, size, resize_mode, overwrite = args
    from ..data.sample import Sample
    from ..io.video import write_video
    p = Track3DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    d = Path(out_dir) / s.subset / s.id
    if not overwrite and (d / "inputs.json").exists():
        return {**_manifest_entry(json.loads((d / "inputs.json").read_text())), "existing": True}
    c = task_case(s, p)
    if not c["ok"]:
        return {"uid": s.uid, "skipped": c["reason"]}
    inp = sample_inputs(s, p, size=size, resize_mode=resize_mode, load_frames=frames != "none", case=c)
    d.mkdir(parents=True, exist_ok=True)
    q = {"query_index": inp["query_index"].astype(np.int32), "query_t": inp["query_t"].astype(np.int32),
         "query_frame": np.full(len(inp["query_index"]), inp["query_frame"], np.int32), "xyz": inp["queries_xyz"],
         "uv": inp["queries_uv"], "visible": inp["queries_visible"], "view_index": np.asarray(inp["views"], np.int32)}
    if "queries_xyz_cam" in inp:
        q["xyz_cam"] = inp["queries_xyz_cam"]
    np.savez_compressed(d / "queries.npz", **q)
    files = {}
    for i, v in enumerate(inp["views"]):
        name = f"view{v}"
        if frames == "none":
            files[name] = {"video": str(Path(s.subset) / s.id / s.meta["views"][v]["video"]),
                           "relative_to": "benchmark root", "frame_indices": "window_indices"}
            continue
        fr = inp["frames"][i]
        if frames == "png":
            import cv2
            vd = d / "frames" / name
            vd.mkdir(parents=True, exist_ok=True)
            for t, f in zip(inp["window_indices"], fr):
                cv2.imwrite(str(vd / f"frame_{int(t):04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            files[name] = {"png_dir": f"frames/{name}", "pattern": "frame_{t:04d}.png (t = GT frame index)"}
        else:
            write_video(d / "frames" / f"{name}.mp4", fr, fps=inp["fps"], lossless=True)
            files[name] = {"video": f"frames/{name}.mp4", "frame_indices": "0..n-1 = window positions"}
    if inp["depth"] is not None:
        np.savez_compressed(d / "depth_query.npz", depth=inp["depth"].astype(np.float32),
                            kind=np.asarray(str(s.meta["depth"].get("kind", ""))))
    n, N = len(inp["window_indices"]), len(inp["query_index"])
    key = "xyz" if p.mode == "world" else "xyz_cam"
    rec = {"task": TASK, "uid": s.uid, "subset": s.subset, "id": s.id, "mode": p.mode, "protocol": p.name,
           "views": inp["views"], "view_names": [s.view_names[v] for v in inp["views"]], "fps": inp["fps"],
           "query_frame": inp["query_frame"], "window_indices": [int(t) for t in inp["window_indices"]],
           "n_window": n, "n_queries": N, "frames": {"kind": frames, "files": files},
           "cameras": {f"view{v}": {"size": inp["image_size"][i], "K": inp["K"][i].tolist(),
                                    "w2c": inp["w2c"][i].tolist()} for i, v in enumerate(inp["views"])},
           "queries": "queries.npz", "depth": "depth_query.npz" if inp["depth"] is not None else None,
           "prediction": {"file": f"{s.subset}/{s.id}.npz", "key": key, "shape": [n, N, 3],
                          "frame": "world" if p.mode == "world" else f"camera of view{inp['views'][0]}",
                          "optional": {"vis": [n, N], "query_index": [N]}}}
    (d / "inputs.json").write_text(json.dumps(rec, indent=1))
    return _manifest_entry(rec)


def export_inputs(bench, out_dir, protocol: Track3DProtocol | None = None, subsets=None, ids=None,
                  frames: str = "none", size=None, resize_mode: str = "resize", jobs: int = 1,
                  overwrite: bool = False) -> dict:
    """Write the task inputs of every evaluable sample (layout in the module docstring) plus ``manifest.jsonl``,
    ``protocol.json``, ``export.json`` (frame options) and ``skipped.json`` in `out_dir`. frames: "none" | "png" |
    "mp4"; size (W, H) only with png / mp4. Resumable: samples whose inputs.json exists are kept unless overwrite;
    an out dir written with another protocol or other frame options is refused unless overwrite.
    Returns {"exported": n (incl. kept), "skipped": m, "existing": k}."""
    p = get_protocol(protocol)
    if frames not in FRAME_EXPORTS:
        raise ValueError(f"frames must be one of {FRAME_EXPORTS}")
    if frames == "none" and size is not None:
        raise ValueError("size needs frames='png' or 'mp4' (frames='none' references the native videos)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"task": TASK, "protocol_hash": p.hash(), "frames": frames,
            "size": None if size is None else [int(x) for x in size], "resize_mode": resize_mode}
    check_out_dir(out_dir, "export.json", opts, overwrite)
    work = [(str(bench.root), s.subset, s.id, str(out_dir), p.to_dict(), frames, size, resize_mode, overwrite)
            for s in select_samples(bench, subsets, ids, p.clip_filter, f"{TASK} export")[0]]
    res = _map(_export_one, work, jobs)
    return _finish_export(out_dir, p, res)


def check_out_dir(out_dir: Path, name: str, content: dict, overwrite: bool):
    """Refuse an out dir whose `name` file differs from `content` (unless overwrite); then write it."""
    f = Path(out_dir) / name
    if f.exists() and not overwrite and json.loads(f.read_text()) != jsonable(content):
        raise RunDirError(f"{out_dir} holds results of a different protocol or options ({f}); use another out dir "
                          "or overwrite=True (--overwrite)")
    f.write_text(json.dumps(jsonable(content), indent=1))


def _finish_export(out_dir: Path, p, res: list) -> dict:
    ok = [r for r in res if "skipped" not in r]
    skipped = [r for r in res if "skipped" in r]
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps({k: v for k, v in r.items() if k != "existing"}) + "\n")
    p.save(out_dir / "protocol.json")
    (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))
    return {"exported": len(ok), "skipped": len(skipped), "existing": sum(bool(r.get("existing")) for r in ok)}


def prediction_path(pred_root, sample) -> Path:
    return Path(pred_root) / sample.subset / f"{sample.id}.npz"


def write_prediction(pred_root, sample, xyz=None, xyz_cam=None, vis=None, query_index=None) -> Path:
    """Write ``<pred_root>/<Subset>/<id>.npz``: xyz (n, N, 3) world metres (world mode) or xyz_cam (n, N, 3) camera
    frame (monocular mode), optional vis (n, N) and query_index (N,)."""
    if (xyz is None) == (xyz_cam is None):
        raise ValueError("give exactly one of xyz (world mode) or xyz_cam (monocular mode)")
    key, a = ("xyz", np.asarray(xyz)) if xyz is not None else ("xyz_cam", np.asarray(xyz_cam))
    d = {key: a if a.dtype in (np.float32, np.float64) else a.astype(np.float32)}
    if vis is not None:
        v = np.asarray(vis)
        d["vis"] = v if v.dtype == bool else v.astype(np.float32)
    if query_index is not None:
        d["query_index"] = np.asarray(query_index, np.int64)
    path = prediction_path(pred_root, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(path)
    return path


def _reorder(a, perm):
    return None if a is None else a[:, perm]


def real_array(a, path, key: str, bool_ok: bool = False) -> np.ndarray:
    """`a` as an array when it holds real numbers (integers / floats; booleans only with bool_ok), else
    InvalidPrediction naming the file and the key."""
    a = np.asarray(a)
    if a.dtype == bool:
        if bool_ok:
            return a
    elif np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating):
        return a
    raise InvalidPrediction(f"{path}: '{key}' must hold real numbers, got dtype {a.dtype}")


def index_array(a, path, key: str) -> np.ndarray:
    """A prediction's column ids (query_index / track_id / query_id) flattened to (n,) int64; InvalidPrediction unless
    they are integers (integral floats accepted)."""
    a = np.asarray(a).reshape(-1)
    ok = a.dtype != bool and (np.issubdtype(a.dtype, np.integer) or (
        np.issubdtype(a.dtype, np.floating) and bool(np.all(np.isfinite(a) & (a == np.round(a))))))
    if not ok:
        raise InvalidPrediction(f"{path}: '{key}' must hold integer ids, got dtype {a.dtype}")
    return a.astype(np.int64)


def load_prediction(pred_root, sample, protocol: Track3DProtocol | None = None, case: dict | None = None):
    """Read and check one prediction file. Returns (pred (n, N, 3) float64 in the prediction frame with non-finite
    entries set to +inf, vis (n, N) bool or None, info {"file", "key", "has_vis", "nonfinite_frac"}); None if the
    file does not exist. Raises InvalidPrediction (a ValueError) when the file cannot be read (not an .npz archive)
    or is malformed (keys, dtypes, shapes, query_index)."""
    p = get_protocol(protocol)
    c = case or task_case(sample, p)
    path = prediction_path(pred_root, sample)
    if not path.exists():
        return None
    n, N = len(c["window"]), len(c["query_index"])
    names, z = read_npz(path, ("xyz", "xyz_cam", "vis", "query_index"))
    key = "xyz" if p.mode == "world" else ("xyz_cam" if "xyz_cam" in z else "xyz")
    if key not in z:
        raise InvalidPrediction(f"{path}: needs '{'xyz' if p.mode == 'world' else 'xyz_cam'}' "
                                f"(keys: {sorted(names)})")
    pred = real_array(z[key], path, key).astype(np.float64)
    vis = real_array(z["vis"], path, "vis", bool_ok=True) if "vis" in z else None
    qi = index_array(z["query_index"], path, "query_index") if "query_index" in z else None
    if pred.shape != (n, N, 3):
        raise InvalidPrediction(f"{path}: '{key}' has shape {pred.shape}, expected (n_window, n_queries, 3) = "
                                f"{(n, N, 3)}")
    if vis is not None:
        if vis.ndim == 3 and vis.shape[:2] == (n, N):
            vis = vis.any(-1) if vis.dtype == bool else (np.nan_to_num(vis.astype(np.float64), nan=0.0) >= 0.5).any(-1)
        if vis.shape != (n, N):
            raise InvalidPrediction(f"{path}: 'vis' has shape {vis.shape}, expected {(n, N)} (or (n, N, V))")
        if vis.dtype != bool:
            vis = np.nan_to_num(vis.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0) >= 0.5
    if qi is not None:
        expect = c["query_index"]
        if qi.shape != expect.shape or not np.array_equal(np.sort(qi), expect):
            raise InvalidPrediction(f"{path}: query_index is not a permutation of the expected {N} query indices")
        perm = np.argsort(qi, kind="stable")
        pred, vis = pred[:, perm], _reorder(vis, perm)
    if p.mode == "monocular" and key == "xyz":
        E = sample.cameras.w2c[c["views"][0]][c["window"]]
        pred = np.einsum("tij,tnj->tni", E[:, :3, :3], pred) + E[:, None, :3, 3]
    bad = ~np.isfinite(pred).all(-1)
    pred[bad] = np.inf
    info = {"file": str(path), "key": key, "has_vis": vis is not None, "nonfinite_frac": float(bad.mean())}
    return pred, vis, info


FLOW_KEYS = ("epe3d", "n", "moving/epe3d", "moving/acc_strict", "moving/acc_relax", "moving/outliers", "moving/n")


def scene_flow_metrics(gt, gt_vis, pred, query_t=0, stride: int = 1, strict=(0.05, 0.05), relax=(0.10, 0.10),
                       outlier=(0.30, 0.10), min_motion_m: float = 0.0) -> dict:
    """Scene-flow metrics implied by point tracks (criteria of FlowNet3D / HPLFlowNet / PointPWC-Net).

    gt (n, N, 3) GT positions in metres (NaN = unknown), gt_vis (n, N) bool, pred (n, N, 3) predicted positions
    (non-finite = failure), query_t (N,) or int window position of each query, stride: position gap s.
    Entries: (j, i) with j >= query_t[i], j + s <= n - 1, gt_vis[j, i] and the GT finite at j and j + s.
    f_gt = gt[j+s] - gt[j], f_pred = pred[j+s] - pred[j], e = ||f_pred - f_gt|| (+inf if a predicted endpoint is not
    finite), rel = e / (||f_gt|| + 1e-4).
    Returns {"epe3d": mean e over all entries (m), "n": entries, and over the MOVING entries (||f_gt|| >=
    min_motion_m; 0 = all entries, the classical definition): "moving/epe3d", "moving/acc_strict" % of (e < strict[0]
    or rel < strict[1]), "moving/acc_relax" same with `relax`, "moving/outliers" % of (e > outlier[0] or
    rel > outlier[1]), "moving/n"}; NaN values where there is no entry.
    """
    g = np.asarray(gt, np.float64)
    pr = np.asarray(pred, np.float64)
    v = np.asarray(gt_vis, bool)
    n, N = g.shape[:2]
    s = int(stride)
    q = np.broadcast_to(np.asarray(query_t), (N,))
    nan = float("nan")
    out = {k: (0 if k.endswith("n") else nan) for k in FLOW_KEYS}
    if n <= s or N == 0:
        return out
    fin = np.isfinite(g).all(-1)
    sel = v[:-s] & fin[:-s] & fin[s:] & (np.arange(n - s)[:, None] >= q[None, :])
    fg = g[s:] - g[:-s]
    pf = np.isfinite(pr).all(-1)
    with np.errstate(invalid="ignore"):
        e = np.linalg.norm((pr[s:] - pr[:-s]) - fg, axis=-1)
    e = np.where(pf[s:] & pf[:-s], e, np.inf)
    e, fn = e[sel], np.linalg.norm(fg, axis=-1)[sel]
    if e.size == 0:
        return out
    out.update({"epe3d": float(np.mean(e)), "n": int(e.size)})
    mv = fn >= float(min_motion_m)
    e, fn = e[mv], fn[mv]
    if e.size == 0:
        return out
    rel = e / (fn + FLOW_EPS)
    out.update({"moving/epe3d": float(np.mean(e)),
                "moving/acc_strict": 100.0 * float(np.mean((e < strict[0]) | (rel < strict[1]))),
                "moving/acc_relax": 100.0 * float(np.mean((e < relax[0]) | (rel < relax[1]))),
                "moving/outliers": 100.0 * float(np.mean((e > outlier[0]) | (rel > outlier[1]))),
                "moving/n": int(e.size)})
    return out


def median_scale(gt, gt_vis, pred) -> float:
    """Monocular global scale: median ||x_gt|| / median ||x_pred|| over scored entries (GT visible, position > 0,
    finite prediction); 1.0 when undefined."""
    g = np.asarray(gt, np.float64)
    pr = np.asarray(pred, np.float64)
    m = np.asarray(gt_vis, bool).copy()
    m[0] = False
    m &= np.isfinite(pr).all(-1) & np.isfinite(g).all(-1)
    if not m.any():
        return 1.0
    a, b = np.median(np.linalg.norm(g[m], axis=-1)), np.median(np.linalg.norm(pr[m], axis=-1))
    return float(a / b) if b > 0 and np.isfinite(a / b) else 1.0


def track_metrics(gt, gt_vis, pred, pred_vis, protocol: Track3DProtocol, subset: str, fps: float | None = None,
                  flow_stride: int | None = None) -> dict:
    """All per-group metrics: `track3d_metrics` keys, ``flow/*`` (`scene_flow_metrics`) and ``allframes/*``.
    gt / pred (n, N, 3) metres (gt NaN = unknown, pred non-finite = failure), gt_vis (n, N), pred_vis (n, N) or None;
    every query at window position 0. The flow stride is `flow_stride`, else ``protocol.flow_stride_for(fps)``
    (fps of the clip; needed when protocol.flow_stride is None)."""
    N = gt.shape[1]
    q = np.zeros(N, np.int64)
    pred = np.where(np.isfinite(pred).all(-1, keepdims=True), pred, np.inf)
    if flow_stride is None:
        if protocol.flow_stride is None and fps is None:
            raise ValueError("track_metrics needs fps (or flow_stride) when protocol.flow_stride is None")
        flow_stride = protocol.flow_stride_for(fps if fps is not None else 1.0)
    r = track3d_metrics(gt, gt_vis, pred, pred_vis, q, protocol.ladder_for(subset), protocol.survival_for(subset),
                        static_m=protocol.static_m, dynamic_m=protocol.dynamic_m)
    f = scene_flow_metrics(gt, gt_vis, pred, q, flow_stride, protocol.flow_strict, protocol.flow_relax,
                           protocol.flow_outlier, float(protocol.get("flow_min_motion_m", subset)))
    r.update({f"flow/{k}": v for k, v in f.items()})
    with np.errstate(invalid="ignore"):
        r["allframes/ade"] = ade(gt, pred)
        r["allframes/mte"] = mte(gt, pred)
        r["allframes/fde"] = fde(gt, pred)
    r["n_queries"] = int(N)
    return r


def evaluate_sample(sample, pred_root, protocol: Track3DProtocol | None = None) -> dict:
    """Per-sample result record (see the module docstring); status "ok", "skipped" or "missing". Raises
    InvalidPrediction (a ValueError) on an unreadable or malformed prediction file (`evaluate` records that as status
    "invalid", any other exception as "error")."""
    p = get_protocol(protocol)
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id, "protocol": p.name,
           "protocol_hash": p.hash(), "mode": p.mode,
           "pred_sig": prediction_signature(pred_root, [prediction_path(pred_root, sample)])}
    c = task_case(sample, p)
    if not c["ok"]:
        return {**rec, "status": "skipped", "reason": c["reason"]}
    stride = p.flow_stride_for(sample.fps)
    rec.update(views=c["views"], window=[int(c["window"][0]), int(c["window"][-1]), int(p.frame_step)],
               n_window=int(len(c["window"])), n_queries=int(len(c["query_index"])), main_group=c["main_group"],
               ladder_m=list(p.ladder_for(sample.subset)), survival_m=p.survival_for(sample.subset),
               flow_stride=stride, depth_input=p.depth if sample.has_depth else "none")
    got = load_prediction(pred_root, sample, p, c)
    if got is None:
        return {**rec, "status": "missing", "reason": f"no {prediction_path(pred_root, sample)}"}
    pred, vis, info = got
    scale = 1.0
    if p.mode == "monocular" and p.mono_scale == "median":
        scale = median_scale(c["gt"], c["gt_vis"], pred)
        pred = pred * scale
    info["scale"] = scale
    rec["prediction"] = info
    groups = {}
    for g, m in c["groups"].items():
        if not m.any():
            groups[g] = {"n_queries": 0}
            continue
        groups[g] = track_metrics(c["gt"][:, m], c["gt_vis"][:, m], pred[:, m],
                                  None if vis is None else vis[:, m], p, sample.subset, flow_stride=stride)
    rec["groups"] = groups
    rec["status"] = "ok"
    return rec


def prediction_signature(pred_root, files) -> str:
    """Short hash of the resolved prediction root and the (relative path, mtime, size) of `files` (missing files
    included): a stored record is reused only while this is unchanged."""
    root = Path(pred_root).resolve()
    h = hashlib.sha1(str(root).encode("utf-8"))
    for f in files:
        f = Path(f).resolve()
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            rel = str(f)
        try:
            st = f.stat()
            h.update(f"{rel}:{st.st_mtime_ns}:{st.st_size};".encode("utf-8"))
        except FileNotFoundError:
            h.update(f"{rel}:-;".encode("utf-8"))
    return h.hexdigest()[:16]


def prediction_files(pred_root, sample, protocol: Track3DProtocol | None = None) -> list:
    """Every prediction file the evaluation of `sample` reads (the one npz)."""
    return [prediction_path(pred_root, sample)]


def run_fingerprint(task: str, protocol, pred_root) -> tuple:
    """(run_id, run dict): one evaluation run = task + protocol content (`protocol_hash`, the name is not part of it)
    + resolved prediction root. The dict is what <out>/run.json holds."""
    run = {"task": task, "pred_root": str(Path(pred_root).resolve()), "protocol_hash": protocol.hash()}
    rid = hashlib.sha1(json.dumps(run, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return rid, dict(run, run_id=rid, protocol=protocol.to_dict())


def evaluate_cached(task: str, sample, protocol, pred_root, out_dir, files, evaluate_fn, overwrite: bool,
                    run_id: Optional[str], strict: bool) -> dict:
    """Shared per-sample step of `run_evaluation` workers: reuse <out_dir>/<Subset>/<id>.json when it belongs to run
    `run_id`, has the protocol's hash, is not a failed record (status invalid / error) and the prediction `files` are
    unchanged (`prediction_signature`; not with overwrite); otherwise write evaluate_fn(sample, pred_root, protocol).
    InvalidPrediction from evaluate_fn gives a status "invalid" record (``reason``), any other exception a status
    "error" record (``reason`` + the innermost traceback frames); `strict` re-raises both. Returns {"uid", "subset",
    "status", "reason", "cached"}."""
    path = Path(out_dir) / sample.subset / f"{sample.id}.json"
    sig = prediction_signature(pred_root, files)
    if not overwrite:
        old = read_json(path)
        if isinstance(old, dict) and old.get("task") == task and old.get("run_id") == run_id and \
                old.get("status") not in FAILED_STATUSES and old.get("protocol_hash") == protocol.hash() and \
                old.get("pred_sig") == sig:
            return {"uid": sample.uid, "subset": sample.subset, "status": old["status"],
                    "reason": old.get("reason", ""), "cached": True}
    try:
        rec = evaluate_fn(sample, pred_root, protocol)
    except Exception as e:
        if strict:
            raise
        rec = {"task": task, "uid": sample.uid, "subset": sample.subset, "id": sample.id, "protocol": protocol.name,
               "protocol_hash": protocol.hash(), "pred_sig": sig}
        if isinstance(e, InvalidPrediction):
            rec.update(status="invalid", reason=str(e))
        else:
            rec.update(status="error", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=-4))
    rec["run_id"] = run_id
    write_json(path, rec)
    return {"uid": sample.uid, "subset": sample.subset, "status": rec["status"], "reason": rec.get("reason", ""),
            "cached": False}


def _evaluate_one(args) -> dict:
    root, subset, sid, pred_root, out_dir, pdict, overwrite, run_id, strict = args
    from ..data.sample import Sample
    p = Track3DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    return evaluate_cached(TASK, s, p, pred_root, out_dir, prediction_files(pred_root, s, p), evaluate_sample,
                           overwrite, run_id, strict)


def _record_files(out_dir: Path, task: str):
    """(path, record) of every readable per-sample JSON of `task` in <out_dir>/<Subset>/."""
    for sub in SUBSETS:
        d = Path(out_dir) / sub
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            r = read_json(f)
            if isinstance(r, dict) and r.get("task") == task:
                yield f, r


def claim_out_dir(task: str, out_dir, protocol, run: dict, bench, pred_root, selected, overwrite: bool,
                  files_fn) -> int:
    """Tie `out_dir` to `run` (`run_fingerprint`) and write <out_dir>/run.json; returns the number of deleted records.

    Another run in run.json is refused unless overwrite. With overwrite, every record of `task` not of this run is
    deleted. An out dir of an older version (no run.json) is adopted: its records with the protocol's hash whose
    prediction files `files_fn(pred_root, sample, protocol)` match their ``pred_sig`` (so: the same resolved
    prediction root and unchanged files) get the run id; the other records of the protocol are refused unless their
    sample is re-scored in this call (uid in `selected`) or overwrite (deleted); records of other protocols stay and
    are counted as stale."""
    from ..data.sample import Sample
    out_dir = Path(out_dir)
    rid = run["run_id"]
    old = read_json(out_dir / RUN_FILE)
    if old is not None and old.get("run_id") != rid and not overwrite:
        what = "another protocol" if old.get("protocol_hash") != run["protocol_hash"] else \
            f"the predictions in {old.get('pred_root')}"
        raise RunDirError(f"{out_dir} holds {task} results of {what}; use another out dir or overwrite=True "
                          "(--overwrite), which deletes them")
    h = protocol.hash()

    def same_predictions(r) -> bool:
        try:
            s = Sample(bench.root, r["subset"], r["id"])
            return r.get("pred_sig") == prediction_signature(pred_root, files_fn(pred_root, s, protocol))
        except Exception:
            return False

    adopt, drop, foreign = [], [], []
    for f, r in _record_files(out_dir, task):
        if r.get("run_id") == rid:
            continue
        legacy = old is None and r.get("run_id") is None and r.get("protocol_hash") == h
        if legacy and same_predictions(r):
            adopt.append((f, r))
        elif overwrite:
            drop.append(f)
        elif legacy and r.get("uid") not in selected:
            foreign.append(r.get("uid"))
    if foreign:
        raise RunDirError(f"{out_dir} holds {len(foreign)} {task} result(s) of this protocol that do not match the "
                          f"prediction files in {run['pred_root']} (another prediction root, or files changed since), "
                          f"e.g. {foreign[0]}; use another out dir, evaluate those samples too, or overwrite=True "
                          "(--overwrite), which deletes them")
    for f, r in adopt:
        write_json(f, dict(r, run_id=rid))
    for f in drop:
        f.unlink()
    write_json(out_dir / RUN_FILE, run)
    return len(drop)


def select_samples(bench, subsets, ids, clip_filter=None, what: str = TASK) -> tuple:
    """(samples, unmatched): the samples of `iter_samples` and the requested ids / uids that match none of them,
    named in one warning prefixed with `what` (a typo, the id of a subset that is not selected, or a sample excluded
    by clip_filter); with no sample selected at all, a warning says so."""
    from ..utils import parse_list
    ids = canonical_ids(parse_list(ids))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\d+ requested id\(s\) match no sample")
        samples = list(iter_samples(bench, subsets, ids, clip_filter))
    seen = {s.id for s in samples} | {s.uid for s in samples}
    unmatched = [i for i in (ids or []) if i not in seen]
    if unmatched:
        warnings.warn(f"{what}: {len(unmatched)} requested id(s) match no selected sample (check --subsets, --ids and "
                      f"the protocol's clip_filter): {unmatched[:10]}")
    elif not samples:
        warnings.warn(f"{what}: no sample selected (subsets {subsets}, clip_filter {clip_filter or {}})")
    return samples, unmatched


def run_evaluation(task: str, worker, protocol, bench, pred_root, out_dir, subsets, ids, on_missing: str, jobs: int,
                   overwrite: bool, verbose: bool, summarize_fn, strict: bool = False, files_fn=None) -> dict:
    """Shared evaluate driver: protocol check of out_dir, the run of out_dir (`claim_out_dir`; files_fn(pred_root,
    sample, protocol) = the prediction files of a sample, default `prediction_files`), `worker` per sample (in `jobs`
    processes; args (root, subset, id, pred_root, out_dir, protocol dict, overwrite, run_id, strict)), status
    counts, warnings for failed samples and unmatched ids, on_missing, summary. The returned summary also has
    "evaluated" = `common.run_status` of the samples selected by this call (status counts, "all_failed", ...) with
    their "unmatched_ids" (not written to summary.json; the command line exits from it, `evaluation_exit_status`)."""
    if on_missing not in ("skip", "error"):
        raise ValueError("on_missing must be 'skip' or 'error'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfile = out_dir / "protocol.json"
    if pfile.exists() and not overwrite:
        old = read_json(pfile)
        try:
            same = type(protocol).from_dict(old).hash() == protocol.hash()
        except (AttributeError, KeyError, TypeError, ValueError):
            same = False
        if not same:
            raise RunDirError(f"{out_dir} holds results of a different protocol ({pfile}); use another out dir or "
                              "overwrite=True (--overwrite), which deletes them")
    samples, unmatched = select_samples(bench, subsets, ids, protocol.clip_filter, task)
    rid, run = run_fingerprint(task, protocol, pred_root)
    n_del = claim_out_dir(task, out_dir, protocol, run, bench, pred_root, {s.uid for s in samples}, overwrite,
                          files_fn or prediction_files)
    if n_del and verbose:
        print(f"[{task}] overwrite: deleted {n_del} record(s) of the previous run in {out_dir}")
    protocol.save(pfile)
    work = [(str(bench.root), s.subset, s.id, str(pred_root), str(out_dir), protocol.to_dict(), overwrite, rid,
             strict) for s in samples]
    res = _map(worker, work, jobs)
    counts = {}
    for r in res:
        c = counts.setdefault(r["subset"], {})
        c[r["status"]] = c.get(r["status"], 0) + 1
    if verbose:
        for sub, c in counts.items():
            n_cached = sum(r["cached"] for r in res if r["subset"] == sub)
            print(f"[{task}] {sub:14s} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                  + (f" ({n_cached} reused)" if n_cached else ""))
    failed = [r for r in res if r["status"] in FAILED_STATUSES]
    if failed:
        warnings.warn(f"{task}: {len(failed)} sample(s) failed (status invalid / error), e.g. {failed[0]['uid']}: "
                      f"{str(failed[0]['reason'])[:300]} (see 'reason' in the per-sample JSON; strict=True / --strict "
                      "raises)")
    bad = [r for r in res if r["status"] in ("missing", "partial")]
    if bad and on_missing == "error":
        raise FileNotFoundError(f"{len(bad)} sample(s) with missing predictions, e.g. {bad[0]['uid']}: "
                                f"{bad[0]['reason']}")
    summary = summarize_fn(out_dir)
    summary["evaluated"] = dict(run_status([r["status"] for r in res]), unmatched_ids=unmatched)
    return summary


def evaluate(bench, pred_root, out_dir, protocol: Track3DProtocol | None = None, subsets=None, ids=None,
             on_missing: str = "skip", verbose: bool = True, jobs: int = 1, overwrite: bool = False,
             strict: bool = False) -> dict:
    """Score every sample: writes ``<out_dir>/<Subset>/<id>.json`` and the summary (`summarize`), returns the summary
    (plus "evaluated": the status counts of this call, see `run_evaluation`).
    on_missing: "skip" (count it; the summary reports n_missing) or "error" (raise FileNotFoundError after scoring).
    jobs: worker processes. out_dir belongs to one run (protocol + prediction root, <out_dir>/run.json): another run is
    refused (RunDirError, a ValueError) unless overwrite, which deletes its records; records of this run with unchanged
    prediction files are reused unless overwrite. An unreadable or malformed prediction file gives status "invalid",
    any other failure of a sample status "error" (strict=True raises instead)."""
    p = get_protocol(protocol)
    return run_evaluation(TASK, _evaluate_one, p, bench, pred_root, out_dir, subsets, ids, on_missing, jobs, overwrite,
                          verbose, summarize, strict=strict, files_fn=prediction_files)


def oracle_prediction(sample, protocol: Track3DProtocol | None = None, case: dict | None = None) -> dict:
    """GT positions (world, or camera frame in monocular mode) and GT visibility; unknown GT entries are filled
    with the frozen value. Returns write_prediction kwargs."""
    p = get_protocol(protocol)
    c = case or task_case(sample, p)
    if not c["ok"]:
        raise ValueError(f"{sample.uid}: {c['reason']}")
    fill = frozen_prediction(sample, p, c)
    key = "xyz" if p.mode == "world" else "xyz_cam"
    gt = np.where(np.isfinite(c["gt"]), c["gt"], fill[key])
    return {key: gt, "vis": c["gt_vis"].copy(), "query_index": c["query_index"]}


def frozen_prediction(sample, protocol: Track3DProtocol | None = None, case: dict | None = None) -> dict:
    """Every query stays at its world position of t_q (monocular: seen through the given camera poses); no
    visibility (= always visible). Returns write_prediction kwargs."""
    p = get_protocol(protocol)
    c = case or task_case(sample, p)
    if not c["ok"]:
        raise ValueError(f"{sample.uid}: {c['reason']}")
    n = len(c["window"])
    X = np.broadcast_to(c["xyz_q"][None], (n,) + c["xyz_q"].shape).copy()
    if p.mode == "world":
        return {"xyz": X, "query_index": c["query_index"]}
    E = sample.cameras.w2c[c["views"][0]][c["window"]]
    return {"xyz_cam": np.einsum("tij,tnj->tni", E[:, :3, :3], X) + E[:, None, :3, 3], "query_index": c["query_index"]}


BASELINES = {"oracle": oracle_prediction, "frozen": frozen_prediction}


def _baseline_one(args) -> bool:
    root, subset, sid, pred_root, pdict, kind = args
    from ..data.sample import Sample
    p = Track3DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    c = task_case(s, p)
    if not c["ok"]:
        return False
    write_prediction(pred_root, s, **BASELINES[kind](s, p, c))
    return True


def run_baseline(bench, pred_root, kind: str = "frozen", protocol: Track3DProtocol | None = None, subsets=None,
                 ids=None, jobs: int = 1) -> dict:
    """Write the predictions of a reference baseline ("oracle" | "frozen") for every evaluable sample (float64,
    so the reference rows are exact); jobs = worker processes."""
    p = get_protocol(protocol)
    if kind not in BASELINES:
        raise KeyError(f"unknown baseline {kind!r}; available: {sorted(BASELINES)}")
    work = [(str(bench.root), s.subset, s.id, str(pred_root), p.to_dict(), kind)
            for s in select_samples(bench, subsets, ids, p.clip_filter, f"{TASK} baseline {kind}")[0]]
    res = _map(_baseline_one, work, jobs)
    return {"written": int(sum(res)), "skipped": int(len(res) - sum(res))}


def lifter_prediction(sample, protocol: Track3DProtocol | None, lifter, case: dict | None = None, size="preset",
                      resize_mode: str = "resize") -> dict:
    """Track the queries in the REAL window frames with a `vpm_worldbench.lifters.base.Lifter` (e.g.
    MVTrackerLifter = MoGe-2 depth + MVTracker). Only task inputs are used: frames on `size` ("preset" = the
    subset's generation eval grid, presets.SUBSET_DEFAULTS; None = stored resolution), K, w2c, the query points, and
    for depth anchoring the given depth of t_q when ``protocol.depth == "query"`` (Anchor "gt_depth"), else the median
    ratio at the queries visible in each view (Anchor "gt_points"); far clip from presets. Monocular mode tracks the
    one view and maps the world tracks into its camera frame. Returns write_prediction kwargs (with "vis")."""
    from ..lifters.base import Anchor
    p = get_protocol(protocol)
    c = case or task_case(sample, p)
    if not c["ok"]:
        raise ValueError(f"{sample.uid}: {c['reason']}")
    if isinstance(size, str):
        if size != "preset":
            raise ValueError("size must be 'preset', None or (W, H)")
        size = tuple(SUBSET_DEFAULTS[sample.subset]["eval_size"])
    inp = sample_inputs(sample, p, size=size, resize_mode=resize_mode, case=c)
    frames = inp["frames"]
    if isinstance(frames, list):
        raise ValueError(f"{sample.uid}: views of different sizes; pass size=(W, H)")
    n = len(c["window"])
    K = np.repeat(inp["K"][:, None], n, 1)
    far = float(SUBSET_DEFAULTS[sample.subset]["far_clip_m"])
    if inp["depth"] is not None:
        anchor = Anchor("gt_depth", depth=inp["depth"], far_clip_m=far)
    else:
        anchor = Anchor("gt_points", points=inp["queries_xyz"], points_vis=inp["queries_visible"], far_clip_m=far)
    anchor = lifter.prepare_anchor(anchor, frames, K, inp["w2c"])
    xyz, vis = lifter.track(frames, K, inp["w2c"], inp["lifter_queries"], anchor)
    xyz = np.asarray(xyz, np.float64)
    out = {"vis": np.asarray(vis, bool), "query_index": c["query_index"]}
    if p.mode == "world":
        out["xyz"] = xyz
    else:
        E = inp["w2c"][0]
        out["xyz_cam"] = np.einsum("tij,tnj->tni", E[:, :3, :3], xyz) + E[:, None, :3, 3]
    return out


def run_lifter_baseline(bench, pred_root, protocol: Track3DProtocol | None = None, lifter="mvtracker",
                        size="preset", subsets=None, ids=None, skip_existing: bool = False, verbose: bool = True,
                        **lifter_kwargs) -> dict:
    """Write `lifter_prediction` for every evaluable sample. lifter: a Lifter instance or a registered name
    (`vpm_worldbench.lifters.base.get_lifter`, kwargs go to its constructor, e.g. device="cuda", hub_repo=...,
    moge=...). Needs the [track] extra (torch, MVTracker, MoGe-2) and a GPU for the default lifter."""
    from ..lifters.base import get_lifter
    p = get_protocol(protocol)
    L = get_lifter(lifter, **lifter_kwargs) if isinstance(lifter, str) else lifter
    n_ok, n_skip, n_have = 0, 0, 0
    for s in select_samples(bench, subsets, ids, p.clip_filter, f"{TASK} baseline lifter")[0]:
        if skip_existing and prediction_path(pred_root, s).exists():
            n_have += 1
            continue
        c = task_case(s, p)
        if not c["ok"]:
            n_skip += 1
            continue
        write_prediction(pred_root, s, **lifter_prediction(s, p, L, c, size=size))
        n_ok += 1
        if verbose:
            print(f"[track3d] lifter {s.uid}: {len(c['query_index'])} queries", flush=True)
    return {"written": n_ok, "skipped": n_skip, "existing": n_have}


def _num(x) -> float:
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return float(x)
    if x is None:
        return float("nan")
    return float(x)


def mean_se(values) -> dict:
    """{"mean", "se", "n", "n_inf"} over the defined values (NaN / None dropped; +-inf kept, making the mean
    infinite: a failed sample is never averaged away; n_inf counts them); se = std(ddof=1) / sqrt(n), NaN for n < 2
    or an infinite value. (The geometry tasks' `common.mean_se` instead leaves +-inf out of the mean and counts it.)"""
    v = np.asarray([_num(x) for x in values], np.float64)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0, "n_inf": 0}
    with np.errstate(invalid="ignore"):
        m = float(np.mean(v))
        se = float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 and np.isfinite(v).all() else float("nan")
    return {"mean": m, "se": se, "n": n, "n_inf": int(np.isinf(v).sum())}


def aggregate(dicts) -> dict:
    """mean_se of every numeric key over a list of flat metric dicts."""
    keys = []
    for d in dicts:
        for k, v in d.items():
            if k not in keys and (isinstance(v, (int, float)) or v is None or v in ("inf", "-inf")) \
                    and not isinstance(v, bool):
                keys.append(k)
    return {k: mean_se([d.get(k) for d in dicts]) for k in keys}


def load_records(out_dir, task: str) -> dict:
    """{subset: [per-sample records]} from <out_dir>/<Subset>/*.json of one task (unreadable files are skipped)."""
    out = {}
    for f, r in _record_files(out_dir, task):
        out.setdefault(f.parent.name, []).append(r)
    return out


def fmt_ms(x: dict, nd: int = 2) -> str:
    m, se = x.get("mean"), x.get("se")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "–"
    if isinstance(m, str) or math.isinf(m):
        return str(m)
    if se is None or (isinstance(se, float) and math.isnan(se)) or isinstance(se, str):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} ± {se:.{nd}f}"


def status_counts(recs) -> dict:
    """{"n_samples", "n_<status>" for ok / partial / skipped / missing / invalid / error, "skip_reasons"}."""
    c = {"n_samples": len(recs)}
    for st in ("ok", "partial", "skipped", "missing", "invalid", "error"):
        c[f"n_{st}"] = sum(r.get("status") == st for r in recs)
    reasons = {}
    for r in recs:
        if r.get("status") == "skipped":
            reasons[r.get("reason", "")] = reasons.get(r.get("reason", ""), 0) + 1
    c["skip_reasons"] = reasons
    return c


def current_records(out_dir, task: str, protocol_cls) -> tuple:
    """(protocol dict or None, {subset: records of the out dir's run}, {subset: n stale}). With <out_dir>/run.json,
    the records of that run (``run_id``); in an out dir of an older version (no run.json), the records whose
    protocol_hash matches protocol.json. Left-out records (other runs, other protocols) are counted as stale; without
    either file every record is kept (a warning names mixed protocols)."""
    out_dir = Path(out_dir)
    by_sub = load_records(out_dir, task)
    pfile = out_dir / "protocol.json"
    proto = read_json(pfile)
    run = read_json(out_dir / RUN_FILE)
    stale = {}
    key, want, why = None, None, ""
    if isinstance(run, dict) and run.get("run_id"):
        key, want = "run_id", run["run_id"]
        why = f"are not of the run in {RUN_FILE} (another protocol or prediction root, or an older version)"
    elif proto is not None:
        try:
            key, want = "protocol_hash", protocol_cls.from_dict(proto).hash()
            why = "were written with another protocol than protocol.json"
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            key = None
            warnings.warn(f"{task}: cannot read {pfile} ({e}); the summary keeps every record")
    if key is not None:
        for sub, recs in list(by_sub.items()):
            kept = [r for r in recs if r.get(key) == want]
            stale[sub] = len(recs) - len(kept)
            by_sub[sub] = kept
        n = sum(stale.values())
        if n:
            warnings.warn(f"{task}: {n} record(s) in {out_dir} {why}; they are left out of the summary (n_stale)")
    elif proto is None:
        hs = {r.get("protocol_hash") for recs in by_sub.values() for r in recs}
        if len(hs) > 1:
            warnings.warn(f"{task}: {out_dir} mixes records of {len(hs)} protocols and has no protocol.json")
    return proto, by_sub, stale


def summarize(out_dir) -> dict:
    """Aggregate the per-sample JSONs of `out_dir` into ``summary.json`` and ``summary.md`` (mean ± s.e. over the
    samples of each subset: the main group of each sample, and every group by name over the samples that have it).
    Only the records of the out dir's run are used (`current_records`; others are counted as ``n_stale``);
    ``pred_root`` = its prediction root (None in an out dir of an older version). Returns the summary."""
    out_dir = Path(out_dir)
    proto, by_sub, stale = current_records(out_dir, TASK, Track3DProtocol)
    summary = {"task": TASK, "protocol": proto, "pred_root": run_pred_root(out_dir), "subsets": {}}
    for sub, recs in by_sub.items():
        if not recs:
            continue
        ok = [r for r in recs if r.get("status") == "ok"]
        mains = sorted({r["main_group"] for r in ok})
        groups = {}
        for r in ok:
            for g, m in r["groups"].items():
                if m.get("n_queries", 0) > 0:
                    groups.setdefault(g, []).append(m)
        summary["subsets"][sub] = {
            **status_counts(recs), "n_stale": stale.get(sub, 0), "main_group": mains[0] if len(mains) == 1 else mains,
            "main": aggregate([r["groups"][r["main_group"]] for r in ok
                               if r["groups"][r["main_group"]].get("n_queries", 0) > 0]),
            "groups": {g: aggregate(v) for g, v in groups.items()},
            "group_n_samples": {g: len(v) for g, v in groups.items()},
            "nonfinite_samples": sum(r["prediction"]["nonfinite_frac"] > 0 for r in ok)}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_summary_md(summary))
    return summary


def _summary_md(summary: dict) -> str:
    p = summary.get("protocol") or {}
    where = f"view {p.get('mono_view')}, camera frame" if p.get("mode") == "monocular" else \
        f"views {p.get('views') or 'all'}, world frame"
    stride = p.get("flow_stride") or f"{p.get('flow_dt_s')} s"
    lines = [f"# track3d ({p.get('name', '?')}, mode {p.get('mode', '?')}: {where})", "",
             f"Queries at frame {p.get('ref_frame')}, horizon {p.get('horizon')}, step {p.get('frame_step')}; "
             "mean ± s.e. over samples; δ / AJ / OA / survival / Acc3D / outliers in %, errors in metres. "
             "(main) = sample.json default_report_group. First column: samples scored / in the subset, missing "
             "predictions (and failed samples: invalid = an unreadable or malformed prediction file, error = another "
             "failure; reason in the per-sample JSON); "
             "n = samples with points of that group. Scene flow over a stride of "
             f"{stride} on the entries moving >= {p.get('flow_min_motion_m')} m over the stride ('mov'; all-entry "
             "EPE3D and Acc3D in summary.json). A non-finite predicted position is a failure: its infinite ADE / "
             "MTE / EPE3D is kept, so the mean shows inf (summary.json n_inf counts such samples).", "",
             "| subset (scored / total, missing) | group | n | " + " | ".join(c[1] for c in _MD_COLUMNS) + " |",
             "|---|---|---|" + "---|" * len(_MD_COLUMNS)]
    for sub, s in summary["subsets"].items():
        main = s["main_group"]
        gn = s.get("group_n_samples", {})
        rows = [] if isinstance(main, str) else [("main", s["main"], s["n_ok"])]
        rows += [(f"{g} (main)" if g == main else g, a, gn.get(g, "?")) for g, a in s["groups"].items()]
        for i, (g, a, n) in enumerate(rows):
            stale = f", {s['n_stale']} stale" if s.get("n_stale") else ""
            err = "".join(f", {s[f'n_{k}']} {k}" for k in FAILED_STATUSES if s.get(f"n_{k}"))
            head = f"| {sub} ({s['n_ok']} / {s['n_samples']}, {s['n_missing']} missing{err}{stale}) |" if i == 0 \
                else "| |"
            lines.append(head + f" {g} | {n} | "
                         + " | ".join(fmt_ms(a.get(k, {}), nd) for k, _, nd in _MD_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


export_track3d_inputs = export_inputs
evaluate_track3d = evaluate


def iter_samples(bench, subsets=None, ids=None, clip_filter=None):
    """Samples of `bench` in the given subsets (all by default; names, aliases, or comma-separated strings),
    restricted to `ids` (ids or uids such as "Real-Robot/episode_000" or "rr/episode_000"; a list or a comma-separated
    string) and the attribute filters. An unknown subset raises KeyError."""
    from ..utils import parse_list, parse_subsets
    subs = parse_subsets(subsets) or bench.subsets()
    ids = canonical_ids(parse_list(ids))
    for sub in subs:
        yield from bench.samples(sub, ids=ids, **(clip_filter or {}))


def canonical_ids(ids):
    """Sample ids / uids with the subset part of a uid canonicalised ("rr/x" -> "Real-Robot/x", `utils.canonical_uid`;
    an unknown subset part is kept as given); None stays None."""
    return None if ids is None else [canonical_uid(i) for i in ids]


def read_json(path):
    """Parsed JSON file, or None when it does not exist or cannot be parsed."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def _map(fn, work, jobs: int):
    """[fn(w) for w in work], in `jobs` worker processes when jobs > 1 ("spawn": video decoders are threaded, and
    fork() of a threaded process can deadlock; a calling script needs an ``if __name__ == "__main__":`` guard)."""
    if jobs and jobs > 1 and len(work) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            return list(ex.map(fn, work, chunksize=1))
    return [fn(w) for w in work]


def jsonable(x):
    """JSON-safe copy: numpy scalars / arrays -> Python, NaN -> None, +-inf -> "inf" / "-inf"."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return x


def write_json(path, obj) -> Path:
    """Atomic write of jsonable(obj) (a killed run never leaves a truncated record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path


def _parse_size(s):
    from ..utils import parse_size
    return parse_size(s)


def add_common_args(q, cmd: str, protocol_help: str):
    """The options every task CLI shares (names fixed across tasks): --root, --pred, --out, --protocol,
    --subsets, --ids, --jobs, --overwrite, --strict."""
    if cmd != "summarize":
        q.add_argument("--root", required=True, help="VP-WorldBench data root")
        q.add_argument("--protocol", default=None, help=protocol_help)
        q.add_argument("--subsets", nargs="*", default=None, help="subset names or aliases (space / comma separated)")
        q.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (space / comma separated)")
    if cmd in ("baseline", "evaluate"):
        q.add_argument("--pred", required=True, help="prediction root")
    if cmd in ("export", "evaluate", "summarize"):
        q.add_argument("--out", required=True, help="output folder")
    if cmd in ("export", "baseline", "evaluate"):
        q.add_argument("--jobs", type=int, default=1, help="worker processes")
    if cmd == "export":
        q.add_argument("--overwrite", action="store_true",
                       help="redo existing samples / replace an out dir of another protocol")
    if cmd == "evaluate":
        q.add_argument("--overwrite", action="store_true", help="re-score the selected samples / replace the results "
                       "of another run (protocol or prediction root; its records are deleted)")
        q.add_argument("--strict", action="store_true",
                       help="raise on the first failed sample (status invalid / error) instead of recording it")


def cli_setup(q, a, protocol_fn):
    """Check a parsed task command line and resolve its protocol (`protocol_fn(a)`): a --root that is no folder, an
    unknown subset, a --protocol that is neither a preset nor a readable protocol file, or an invalid override is a
    parser error of the subcommand parser `q` (usage + one line on stderr, exit status 2) instead of a traceback, as
    in `common.task_cli`."""
    from ..utils import parse_subsets
    if not Path(a.root).is_dir():
        q.error(f"--root {a.root}: no such directory")
    try:
        parse_subsets(getattr(a, "subsets", None))
    except KeyError as e:
        q.error(f"argument --subsets: {e.args[0] if e.args else e}")
    try:
        return protocol_fn(a)
    except (OSError, ImportError, KeyError, TypeError, ValueError) as e:
        q.error(f"invalid protocol (--protocol {a.protocol or 'default'} and the protocol options): "
                f"{e.args[0] if isinstance(e, KeyError) and e.args else e}")


def _device(name: str) -> str:
    """'auto' -> 'cuda' when PyTorch sees a GPU, else 'cpu'; other names as given."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _protocol_from_args(a) -> Track3DProtocol:
    p = get_protocol(a.protocol or None)
    over = {}
    if a.mode:
        over["mode"] = a.mode
    if a.view is not None:
        over["mono_view"] = a.view
    if a.ref_frame is not None:
        over["ref_frame"] = a.ref_frame
    if a.horizon is not None:
        over["horizon"] = None if a.horizon <= 0 else a.horizon
    if a.frame_step is not None:
        over["frame_step"] = a.frame_step
    if a.views:
        from ..utils import parse_list
        over["views"] = [int(v) for v in parse_list(a.views)]
    if getattr(a, "depth", None):
        over["depth"] = a.depth
    if a.flow_stride is not None:
        over["flow_stride"] = None if a.flow_stride <= 0 else a.flow_stride
    return override_protocol(p, over, Track3DProtocol)


def main(argv=None) -> int:
    import argparse
    from ..data.benchmark import Benchmark
    ap = argparse.ArgumentParser(prog="vpmwb track3d",
                                 description="track3d (multi-view / monocular 3D point tracking): export inputs, "
                                             "write baselines, evaluate, summarize")
    sp = ap.add_subparsers(dest="cmd", required=True)
    helps = {"export": "write the model inputs", "baseline": "write reference predictions (--kind)",
             "evaluate": "score predictions", "summarize": "rebuild summary.json / summary.md"}
    parsers = {}
    for name in ("export", "baseline", "evaluate", "summarize"):
        q = parsers[name] = sp.add_parser(name, help=helps[name], description=helps[name])
        add_common_args(q, name, "track3d-v1 (default) | legacy-prerelease | <json/yaml file>")
        if name != "summarize":
            q.add_argument("--mode", choices=MODES, default=None)
            q.add_argument("--view", type=int, default=None, help="monocular view")
            q.add_argument("--views", nargs="*", default=None, help="world mode: view indices")
            q.add_argument("--ref-frame", type=int, default=None)
            q.add_argument("--horizon", type=int, default=None, help="<= 0: to the end of the clip")
            q.add_argument("--frame-step", type=int, default=None)
            q.add_argument("--flow-stride", type=int, default=None, help="scene-flow stride; <= 0: from flow_dt_s")
        if name == "export":
            q.add_argument("--frames", choices=FRAME_EXPORTS, default="none")
            q.add_argument("--size", default=None, help="WxH (png / mp4 frames only)")
            q.add_argument("--resize-mode", default="resize", choices=("resize", "center_crop", "letterbox"))
            q.add_argument("--depth", choices=DEPTH_INPUTS, default=None)
        elif name == "baseline":
            q.add_argument("--kind", choices=sorted(BASELINES) + ["lifter"], default="frozen")
            q.add_argument("--lifter", default="mvtracker", help="kind lifter: registered lifter name")
            q.add_argument("--device", default="auto", help="kind lifter: torch device ('auto' = cuda if available)")
            q.add_argument("--hub-repo", default=None, help="kind lifter: MVTracker torch.hub repo or local clone")
            q.add_argument("--moge", default=None, help="kind lifter: MoGe-2 HF repo id or local model.pt")
            q.add_argument("--lifter-size", default="preset", help="kind lifter: 'preset', 'native' or WxH")
            q.add_argument("--skip-existing", action="store_true", help="kind lifter: keep existing predictions")
        elif name == "evaluate":
            q.add_argument("--on-missing", choices=("skip", "error"), default="skip")
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        if not Path(a.out).is_dir():
            parsers["summarize"].error(f"--out {a.out}: no such directory")
        summarize(a.out)
        print((Path(a.out) / "summary.md").read_text())
        return 0
    q = parsers[a.cmd]
    p = cli_setup(q, a, _protocol_from_args)
    bench = Benchmark(a.root)
    if a.cmd == "export":
        size = cli_size(q, "--size", a.size)
        if size is not None and a.frames == "none":
            q.error("--size needs --frames png or mp4 (--frames none references the native videos)")
        res = export_inputs(bench, a.out, p, a.subsets, a.ids, frames=a.frames, size=size, resize_mode=a.resize_mode,
                            jobs=a.jobs, overwrite=a.overwrite)
    elif a.cmd == "baseline" and a.kind == "lifter":
        from ..lifters.base import get_lifter
        size = "preset" if a.lifter_size == "preset" else cli_size(q, "--lifter-size", a.lifter_size)
        kw = {"device": _device(a.device)}
        if a.hub_repo:
            kw["hub_repo"] = a.hub_repo
        if a.moge:
            kw["moge"] = a.moge
        try:
            lifter = get_lifter(a.lifter, **kw)
        except KeyError as e:
            q.error(f"--lifter: {e.args[0] if e.args else e}")
        res = run_lifter_baseline(bench, a.pred, p, lifter, size, a.subsets, a.ids, skip_existing=a.skip_existing)
    elif a.cmd == "baseline":
        res = run_baseline(bench, a.pred, a.kind, p, a.subsets, a.ids, jobs=a.jobs)
    else:
        summ = evaluate(bench, a.pred, a.out, p, a.subsets, a.ids, on_missing=a.on_missing, jobs=a.jobs,
                        overwrite=a.overwrite, strict=a.strict)
        print((Path(a.out) / "summary.md").read_text())
        return evaluation_exit_status(summ, TASK, a.out)
    print(res)
    return selection_exit_status(res, TASK)


if __name__ == "__main__":
    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
