"""Task ``track2d``: TAP-Vid style 2D point tracking per view, plus a cross-view correspondence sub-task.

Tracking (per view)
-------------------
Each view of each sample is one video: given the frames of the window on the eval grid and 2D query points
(t, u, v), predict the 2D position and the visibility of every query in every frame of the window. Views are scored
independently; a method may still use all views jointly (multi-view 2D tracking) - the cameras are in the inputs.

Protocol (`Track2DProtocol`, conventions of `vpm_worldbench.protocol.Protocol`)
    window      positions j = 0..n-1 <-> GT frames ``ref_frame + j * frame_step`` (default 16..80, 65 frames;
                ``horizon=None`` = up to the end of the clip).
    eval grid   every view is mapped to ``eval_size`` (W, H) (default 256x256, TAP-Vid) with ``resize_mode``
                (geometry.image_map: "resize" may change the aspect, "center_crop", "letterbox"); all coordinates
                (queries, predictions, GT) are pixels of that grid, integer = pixel centre, x = u right, y = v down.
                Both accept a {subset: value} dict (subset names or aliases; unknown subsets raise ValueError).
    GT          `Sample.tracks2d` of the view on the eval grid (projected GT tracks; visible = the stored
                occlusion-aware flag of that view & inside the grid), restricted to ``default_gt`` frames: a frame
                whose GT position is unknown counts as occluded.
    queries     ``query_mode``:
                "first"    one track per point visible at least once in the window, queried at its first visible
                           window position (TAP-Vid "query first"); scored frames are those after the query.
                "strided"  one track per (point, j) for j = 0, query_stride, 2*query_stride, ... with the point
                           visible at j (TAP-Vid "strided", stride 5); every frame except the query frame is scored.
                           About 10x more tracks than "first" (default window: Simul-Robot ~8.5k per view on
                           average, up to 15k); use ``max_points`` to bound the cost.
                "fixed"    one track per point visible at window position 0 (= t_q, the benchmark query frame);
                           frames after it are scored.
                ``max_points`` (default None = all) keeps a seeded, deterministic subset of the eligible points.
                The tracks of a view are listed in a seeded random order with OPAQUE ids (``track_id`` = row
                number): the inputs never say which GT point a query belongs to, so queries of the same point (the
                strided queries of one point, or one point seen in several views) cannot be linked by an id. The
                (point, query) mapping stays inside the evaluator. Treat every query independently, as in TAP-Vid:
                a method must not snap its tracks onto other queries (they are GT positions of visible points).
    Metrics     `metrics.tracking.track2d_metrics` (port of tapnet's compute_tapvid_metrics): occlusion accuracy
                (OA), pts_within_{1,2,4,8,16}, jaccard_{x}, average_jaccard (AJ), average_pts_within_thresh
                (<δ^x_avg>), in percent, pooled over the tracks and frames of one view; thresholds apply after
                rescaling to ``normalize_to`` (default 256x256, i.e. the eval grid itself). Per sample: mean over
                its views; per subset: mean ± s.e. over samples. A sample's tracking result counts only when EVERY
                view with tracks has a prediction file; otherwise it is "missing" (the per-view metrics of the
                present views stay in the record for diagnosis, but the sample is not averaged).

Cross-view correspondence (``corr=True``)
-----------------------------------------
At each corr frame t (``corr_frames``, GT frames; default [ref_frame // 2] = [8], BEFORE the tracking window - corr
frames on the tracking window are rejected, because tracking queries there would reveal the targets) every point
GT-visible (and ``default_gt``, inside the grid) in at least two views is a correspondence point. For each such
point ONE source view a is drawn (seeded) among the views that see it, and it yields one query per other view b that
sees it: given (u, v) of the point in view a at t, predict its (u, v) in view b at the same t, on the eval grid. GT
points at the same 3D position at t count once (Real-Robot has one duplicated hand track per episode). So the
inputs never contain the position of a correspondence point in a target view at a corr frame (no reverse pairs). Queries are listed in a seeded random order with opaque ids (``query_id`` = row number); ``corr_max_points``
caps the points per frame (seeded). Corr frames outside the clip are dropped for that sample (listed in the
record); a sample without any correspondence point has corr status "no_queries".
Metrics: ``pct_within_{x}`` = % of queries with error < x px for x in {1, 2, 4, 8, 16} (on the ``normalize_to`` grid
= the eval grid by default), ``avg_pct_within`` (their mean), ``mean_err_px`` / ``median_err_px``; pooled over all
queries of a sample (per ordered view pair in ``pairs``), then mean ± s.e. over samples. The cameras of the views
at the corr frames are in the inputs: geometry-aware methods may use them (please report it). Use only the corr
inputs for this sub-task: the benchmark folder holds the GT itself, and the inputs of the other tasks (tracking
queries, track3d queries at frame 16) contain the same points at other frames - for a static point seen by a
static camera these positions equal the corr-frame positions.

Inputs (`sample_inputs` in Python, `export_inputs` on disk under ``<out>/<Subset>/<id>/``)
    frames          per view, the n window frames on the eval grid: ``frames="mp4"`` (default; frames/view{k}.mp4,
                    lossless libx264rgb), "png" (frames/view{k}/frame_{t:04d}.png, t = GT frame), "none" (use the
                    benchmark's videos/view{k}.mp4 at its native size; inputs.json gives the native -> eval-grid
                    map u' = sx (u + 0.5) - 0.5 + ox, same for v).
    queries         queries_view{k}.npz: ``track_id`` (Nt,) int32 (= 0..Nt-1, the row number), ``query_t`` (Nt,)
                    int32 window position, ``query_frame`` (Nt,) GT frame, ``query_uv`` (Nt, 2) float64 [u, v] on the
                    eval grid. NOTE the order [u, v] = [x, y] (TAP-Vid's own query_points are [t, y, x]).
    cameras         inputs.json ``cameras.view{k}``: K (3, 3) on the eval grid, w2c (n, 4, 4) of the window frames.
    correspondence  corr/queries.npz: ``query_id`` (P,) int32 (row number), ``frame`` (P,) GT frame, ``view_a`` /
                    ``view_b`` (P,) int32, ``uv_a`` (P, 2) float64 on the eval grid; images
                    corr/view{k}_frame_{t:04d}.png on the eval grid (frames != "none"; with "none" decode frame t of
                    the native video); inputs.json ``corr.cameras.view{k}``: K on the eval grid and w2c
                    (len(frames), 4, 4) at the corr frames.

Prediction files
    ``<pred_root>/<Subset>/<id>/view{k}.npz``: ``uv`` (n, Nt, 2) float [u, v] on the eval grid, rows = window
        positions, columns = tracks in the order of queries_view{k}.npz; optional ``vis`` (n, Nt) bool or score
        (occluded iff < 0.5; non-finite = occluded; missing = always visible); optional ``track_id`` (Nt,): a
        permutation of 0..Nt-1 naming the query of each column (columns are reordered).
    ``<pred_root>/<Subset>/<id>/corr.npz``: ``uv_b`` (P, 2) [u, v] in view b on the eval grid, in the order of
        corr/queries.npz (``uv`` accepted as an alias); optional ``query_id`` (P,) permutation.
    Non-finite coordinates are FAILURES (+inf distance: never within a threshold, an FP when predicted visible).

Ground truth per subset (see track3d): Simul-* exact synthetic tracks and renderer/depth-test visibility;
Real-Robot forward kinematics (robot self-occlusion only, calibration ~2 px - a few px of GT noise at native
resolution, ~0.4-0.7 px on the 256 grid); Real-General PSEUDO-GT (triangulated / body-model tracks, reconstruction
visibility). Every subset has 2D GT (projections of the 3D tracks).

Baselines (`run_baseline` / `BASELINES`, float64 predictions)
    oracle  GT positions and visibility (unknown entries filled with the query) and the GT correspondences -> 100 %.
    frozen  every track stays at its query position, always visible ("zero motion"); correspondence = identity
            (the query's (u, v) copied to view b, a floor: ~0 % on the wide-baseline Real-* rigs).

Results: ``evaluate`` writes ``<out>/<Subset>/<id>.json`` ("track" and "corr" each with their own status: ok / missing /
no_tracks / no_queries / disabled) and calls ``summarize`` -> ``<out>/summary.json`` + ``summary.md`` (mean ± s.e. over
samples; scored / total samples, missing samples and missing view files per subset). Sample statuses (the vocabulary of
every task, `common.STATUSES`): "ok" every part with queries is scored; "partial" one part (tracking or corr) is scored
and the other is missing; "missing" nothing scored for want of prediction files; "skipped" not evaluable; "invalid" a
prediction file that cannot be read or is malformed (keys, dtypes, shapes, ids; ``reason`` names the file); "error"
any other failure (``reason`` + ``traceback``). Invalid / error samples are counted (``n_invalid`` / ``n_error``),
named in a warning and scored again by the next ``evaluate``; ``strict=True`` / ``--strict`` raises instead;
``on_missing="error"`` raises for partial and missing samples. NaN is written as null, infinities as "inf" / "-inf"
(an infinite corr error is kept in the mean, see track3d `mean_se`). Run bookkeeping as in track3d: an out dir holds
one run (protocol + resolved prediction root, ``<out>/run.json``; another run is refused unless ``overwrite``, which
deletes its records; the summary covers only the run's records, writes its ``pred_root`` and counts the others as
``n_stale``), reuse of records with unchanged prediction files, a warning for requested ids that match no selected
sample, renamed protocols for command-line overrides (a protocol file without ``name`` is named after the file),
``jobs`` worker processes that need an ``if __name__ == "__main__":`` guard in scripts.
Every function taking ``protocol`` also accepts None (default), "track2d-v1" or a json / yaml file (`get_protocol`;
anything else raises ValueError).

    from vpm_worldbench import Benchmark
    from vpm_worldbench.tasks import track2d
    bench = Benchmark("/data/VP-WorldBench")
    p = track2d.Track2DProtocol(query_mode="first")
    for s in bench.samples("Simul-Robot"):
        inp = track2d.sample_inputs(s, p)
        for k, v in enumerate(inp["views"]):
            q = inp["queries"][k]                                     # query_t, query_uv (rows = track_id)
            uv, vis = my_tracker(inp["frames"][k], q["query_t"], q["query_uv"])
            track2d.write_prediction("preds", s, v, uv, vis)
        c = inp["corr"]                                               # frame, view_a, view_b, uv_a, images
        track2d.write_corr_prediction("preds", s, my_matcher(c))
    track2d.evaluate(bench, "preds", "results/track2d", p)

    vpmwb track2d export   --root R --out inputs/ [--query-mode strided] [--size 256x256] [--jobs 8]
    vpmwb track2d baseline --root R --pred preds/ --kind frozen          (oracle | frozen)
    vpmwb track2d evaluate --root R --pred preds/ --out results/ [--no-corr] [--jobs 8] [--overwrite] [--strict]
    vpmwb track2d summarize --out results/
(``--horizon`` <= 0 = to the end of the clip; ``--frame-step k`` scores every k-th frame. Exit status as for track3d: 1
when the selection matches no sample or none of its evaluable samples is scored, 2 for usage errors.)
"""
from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..geometry import RESIZE_MODES
from ..metrics.tracking import TAPVID_THRESHOLDS_PX, threshold_key, track2d_metrics
from .common import FAILED_STATUSES, InvalidPrediction, evaluation_exit_status, read_npz, selection_exit_status
from .track3d import (_finish_export, _map, _parse_size, add_common_args, aggregate, canonical_subset_keys,
                      check_out_dir, check_protocol_spec, cli_setup, current_records, evaluate_cached, fmt_ms,
                      index_array, load_protocol_file, override_protocol, prediction_signature, protocol_hash,
                      real_array, run_evaluation, run_pred_root, select_samples, status_counts, write_json)

TASK = "track2d"
PRESETS = ("track2d-v1",)
QUERY_MODES = ("first", "strided", "fixed")
FRAME_EXPORTS = ("mp4", "png", "none")
_RNG_CAP, _RNG_ORDER, _RNG_CORR_CAP, _RNG_CORR_SOURCE, _RNG_CORR_ORDER = range(5)


@dataclass
class Track2DProtocol:
    """Protocol of the track2d task (see the module docstring). ``eval_size`` and ``resize_mode`` accept a dict
    {subset: value} (aliases allowed; missing subsets use (256, 256) / "resize").

    name (a protocol file without ``name`` is named after the file), ref_frame (window start; 'fixed' query
    frame), horizon (None = to the end of the clip), frame_step,
    views (None = all), eval_size (W, H), resize_mode, query_mode ('first' | 'strided' | 'fixed'), query_stride
    (strided: query every k window positions), thresholds_px, normalize_to ((W', H') grid of the thresholds; None =
    eval-grid pixels), max_points (None = all eligible points), seed, corr (run the correspondence sub-task),
    corr_frames (GT frames off the tracking window; None = [ref_frame // 2], none when ref_frame = 0),
    corr_max_points (points per corr frame; None = all), clip_filter.
    """
    name: str = "track2d-v1"
    ref_frame: int = 16
    horizon: Optional[int] = 64
    frame_step: int = 1
    views: Optional[list] = None
    eval_size: Any = (256, 256)
    resize_mode: Any = "resize"
    query_mode: str = "first"
    query_stride: int = 5
    thresholds_px: tuple = TAPVID_THRESHOLDS_PX
    normalize_to: Optional[tuple] = (256, 256)
    max_points: Optional[int] = None
    seed: int = 0
    corr: bool = True
    corr_frames: Optional[list] = None
    corr_max_points: Optional[int] = None
    clip_filter: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {self.query_mode!r}")
        if self.ref_frame < 0 or self.frame_step < 1 or self.query_stride < 1 or \
                (self.horizon is not None and self.horizon < 1):
            raise ValueError("need ref_frame >= 0, frame_step >= 1, query_stride >= 1, horizon >= 1 (or None)")
        for k in ("max_points", "corr_max_points"):
            if getattr(self, k) is not None and getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1 or None")
        self.thresholds_px = tuple(float(x) for x in self.thresholds_px)
        if self.normalize_to is not None:
            self.normalize_to = tuple(int(x) for x in self.normalize_to)
        self.eval_size = canonical_subset_keys(self.eval_size, "eval_size")
        self.resize_mode = canonical_subset_keys(self.resize_mode, "resize_mode")
        if isinstance(self.eval_size, dict):
            self.eval_size = {k: tuple(int(x) for x in v) for k, v in self.eval_size.items()}
        else:
            self.eval_size = tuple(int(x) for x in self.eval_size)
        sizes = self.eval_size.values() if isinstance(self.eval_size, dict) else [self.eval_size]
        if any(len(z) != 2 or min(z) < 1 for z in sizes):
            raise ValueError(f"eval_size must be (W, H) with W, H >= 1 (or a dict of them), got {self.eval_size}")
        modes = self.resize_mode.values() if isinstance(self.resize_mode, dict) else [self.resize_mode]
        if any(m not in RESIZE_MODES for m in modes):
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if self.views is not None:
            self.views = [int(v) for v in self.views]
        if self.corr_frames is not None:
            self.corr_frames = [int(t) for t in self.corr_frames]
        bad = [t for t in self.corr_frames_for() if self.on_window(t)]
        if self.corr and bad:
            raise ValueError(f"corr_frames {bad} are tracking-window frames (window {self.ref_frame}, "
                             f"{self.ref_frame + self.frame_step}, ..., horizon {self.horizon}): the tracking queries "
                             "there would reveal the correspondence targets; choose frames outside the window")

    def eval_size_for(self, subset: str) -> tuple:
        from ..constants import canonical_subset
        v = self.eval_size
        if isinstance(v, dict):
            v = v.get(canonical_subset(subset), (256, 256))
        return tuple(int(x) for x in v)

    def resize_mode_for(self, subset: str) -> str:
        from ..constants import canonical_subset
        v = self.resize_mode
        return v.get(canonical_subset(subset), "resize") if isinstance(v, dict) else v

    def window(self, sample) -> np.ndarray:
        """(n,) GT frame indices of the window positions."""
        h = (sample.n_frames - 1 - self.ref_frame) // self.frame_step if self.horizon is None else self.horizon
        return self.ref_frame + self.frame_step * np.arange(max(h, 0) + 1)

    def on_window(self, t: int) -> bool:
        """True if GT frame t is (or, with horizon None, may be) a window position of some clip."""
        d = int(t) - self.ref_frame
        return d >= 0 and d % self.frame_step == 0 and (self.horizon is None or d // self.frame_step <= self.horizon)

    def views_for(self, sample) -> list:
        return list(range(sample.n_views)) if self.views is None else list(self.views)

    def corr_frames_for(self, sample=None) -> list:
        """Corr frames of the protocol (sample=None) or of one sample (frames outside its clip dropped)."""
        fr = list(self.corr_frames) if self.corr_frames is not None else \
            ([self.ref_frame // 2] if self.ref_frame >= 1 else [])
        if sample is None:
            return fr
        return [t for t in fr if 0 <= t <= sample.n_frames - 1]

    def evaluable(self, sample) -> tuple:
        if self.ref_frame > sample.n_frames - 1:
            return False, f"window start {self.ref_frame} > last frame {sample.n_frames - 1}"
        win = self.window(sample)
        if len(win) < 2:
            return False, "window has fewer than 2 frames"
        if win[-1] > sample.n_frames - 1:
            return False, f"window ends at {int(win[-1])} > last frame {sample.n_frames - 1}"
        views = self.views_for(sample)
        if not views or min(views) < 0 or max(views) >= sample.n_views:
            return False, f"views {views} not in 0..{sample.n_views - 1}"
        return True, ""

    def hash(self) -> str:
        return protocol_hash(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                d[k] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Track2DProtocol":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown track2d protocol fields {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path) -> "Track2DProtocol":
        """A json / yaml file of protocol fields (named after the file when it has no ``name``)."""
        return cls.from_dict(load_protocol_file(path))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


def get_protocol(spec=None) -> Track2DProtocol:
    """None / "" / "track2d-v1" -> default; a Track2DProtocol as is; otherwise a json / yaml file path (ValueError if
    it is not a file)."""
    if isinstance(spec, Track2DProtocol):
        return spec
    if spec in (None, "", "track2d-v1", "default"):
        return Track2DProtocol()
    check_protocol_spec(spec, TASK, PRESETS)
    return Track2DProtocol.load(spec)


def _rng(protocol: Track2DProtocol, uid: str, *keys) -> np.random.Generator:
    return np.random.default_rng([int(protocol.seed), zlib.crc32(uid.encode("utf-8"))] + [int(k) for k in keys])


def _cap(points: np.ndarray, limit, rng: np.random.Generator) -> np.ndarray:
    """Deterministic subset of at most `limit` of the sorted point indices `points` (kept sorted)."""
    if limit is None or len(points) <= limit:
        return points
    return points[np.sort(rng.choice(len(points), limit, replace=False))]


def view_gt(sample, view: int, protocol: Track2DProtocol, indices) -> tuple:
    """GT on the eval grid for GT frames `indices`: uv (n, N, 2) float64 (NaN where unknown / behind the camera),
    vis (n, N) bool = visible in the view & inside the grid & default_gt."""
    size, mode = protocol.eval_size_for(sample.subset), protocol.resize_mode_for(sample.subset)
    idx = np.asarray(indices)
    uv, vis, _ = sample.tracks2d(view, size=size, mode=mode, indices=idx)
    tr = sample.tracks
    known = tr.default_gt[idx] & np.isfinite(tr.xyz[idx]).all(-1)
    uv = np.where(known[..., None], uv, np.nan)
    return uv, vis & known & np.isfinite(uv).all(-1)


def view_case(sample, view: int, protocol: Track2DProtocol, window=None) -> dict:
    """Tracks of one view in their (seeded, shuffled) export order: {"point_index" (Nt,) GT point of each track
    (evaluator-side only, never exported), "query_t" (Nt,), "query_uv" (Nt, 2), "gt_uv" (n, Nt, 2), "gt_vis"
    (n, Nt), "n_points"} for the protocol's query mode."""
    win = protocol.window(sample) if window is None else window
    uv, vis = view_gt(sample, view, protocol, win)
    n = len(win)
    cap = _rng(protocol, sample.uid, _RNG_CAP, view)
    if protocol.query_mode == "fixed":
        pidx = _cap(np.nonzero(vis[0])[0], protocol.max_points, cap)
        qt = np.zeros(len(pidx), np.int64)
    elif protocol.query_mode == "first":
        pidx = _cap(np.nonzero(vis.any(0))[0], protocol.max_points, cap)
        qt = np.argmax(vis[:, pidx], axis=0).astype(np.int64)
    else:
        js = np.arange(0, n, protocol.query_stride)
        pts = _cap(np.nonzero(vis[js].any(0))[0], protocol.max_points, cap)
        keep = np.zeros(vis.shape[1], bool)
        keep[pts] = True
        pl = [np.nonzero(vis[j] & keep)[0] for j in js]
        pidx = np.concatenate(pl) if pl else np.zeros(0, np.int64)
        qt = np.concatenate([np.full(len(p), j, np.int64) for j, p in zip(js, pl)]) if pl else np.zeros(0, np.int64)
    order = _rng(protocol, sample.uid, _RNG_ORDER, view).permutation(len(pidx))
    pidx, qt = pidx.astype(np.int64)[order], qt[order]
    return {"point_index": pidx, "query_t": qt, "query_uv": uv[qt, pidx].copy() if len(pidx) else np.zeros((0, 2)),
            "gt_uv": uv[:, pidx], "gt_vis": vis[:, pidx], "n_points": int(len(np.unique(pidx)))}


def _distinct_positions(points: np.ndarray, xyz_t: np.ndarray) -> np.ndarray:
    """`points` without duplicates of the same GT position at that frame (lowest index kept): two copies of one
    point with different source views would reveal each other's target."""
    if len(points) < 2:
        return points
    _, first = np.unique(np.round(np.asarray(xyz_t[points], np.float64), 7), axis=0, return_index=True)
    return points[np.sort(first)]


def corr_case(sample, protocol: Track2DProtocol) -> dict:
    """Correspondence queries in their (seeded, shuffled) export order: {"frame" (P,), "view_a" (P,), "view_b" (P,),
    "uv_a" (P, 2), "uv_b" (P, 2) GT target, "point_index" (P,) (evaluator-side only), "frames" corr frames used,
    "frames_dropped" corr frames outside the clip}. One source view per (frame, point); P may be 0."""
    views = protocol.views_for(sample)
    frames = protocol.corr_frames_for(sample)
    rows = {k: [] for k in ("frame", "view_a", "view_b", "point_index", "uv_a", "uv_b")}
    if frames and len(views) >= 2:
        gts = [view_gt(sample, v, protocol, frames) for v in views]
        xyz = sample.tracks.xyz
        for i, t in enumerate(frames):
            vis = np.stack([g[1][i] for g in gts])
            uv = np.stack([g[0][i] for g in gts])
            pts = _distinct_positions(np.nonzero(vis.sum(0) >= 2)[0], xyz[t])
            pts = _cap(pts, protocol.corr_max_points, _rng(protocol, sample.uid, _RNG_CORR_CAP, t))
            draw = _rng(protocol, sample.uid, _RNG_CORR_SOURCE, t).random(len(pts))
            for p_, r in zip(pts, draw):
                seen = np.nonzero(vis[:, p_])[0]
                a = seen[min(int(r * len(seen)), len(seen) - 1)]
                for b in seen:
                    if b == a:
                        continue
                    rows["frame"].append(t)
                    rows["view_a"].append(views[a])
                    rows["view_b"].append(views[b])
                    rows["point_index"].append(p_)
                    rows["uv_a"].append(uv[a, p_])
                    rows["uv_b"].append(uv[b, p_])
    P = len(rows["frame"])
    order = _rng(protocol, sample.uid, _RNG_CORR_ORDER).permutation(P)
    out = {}
    for k, v in rows.items():
        if k.startswith("uv"):
            a = np.asarray(v, np.float64).reshape(P, 2)
        else:
            a = np.asarray(v, np.int64).reshape(P)
        out[k] = a[order]
    out["frames"] = frames
    out["frames_dropped"] = [t for t in protocol.corr_frames_for() if t not in frames]
    return out


def sample_inputs(sample, protocol: Track2DProtocol | None = None, load_frames: bool = True) -> dict:
    """In-memory inputs: {"uid", "subset", "id", "views", "window_indices", "fps", "eval_size", "resize_mode",
    "query_mode", "frames" [(n, H, W, 3) uint8 per view on the eval grid] (None without load_frames), "K" (V, 3, 3)
    on the eval grid, "w2c" (V, n, 4, 4), "queries" [per view {"track_id", "query_t", "query_frame", "query_uv"}],
    "corr" None or {"query_id", "frame", "view_a", "view_b", "uv_a", "frames" [GT frames], "w2c" (V, F, 4, 4),
    "images" [(F, H, W, 3) per view on the eval grid] or None}}. No GT beyond the queries. Raises ValueError for a
    sample that is not evaluable."""
    p = get_protocol(protocol)
    ok, why = p.evaluable(sample)
    if not ok:
        raise ValueError(f"{sample.uid}: {why}")
    win = p.window(sample)
    size, mode = p.eval_size_for(sample.subset), p.resize_mode_for(sample.subset)
    views = p.views_for(sample)
    queries = []
    for v in views:
        c = view_case(sample, v, p, win)
        queries.append({"track_id": np.arange(len(c["query_t"]), dtype=np.int64), "query_t": c["query_t"],
                        "query_frame": win[c["query_t"]] if len(c["query_t"]) else np.zeros(0, np.int64),
                        "query_uv": c["query_uv"]})
    out = {"uid": sample.uid, "subset": sample.subset, "id": sample.id, "views": views,
           "window_indices": win.copy(), "fps": sample.fps / p.frame_step, "eval_size": list(size),
           "resize_mode": mode, "query_mode": p.query_mode,
           "K": np.stack([sample.K(v, size, mode) for v in views]), "w2c": np.stack([sample.w2c(v, win) for v in views]),
           "queries": queries, "frames": None, "corr": None}
    if p.corr:
        c = corr_case(sample, p)
        fr = c["frames"]
        out["corr"] = {"query_id": np.arange(len(c["frame"]), dtype=np.int64), "frame": c["frame"],
                       "view_a": c["view_a"], "view_b": c["view_b"], "uv_a": c["uv_a"], "frames": fr,
                       "frames_dropped": c["frames_dropped"],
                       "w2c": np.stack([sample.w2c(v, fr) for v in views]) if fr else np.zeros((len(views), 0, 4, 4)),
                       "images": None}
        if load_frames and fr:
            out["corr"]["images"] = [sample.frames(v, fr, size=size, mode=mode) for v in views]
    if load_frames:
        out["frames"] = [sample.frames(v, win, size=size, mode=mode) for v in views]
    return out


def _manifest_entry(rec: dict) -> dict:
    return {"uid": rec["uid"], "subset": rec["subset"], "id": rec["id"], "n_window": rec["n_window"],
            "n_tracks": rec["n_tracks"], "n_corr": None if rec["corr"] is None else rec["corr"]["n"],
            "dir": f"{rec['subset']}/{rec['id']}"}


def _export_one(args) -> dict:
    root, subset, sid, out_dir, pdict, frames, overwrite = args
    import cv2
    from ..data.sample import Sample
    from ..io.video import write_video
    p = Track2DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    d = Path(out_dir) / s.subset / s.id
    if not overwrite and (d / "inputs.json").exists():
        return {**_manifest_entry(json.loads((d / "inputs.json").read_text())), "existing": True}
    ok, why = p.evaluable(s)
    if not ok:
        return {"uid": s.uid, "skipped": why}
    inp = sample_inputs(s, p, load_frames=frames != "none")
    d.mkdir(parents=True, exist_ok=True)
    size, mode = tuple(inp["eval_size"]), inp["resize_mode"]
    files, cams, n_tracks = {}, {}, {}
    for i, v in enumerate(inp["views"]):
        name = f"view{v}"
        q = inp["queries"][i]
        np.savez_compressed(d / f"queries_{name}.npz", track_id=q["track_id"].astype(np.int32),
                            query_t=q["query_t"].astype(np.int32), query_frame=np.asarray(q["query_frame"], np.int32),
                            query_uv=q["query_uv"].astype(np.float64))
        n_tracks[name] = int(len(q["track_id"]))
        cams[name] = {"K": inp["K"][i].tolist(), "w2c": inp["w2c"][i].tolist()}
        if frames == "none":
            m = s.image_map(v, size, mode)
            files[name] = {"video": str(Path(s.subset) / s.id / s.meta["views"][v]["video"]),
                           "relative_to": "benchmark root", "frame_indices": "window_indices",
                           "native_to_eval": {"src_size": list(m.src_size), "dst_size": list(m.dst_size), "sx": m.sx,
                                              "sy": m.sy, "ox": m.ox, "oy": m.oy, "mode": m.mode}}
        elif frames == "png":
            vd = d / "frames" / name
            vd.mkdir(parents=True, exist_ok=True)
            for t, f in zip(inp["window_indices"], inp["frames"][i]):
                cv2.imwrite(str(vd / f"frame_{int(t):04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            files[name] = {"png_dir": f"frames/{name}", "pattern": "frame_{t:04d}.png (t = GT frame index)"}
        else:
            write_video(d / "frames" / f"{name}.mp4", inp["frames"][i], fps=inp["fps"], lossless=True)
            files[name] = {"video": f"frames/{name}.mp4", "frame_indices": "0..n-1 = window positions"}
    corr = None
    if inp["corr"] is not None:
        c = inp["corr"]
        (d / "corr").mkdir(exist_ok=True)
        np.savez_compressed(d / "corr" / "queries.npz", query_id=c["query_id"].astype(np.int32),
                            frame=c["frame"].astype(np.int32), view_a=c["view_a"].astype(np.int32),
                            view_b=c["view_b"].astype(np.int32), uv_a=c["uv_a"].astype(np.float64))
        P = int(len(c["frame"]))
        corr = {"queries": "corr/queries.npz", "n": P, "frames": c["frames"], "frames_dropped": c["frames_dropped"],
                "cameras": {f"view{v}": {"K": inp["K"][i].tolist(), "w2c": c["w2c"][i].tolist()}
                            for i, v in enumerate(inp["views"])},
                "prediction": {"file": f"{s.subset}/{s.id}/corr.npz", "key": "uv_b", "shape": [P, 2]}}
        if frames != "none":
            for i, v in enumerate(inp["views"]):
                for t, f in zip(c["frames"], c["images"][i]):
                    cv2.imwrite(str(d / "corr" / f"view{v}_frame_{int(t):04d}.png"),
                                cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            corr["images"] = "corr/view{k}_frame_{t:04d}.png"
        else:
            corr["images"] = "frames.files.view{k}.video, frame t (native size; map with native_to_eval)"
    n = len(inp["window_indices"])
    rec = {"task": TASK, "uid": s.uid, "subset": s.subset, "id": s.id, "protocol": p.name, "views": inp["views"],
           "fps": inp["fps"], "window_indices": [int(t) for t in inp["window_indices"]], "n_window": n,
           "eval_size": list(size), "resize_mode": mode, "query_mode": p.query_mode,
           "normalize_to": None if p.normalize_to is None else list(p.normalize_to),
           "frames": {"kind": frames, "files": files}, "cameras": cams,
           "queries": {f"view{v}": f"queries_view{v}.npz" for v in inp["views"]}, "n_tracks": n_tracks,
           "prediction": {f"view{v}": {"file": f"{s.subset}/{s.id}/view{v}.npz", "key": "uv",
                                       "shape": [n, n_tracks[f'view{v}'], 2], "optional": {"vis": [n, n_tracks[f'view{v}']]}}
                          for v in inp["views"]},
           "corr": corr}
    (d / "inputs.json").write_text(json.dumps(rec, indent=1))
    return _manifest_entry(rec)


def export_inputs(bench, out_dir, protocol: Track2DProtocol | None = None, subsets=None, ids=None,
                  frames: str = "mp4", jobs: int = 1, overwrite: bool = False) -> dict:
    """Write the inputs of every evaluable sample (layout in the module docstring), ``manifest.jsonl``,
    ``protocol.json``, ``export.json`` and ``skipped.json``. frames: "mp4" | "png" | "none". Resumable: samples
    whose inputs.json exists are kept unless overwrite; an out dir of another protocol / frames option is refused
    unless overwrite. Returns {"exported" (incl. kept), "skipped", "existing"}."""
    p = get_protocol(protocol)
    if frames not in FRAME_EXPORTS:
        raise ValueError(f"frames must be one of {FRAME_EXPORTS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_out_dir(out_dir, "export.json", {"task": TASK, "protocol_hash": p.hash(), "frames": frames}, overwrite)
    work = [(str(bench.root), s.subset, s.id, str(out_dir), p.to_dict(), frames, overwrite)
            for s in select_samples(bench, subsets, ids, p.clip_filter, f"{TASK} export")[0]]
    return _finish_export(out_dir, p, _map(_export_one, work, jobs))


def prediction_path(pred_root, sample, view: int | str) -> Path:
    name = view if isinstance(view, str) else f"view{int(view)}"
    return Path(pred_root) / sample.subset / sample.id / f"{name}.npz"


def prediction_files(pred_root, sample, protocol: Track2DProtocol) -> list:
    """Every prediction file the evaluation of `sample` reads (views, then corr.npz when corr is on)."""
    files = [prediction_path(pred_root, sample, v) for v in protocol.views_for(sample)]
    return files + ([prediction_path(pred_root, sample, "corr")] if protocol.corr else [])


def write_prediction(pred_root, sample, view: int, uv, vis=None, track_id=None) -> Path:
    """Write ``<pred_root>/<Subset>/<id>/view{k}.npz`` with uv (n, Nt, 2) on the eval grid (float32 / float64 kept,
    other dtypes -> float32), optional vis (n, Nt) and the optional column ids track_id (Nt,)."""
    a = np.asarray(uv)
    d = {"uv": a if a.dtype in (np.float32, np.float64) else a.astype(np.float32)}
    if vis is not None:
        v = np.asarray(vis)
        d["vis"] = v if v.dtype == bool else v.astype(np.float32)
    if track_id is not None:
        d["track_id"] = np.asarray(track_id, np.int64)
    return _save(prediction_path(pred_root, sample, view), d)


def write_corr_prediction(pred_root, sample, uv_b, query_id=None) -> Path:
    """Write ``<pred_root>/<Subset>/<id>/corr.npz`` with uv_b (P, 2) on the eval grid of view b (optional query_id)."""
    d = {"uv_b": np.asarray(uv_b, np.float64)}
    if query_id is not None:
        d["query_id"] = np.asarray(query_id, np.int64)
    return _save(prediction_path(pred_root, sample, "corr"), d)


def _save(path: Path, d: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(path)
    return path


def _vis_bool(vis: np.ndarray) -> np.ndarray:
    if vis.dtype == bool:
        return vis
    return np.nan_to_num(vis.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0) >= 0.5


def _id_order(ids, n: int, path: Path, key: str) -> np.ndarray:
    """Column order that sorts a prediction by its ids (a permutation of 0..n-1); InvalidPrediction otherwise."""
    ids = index_array(ids, path, key)
    if ids.shape != (n,) or not np.array_equal(np.sort(ids), np.arange(n)):
        raise InvalidPrediction(f"{path}: '{key}' must be a permutation of 0..{n - 1} (the query rows)")
    return np.argsort(ids, kind="stable")


def load_view_prediction(pred_root, sample, view: int, case: dict, n: int):
    """(uv (n, Nt, 2) float64 with non-finite -> +inf, vis (n, Nt) bool or None, info) or None if missing. Raises
    InvalidPrediction (a ValueError) when the file cannot be read or is malformed (keys, dtypes, shapes, track_id)."""
    path = prediction_path(pred_root, sample, view)
    if not path.exists():
        return None
    Nt = len(case["query_t"])
    names, z = read_npz(path, ("uv", "vis", "track_id"))
    if "uv" not in z:
        raise InvalidPrediction(f"{path}: needs 'uv' (keys: {sorted(names)})")
    uv = real_array(z["uv"], path, "uv").astype(np.float64)
    vis = real_array(z["vis"], path, "vis", bool_ok=True) if "vis" in z else None
    tid = z.get("track_id")
    if uv.shape != (n, Nt, 2):
        raise InvalidPrediction(f"{path}: 'uv' has shape {uv.shape}, expected (n_window, n_tracks, 2) = {(n, Nt, 2)}")
    if vis is not None and vis.shape != (n, Nt):
        raise InvalidPrediction(f"{path}: 'vis' has shape {vis.shape}, expected {(n, Nt)}")
    if tid is not None:
        perm = _id_order(tid, Nt, path, "track_id")
        uv = uv[:, perm]
        vis = None if vis is None else vis[:, perm]
    bad = ~np.isfinite(uv).all(-1)
    uv[bad] = np.inf
    vis = None if vis is None else _vis_bool(vis)
    return uv, vis, {"file": str(path), "has_vis": vis is not None, "nonfinite_frac": float(bad.mean()) if bad.size else 0.0}


def load_corr_prediction(pred_root, sample, P: int):
    """(uv_b (P, 2) float64 with non-finite -> +inf, info) or None if missing. Raises InvalidPrediction (a ValueError)
    when the file cannot be read or is malformed."""
    path = prediction_path(pred_root, sample, "corr")
    if not path.exists():
        return None
    names, z = read_npz(path, ("uv_b", "uv", "query_id"))
    key = "uv_b" if "uv_b" in z else ("uv" if "uv" in z else None)
    if key is None:
        raise InvalidPrediction(f"{path}: needs 'uv_b' (keys: {sorted(names)})")
    uv = real_array(z[key], path, key).astype(np.float64)
    qid = z.get("query_id")
    if uv.shape != (P, 2):
        raise InvalidPrediction(f"{path}: '{key}' has shape {uv.shape}, expected (n_corr_queries, 2) = {(P, 2)}")
    if qid is not None:
        uv = uv[_id_order(qid, P, path, "query_id")]
    bad = ~np.isfinite(uv).all(-1)
    uv[bad] = np.inf
    return uv, {"file": str(path), "nonfinite_frac": float(bad.mean()) if bad.size else 0.0}


def correspondence_metrics(pred_uv, gt_uv, image_size, thresholds_px=TAPVID_THRESHOLDS_PX,
                           normalize_to=(256, 256)) -> dict:
    """Cross-view point-transfer accuracy. pred_uv, gt_uv (P, 2) [u, v] pixels on an image of `image_size` (W, H);
    coordinates are rescaled to `normalize_to` (None = keep) before the distance. Non-finite predictions = +inf.
    Returns {"pct_within_{x}" (%, strict <), "avg_pct_within", "mean_err_px", "median_err_px", "n"}."""
    pr = np.asarray(pred_uv, np.float64).reshape(-1, 2)
    gt = np.asarray(gt_uv, np.float64).reshape(-1, 2)
    if pr.shape != gt.shape:
        raise ValueError(f"pred {pr.shape} and gt {gt.shape} differ")
    W, H = (float(x) for x in image_size)
    sc = np.ones(2) if normalize_to is None else np.array([normalize_to[0] / W, normalize_to[1] / H])
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm((pr - gt) * sc, axis=-1)
    d = np.where(np.isfinite(pr).all(-1), d, np.inf)
    out, pcts = {}, []
    nan = float("nan")
    for t in thresholds_px:
        v = 100.0 * float(np.mean(d < float(t))) if d.size else nan
        out[f"pct_within_{threshold_key(t)}"] = v
        pcts.append(v)
    out["avg_pct_within"] = float(np.mean(pcts)) if d.size else nan
    out["mean_err_px"] = float(np.mean(d)) if d.size else nan
    out["median_err_px"] = float(np.median(d)) if d.size else nan
    out["n"] = int(d.size)
    return out


def evaluate_sample(sample, pred_root, protocol: Track2DProtocol | None = None) -> dict:
    """Per-sample record: per-view TAP-Vid metrics, "track" (status + their mean over views when every view with
    tracks is predicted) and "corr" (status + correspondence metrics). Sample status: "ok" (every part with queries
    scored), "partial" (one part scored, another missing), "missing" (nothing scored) or "skipped". Raises
    InvalidPrediction (a ValueError) on an unreadable or malformed prediction file (`evaluate` records that as status
    "invalid", any other exception as "error")."""
    p = get_protocol(protocol)
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id, "protocol": p.name,
           "protocol_hash": p.hash(), "pred_sig": prediction_signature(pred_root, prediction_files(pred_root, sample, p))}
    ok, why = p.evaluable(sample)
    if not ok:
        return {**rec, "status": "skipped", "reason": why}
    win = p.window(sample)
    size = p.eval_size_for(sample.subset)
    qmode = "strided" if p.query_mode == "strided" else "first"
    rec.update(eval_size=list(size), query_mode=p.query_mode, window=[int(win[0]), int(win[-1]), int(p.frame_step)],
               n_window=int(len(win)))
    views, per_view, missing = {}, [], []
    n_with_tracks = 0
    for v in p.views_for(sample):
        c = view_case(sample, v, p, win)
        name = f"view{v}"
        if len(c["query_t"]) == 0:
            views[name] = {"status": "no_tracks", "n_tracks": 0}
            continue
        n_with_tracks += 1
        got = load_view_prediction(pred_root, sample, v, c, len(win))
        if got is None:
            views[name] = {"status": "missing", "n_tracks": int(len(c["query_t"]))}
            missing.append(name)
            continue
        uv, vis, info = got
        m = track2d_metrics(c["gt_uv"], c["gt_vis"], uv, vis, c["query_t"], size, p.thresholds_px,
                            p.normalize_to, query_mode=qmode)
        m.update(status="ok", n_tracks=int(len(c["query_t"])), n_points=c["n_points"],
                 has_vis=info["has_vis"], nonfinite_frac=info["nonfinite_frac"])
        views[name] = m
        per_view.append(m)
    rec["views"] = views
    if n_with_tracks == 0:
        rec["track"] = {"status": "no_tracks"}
    elif missing:
        rec["track"] = {"status": "missing", "missing_views": missing, "n_views": n_with_tracks}
    else:
        keys = [k for k in per_view[0] if isinstance(per_view[0][k], float) and k != "nonfinite_frac"]
        tr = {"status": "ok"}
        for k in keys:
            vals = np.asarray([m[k] for m in per_view], np.float64)
            vals = vals[~np.isnan(vals)]
            tr[k] = float(vals.mean()) if vals.size else float("nan")
        tr["n_views"] = len(per_view)
        tr["n_tracks"] = int(sum(m["n_tracks"] for m in per_view))
        rec["track"] = tr
    if not p.corr:
        rec["corr"] = {"status": "disabled"}
    else:
        c = corr_case(sample, p)
        P = int(len(c["frame"]))
        base = {"frames": c["frames"], "frames_dropped": c["frames_dropped"]}
        if P == 0:
            rec["corr"] = {"status": "no_queries", "n": 0, **base}
        else:
            got = load_corr_prediction(pred_root, sample, P)
            if got is None:
                rec["corr"] = {"status": "missing", "n": P, **base}
            else:
                uvb, info = got
                m = correspondence_metrics(uvb, c["uv_b"], size, p.thresholds_px, p.normalize_to)
                pairs = {}
                for a, b in sorted(set(zip(c["view_a"].tolist(), c["view_b"].tolist()))):
                    sel = (c["view_a"] == a) & (c["view_b"] == b)
                    pm = correspondence_metrics(uvb[sel], c["uv_b"][sel], size, p.thresholds_px, p.normalize_to)
                    pairs[f"{a}->{b}"] = {k: pm[k] for k in ("avg_pct_within", "mean_err_px", "n")}
                rec["corr"] = {"status": "ok", **m, "nonfinite_frac": info["nonfinite_frac"], "pairs": pairs, **base}
    parts = {"track": rec["track"]["status"], "corr": rec["corr"]["status"]}
    need = {k: v for k, v in parts.items() if v in ("ok", "missing")}
    lacking = missing + (["corr"] if parts["corr"] == "missing" else [])
    if not need:
        rec.update(status="skipped", reason="no tracks and no correspondence queries")
    elif all(v == "ok" for v in need.values()):
        rec["status"] = "ok"
    else:
        rec["status"] = "partial" if any(v == "ok" for v in need.values()) else "missing"
        rec["reason"] = (f"no prediction for {', '.join(lacking)} under "
                         f"{Path(pred_root) / sample.subset / sample.id}")
    return rec


def _evaluate_one(args) -> dict:
    root, subset, sid, pred_root, out_dir, pdict, overwrite, run_id, strict = args
    from ..data.sample import Sample
    p = Track2DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    return evaluate_cached(TASK, s, p, pred_root, out_dir, prediction_files(pred_root, s, p), evaluate_sample,
                           overwrite, run_id, strict)


def evaluate(bench, pred_root, out_dir, protocol: Track2DProtocol | None = None, subsets=None, ids=None,
             on_missing: str = "skip", verbose: bool = True, jobs: int = 1, overwrite: bool = False,
             strict: bool = False) -> dict:
    """Score every sample: ``<out_dir>/<Subset>/<id>.json`` + summary (`summarize`); returns the summary (plus
    "evaluated": the status counts of this call). on_missing: "skip" (count) or "error" (raise FileNotFoundError,
    after scoring, when a sample lacks any prediction file it needs: a view with tracks, or corr.npz when corr is on -
    pass corr=False to evaluate tracking only). jobs: worker processes. out_dir belongs to one run (protocol +
    prediction root): another run is refused (common.RunDirError, a ValueError) unless overwrite, which deletes its
    records; unchanged records of the run are reused unless overwrite. An unreadable or malformed prediction file gives
    status "invalid", any other failure of a sample status "error" (strict=True raises instead); see
    track3d.evaluate."""
    p = get_protocol(protocol)
    return run_evaluation(TASK, _evaluate_one, p, bench, pred_root, out_dir, subsets, ids, on_missing, jobs, overwrite,
                          verbose, summarize, strict=strict, files_fn=prediction_files)


def oracle_prediction(sample, view: int, protocol: Track2DProtocol | None = None, case: dict | None = None) -> dict:
    """GT uv / visibility of one view (unknown GT entries filled with the query position), in query order."""
    p = get_protocol(protocol)
    c = case or view_case(sample, view, p)
    uv = np.where(np.isfinite(c["gt_uv"]), c["gt_uv"], c["query_uv"][None])
    return {"uv": uv, "vis": c["gt_vis"].copy()}


def frozen_prediction(sample, view: int, protocol: Track2DProtocol | None = None, case: dict | None = None) -> dict:
    """Zero motion: every track stays at its query position, always visible."""
    p = get_protocol(protocol)
    c = case or view_case(sample, view, p)
    n = c["gt_uv"].shape[0]
    return {"uv": np.broadcast_to(c["query_uv"][None], (n,) + c["query_uv"].shape).copy()}


BASELINES = {"oracle": oracle_prediction, "frozen": frozen_prediction}


def _baseline_one(args) -> bool:
    root, subset, sid, pred_root, pdict, kind = args
    from ..data.sample import Sample
    p = Track2DProtocol.from_dict(pdict)
    s = Sample(root, subset, sid)
    ok, _ = p.evaluable(s)
    if not ok:
        return False
    win = p.window(s)
    for v in p.views_for(s):
        c = view_case(s, v, p, win)
        if len(c["query_t"]):
            write_prediction(pred_root, s, v, **BASELINES[kind](s, v, p, c))
    if p.corr:
        c = corr_case(s, p)
        if len(c["frame"]):
            write_corr_prediction(pred_root, s, c["uv_b"] if kind == "oracle" else c["uv_a"])
    return True


def run_baseline(bench, pred_root, kind: str = "frozen", protocol: Track2DProtocol | None = None, subsets=None,
                 ids=None, jobs: int = 1) -> dict:
    """Write a reference baseline for every evaluable sample: "oracle" (GT tracks + GT correspondences) or "frozen"
    (zero motion + identity correspondences). jobs = worker processes."""
    p = get_protocol(protocol)
    if kind not in BASELINES:
        raise KeyError(f"unknown baseline {kind!r}; available: {sorted(BASELINES)}")
    work = [(str(bench.root), s.subset, s.id, str(pred_root), p.to_dict(), kind)
            for s in select_samples(bench, subsets, ids, p.clip_filter, f"{TASK} baseline {kind}")[0]]
    res = _map(_baseline_one, work, jobs)
    return {"written": int(sum(res)), "skipped": int(len(res) - sum(res))}


_MD_TRACK = (("average_jaccard", "AJ ↑"), ("average_pts_within_thresh", "<δ^x_avg> ↑"),
             ("occlusion_accuracy", "OA ↑"))


def summarize(out_dir) -> dict:
    """Aggregate the per-sample JSONs into ``summary.json`` / ``summary.md``: mean ± s.e. over samples per subset of
    the per-sample view-mean tracking metrics (samples with every view predicted) and of the correspondence
    metrics. Only the records of the out dir's run are used (track3d `current_records`; others are counted as
    ``n_stale``); ``pred_root`` = its prediction root (None in an out dir of an older version)."""
    out_dir = Path(out_dir)
    proto, by_sub, stale = current_records(out_dir, TASK, Track2DProtocol)
    summary = {"task": TASK, "protocol": proto, "pred_root": run_pred_root(out_dir), "subsets": {}}
    for sub, recs in by_sub.items():
        if not recs:
            continue
        tst = [r.get("track", {}).get("status") for r in recs]
        cst = [r.get("corr", {}).get("status") for r in recs]
        tr = [{k: v for k, v in r["track"].items() if k != "status"} for r, st in zip(recs, tst) if st == "ok"]
        cr = [{k: v for k, v in r["corr"].items() if k not in ("status", "pairs", "frames", "frames_dropped")}
              for r, st in zip(recs, cst) if st == "ok"]
        summary["subsets"][sub] = {
            **status_counts(recs), "n_stale": stale.get(sub, 0),
            "n_track_samples": len(tr), "n_track_missing": tst.count("missing"), "n_track_none": tst.count("no_tracks"),
            "n_views_missing": sum(v.get("status") == "missing" for r in recs for v in r.get("views", {}).values()),
            "n_corr_samples": len(cr), "n_corr_missing": cst.count("missing"), "n_corr_none": cst.count("no_queries"),
            "corr_frames_dropped": sum(bool(r.get("corr", {}).get("frames_dropped")) for r in recs),
            "track": aggregate(tr), "corr": aggregate(cr)}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_summary_md(summary))
    return summary


def _summary_md(summary: dict) -> str:
    p = summary.get("protocol") or {}
    thr = [threshold_key(t) for t in (p.get("thresholds_px") or TAPVID_THRESHOLDS_PX)]
    size = p.get("eval_size")
    lines = [f"# track2d ({p.get('name', '?')}, query mode {p.get('query_mode', '?')})", "",
             f"Window from frame {p.get('ref_frame')}, horizon {p.get('horizon')}, step {p.get('frame_step')}; eval grid "
             f"{size} ({p.get('resize_mode')}), thresholds {thr} px on a {p.get('normalize_to')} grid. TAP-Vid metrics "
             "per view (pooled over tracks and frames), averaged over the views of a sample; mean ± s.e. over samples "
             "(%). scored / total = samples with every view predicted / samples in the subset; missing = samples with "
             "a missing view file (number of missing view files); invalid / error = samples with an unreadable or "
             "malformed prediction file / another failure (reason in the per-sample JSON).", "",
             "| subset | scored / total | missing (views) | " + " | ".join(h for _, h in _MD_TRACK) + " | "
             + " | ".join(f"δ<{t}" for t in thr) + " |",
             "|---|---|---|" + "---|" * (len(_MD_TRACK) + len(thr))]
    for sub, s in summary["subsets"].items():
        a = s["track"]
        stale = f"; {s['n_stale']} stale" if s.get("n_stale") else ""
        err = "".join(f"; {s[f'n_{k}']} {k}" for k in FAILED_STATUSES if s.get(f"n_{k}"))
        lines.append(f"| {sub} | {s['n_track_samples']} / {s['n_samples']} | {s['n_track_missing']} "
                     f"({s['n_views_missing']}){err}{stale} | "
                     + " | ".join(fmt_ms(a.get(k, {}), 1) for k, _ in _MD_TRACK) + " | "
                     + " | ".join(fmt_ms(a.get(f"pts_within_{t}", {}), 1) for t in thr) + " |")
    if p.get("corr"):
        fr = p.get("corr_frames") or ([p["ref_frame"] // 2] if p.get("ref_frame") else [])
        lines += ["", f"## Cross-view correspondence (frames {fr}, one source view per point, every other view that "
                  "sees it is a target)", "",
                  "| subset | scored / total | missing | no queries | avg % within ↑ | "
                  + " | ".join(f"% <{t} px" for t in thr) + " | mean err px ↓ | median err px ↓ |",
                  "|---|---|---|---|---|" + "---|" * (len(thr) + 2)]
        for sub, s in summary["subsets"].items():
            a = s["corr"]
            lines.append(f"| {sub} | {s['n_corr_samples']} / {s['n_samples']} | {s['n_corr_missing']} | "
                         f"{s['n_corr_none']} | " + fmt_ms(a.get("avg_pct_within", {}), 1) + " | "
                         + " | ".join(fmt_ms(a.get(f"pct_within_{t}", {}), 1) for t in thr) + " | "
                         + fmt_ms(a.get("mean_err_px", {}), 2) + " | " + fmt_ms(a.get("median_err_px", {}), 2) + " |")
    return "\n".join(lines) + "\n"


export_track2d_inputs = export_inputs
evaluate_track2d = evaluate


def _protocol_from_args(a) -> Track2DProtocol:
    p = get_protocol(a.protocol or None)
    over = {}
    for k, v in (("query_mode", a.query_mode), ("resize_mode", a.resize_mode), ("ref_frame", a.ref_frame),
                 ("max_points", a.max_points), ("corr_max_points", a.corr_max_points)):
        if v is not None:
            over[k] = v
    if a.size:
        try:
            size = _parse_size(a.size)
        except ValueError:
            size = None
        if size is None:
            raise ValueError(f"--size must be WxH (e.g. 256x256), got {a.size!r}")
        over["eval_size"] = list(size)
    if a.horizon is not None:
        over["horizon"] = None if a.horizon <= 0 else a.horizon
    if a.frame_step is not None:
        over["frame_step"] = a.frame_step
    if a.corr_frames is not None:
        from ..utils import parse_list
        over["corr_frames"] = [int(t) for t in (parse_list(a.corr_frames) or [])]
    if a.no_corr:
        over["corr"] = False
    return override_protocol(p, over, Track2DProtocol)


def main(argv=None) -> int:
    import argparse
    from ..data.benchmark import Benchmark
    ap = argparse.ArgumentParser(prog="vpmwb track2d",
                                 description="track2d (TAP-Vid style 2D tracking + cross-view correspondence): export "
                                             "inputs, write baselines, evaluate, summarize")
    sp = ap.add_subparsers(dest="cmd", required=True)
    helps = {"export": "write the model inputs", "baseline": "write reference predictions (--kind)",
             "evaluate": "score predictions", "summarize": "rebuild summary.json / summary.md"}
    parsers = {}
    for name in ("export", "baseline", "evaluate", "summarize"):
        q = parsers[name] = sp.add_parser(name, help=helps[name], description=helps[name])
        add_common_args(q, name, "track2d-v1 (default) | <json/yaml file>")
        if name != "summarize":
            q.add_argument("--query-mode", choices=QUERY_MODES, default=None)
            q.add_argument("--size", default=None, help="eval grid WxH (default 256x256)")
            q.add_argument("--resize-mode", choices=RESIZE_MODES, default=None)
            q.add_argument("--ref-frame", type=int, default=None)
            q.add_argument("--horizon", type=int, default=None, help="<= 0: to the end of the clip")
            q.add_argument("--frame-step", type=int, default=None, help="GT frames between window positions")
            q.add_argument("--max-points", type=int, default=None)
            q.add_argument("--corr-frames", nargs="*", default=None, help="GT frames outside the tracking window")
            q.add_argument("--corr-max-points", type=int, default=None)
            q.add_argument("--no-corr", action="store_true", help="tracking only")
        if name == "export":
            q.add_argument("--frames", choices=FRAME_EXPORTS, default="mp4")
        elif name == "baseline":
            q.add_argument("--kind", choices=sorted(BASELINES), default="frozen")
        elif name == "evaluate":
            q.add_argument("--on-missing", choices=("skip", "error"), default="skip")
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        if not Path(a.out).is_dir():
            parsers["summarize"].error(f"--out {a.out}: no such directory")
        summarize(a.out)
        print((Path(a.out) / "summary.md").read_text())
        return 0
    p = cli_setup(parsers[a.cmd], a, _protocol_from_args)
    bench = Benchmark(a.root)
    if a.cmd == "export":
        res = export_inputs(bench, a.out, p, a.subsets, a.ids, frames=a.frames, jobs=a.jobs, overwrite=a.overwrite)
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
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
