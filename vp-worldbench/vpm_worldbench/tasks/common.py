"""Utilities shared by the geometry tasks (`depth`, `recon4d`, `nvs`); `segmentation` also uses the status vocabulary,
the prediction reading and the command-line helpers, `understanding` the sample selection and command-line helpers.

Contents
    TaskProtocol            frame window t = ref_frame + j * frame_step (j = 0..horizon) + per-subset option dicts
                            (subset aliases canonicalised, unknown subsets rejected); a protocol file without "name" is
                            named after the file (`file_protocol_fields`)
    resampling              exact pixel-centre nearest resize (`nearest_index`, `resize_nearest`: the rule of
                            `geometry.warp_image(interp="nearest")`), NaN-aware map resize (`resize_map`), sub-pixel
                            map sampling (`sample_map`)
    fov_mask                valid field of view of an undistorted video (Real-General's black undistortion border)
    GT accessors            `read_gt_depth` / `gt_depth` (GT depth on a grid + the dense-vs-track consistency test),
                            `track_observations` (GT track points seen in a view), `track_path_length`
    export_view_frames      window frames of one view as mp4 / png / nothing, with K at the exported size
    sample selection        `iter_samples` / `select_samples` (subset aliases also in uids, `canonical_ids`; requested
                            ids that match no sample are reported by `Benchmark.samples`)
    predictions             `MissingPrediction` / `InvalidPrediction` (status missing / invalid), `read_npz`,
                            `transposed_hint` (maps saved as (W, H) instead of (H, W))
    evaluation driver       `evaluate_task` / `evaluate_one`: one JSON per sample, resumable, statuses of `STATUSES`
                            (ok / skipped / missing / invalid / error; the names not_evaluable / missing_prediction of
                            earlier records are read as skipped / missing); results are tied to one run (task,
                            protocol, prediction root) and to the prediction files' size / mtime
    one run per out dir     `RunDirError` (a `utils.UsageError` and a ValueError) for an out dir of another run or of
                            other export options, raised by depth / recon4d / nvs (`evaluate_task`), segmentation /
                            understanding (`check_out_stamp`) and track3d / track2d; `run_pred_root` (the prediction
                            root recorded in <out>/run.json)
    summary                 `mean_se`, `summarize_records` (summary.json + summary.md, mean +- s.e. over samples)
    task_cli                the `vpmwb <task> {export,baseline,evaluate,summarize}` command line; `cli_protocol`,
                            `cli_subsets`, `cli_size`, `override_protocol` and the exit statuses are shared with the
                            other task CLIs

Statuses (`STATUSES`, the per-sample "status" of every task): ok = scored; partial = scored in part (track2d);
skipped = not evaluable under the protocol (window, views, no prompt); missing = no prediction file; invalid = a
prediction file that cannot be read or has the wrong shape / content; error = any other failure ("reason" holds the
message). An evaluate command exits with status 1 when its selection matches no sample or none of its evaluable
(not skipped) samples is scored (`run_status`, `evaluation_exit_status`); usage and setup problems (`utils.UsageError`
such as `RunDirError`, a missing optional package) end `vpmwb` with one line `vpmwb: error: ...` and exit status 2
(`cli.main`; VPMWB_TRACEBACK=1 shows the traceback instead).

Conventions: images are (H, W[, C]), sizes are (W, H), pixel coordinates follow OpenCV (integer = pixel centre), depth
is planar camera z in metres. JSON files hold no NaN / Infinity tokens: NaN -> null, +-inf -> "inf" / "-inf".
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys
import traceback
import warnings
import zipfile
import zlib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from ..constants import SUBSETS, canonical_subset
from ..geometry import image_map
from ..utils import UsageError

__all__ = [
    "FRAME_MODES", "STATUSES", "STATUS_ALIASES", "SCORED_STATUSES", "FAILED_STATUSES", "canonical_status", "json_safe",
    "json_number", "write_json", "sample_json_path", "TaskProtocol", "file_protocol_fields", "as_size",
    "canonical_subset_dict", "nearest_index", "resize_nearest", "resize_map", "sample_map", "fov_mask", "gt_depth",
    "read_gt_depth", "TrackObs", "track_observations", "track_path_length", "export_view_frames", "canonical_ids",
    "iter_samples", "select_samples", "run_jobs", "MissingPrediction", "InvalidPrediction", "read_npz",
    "frame_index_array", "ASPECT_TOL", "transposed_hint", "base_record", "prediction_stamp", "run_fingerprint",
    "RunDirError", "check_out_stamp", "run_pred_root", "evaluate_one", "run_status", "evaluate_task", "mean_se",
    "summarize_records", "write_manifest", "baseline_args", "run_baseline_task", "override_protocol", "cli_protocol",
    "cli_subsets", "cli_size", "evaluation_exit_status", "selection_exit_status", "task_cli",
]

FRAME_MODES = ("mp4", "png", "none")
RUN_FILE = "run.json"
STATUSES = ("ok", "partial", "skipped", "missing", "invalid", "error")
STATUS_ALIASES = {"not_evaluable": "skipped", "missing_prediction": "missing"}
SCORED_STATUSES = ("ok", "partial")
FAILED_STATUSES = ("invalid", "error")


def canonical_status(status):
    """The `STATUSES` name of a record status (earlier records' not_evaluable / missing_prediction -> skipped /
    missing; other values unchanged)."""
    return STATUS_ALIASES.get(status, status)


def json_safe(x):
    """Recursively convert to JSON-standard values: numpy scalars/arrays -> Python, NaN -> None, +-inf -> "inf" / "-inf"
    (the per-sample and summary JSON files contain no NaN / Infinity tokens)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return x


def json_number(x) -> float:
    """Inverse of `json_safe` for one number: None -> nan, "inf" / "-inf" -> +-inf."""
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return {"inf": float("inf"), "-inf": float("-inf")}.get(x, float("nan"))
    return float(x)


def write_json(path, obj) -> Path:
    """Atomic write of json_safe(obj)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_safe(obj), indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path


def sample_json_path(out_dir, sample) -> Path:
    return Path(out_dir) / sample.subset / f"{sample.id}.json"


def canonical_subset_dict(d: dict, what: str = "per-subset dict") -> dict:
    """{alias or subset name: v} -> {canonical subset name: v}; raises ValueError on an unknown subset or on two keys
    naming the same subset."""
    out = {}
    for k, v in d.items():
        try:
            c = canonical_subset(str(k))
        except KeyError:
            raise ValueError(f"{what}: unknown subset {k!r}; expected one of {SUBSETS}") from None
        if c in out:
            raise ValueError(f"{what}: subset {c!r} given twice")
        out[c] = v
    return out


@dataclass
class TaskProtocol:
    """Frame window and per-subset options shared by the geometry tasks.

    The window is t = ref_frame + j * frame_step for j = 0..horizon (horizon + 1 frames): the grid of the generation
    protocol's `window_indices` (`vpm_worldbench.protocol.Protocol`); here every window frame is an input and is scored.
    The fields named in `SUBSET_FIELDS` may be dicts {subset: value} (subset aliases such as "sr" are canonicalised,
    unknown subsets raise ValueError); subsets missing from such a dict fall back to the task defaults."""
    SUBSET_FIELDS = ()

    name: str = "task"
    ref_frame: int = 16
    horizon: int = 64
    frame_step: int = 1
    views: Optional[list] = None
    clip_filter: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.ref_frame < 0 or self.horizon < 0 or self.frame_step < 1:
            raise ValueError("need ref_frame >= 0, horizon >= 0, frame_step >= 1")
        for name in self.SUBSET_FIELDS:
            v = getattr(self, name)
            if isinstance(v, dict):
                setattr(self, name, canonical_subset_dict(v, f"{type(self).__name__}.{name}"))

    @property
    def window_indices(self) -> list:
        return [self.ref_frame + j * self.frame_step for j in range(self.horizon + 1)]

    @property
    def last_frame(self) -> int:
        return self.ref_frame + self.horizon * self.frame_step

    def evaluable(self, sample) -> tuple:
        """(ok, reason): the window must lie inside the clip and the views must exist."""
        if self.last_frame > sample.n_frames - 1:
            return False, f"window ends at {self.last_frame} > last frame {sample.n_frames - 1}"
        if self.views is not None and (not self.views or max(self.views) >= sample.n_views or min(self.views) < 0):
            return False, f"views {self.views} not in 0..{sample.n_views - 1}"
        return True, ""

    def views_for(self, sample) -> list:
        return list(range(sample.n_views)) if self.views is None else [int(v) for v in self.views]

    def per_subset(self, name: str, subset: str, defaults: dict):
        """Value of a dict-valued field for `subset` (falls back to defaults[subset])."""
        subset = canonical_subset(subset)
        own = getattr(self, name)
        if isinstance(own, dict) and subset in own:
            return own[subset]
        return defaults[subset]

    def to_dict(self) -> dict:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown {cls.__name__} fields {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path):
        """Protocol from a JSON / YAML file of fields. A file without "name" is named after the file (its stem), so
        the results of a custom window are never labelled as the default protocol."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("YAML protocol files need PyYAML (pip install pyyaml); or use JSON") from e
            return cls.from_dict(file_protocol_fields(yaml.safe_load(text), path))
        return cls.from_dict(file_protocol_fields(json.loads(text), path))

    def save(self, path):
        write_json(path, self.to_dict())


def file_protocol_fields(d, path) -> dict:
    """The protocol fields `d` read from the file `path` (None = an empty file), with "name" = the file stem when the
    file gives no name (or an empty / null one). Raises ValueError when the file does not hold a mapping."""
    d = {} if d is None else d
    if not isinstance(d, dict):
        raise ValueError(f"{path}: a protocol file holds a mapping of protocol fields, got {type(d).__name__}")
    return d if d.get("name") not in (None, "") else dict(d, name=Path(path).stem)


def as_size(x) -> Optional[tuple]:
    """(W, H) as a tuple of ints (None stays None); accepts lists from JSON protocol files."""
    return None if x is None else (int(x[0]), int(x[1]))


def nearest_index(src_len: int, dst_len: int) -> np.ndarray:
    """Source index of every target pixel for an exact pixel-centre nearest-neighbour resize of one axis:
    floor((i + 0.5) * src / dst), in exact integer arithmetic (the rule of `geometry.warp_image(interp="nearest")`)."""
    src_len, dst_len = int(src_len), int(dst_len)
    return ((2 * np.arange(dst_len, dtype=np.int64) + 1) * src_len) // (2 * dst_len)


def resize_nearest(x: np.ndarray, size, axes=(-2, -1)) -> np.ndarray:
    """Exact pixel-centre nearest-neighbour resize of the two image axes `axes` (default the last two, for (..., H, W)
    maps; use axes=(-3, -2) for (..., H, W, C)) to size = (W, H). Any dtype, values copied exactly."""
    x = np.asarray(x)
    w, h = int(size[0]), int(size[1])
    ay, ax = axes
    H, W = x.shape[ay], x.shape[ax]
    if (W, H) == (w, h):
        return x
    x = np.take(x, nearest_index(H, h), axis=ay)
    return np.take(x, nearest_index(W, w), axis=ax)


def resize_map(x: np.ndarray, size, interp: str = "bilinear") -> np.ndarray:
    """Resize float maps (n, H, W) to size = (W, H) on the 'resize' pixel map; NaN marks missing values. Returns
    float32.

    "nearest"   exact pixel-centre nearest neighbour (`resize_nearest`).
    "bilinear"  missing-value-aware bilinear (cv2 INTER_LINEAR support, pixel-centre aligned): every output pixel is
                the weighted mean of the VALID source pixels of its 2 x 2 support (weights renormalised), and is
                missing exactly where its nearest source pixel is missing. Holes therefore keep the footprint they have
                under nearest-neighbour resampling (they neither grow nor get filled in) and missing values never mix
                into valid ones."""
    x = np.asarray(x, np.float32)
    w, h = int(size[0]), int(size[1])
    if x.shape[-2:] == (h, w):
        return x
    if interp == "nearest":
        return resize_nearest(x, (w, h))
    if interp != "bilinear":
        raise ValueError(f"interp={interp!r} not in ('bilinear', 'nearest')")
    import cv2
    lead = x.shape[:-2]
    flat = x.reshape((-1,) + x.shape[-2:])
    out = np.empty((len(flat), h, w), np.float32)
    for i, f in enumerate(flat):
        ok = np.isfinite(f)
        if ok.all():
            out[i] = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            continue
        num = cv2.resize(np.where(ok, f, 0.0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        den = cv2.resize(ok.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        r[~resize_nearest(ok, (w, h)) | (den <= 1e-6)] = np.nan
        out[i] = r
    return out.reshape(lead + (h, w))


def sample_map(img: np.ndarray, uv: np.ndarray, interp: str = "nearest") -> np.ndarray:
    """Values of a map (H, W) or (H, W, C) at pixel positions uv (..., 2) (OpenCV convention, integer = pixel centre).

    nearest: pixel (floor(u + 0.5), floor(v + 0.5)); bilinear: the four neighbours (edge-replicated in the outer half
    pixel), NaN if any neighbour with non-zero weight is NaN. Positions outside [-0.5, W - 0.5) x [-0.5, H - 0.5) or
    non-finite give NaN. Returns float64 (...,) or (..., C)."""
    a = np.asarray(img)
    H, W = a.shape[:2]
    uv = np.asarray(uv, np.float64)
    u, v = uv[..., 0], uv[..., 1]
    extra = a.shape[2:]
    out = np.full(u.shape + extra, np.nan)
    with np.errstate(invalid="ignore"):
        ins = np.isfinite(u) & np.isfinite(v) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    uu, vv = u[ins], v[ins]
    if interp == "nearest":
        iu = np.clip(np.floor(uu + 0.5).astype(np.int64), 0, W - 1)
        iv = np.clip(np.floor(vv + 0.5).astype(np.int64), 0, H - 1)
        out[ins] = a[iv, iu]
        return out
    if interp != "bilinear":
        raise ValueError(f"interp={interp!r} not in ('nearest', 'bilinear')")
    uc, vc = np.clip(uu, 0, W - 1), np.clip(vv, 0, H - 1)
    u0, v0 = np.floor(uc).astype(np.int64), np.floor(vc).astype(np.int64)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    wu, wv = uc - u0, vc - v0
    acc = np.zeros(uc.shape + extra)
    bad = np.zeros(uc.shape + extra, bool)
    for iu, iv, wt in ((u0, v0, (1 - wu) * (1 - wv)), (u1, v0, wu * (1 - wv)), (u0, v1, (1 - wu) * wv),
                       (u1, v1, wu * wv)):
        val = a[iv, iu].astype(np.float64)
        wt = wt.reshape(wt.shape + (1,) * len(extra))
        use = np.broadcast_to(wt > 0, val.shape)
        okv = np.isfinite(val)
        bad |= use & ~okv
        acc += np.where(use & okv, wt * np.where(okv, val, 0.0), 0.0)
    acc[bad] = np.nan
    out[ins] = acc
    return out


def fov_mask(sample, view: int, size=None, mode: str = "resize") -> Optional[np.ndarray]:
    """Valid field of view of an undistorted video: (H, W) bool on the (W, H) grid (default the stored video) obtained
    with resize mode `mode`, True where the pixel's ray lands inside the raw (distorted) capture, so the pixel shows
    scene content. False on the black border that undistortion leaves (Real-General, whose cameras.npz has `K_raw`,
    radtan5 `dist`, `raw_size`); a raw pixel must be at least 0.5 px inside the raw image (bilinear support), so the
    one-pixel anti-aliased ring of the border is excluded too. Returns None when the sample has no distortion model
    (every pixel valid)."""
    ex = sample.cameras.extra
    if not all(k in ex for k in ("K_raw", "dist", "raw_size")):
        return None
    W, H = sample.image_size(view) if size is None else (int(size[0]), int(size[1]))
    K = sample.K(view, (W, H), mode)
    k1, k2, p1, p2, k3 = (float(c) for c in np.asarray(ex["dist"][view], np.float64)[:5])
    Kr = np.asarray(ex["K_raw"][view], np.float64)
    Wr, Hr = (int(c) for c in np.asarray(ex["raw_size"]).reshape(-1)[:2])
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    y = (v - K[1, 2]) / K[1, 1]
    x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
    r2 = x * x + y * y
    rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    ur = Kr[0, 0] * xd + Kr[0, 1] * yd + Kr[0, 2]
    vr = Kr[1, 1] * yd + Kr[1, 2]
    m = (ur >= 0.0) & (ur <= Wr - 1.0) & (vr >= 0.0) & (vr <= Hr - 1.0)
    if mode == "letterbox":
        imap = sample.image_map(view, (W, H), mode)
        x0, y0 = int(imap.ox), int(imap.oy)
        nw, nh = int(round(sample.image_size(view)[0] * imap.sx)), int(round(sample.image_size(view)[1] * imap.sy))
        pad = np.ones((H, W), bool)
        pad[y0:y0 + nh, x0:x0 + nw] = False
        m &= ~pad
    return m


def gt_depth(sample, view: int, indices, size=None, chunk: int = 8) -> Optional[np.ndarray]:
    """GT depth (n, h, w) float32 metres (0 = no GT) of `view` at frames `indices` on the 'resize' grid size = (w, h)
    (default: the stored video), exact pixel-centre nearest resampling (= `Sample.depth(view, indices, size)`). None
    when the sample has no depth. Reads `chunk` frames at a time."""
    return read_gt_depth(sample, view, indices, size, chunk=chunk)[0]


def read_gt_depth(sample, view: int, indices, size=None, obs: Optional["TrackObs"] = None, tol=None,
                  chunk: int = 8) -> tuple:
    """Read the GT depth of `view` at frames `indices` once and return (depth (n, h, w) float32 on the 'resize' grid
    `size` (exact nearest; None = video grid) or None without depth, track mask (n, N) or None).

    With `obs` (track observations of the same view and frames) and tol = (abs_m, rel), the track mask is obs.mask
    restricted to the points whose dense GT depth D agrees with their camera z, |D - z| <= max(abs_m, rel * z), both at
    their nearest pixel of the video grid and at their nearest pixel of the output grid `size`. The first test drops
    points that are not the visible surface at their pixel (silhouette points seen against a farther background, points
    behind the surface within the visibility tolerance); the second drops points whose pixel of the coarser output grid
    shows another surface (object edges), so the kept points are unambiguous at the evaluation resolution. Without depth
    or tol, obs.mask is returned unchanged (None without obs)."""
    mask = None if obs is None else obs.mask.copy()
    if not sample.has_depth:
        return None, mask
    idx = [int(t) for t in indices]
    native = sample.image_size(view)
    to_out = None if size is None or tuple(size) == tuple(native) else image_map(native, size, "resize")

    def test(dmap, i, uv_map=None):
        m = mask[i]
        if not m.any():
            return
        uv = obs.uv[i, m] if uv_map is None else uv_map.apply_uv(obs.uv[i, m])
        D = sample_map(dmap, uv, "nearest")
        z = obs.z[i, m]
        with np.errstate(invalid="ignore"):
            good = (D > 0) & (np.abs(D - z) <= np.maximum(float(tol[0]), float(tol[1]) * z))
        mask[i, np.flatnonzero(m)[~good]] = False

    out = []
    for a in range(0, len(idx), chunk):
        d = sample.depth(view, idx[a:a + chunk])
        r = d if to_out is None else resize_nearest(d, size)
        if obs is not None and tol is not None:
            for i in range(len(d)):
                test(d[i], a + i)
                if to_out is not None:
                    test(r[i], a + i, to_out)
        out.append(r)
    return np.concatenate(out, 0).astype(np.float32, copy=False), mask


@dataclass
class TrackObs:
    """GT track points observed in one view over frames `indices` (n frames, N points):
    uv (n, N, 2) float64 pixel coordinates on the stored video grid (NaN where not observed), z (n, N) camera z [m],
    xyz (n, N, 3) world [m], mask (n, N) bool: default_gt & visible (or in_frustum) in the view & inside the valid
    field of view."""
    uv: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    mask: np.ndarray


def track_observations(sample, view: int, indices, visibility: str = "visible", use_default_gt: bool = True,
                       fov: Optional[np.ndarray] = None) -> TrackObs:
    """Observations of the GT tracks in `view` at frames `indices` (see `TrackObs`). visibility: "visible"
    (occlusion-aware flag) or "in_frustum". `fov` = valid field-of-view mask on the stored video grid (points whose
    nearest pixel is outside it are dropped)."""
    if visibility not in ("visible", "in_frustum"):
        raise ValueError("visibility must be 'visible' or 'in_frustum'")
    tr = sample.tracks
    idx = np.asarray(indices, np.int64)
    K = np.asarray(sample.cameras.K[view], np.float64)
    W, H = sample.image_size(view)
    xyz = tr.xyz[idx].astype(np.float64)
    flag = (tr.visible if visibility == "visible" else tr.in_frustum)[idx, :, view]
    mask = flag & tr.valid[idx]
    if use_default_gt:
        mask &= tr.default_gt[idx]
    w2c = np.asarray(sample.cameras.w2c[view, idx], np.float64)
    with np.errstate(invalid="ignore"):
        Xc = np.einsum("nij,nkj->nki", w2c[:, :3, :3], xyz) + w2c[:, None, :3, 3]
        z = Xc[..., 2]
        u = (K[0, 0] * Xc[..., 0] + K[0, 1] * Xc[..., 1]) / z + K[0, 2]
        v = K[1, 1] * Xc[..., 1] / z + K[1, 2]
        mask &= np.isfinite(u) & np.isfinite(v) & (z > 0) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
    uv = np.stack([u, v], -1)
    if fov is not None and mask.any():
        iu = np.clip(np.floor(np.where(mask, u, 0) + 0.5).astype(np.int64), 0, W - 1)
        iv = np.clip(np.floor(np.where(mask, v, 0) + 0.5).astype(np.int64), 0, H - 1)
        mask &= fov[iv, iu]
    uv[~mask] = np.nan
    z = np.where(mask, z, np.nan)
    return TrackObs(uv=uv, z=z, xyz=xyz, mask=mask)


def track_path_length(xyz: np.ndarray, valid: np.ndarray) -> tuple:
    """GT path length of every track over a frame range: (path (N,) metres, n_valid (N,) int).

    xyz (T, N, 3), valid (T, N) bool (positions that are not finite are treated as invalid). The path sums the steps
    between consecutive VALID frames of each track, bridging gaps (the rule of `metrics.tracking`'s path length);
    tracks with fewer than 2 valid frames get path 0 and n_valid < 2 (no motion information)."""
    xyz = np.asarray(xyz, np.float64)
    ok = np.asarray(valid, bool) & np.isfinite(xyz).all(-1)
    T, N = ok.shape
    path = np.zeros(N)
    for n in np.flatnonzero(ok.sum(0) >= 2):
        p = xyz[ok[:, n], n]
        path[n] = float(np.linalg.norm(p[1:] - p[:-1], axis=-1).sum())
    return path, ok.sum(0).astype(np.int64)


def export_view_frames(sample, view: int, indices, out_dir, frames: str = "mp4", size=None, mode: str = "resize",
                       crf: int = 10, lossless: bool = False, name: Optional[str] = None) -> dict:
    """Write the frames `indices` of `view` for a model: <out_dir>/<name>.mp4 (H.264 crf `crf`, or lossless RGB) at
    fps = clip fps / frame step, or <out_dir>/<name>/frame_{t:04d}.png, or nothing ("none"). Returns the inputs.json
    entry {"file", "size" [W, H], "K" (at that size), "source_video" (relative to the benchmark root),
    "resize_mode"}."""
    if frames not in FRAME_MODES:
        raise ValueError(f"frames={frames!r} not in {FRAME_MODES}")
    out_dir = Path(out_dir)
    name = name or f"view{view}"
    idx = [int(t) for t in indices]
    W, H = sample.image_size(view) if size is None else (int(size[0]), int(size[1]))
    rec = {"file": None, "size": [W, H], "K": sample.K(view, (W, H), mode).tolist(),
           "source_video": f"{sample.subset}/{sample.id}/{sample.meta['views'][view]['video']}",
           "resize_mode": mode}
    if frames == "none":
        return rec
    sz = None if size is None else (W, H)
    if frames == "mp4":
        from ..io.video import write_video
        step = idx[1] - idx[0] if len(idx) > 1 else 1
        fr = sample.frames(view, idx, size=sz, mode=mode)
        write_video(out_dir / f"{name}.mp4", fr, fps=sample.fps / max(step, 1), crf=crf, lossless=lossless)
        rec["file"] = f"{name}.mp4"
        return rec
    import cv2
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for a in range(0, len(idx), 16):
        fr = sample.frames(view, idx[a:a + 16], size=sz, mode=mode)
        for t, f in zip(idx[a:a + 16], fr):
            cv2.imwrite(str(d / f"frame_{t:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    rec["file"] = f"{name}/"
    return rec


def write_manifest(out_dir, records: list, skipped: list, protocol: TaskProtocol, keys: Sequence[str]) -> None:
    """<out_dir>/manifest.jsonl (one line per exported sample, `keys` only), protocol.json and skipped.json."""
    out_dir = Path(out_dir)
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(json_safe({k: r[k] for k in keys if k in r}), ensure_ascii=False) + "\n")
    protocol.save(out_dir / "protocol.json")
    write_json(out_dir / "skipped.json", skipped)


def canonical_ids(ids) -> Optional[list]:
    """Sample ids / uids with a subset alias in a uid resolved ("sr/s000" -> "Simul-Robot/s000", `utils.canonical_uid`);
    None stays None. An unknown subset part is left as given (it then matches no sample)."""
    from ..utils import canonical_uid
    return None if ids is None else [canonical_uid(i) for i in ([ids] if isinstance(ids, str) else ids)]


def iter_samples(bench, subsets=None, ids=None, clip_filter: Optional[dict] = None):
    """Samples of `subsets` (default all; aliases accepted), optionally restricted to `ids` (ids or uids; subset aliases
    accepted in uids) and to attribute filters (index.json). One `Benchmark.samples` call over all the subsets: after
    a complete iteration it warns about requested ids that match no selected sample."""
    subsets = [subsets] if isinstance(subsets, str) else subsets
    yield from bench.samples(subsets or None, ids=ids, **(clip_filter or {}))


def select_samples(bench, subsets=None, ids=None, clip_filter: Optional[dict] = None) -> list:
    """`iter_samples` as a list: a complete iteration, so requested ids that match no selected sample (a typo, the id of
    a subset that is not selected, or a sample excluded by clip_filter) are reported in a warning
    (`Benchmark.samples`; `Benchmark.unmatched_ids` lists them)."""
    return list(iter_samples(bench, subsets, ids, clip_filter))


def run_jobs(fn: Callable, args: list, jobs: int = 1, desc: str = "", progress: bool = True) -> list:
    """fn(*a) for every a in args, in a process pool when jobs > 1 (results in input order)."""
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(args), desc=desc, leave=False)
        except ImportError:
            bar = None
    out = [None] * len(args)
    if jobs <= 1 or len(args) <= 1:
        for i, a in enumerate(args):
            out[i] = fn(*a)
            if bar is not None:
                bar.update(1)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futs = {ex.submit(fn, *a): i for i, a in enumerate(args)}
            for f in as_completed(futs):
                out[futs[f]] = f.result()
                if bar is not None:
                    bar.update(1)
    if bar is not None:
        bar.close()
    return out


class MissingPrediction(FileNotFoundError):
    """A prediction file of an evaluable sample does not exist (status "missing")."""


class InvalidPrediction(ValueError):
    """A prediction file exists but cannot be used: unreadable, wrong keys, shapes, frame count or meta data (status
    "invalid"; a ValueError, so code that catches ValueError keeps working)."""


def read_npz(path, keys: Optional[Sequence[str]] = None) -> tuple:
    """(names, {name: array}) of an .npz prediction file; only the arrays named in `keys` are read (all when None).
    Raises InvalidPrediction when the file is not a readable .npz archive (corrupt, a plain .npy, pickled objects)."""
    errors = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)
    try:
        z = np.load(path, allow_pickle=False)
    except errors as e:
        raise InvalidPrediction(f"{path}: not a readable .npz file ({type(e).__name__}: {e})") from e
    if not hasattr(z, "files"):
        raise InvalidPrediction(f"{path}: not an .npz archive of named arrays (write it with np.savez_compressed)")
    try:
        with z:
            names = list(z.files)
            return names, {k: np.asarray(z[k]) for k in names if keys is None or k in keys}
    except errors as e:
        raise InvalidPrediction(f"{path}: cannot read the arrays ({type(e).__name__}: {e})") from e


def frame_index_array(a, path) -> np.ndarray:
    """A prediction's 'frame_indices' as (n,) int64. Raises InvalidPrediction unless it is 1-D with integer values
    (integral floats accepted)."""
    a = np.asarray(a)
    ok = a.ndim == 1 and a.dtype != bool and (np.issubdtype(a.dtype, np.integer) or (
        np.issubdtype(a.dtype, np.floating) and bool(np.all(np.isfinite(a) & (a == np.round(a))))))
    if not ok:
        raise InvalidPrediction(f"{path}: 'frame_indices' must be (n,) integer frame indices, got {a.dtype} {a.shape}")
    return a.astype(np.int64)


ASPECT_TOL = 0.02


def transposed_hint(pred_size, video_size, tol: float = ASPECT_TOL) -> Optional[str]:
    """A warning text when prediction maps of size (w, h) have the TRANSPOSED aspect of the video (W, H) (within tol)
    and not its own: arrays saved as (n, W, H) instead of (n, H, W). None otherwise; any other size is accepted (maps
    may have any resolution and aspect) and a square video cannot be checked."""
    w, h = (int(x) for x in pred_size)
    W, H = (int(x) for x in video_size)
    if min(w, h, W, H) < 1:
        return None
    if abs((w / h) / (W / H) - 1.0) > tol and abs((h / w) / (W / H) - 1.0) <= tol:
        return (f"prediction maps are {w}x{h} (W x H) but the video is {W}x{H}: H and W swapped? (maps are (..., H, W) "
                "covering the full frame; scored as given)")
    return None


def base_record(task: str, sample, protocol: TaskProtocol) -> dict:
    return {"task": task, "protocol": protocol.name, "uid": sample.uid, "subset": sample.subset, "id": sample.id,
            "window": {"first": protocol.ref_frame, "last": protocol.last_frame, "step": protocol.frame_step,
                       "n": protocol.horizon + 1}}


def prediction_stamp(pred_root, sample) -> Optional[str]:
    """Fingerprint of the sample's prediction folder <pred_root>/<Subset>/<id>/: a hash of the relative path, size and
    modification time of every file in it (None when the folder does not exist). A cached per-sample result is reused
    only while the stamp is unchanged."""
    d = Path(pred_root) / sample.subset / sample.id
    if not d.is_dir():
        return None
    h = hashlib.sha1()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(d).as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def run_fingerprint(task: str, protocol: TaskProtocol, pred_root) -> tuple:
    """(run_id, run dict): one evaluation run = task + protocol + resolved prediction root."""
    run = {"task": task, "pred_root": str(Path(pred_root).resolve()), "protocol": protocol.to_dict()}
    rid = hashlib.sha1(json.dumps(run, sort_keys=True).encode()).hexdigest()[:16]
    return rid, dict(run, run_id=rid)


class RunDirError(UsageError, ValueError):
    """An out dir that belongs to another run (task, protocol, prediction root; recorded in <out>/run.json) or holds
    the inputs of other export options (<out>/export.json): refused unless overwrite (--overwrite). Raised by depth /
    recon4d / nvs (`evaluate_task`), segmentation / understanding (`check_out_stamp`) and track3d / track2d. A
    `utils.UsageError`, so `vpmwb` reports it as one line `vpmwb: error: ...` with exit status 2, and a ValueError,
    so code that catches ValueError keeps working."""


def run_pred_root(out_dir) -> Optional[str]:
    """The resolved prediction root of the run recorded in <out_dir>/run.json (None without a readable run.json)."""
    try:
        run = json.loads((Path(out_dir) / RUN_FILE).read_text())
    except (OSError, ValueError):
        return None
    return run.get("pred_root") if isinstance(run, dict) else None


def check_out_stamp(out_dir, name: str, content: dict, overwrite: bool, task: str) -> bool:
    """Compare the stamp file <out_dir>/<name> with `content` (run.json of an evaluation: task, protocol, prediction
    root, ...; export.json of an export: task, protocol, export options). Returns True when the file exists and differs
    (an unreadable file differs), else False. A difference without overwrite raises `RunDirError`, naming what differs
    (another protocol, the prediction root of the other run, other options) and the way out (--overwrite; it deletes
    the other run's results, or rewrites the inputs)."""
    f = Path(out_dir) / name
    if not f.exists():
        return False
    try:
        old = json.loads(f.read_text())
    except ValueError:
        old = None
    new = json_safe(content)
    if old == new:
        return False
    if overwrite:
        return True
    if not isinstance(old, dict):
        raise RunDirError(f"{out_dir} holds an unreadable {name}; use another out dir or overwrite=True (--overwrite)")
    diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    owner, other = old.get("task") or task, f"{name}: other {', '.join(diff)}"
    if name != RUN_FILE:
        what, then = f"inputs of other export options ({other})", "which rewrites them"
    elif "task" in diff or "protocol" in diff:
        what, then = f"results of another protocol ({name})", "which deletes them"
    elif "pred_root" in diff:
        what, then = f"results of the predictions in {old.get('pred_root')} ({name})", "which deletes them"
    else:
        what, then = f"results of another run ({other})", "which deletes them"
    raise RunDirError(f"{out_dir} holds {owner} {what}; use another out dir or overwrite=True (--overwrite), {then}")


def evaluate_one(task: str, fn: Callable, root, subset: str, sid: str, pred_root, out_dir, protocol_dict: dict,
                 protocol_cls, overwrite: bool, strict: bool, opts: dict, run_id: Optional[str] = None) -> dict:
    """Shared driver of one sample: reuse a finished result (status ok / skipped, or not_evaluable in records of
    earlier versions) of the same run whose prediction folder is unchanged (`prediction_stamp`) unless overwrite,
    check evaluability (status "skipped"), run fn(sample, pred_root, protocol, **opts) and write
    <out_dir>/<Subset>/<id>.json. MissingPrediction gives status "missing", InvalidPrediction status "invalid", other
    exceptions status "error" with the message (invalid / error re-raised when strict). Returns {"uid", "status",
    "cached", "reason", "warnings"}."""
    from ..data.sample import Sample
    sample = Sample(root, subset, sid)
    path = sample_json_path(out_dir, sample)
    stamp = prediction_stamp(pred_root, sample)
    if path.exists() and not overwrite:
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            rec = {}
        if (rec.get("task") == task and canonical_status(rec.get("status")) in ("ok", "skipped")
                and rec.get("run_id") == run_id and rec.get("pred_stamp") == stamp):
            return {"uid": sample.uid, "status": canonical_status(rec["status"]), "cached": True,
                    "reason": rec.get("reason"), "warnings": rec.get("warnings") or []}
    protocol = protocol_cls.from_dict(protocol_dict)
    rec = base_record(task, sample, protocol)
    rec.update(run_id=run_id, pred_stamp=stamp)
    ok, why = protocol.evaluable(sample)
    if not ok:
        rec.update(status="skipped", reason=why)
    else:
        try:
            rec.update(fn(sample, pred_root, protocol, **opts))
        except MissingPrediction as e:
            rec.update(status="missing", reason=str(e))
        except InvalidPrediction as e:
            if strict:
                raise
            rec.update(status="invalid", reason=str(e))
        except Exception as e:
            if strict:
                raise
            rec.update(status="error", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=4))
    write_json(path, rec)
    return {"uid": sample.uid, "status": rec["status"], "cached": False, "reason": rec.get("reason"),
            "warnings": rec.get("warnings") or []}


def _remove_stale(out_dir: Path, task: str, run_id: str) -> int:
    n = 0
    for sub in SUBSETS:
        for p in sorted((out_dir / sub).glob("*.json")) if (out_dir / sub).is_dir() else []:
            try:
                r = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if r.get("task") == task and r.get("run_id") != run_id:
                p.unlink()
                n += 1
    return n


def run_status(statuses, unmatched_ids=()) -> dict:
    """Status counts of the samples selected by one evaluate call: {"n_samples", "status" {name: n}, "n_failed"
    (status invalid / error), "all_failed", "unmatched_ids" (requested ids that match no sample)}. all_failed: at least
    one selected sample is evaluable (status other than skipped) and none of them is scored (ok / partial; results
    reused from an earlier call count): every one is missing, invalid or error, so there is no number to report (the
    task command lines then exit with status 1)."""
    st = [canonical_status(s) for s in statuses]
    counts = {}
    for s in st:
        counts[s] = counts.get(s, 0) + 1
    evaluable = [s for s in st if s != "skipped"]
    return {"n_samples": len(st), "status": dict(sorted(counts.items())),
            "n_failed": int(sum(s in FAILED_STATUSES for s in st)),
            "all_failed": bool(evaluable) and not any(s in SCORED_STATUSES for s in evaluable),
            "unmatched_ids": list(unmatched_ids or [])}


def evaluate_task(task: str, fn: Callable, protocol, protocol_cls, bench, pred_root, out_dir, subsets=None, ids=None,
                  jobs: int = 1, overwrite: bool = False, strict: bool = False, progress: bool = True,
                  summarize_fn: Optional[Callable] = None, **opts) -> dict:
    """Evaluate every selected sample (in parallel with `jobs` processes) and summarise.

    <out_dir> belongs to one run (task, protocol, resolved prediction root; recorded in <out_dir>/run.json and in
    every per-sample JSON). Evaluating another run into the same folder raises `RunDirError` (a ValueError; `vpmwb`:
    one line, exit status 2) unless overwrite=True (--overwrite), which deletes the per-sample results of the previous
    run; within a run, finished samples are reused while their
    prediction files are unchanged (overwrite=True re-evaluates the selected samples). Warns about requested ids that
    match no sample, failed samples (status invalid / error) and samples whose record carries "warnings". Returns the
    summary of `summarize_fn` plus "evaluated" = `run_status` of the samples selected by this call (not written to
    summary.json): the command line exits with status 1 when nothing is selected or no evaluable sample is scored."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rid, run = run_fingerprint(task, protocol, pred_root)
    rfile, pfile = out_dir / RUN_FILE, out_dir / "protocol.json"
    old = json.loads(rfile.read_text()) if rfile.exists() else None
    if old is None and pfile.exists():
        old = {"run_id": None, "pred_root": "?", "protocol": json.loads(pfile.read_text())}
    if old is not None and old.get("run_id") != rid:
        if not overwrite:
            if not rfile.exists():
                what = f"a protocol.json without {RUN_FILE} (inputs, or results of an earlier version)"
            elif old.get("protocol") != run["protocol"]:
                what = f"{old.get('task', task)} results of another protocol"
            else:
                what = f"{old.get('task', task)} results of the predictions in {old.get('pred_root')}"
            raise RunDirError(f"{out_dir} holds {what}; use another out dir or overwrite=True (--overwrite), which "
                              "deletes them")
        _remove_stale(out_dir, task, rid)
    write_json(rfile, run)
    protocol.save(pfile)
    args = [(task, fn, str(bench.root), s.subset, s.id, str(pred_root), str(out_dir), protocol.to_dict(), protocol_cls,
             overwrite, strict, opts, rid) for s in select_samples(bench, subsets, ids, protocol.clip_filter)]
    res = run_jobs(evaluate_one, args, jobs, desc=task, progress=progress)
    bad = [r for r in res if r["status"] in FAILED_STATUSES]
    if bad:
        warnings.warn(f"{task}: {len(bad)} sample(s) failed (status invalid / error), e.g. {bad[0]['uid']}: "
                      f"{str(bad[0].get('reason') or '')[:300]} (see the per-sample JSON 'reason')")
    warned = [r for r in res if r.get("warnings")]
    if warned:
        warnings.warn(f"{task}: {len(warned)} sample(s) with warnings, e.g. {warned[0]['uid']}: "
                      f"{warned[0]['warnings'][0]}")
    summary = summarize_fn(out_dir) if summarize_fn is not None else {}
    unmatched = bench.unmatched_ids(ids, subsets, **protocol.clip_filter) if ids else []
    summary["evaluated"] = run_status([r["status"] for r in res], unmatched)
    return summary


def mean_se(values) -> dict:
    """{"mean", "se", "n", "n_undefined", "n_inf"} over finite values (se = sample std / sqrt(n), NaN for n < 2).
    NaN / None values (metric undefined for that sample) and +-inf (failures) are counted, not averaged."""
    x = np.array([json_number(v) for v in values], np.float64)
    fin = x[np.isfinite(x)]
    n = int(fin.size)
    return {"mean": float(fin.mean()) if n else float("nan"),
            "se": float(fin.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
            "n": n, "n_undefined": int(np.isnan(x).sum()), "n_inf": int(np.isinf(x).sum())}


def _fmt(v: float, digits: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{digits}f}"


def summarize_records(out_dir, task: str, title: str, headline: Sequence[tuple], notes: Sequence[str] = (),
                      subset_notes: Optional[dict] = None) -> dict:
    """Aggregate <out_dir>/<Subset>/<id>.json of `task` (only the records of the run in <out_dir>/run.json, when it
    exists; its prediction root is recorded as "pred_root"): per subset the status counts ("status", `STATUSES` names;
    earlier records' names mapped by `canonical_status`), the failed samples ("n_invalid" / "n_error"), the number of
    records with "warnings" and, for every key of the per-sample "metrics" dicts of the scored samples (status "ok"),
    mean +- s.e. over samples (`mean_se`). Writes <out_dir>/summary.json and summary.md (rows = `headline`: (key, label,
    digits) tuples; absent keys skipped; `subset_notes` {subset: [lines]} under each subset table). Returns the summary
    dict."""
    out_dir = Path(out_dir)
    proto = json.loads((out_dir / "protocol.json").read_text()) if (out_dir / "protocol.json").exists() else None
    run = json.loads((out_dir / RUN_FILE).read_text()) if (out_dir / RUN_FILE).exists() else None
    summary = {"task": task, "generated": datetime.datetime.now().isoformat(timespec="seconds"), "protocol": proto,
               "pred_root": None if run is None else run.get("pred_root"), "subsets": {}}
    for sub in SUBSETS:
        d = out_dir / sub
        if not d.is_dir():
            continue
        recs, stale = [], 0
        for p in sorted(d.glob("*.json")):
            try:
                r = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if r.get("task") != task:
                continue
            if run is not None and r.get("run_id") != run.get("run_id"):
                stale += 1
                continue
            recs.append(r)
        if not recs:
            continue
        status = {}
        for r in recs:
            st = canonical_status(r.get("status", "?"))
            status[st] = status.get(st, 0) + 1
        ok = [r for r in recs if r.get("status") == "ok"]
        keys = sorted({k for r in ok for k in r.get("metrics", {})})
        metrics = {k: mean_se([r["metrics"][k] for r in ok if k in r["metrics"]]) for k in keys}
        summary["subsets"][sub] = {"n_samples": len(recs), "status": status, "n_invalid": status.get("invalid", 0),
                                   "n_error": status.get("error", 0), "metrics": metrics, "n_stale": stale,
                                   "n_warnings": sum(1 for r in recs if r.get("warnings"))}
    write_json(out_dir / "summary.json", summary)
    lines = [f"# {title}", ""]
    if proto:
        lines += [f"Protocol `{proto.get('name')}`: frames {proto.get('ref_frame')}.."
                  f"{proto.get('ref_frame', 0) + proto.get('horizon', 0) * proto.get('frame_step', 1)} step "
                  f"{proto.get('frame_step')} ({proto.get('horizon', 0) + 1} frames). Mean +- s.e. over samples; n = "
                  "samples with a finite value; '(k inf)' = k failures (e.g. a reconstruction that could not be "
                  "aligned), excluded from the mean.", ""]
    for n in notes:
        lines.append(f"- {n}")
    if notes:
        lines.append("")
    for sub, s in summary["subsets"].items():
        st = ", ".join(f"{k} {v}" for k, v in sorted(s["status"].items()))
        lines += [f"## {sub}", "", f"{s['n_samples']} samples ({st})", "", "| metric | mean +- s.e. | n |",
                  "|---|---|---|"]
        for key, label, digits in headline:
            m = s["metrics"].get(key)
            if m is None:
                continue
            extra = f" ({m['n_inf']} inf)" if m["n_inf"] else ""
            lines.append(f"| {label} | {_fmt(m['mean'], digits)} +- {_fmt(m['se'], digits)} | {m['n']}{extra} |")
        lines.append("")
        if s.get("n_warnings"):
            lines += [f"- {s['n_warnings']} sample(s) with warnings (e.g. H and W swapped?): see \"warnings\" in the "
                      "per-sample JSON", ""]
        for n in (subset_notes or {}).get(sub, []):
            lines.append(f"- {n}")
        if (subset_notes or {}).get(sub):
            lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))
    return summary


def _baseline_one(name, root, subset, sid, pred_root, protocol_dict, protocol_cls, baselines, opts):
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = protocol_cls.from_dict(protocol_dict)
    ok, why = p.evaluable(s)
    if not ok:
        return {"uid": s.uid, "skipped": why}
    baselines[name](s, pred_root, p, **opts)
    return {"uid": s.uid}


def baseline_args(bench, pred_root, kind, opts: dict) -> tuple:
    """(bench, pred_root, kind) of a `run_baseline` call. Every task takes run_baseline(bench, pred_root, kind=...,
    protocol=..., ...); the form of earlier depth / recon4d / nvs versions, run_baseline(kind, bench, pred_root, ...),
    and the keyword name=... are still accepted (`opts` loses "name")."""
    if isinstance(bench, str) and hasattr(pred_root, "samples"):
        bench, pred_root, kind = pred_root, kind, bench
    if "name" in opts:
        kind = opts.pop("name")
    if not hasattr(bench, "samples") or pred_root is None:
        raise TypeError("run_baseline(bench, pred_root, kind=...) needs a vpm_worldbench.Benchmark and the prediction "
                        f"root, got bench={type(bench).__name__}, pred_root={pred_root!r}")
    return bench, pred_root, kind


def run_baseline_task(name: str, baselines: dict, protocol, protocol_cls, bench, pred_root, subsets=None, ids=None,
                      jobs: int = 1, progress: bool = True, **opts) -> dict:
    """Write the predictions of baseline `name` (baselines[name](sample, pred_root, protocol, **opts)) for every
    evaluable sample. Returns {"written": n, "skipped": m}."""
    if name not in baselines:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(baselines)}")
    args = [(name, str(bench.root), s.subset, s.id, str(pred_root), protocol.to_dict(), protocol_cls, baselines, opts)
            for s in select_samples(bench, subsets, ids, protocol.clip_filter)]
    res = run_jobs(_baseline_one, args, jobs, desc=f"baseline {name}", progress=progress)
    return {"written": sum("skipped" not in r for r in res), "skipped": sum("skipped" in r for r in res)}


def _fmt_value(v) -> str:
    if v is None or isinstance(v, (str, bool, int, float)):
        return str(v)
    return json.dumps(json_safe(v)).replace(" ", "")


def override_protocol(p, overrides: dict, cls=None):
    """`p` with the fields `overrides` replaced. When a value changes, the name gets a "+field=value,..." suffix (the
    rule of track3d / track2d), so the results of a modified protocol are not labelled as the preset or the file."""
    cls = cls or type(p)
    d = p.to_dict()
    changed = {k: v for k, v in overrides.items() if json_safe(d.get(k)) != json_safe(v)}
    if not changed:
        return p
    d.update(changed)
    d["name"] = f"{p.name}+" + ",".join(f"{k}={_fmt_value(v)}" for k, v in changed.items())
    return cls.from_dict(d)


def cli_protocol(parser, spec, protocol_cls, load: Optional[Callable] = None):
    """The --protocol option of a task command: None / "" / the preset name -> protocol_cls(); an existing JSON / YAML
    file -> load(path) (default protocol_cls.load). Anything else, or a file that does not load, is a usage error
    (parser.error, exit status 2) naming the preset."""
    name = protocol_cls().name
    if spec in (None, "", name):
        return protocol_cls()
    if not Path(spec).is_file():
        parser.error(f"--protocol {spec!r}: neither the preset {name!r} nor an existing JSON / YAML file of "
                     f"{protocol_cls.__name__} fields")
    try:
        return (load or protocol_cls.load)(spec)
    except Exception as e:
        parser.error(f"--protocol {spec}: {type(e).__name__}: {e}")


def cli_subsets(parser, values) -> Optional[list]:
    """`utils.parse_subsets` of a --subsets option; an unknown subset is a usage error (exit status 2)."""
    from ..utils import parse_subsets
    try:
        return parse_subsets(values)
    except KeyError as e:
        parser.error(str(e.args[0]) if e.args else str(e))


def cli_size(parser, flag: str, value) -> Optional[tuple]:
    """(W, H) of a WxH option such as --size (`utils.parse_size`; None / '' / 'native' -> None). A value that is not two
    positive integers (`utils.SizeError`) is a usage error of `parser` (usage + one line, exit status 2)."""
    from ..utils import parse_size
    try:
        return parse_size(value, flag)
    except ValueError as e:
        parser.error(str(e))


def evaluation_exit_status(summary: dict, task: str, out_dir) -> int:
    """Exit status of an evaluate command from the "evaluated" entry of its summary (`run_status`): 1 with a one-line
    message on stderr when the selection matched no sample or no evaluable sample is scored (every one missing,
    invalid or error), else 0."""
    ev = (summary or {}).get("evaluated") or {}
    if ev.get("n_samples") == 0:
        print(f"error: {task}: no sample matches the selection (--subsets / --ids / clip_filter)", file=sys.stderr)
        return 1
    if ev.get("all_failed"):
        counts = ", ".join(f"{k} {v}" for k, v in ev["status"].items())
        print(f"error: {task}: every evaluated sample failed ({counts}), nothing is scored; see \"reason\" in the "
              f"per-sample JSONs under {out_dir}", file=sys.stderr)
        return 1
    return 0


def selection_exit_status(counts, task: str) -> int:
    """Exit status of an export / baseline command from its {name: count} result: 1 with a one-line message on
    stderr when the selection matched no sample (all counts 0), else 0."""
    nums = [v for v in (counts or {}).values() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)]
    if nums and sum(nums) == 0:
        print(f"error: {task}: no sample matches the selection (--subsets / --ids / clip_filter)", file=sys.stderr)
        return 1
    return 0


def _dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def task_cli(argv, task: str, protocol_cls, *, export_fn, evaluate_fn, summarize_fn, baseline_fn, baselines: dict,
             default_baseline: str, description: str = "", protocol_args: Sequence[tuple] = (),
             evaluate_args: Sequence[tuple] = (), evaluate_flags: Sequence[tuple] = ()) -> int:
    """`vpmwb <task> {export,baseline,evaluate,summarize}` (also `python -m vpm_worldbench.tasks.<task>`).

    Common options: --root (data root), --protocol (the default protocol name or a JSON / YAML file of protocol
    fields; anything else is a usage error), --ref-frame / --horizon / --frame-step (override the window; horizon >= 1:
    these tasks score a fixed window of horizon + 1 frames; a changed value adds "+field=value" to the protocol name),
    --subsets / --ids (comma- and/or space-separated; subset aliases accepted, also in uids), --jobs; export: --out,
    --frames, --size, --lossless; baseline: --kind, --pred (where the baseline predictions are written); evaluate:
    --pred, --out, --overwrite, --strict; summarize: --out. protocol_args = [(flag, field, type, help)] adds
    protocol-field overrides; evaluate_args = [(flag, kwarg, type, default, help)] adds keyword arguments of
    evaluate_fn; evaluate_flags = [(flag, field, value, help)] adds evaluate switches that set a protocol field.

    Exit status: 0 done; 1 the selection matches no sample, or no evaluable sample is scored (every one missing,
    invalid or error); 2 usage error (unknown subset, --protocol neither the preset nor a file, --horizon < 1, a
    malformed --size, a summarize --out that is no folder, ...: usage + one line) or setup problem: an out dir of
    another run without --overwrite (`RunDirError`) or a missing optional dependency (ImportError), which propagate:
    `vpmwb` (`cli.main`, also behind `python -m vpm_worldbench.tasks.<task>`) reports them as one line
    `vpmwb: error: ...` (VPMWB_TRACEBACK=1: the traceback)."""
    import argparse

    from ..utils import parse_list
    ap = argparse.ArgumentParser(prog=f"vpmwb {task}", description=description or None)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{export,baseline,evaluate,summarize}")

    def common(p):
        p.add_argument("--root", required=True, help="VP-WorldBench data root")
        p.add_argument("--protocol", default=None, help=f"'{protocol_cls().name}' (default) or a JSON / YAML file "
                       f"of {protocol_cls.__name__} fields (a file without 'name' is named after the file)")
        p.add_argument("--ref-frame", type=int, default=None, help="override the first window frame")
        p.add_argument("--horizon", type=int, default=None,
                       help="override the window length (>= 1; horizon + 1 frames)")
        p.add_argument("--frame-step", type=int, default=None, help="override the window frame step")
        for flag, _, typ, hlp in protocol_args:
            p.add_argument(flag, type=typ, default=None, help=hlp)
        p.add_argument("--subsets", nargs="*", default=None, help="subsets (names or aliases; default all)")
        p.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (default all)")
        p.add_argument("--jobs", type=int, default=1, help="worker processes")

    parsers = {}
    pe = parsers["export"] = sub.add_parser("export", help="write the model inputs")
    common(pe)
    pe.add_argument("--out", required=True, help="input folder to write")
    pe.add_argument("--frames", choices=FRAME_MODES, default="mp4")
    pe.add_argument("--size", default=None, help="WxH (default: native resolution)")
    pe.add_argument("--lossless", action="store_true", help="lossless mp4")
    pb = parsers["baseline"] = sub.add_parser("baseline", help="write the predictions of a reference baseline")
    common(pb)
    pb.add_argument("--kind", choices=sorted(baselines), default=default_baseline)
    pb.add_argument("--pred", required=True, help="prediction root to write")
    pv = parsers["evaluate"] = sub.add_parser("evaluate", help="score predictions")
    common(pv)
    pv.add_argument("--pred", required=True, help="prediction root")
    pv.add_argument("--out", required=True, help="result folder")
    pv.add_argument("--overwrite", action="store_true", help="re-evaluate / replace the results of another run")
    pv.add_argument("--strict", action="store_true", help="raise on the first per-sample error")
    for flag, _, typ, default, hlp in evaluate_args:
        pv.add_argument(flag, type=typ, default=default, help=hlp)
    for flag, _, _, hlp in evaluate_flags:
        pv.add_argument(flag, action="store_true", help=hlp)
    ps = sub.add_parser("summarize", help="rebuild summary.json / summary.md")
    ps.add_argument("--out", required=True, help="result folder")
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        if not Path(a.out).is_dir():
            ps.error(f"--out {a.out}: no such directory")
        summarize_fn(a.out)
        print((Path(a.out) / "summary.md").read_text())
        return 0
    q = parsers[a.cmd]
    if not Path(a.root).is_dir():
        q.error(f"--root {a.root}: no such directory")
    if a.horizon is not None and a.horizon < 1:
        q.error(f"--horizon {a.horizon}: must be >= 1 ({task} scores a fixed window of horizon + 1 frames)")
    size = cli_size(q, "--size", a.size) if a.cmd == "export" else None
    proto = cli_protocol(q, a.protocol, protocol_cls)
    over = {k: v for k, v in (("ref_frame", a.ref_frame), ("horizon", a.horizon), ("frame_step", a.frame_step))
            if v is not None}
    for flag, name, _, _ in protocol_args:
        v = getattr(a, _dest(flag))
        if v is not None:
            over[name] = v
    if a.cmd == "evaluate":
        for flag, name, value, _ in evaluate_flags:
            if getattr(a, _dest(flag)):
                over[name] = value
    if over:
        try:
            proto = override_protocol(proto, over, protocol_cls)
        except (KeyError, TypeError, ValueError) as e:
            q.error(str(e))
    subsets, ids = cli_subsets(q, a.subsets), parse_list(a.ids)
    from ..data.benchmark import Benchmark
    bench = Benchmark(a.root)
    if a.cmd == "export":
        res = export_fn(bench, a.out, proto, subsets=subsets, ids=ids, jobs=a.jobs, frames=a.frames, size=size,
                        lossless=a.lossless)
        print(res)
        return selection_exit_status(res, task)
    if a.cmd == "baseline":
        res = baseline_fn(bench, a.pred, kind=a.kind, protocol=proto, subsets=subsets, ids=ids, jobs=a.jobs)
        print(res)
        return selection_exit_status(res, task)
    kw = {name: getattr(a, _dest(flag)) for flag, name, _, _, _ in evaluate_args}
    res = evaluate_fn(bench, a.pred, a.out, proto, subsets=subsets, ids=ids, jobs=a.jobs, overwrite=a.overwrite,
                      strict=a.strict, **kw)
    print((Path(a.out) / "summary.md").read_text())
    return evaluation_exit_status(res, task, a.out)
