"""Task ``generation`` (the main VP-WorldBench task): generate every view of a multi-view clip; score the videos with
FVD, PSNR / SSIM / LPIPS, cross-view MEt3R and the 3D tracking accuracy δ of the lifted video.

Task (docs/PROTOCOL.md, `vpm_worldbench.protocol.Protocol`)
    Per sample and view, the model gets the context frames t_ref-c+1 .. t_ref (c = 1: image-to-video) and the prompt
    (`vpm_worldbench.inputs.export_inputs`) and generates the frames t_ref+1 .. t_ref+horizon (every `frame_step`-th GT
    frame). All views of a sample are scored together (cross-view consistency, multi-view 3D lifting).

Prediction layout (`vpm_worldbench.predictions`)
    <pred_root>/<Subset>/<id>/view{k}.mp4    one video per view; frame 0 = t_ref (the conditioning frame), then the
                                             window frames (>= horizon + 1 frames), any resolution
    <pred_root>/<Subset>/<id>/meta.json      optional {"fps": 24.0, "first_frame": "ref" | "future"}: frames taken by
                                             time (nearest frame when fps >= 12, else a linear blend); "future" = the
                                             video starts at t_ref+1 and the real conditioning frame becomes frame 0
    Frames are resized to the protocol eval grid of the subset (Simul 480x480, Real-Robot 848x480, Real-General 640x480;
    `resize_mode`). The real reference = the sample's video on the same window and grid (`predictions.load_reference`).

Metrics per sample (status "ok" once every requested metric is done)
    cond_psnr    I2V sanity: PSNR of prediction position 0 vs the real frame t_ref per view (flag below 25 dB).
    appearance   PSNR / SSIM / LPIPS (metrics.appearance) per view over positions 1..horizon (the conditioning frame
                 excluded), clip value = mean over views; per-frame values are stored.
    fvd          I3D (Kinetics-400, TATS weights) features of the non-overlapping `fvd_window`-frame windows from position
                 1 (`fvd_windows` of them, spread over the horizon) of every view -> <out_dir>/_fvd/<Subset>/<id>.npz
                 ("feats" (V*k, 400) float32 = the exact float32 logits). The FVD itself is pooled over a subset by
                 `summarize` (real windows from the reference directory). The cuDNN / matmul TF32 state is recorded.
                 When not one window fits after the conditioning frame (short horizons), FVD is recorded as skipped.
    met3r        official MEt3R (metrics.met3r) on every view pair at window positions [0] + protocol.met3r_positions()
                 (t_ref, t_ref+4, ...; position 0 is stored but not scored), frames made square with `met3r_aspect`;
                 cross[k][p] per position and pair, overlaps, and the clip mean m(c) over the scored positions.
    delta        the prediction's window frames of all views are lifted to 3D tracks by the protocol lifter (default
                 MVTracker + MoGe-2, lifters/mvtracker.py) on the TRACK grid (= the eval grid unless a per-subset
                 `track_size` is set; legacy Real-General 640x480) with K on that grid and the GT w2c of the window
                 frames; queries = GT points with default_gt (and visible in >= 1 used view at t_ref when
                 protocol.query_requires_visible) at t_ref, ascending point index; depth anchored per presets
                 depth_anchor: "gt_depth" = scale+shift of MoGe against the GT / reconstruction depth of t_ref on the
                 track grid (Sample.depth), "gt_points" = per-view median ratio at the GT points visible at t_ref; the
                 scene normalisation always comes from the REFERENCE depth (the GT depth, or the anchored depth of the
                 real video's t_ref for "gt_points", computed once and cached), so it is identical for every source.
                 Scored with metrics.tracking.track3d_metrics: per-subset ladder / survival (presets), static (< 1 cm
                 path) / dynamic (> 10 cm) splits, per point group ([main group] + other groups + "all"; main = protocol
                 delta_group or sample.json tracks.default_report_group). GT visibility per frame = default_gt AND
                 (visible | in_frustum per protocol.gt_visibility).any(used views); with a cropping eval grid
                 (resize_mode center_crop) the point must also project inside the view's grid (queries likewise).
                 Non-finite predicted positions are failures (+inf error). Also "allframes/ade|mte|fde" (every window
                 frame, the pre-release Simul ADE). As in MVTracker's evaluate_predictions, t_ref itself counts for the
                 ">= 2 visible frames" track filter and for survival (an error > survival_m at t_ref is a failure).
                 Read delta with the frozen row and the dynamic split: on mostly static windows a video without motion
                 scores near the frozen row (above the baselines on some Real-Robot clips), and uniform noise keeps the
                 queries near their start and scores like a weak model (measured on the released data, 2026-10-08).
    Reference rows (computed once per sample and protocol, cached in the reference directory and shared by all models):
    the real video's FVD features and cross-view MEt3R, the real video through the same lifter, frozen points (the
    query positions at every frame, always visible).

Outputs
    <out_dir>/protocol.json                      protocol, its hash (`protocol_hash`), resolved per-subset settings and
                                                 "pred_root" (the resolved prediction root); a run directory holds one
                                                 hash and one prediction root (evaluating another hash into it raises,
                                                 another root raises unless overwrite=True, which rescores the selected
                                                 samples; `summarize` excludes records left from the previous root)
    <out_dir>/<Subset>/<id>.json                 per-sample record: status ok | skipped (not evaluable) | missing
                                                 (prediction files) | invalid (unusable prediction) | error, "reason",
                                                 "metrics_done", one entry per metric, "pred_root" and "pred_stamp" (the
                                                 predictions it was scored from: resolved root, `prediction_stamp` of
                                                 the files), "models" (per metric: weight fingerprints / md5 and
                                                 options, `Models.identity`), "data_version" (index.json), sample
                                                 "attributes" / "flags", and "runtime" (stage seconds, peak GPU memory,
                                                 TF32 flags, host: the only part that differs between two evaluations
                                                 of the same predictions)
    <out_dir>/_selection/<hash>_<digest>.json    the samples an evaluate() call was asked for (before sharding), with
                                                 attributes: `summarize` reports expected samples without a record
    <out_dir>/_fvd/<Subset>/<id>.npz             I3D features of the prediction windows
    <out_dir>/_tracks/<Subset>/<id>.npz          (save_tracks=True) lifted tracks xyz (n, N, 3) / vis / query_index
    <reference_dir>/<hash>/...                   the same layout for the reference rows (+ _norm/ scene-normalisation
                                                 depth for "gt_points" anchors). Default reference_dir = _reference next
                                                 to the run directory (<parent of out_dir>/_reference; with several
                                                 models <out_dir>/_reference), so runs in one parent directory share it;
                                                 else pass one --reference-dir for all models of a comparison.
    The hash covers only what changes per-sample values, after resolution (see `protocol_hash`): configs/vpm-v1.yaml,
    Protocol() and the camera-control / V2V / clip-filter configs share one reference directory. The weights (I3D,
    MEt3R, MVTracker, MoGe-2) and the data version are not part of it: they are stored per record and `summarize`
    warns when the records of a table (models and reference rows) disagree.
    Resumable: existing records are skipped unless overwrite=True; a record missing some requested metric gets only those
    computed. A record is reused only while it was scored from the same predictions (same resolved root, unchanged
    view*.mp4 / meta.json: size and modification time); predictions regenerated in place, or another root, are scored
    afresh. Records written before roots / stamps were stored carry neither and are reused as they are. Records with
    status missing / invalid / error are retried. A failure inside one metric is recorded in that
    metric (status "error"); a failure outside them (unreadable sample data) is recorded for the sample; neither stops
    the run. Prediction videos are decoded only up to the frames the window needs. Deterministic: no sampling anywhere;
    models are loaded once per process (`Models`, `get_models`). JSON: NaN -> null, infinities -> "inf" / "-inf".

Numerics (measured on GH200 by an independent check, 2026-10-08)
    MEt3R and the lifter amplify sub-level input changes far beyond the change in PSNR. Decoding the same mp4 with
    OpenCV instead of PyAV (<= 1 level apart) or re-encoding at CRF 4 (47-53 dB) moved per-clip MEt3R means by up to
    ~0.04 (single pairs by up to ~0.45, low-overlap pairs). The published vpm-v1 tables have floors of
    0.0038-0.0296 and noise floors of 0.0008-0.0111, so such a shift is only ~1.4x the floor on Simul-General and
    ~4-10x on Simul-Robot and Real-*. A CRF-18 re-encode (37-44 dB, PSNR vs GT changed by <= 0.025 dB) moved per-clip
    δ_avg by up to ~7 points. Deliver lossless (or near-lossless, CRF <= ~4) prediction videos, and compare models on
    subset means, not single clips.

Legacy protocol (`legacy-prerelease`, `vpm_worldbench.protocol.legacy_prerelease`; not for new results)
    Applies the per-subset switches of presets.LEGACY_PRERELEASE (FVD windows / covariance, MEt3R aspect, query and
    visibility rules, RGB/255 MVTracker input, plain K scaling, Real-General _reproduce 480x480 frames) and the recipes
    of `vpm_worldbench.legacy` (vendor K with cx = 640 on Simul-*, the source world frame for the lifter on Simul-*,
    raw/100 depth and the _reproduce reconstruction depth resized with cv2.INTER_NEAREST, OpenCV decoding, MoGe FoV from
    metadata, 640x480 Real-General track grid, MEt3R batching 64 / 32 with cross + temporal pairs). Remaining
    differences from the internal pre-release numbers (details in vpm_worldbench/legacy.py):
      - Simul-* / Real-Robot appearance and delta of the generated videos: the pre-release scored the in-memory frames
        before they were encoded to the mp4s that exist now (~0.03 dB PSNR); MVTracker amplifies such differences.
      - Real-Robot GT: the released tracks are aligned to the videos (-2.3 frames) and their visibility differs
        (12 px z-buffer at 1280x720 vs 8 px at 848x480), so queries and scored entries differ.
      - Simul-* GT visibility was recomputed (corrected K, renderer near plane); the pre-release frozen row was
        "camera_only" (points frozen in camera 0), which differs where camera 0 moves.
      - MEt3R: binned vs naive PyTorch3D rasterizer (<= 5e-5 per pair), 5-decimal rounding; FVD: float32 feature means
        in the pre-release Simul-* tables (<= 1.2e-4); OpenCV decoding depends on the OpenCV / FFmpeg build.
    Observed differences on 28 samples x 3 baselines: see the end of the `vpm_worldbench.legacy` docstring.

Usage
    from vpm_worldbench.tasks import generation as G
    G.evaluate("/data/VP-WorldBench", "preds/my_model", "results/my_model", "vpm-v1",
               reference_dir="results/_reference")
    G.evaluate("/data/VP-WorldBench", {"a": "preds/a", "b": "preds/b"}, "results", "vpm-v1")   # one pass, 2 models
    G.summarize({"my_model": "results/my_model", "other": "results/other"}, reference_dir="results/_reference",
                out="results/summary")

    vpmwb generation export   --root R --out inputs/ [--protocol P] [--subsets ...] [--ids ...] [--size WxH]
    vpmwb generation evaluate --root R --pred preds/my_model --out results/my_model \\
        [--reference-dir results/_reference] [--protocol vpm-v1|legacy-prerelease|p.yaml] [--subsets sr rg | sr,rg]
        [--ids ...] [--metrics appearance fvd met3r delta] [--shard k/n] [--device auto|cuda|cpu] [--overwrite]
    vpmwb generation evaluate --root R --pred a=preds/a b=preds/b --out results [...]    # several models, one pass
    vpmwb generation reference --root R --reference-dir results/_reference [...]       # reference rows only
    vpmwb generation summarize --run my_model=results/my_model --run other=results/other \\
        [--reference-dir results/_reference] --out results/summary [--split difficulty=hard] [--protocol P] \
        [--require-complete]       # exit status 3 when expected records are missing or the summary has warnings
    (`eval` is an alias of `evaluate`; `python -m vpm_worldbench.tasks.generation ...` takes the same arguments.)
    Weights: --mvtracker-repo / --moge / --featup-repo / --mast3r-weights / --i3d-weights point to local copies.
    Exit status of evaluate / reference: 0; 1 when the selection matches no sample, or every sample this call evaluated
    failed for some model (status error, missing or invalid; cached and not evaluable samples do not count) or, for
    `reference`, every reference row failed; 2 for a setup problem (missing package or weights, a run directory of
    another protocol / prediction root), reported on one line `vpmwb: error: ...`, and for bad arguments (argparse
    usage message). summarize: 2 for bad arguments and unusable run / reference directories, 3 as above.
    VPMWB_TRACEBACK=1 (anything but empty or 0) re-raises setup problems, which shows their traceback.

Cost (one GH200 per process, vpm-v1, measured on 2 samples per subset x 3 baselines, 2026-10-08)
    model loading ~40-80 s per process (once; + a few s to fingerprint the weights). Per sample: reference rows 8-22 s (decode, FVD, MEt3R, MVTracker + MoGe-2)
    and every model 10-21 s (MEt3R 2-9 s, delta 5-9 s, appearance 1.5-2.4 s, FVD < 0.3 s, decode < 0.5 s), i.e. with 3
    models + reference: Simul-Robot ~58 s, Simul-General ~81 s, Real-Robot ~37 s, Real-General ~81 s, ~68 s on average
    over the 1200 samples (~23 GPU-hours). Peak GPU memory ~32 GiB, set by MEt3R (`met3r_batch` 16,
    `met3r_feature_chunk` 8). legacy-prerelease: ~96 s per sample on average, peak ~94 GiB (pre-release MEt3R batching
    64 / 32 with temporal pairs).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from .. import legacy as L
from ..constants import SUBSETS, canonical_subset
from ..geometry import image_map, inside, project, warp_image
from ..lifters.base import ANCHOR_MODES
from ..predictions import _resample, has_prediction, load_reference, prediction_dir, prediction_meta
from ..presets import DYNAMIC_MIN_M, STATIC_MAX_M, SUBSET_DEFAULTS
from ..protocol import Protocol, get_protocol, legacy_option
from ..report import SELECTION_DIR, jsonable, read_json, summarize_generation, write_json
from ..utils import UsageError

TASK = "generation"
REFERENCE_TASK = "generation-reference"
METRICS = ("appearance", "fvd", "met3r", "delta")
COND_PSNR_MIN_DB = 25.0
REFERENCE_VERSION = 2
LOCK_STALE_S = 3600
DEFAULT_MET3R_BATCH, DEFAULT_MET3R_CHUNK = 16, 8


def resolve_protocol(protocol) -> Protocol:
    """A Protocol, a preset name ('vpm-v1', 'legacy-prerelease'), a json / yaml file or None (vpm-v1)."""
    if isinstance(protocol, Protocol):
        return protocol
    return get_protocol(protocol)


def _per_subset(value, subset):
    if isinstance(value, dict):
        return value.get(subset)
    return value


class SettingsError(UsageError, ValueError):
    """A protocol setting that resolves to no valid value (an unknown depth_anchor mode or decoder): `vpmwb` reports it
    on one line with exit status 2."""


def subset_settings(protocol: Protocol, subset: str, track_size=None, decoder: str | None = None) -> dict:
    """Every setting the metrics use for `subset`, resolved from the protocol, presets.SUBSET_DEFAULTS, the legacy
    switches (presets.LEGACY_PRERELEASE via protocol.legacy_option) and the legacy recipes (vpm_worldbench.legacy).

    track_size: (W, H) or {subset: (W, H)} of the depth-estimation / tracking grid (None: legacy recipe, else the
    eval grid). decoder: 'pyav' | 'opencv' | None (legacy: 'opencv', else 'pyav'). JSON-friendly dict. An unknown
    decoder or depth_anchor mode raises SettingsError (a ValueError)."""
    p = protocol
    sub = canonical_subset(subset)
    rec = L.generation_recipe(p, sub)
    eval_size = tuple(int(v) for v in p.eval_size_for(sub))
    ts = _per_subset(track_size, sub)
    if ts is None:
        ts = rec["track_size"]
    ts = eval_size if ts is None else tuple(int(v) for v in ts)
    dec = decoder or ("opencv" if L.is_legacy(p) else "pyav")
    if dec not in L.DECODERS:
        raise SettingsError(f"decoder must be one of {L.DECODERS}, got {dec!r}")
    anchor = p.get("depth_anchor", sub)
    if anchor not in ANCHOR_MODES:
        raise SettingsError(f"{sub}: depth_anchor must be one of {', '.join(ANCHOR_MODES)}, got {anchor!r}")
    window = [int(t) for t in p.window_indices]
    return {
        "subset": sub, "legacy": L.is_legacy(p), "window": [window[0], window[-1], int(p.frame_step)],
        "eval_size": list(eval_size), "resize_mode": p.resize_mode_for(sub), "track_size": list(ts),
        "reference_frames": p.reference_frames, "decoder": dec,
        "fvd_window": int(p.fvd_window), "fvd_windows": legacy_option(p, sub, "fvd_windows", p.fvd_windows),
        "fvd_cov": legacy_option(p, sub, "fvd_cov", p.fvd_cov),
        "met3r_aspect": legacy_option(p, sub, "met3r_aspect", p.met3r_aspect), "met3r_size": int(p.met3r_size),
        "met3r_positions": [0] + [int(j) for j in p.met3r_positions()],
        "met3r_temporal": bool(L.is_legacy(p) and L.MET3R_LEGACY["temporal"]),
        "query_requires_visible": bool(legacy_option(p, sub, "query_requires_visible", p.query_requires_visible)),
        "gt_visibility": str(legacy_option(p, sub, "gt_visibility", p.gt_visibility)),
        "depth_anchor": str(anchor), "far_clip_m": float(SUBSET_DEFAULTS[sub]["far_clip_m"]),
        "ladder_m": [float(x) for x in p.get("delta_ladder_m", sub)],
        "survival_m": float(p.get("delta_survival_m", sub)), "main_group": p.get("delta_group", sub),
        "static_m": STATIC_MAX_M, "dynamic_m": DYNAMIC_MIN_M, "lifter": p.lifter,
        "rgb_scale_quirk": bool(legacy_option(p, sub, "rgb_scale_quirk", p.rgb_scale_quirk)),
        "plain_K_scaling": bool(legacy_option(p, sub, "plain_K_scaling", p.plain_K_scaling)),
        "K_source": rec["K"], "fov": rec["fov"], "world": rec["world"], "depth_recipe": rec["depth"],
        "depth_min_m": float(rec["depth_min_m"]),
        "visibility_in_grid": p.resize_mode_for(sub) == "center_crop",
    }


_HASH_EXCLUDED = ("legacy", "fvd_cov", "fvd_windows")
_HASH_OMIT_IF = {"visibility_in_grid": False}


def fvd_starts(protocol, settings: dict):
    """Window positions where the FVD windows start (metrics.fvd.window_starts on the protocol window), or None when not
    one `fvd_window`-frame window fits after the conditioning frame (FVD is then skipped)."""
    from ..metrics.fvd import window_starts
    p = resolve_protocol(protocol)
    try:
        return [int(x) for x in window_starts(len(p.window_indices), settings["fvd_window"], settings["fvd_windows"], 1)]
    except ValueError:
        return None


def hash_settings(protocol, subset: str, track_size=None, decoder: str | None = None) -> dict:
    """The resolved settings of `subset` that determine the per-sample values: `subset_settings` without
    _HASH_EXCLUDED, plus the resolved FVD window starts."""
    p = resolve_protocol(protocol)
    s = subset_settings(p, subset, track_size, decoder)
    out = {k: v for k, v in s.items()
           if k not in _HASH_EXCLUDED and not (k in _HASH_OMIT_IF and v == _HASH_OMIT_IF[k])}
    out["fvd_starts"] = fvd_starts(p, s)
    return out


def protocol_hash(protocol, track_size=None, decoder: str | None = None) -> str:
    """12-hex key of the per-sample results: a digest of the window (ref_frame, horizon, frame_step), the views and the
    RESOLVED per-subset settings of all four subsets (`hash_settings`). It names the reference directory
    <reference_dir>/<hash>/ and is stored in every record; results and reference rows of different keys are never
    mixed, and a run directory holds one key.

    Only settings that change a per-sample value enter it, after resolution, so equivalent spellings share one key
    (configs/vpm-v1.yaml, `Protocol()` and 'vpm-v1'; eval sizes as lists or tuples; fvd_windows None or 4 with a 64-frame
    horizon; track_size / decoder None or their resolved defaults). Not part of it: the protocol name (the legacy preset
    is recognised by its name, but each of its switches is a resolved value), the model inputs (text, camera_conditioning,
    context_frames), clip_filter (selects samples only) and fvd_cov (applied by `summarize`). Also not part of it: the
    identity of the metric weights (I3D, MEt3R, MVTracker, MoGe-2) and the data version; every record stores them
    ("models", "data_version") and `summarize` warns when the records of one table disagree."""
    p = resolve_protocol(protocol)
    payload = {"version": REFERENCE_VERSION, "ref_frame": int(p.ref_frame), "horizon": int(p.horizon),
               "frame_step": int(p.frame_step), "views": None if p.views is None else [int(v) for v in p.views],
               "subsets": {s: hash_settings(p, s, track_size, decoder) for s in SUBSETS}}
    blob = json.dumps(jsonable(payload), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def report_groups(sample, main: str) -> list:
    """[main] + the sample's other groups + "all" (when it has several groups), without duplicates."""
    names = [str(g) for g in sample.tracks.group_names]
    out = [main]
    if len(names) > 1:
        out += [g for g in names if g != main]
        if main != "all":
            out.append("all")
    return out


def _plain_K(K: np.ndarray, m) -> np.ndarray:
    """K rows scaled by (sx, sy) plus the crop / pad offset of an ImageMap (the pre-release 'plain' scaling)."""
    out = np.array(K, np.float64, copy=True)
    out[0, :] *= m.sx
    out[1, :] *= m.sy
    out[0, 2] += m.ox
    out[1, 2] += m.oy
    return out


def track_intrinsics(sample, view: int, settings: dict) -> np.ndarray:
    """(3, 3) K of `view` on the track grid. 'resize' eval grids: one map from the stored resolution to the track grid
    (pixel-centre ImageMap.apply_K, or plain row scaling); other modes: stored -> eval grid (resize_mode) -> track grid.
    The legacy recipes start from K_delivered (Simul-*) or float32(K_undistort_raw) on the raw grid (Real-General)."""
    src = sample.image_size(view)
    extra = sample.cameras.extra
    if settings["K_source"] == "delivered" and "K_delivered" in extra:
        K = extra["K_delivered"][view]
    elif settings["K_source"] == "undistort_raw_f32" and "K_undistort_raw" in extra:
        K = extra["K_undistort_raw"][view].astype(np.float32).astype(np.float64)
        src = tuple(int(x) for x in extra["raw_size"])
    else:
        K = sample.cameras.K[view]
    ev, ts, mode = tuple(settings["eval_size"]), tuple(settings["track_size"]), settings["resize_mode"]
    maps = [image_map(src, ts, "resize")] if mode == "resize" else \
        [image_map(src, ev, mode)] + ([image_map(ev, ts, "resize")] if ts != ev else [])
    for m in maps:
        K = _plain_K(K, m) if settings["plain_K_scaling"] else m.apply_K(K)
    return np.asarray(K, np.float64)


class SampleCase:
    """Everything the metrics need for one sample under one protocol, deterministic and without pixels:
    views, window, queries and GT (`ok`, `reason` when the sample is not evaluable). Depth / anchors are built lazily.

    GT visibility of point i at window position j = default_gt AND finite AND (visible | in_frustum, per
    gt_visibility) in at least one used view; when the eval grid crops the native image (resize_mode 'center_crop',
    settings["visibility_in_grid"]) the point must also project inside that view's grid (GT xyz, K on the track grid,
    GT w2c of the frame; geometry.inside), as in Sample.tracks2d. Queries = default GT points at t_ref (visible in >= 1
    view at t_ref under that rule when query_requires_visible)."""

    def __init__(self, sample, protocol: Protocol, settings: dict):
        self.sample, self.protocol, self.settings = sample, protocol, settings
        self.ok, self.reason = protocol.evaluable(sample)
        if not self.ok:
            return
        p, s = protocol, settings
        self.views = protocol.views_for(sample)
        self.window = np.asarray(p.window_indices)
        self.t_ref = int(p.ref_frame)
        self.K = np.stack([track_intrinsics(sample, v, s) for v in self.views])
        self.w2c = np.stack([sample.w2c(v, self.window) for v in self.views])
        tr = sample.tracks
        flag = tr.visible if s["gt_visibility"] == "visible" else tr.in_frustum
        xyz = tr.xyz[self.window]
        dgt = tr.default_gt[self.window] & np.isfinite(xyz).all(-1)
        seen = flag[self.window][:, :, self.views] & dgt[:, :, None]
        self.in_grid = None
        if s.get("visibility_in_grid"):
            self.in_grid = self._inside_grid(xyz)
            seen &= self.in_grid
        q = dgt[0].copy()
        if s["query_requires_visible"]:
            q &= seen[0].any(-1)
        self.query_index = np.nonzero(q)[0]
        X = xyz[:, self.query_index].astype(np.float64)
        known = dgt[:, self.query_index]
        X[~known] = np.nan
        self.gt = X
        self.gt_vis = seen[:, self.query_index].any(-1) & known
        self.queries = np.concatenate([np.zeros((len(self.query_index), 1)), X[0]], 1)
        main = s["main_group"] or sample.meta.get("tracks", {}).get("default_report_group") or "all"
        names = [str(g) for g in tr.group_names]
        if main != "all" and main not in names:
            raise KeyError(f"{sample.uid}: main point group {main!r} not in {names}")
        self.main_group = main
        self.groups = {g: tr.group_mask(g)[self.query_index] for g in report_groups(sample, main)}
        self.fov_x = None
        if s["fov"] == "hfov_deg":
            self.fov_x = [float(sample.meta["views"][v]["hfov_deg"]) for v in self.views]
        elif s["fov"] == "native_K":
            self.fov_x = [L.native_fov_deg(sample, v) for v in self.views]
        self.world_transform = None
        if s["world"] == "source":
            from ..lifters.base import source_world_transform
            self.world_transform = source_world_transform(sample.meta["world"]["to_source"])
        self._anchor = None
        self.depth_source = None

    def _inside_grid(self, xyz: np.ndarray) -> np.ndarray:
        """(n, N, V) bool: GT point in front of the camera and inside the track grid (= the eval grid, resized) of
        each used view at each window frame. xyz (n, N, 3) world metres (non-finite -> False)."""
        ts = tuple(int(v) for v in self.settings["track_size"])
        out = np.zeros(xyz.shape[:2] + (len(self.views),), bool)
        for iv in range(len(self.views)):
            for j in range(len(self.window)):
                uv, z = project(xyz[j], self.K[iv], self.w2c[iv, j])
                with np.errstate(invalid="ignore"):
                    out[j, :, iv] = (z > 0) & inside(uv, ts)
        return out

    @property
    def n_queries(self) -> int:
        return int(len(self.query_index)) if self.ok else 0

    def K_window(self) -> np.ndarray:
        return np.repeat(self.K[:, None], len(self.window), 1)

    def anchor_depth(self) -> np.ndarray:
        """(V, h, w) float32 metres (0 = invalid) of t_ref on the track grid (anchor modes 'gt_depth')."""
        s, sample = self.settings, self.sample
        ts, ev = tuple(s["track_size"]), tuple(s["eval_size"])
        far = s["far_clip_m"]
        recipe = s["depth_recipe"]
        if recipe == "_reproduce":
            try:
                d = L.reproduce_depth(sample, self.views, self.t_ref, size=ts)
                self.depth_source = "_reproduce depth_cond.npz (480x480, cv2 nearest)"
            except (FileNotFoundError, ValueError):
                recipe = None
        elif recipe == "raw_div100":
            d = np.stack([L.simul_depth_raw_div100(sample, v, self.t_ref, ev, far, s["depth_min_m"])
                          for v in self.views])
            self.depth_source = "depth.zip raw/100 (legacy)"
        if recipe is None:
            if s["resize_mode"] == "resize":
                d = np.stack([sample.depth(v, [self.t_ref], size=ts, mode="resize")[0] for v in self.views])
            else:
                d = np.stack([sample.depth(v, [self.t_ref], size=ev, mode=s["resize_mode"])[0] for v in self.views])
            self.depth_source = "Sample.depth"
        d = np.asarray(d, np.float32)
        if d.shape[1:] != (ts[1], ts[0]):
            m = image_map((d.shape[2], d.shape[1]), ts, "resize")
            d = np.stack([warp_image(x, m, interp="nearest") for x in d])
        if s["depth_min_m"] > 0:
            d = np.where(d <= s["depth_min_m"], 0.0, d).astype(np.float32)
        return d

    def base_anchor(self):
        """The Anchor without the scene-normalisation depth for estimation-based modes (see `Lifter.prepare_anchor`).
        settings["depth_anchor"]: one of lifters.base.ANCHOR_MODES; any other value raises ValueError."""
        from ..lifters.base import Anchor
        if self._anchor is not None:
            return self._anchor
        s = self.settings
        mode = s["depth_anchor"]
        if mode not in ANCHOR_MODES:
            raise ValueError(f"{self.sample.uid}: depth_anchor must be one of {', '.join(ANCHOR_MODES)}, got {mode!r}")
        if mode == "gt_depth" and not self.sample.has_depth:
            raise ValueError(f"{self.sample.uid}: depth_anchor 'gt_depth' but the sample has no depth")
        if mode == "gt_depth":
            d = self.anchor_depth()
            a = Anchor("gt_depth", depth=d, far_clip_m=s["far_clip_m"], norm_depth=d)
        elif mode == "gt_points":
            tr = self.sample.tracks
            dgt = tr.default_gt[self.t_ref] & np.isfinite(tr.xyz[self.t_ref]).all(-1)
            vis = tr.visible[self.t_ref][:, self.views] & dgt[:, None]
            if self.in_grid is not None:
                vis &= self.in_grid[0]
            vis = vis.T
            a = Anchor("gt_points", points=tr.xyz[self.t_ref].astype(np.float64), points_vis=vis,
                       far_clip_m=s["far_clip_m"])
        else:
            a = Anchor("none", far_clip_m=s["far_clip_m"])
        self._anchor = a
        return a

    def to_track_grid(self, frames: np.ndarray) -> np.ndarray:
        """(V, n, H, W, 3) eval-grid frames -> the track grid (INTER_AREA when shrinking an axis, else INTER_CUBIC)."""
        ts = tuple(self.settings["track_size"])
        if (frames.shape[3], frames.shape[2]) == ts:
            return frames
        from ..lifters.base import resample_inputs
        V, n = frames.shape[:2]
        out, _, _ = resample_inputs(frames, np.broadcast_to(np.eye(3), (V, n, 3, 3)), ts,
                                    self.settings["plain_K_scaling"])
        return out


def load_reference_frames(sample, view: int, protocol: Protocol, settings: dict):
    """The real window frames of `view` on the eval grid -> (frames (n, H, W, 3) uint8, source description)."""
    if protocol.reference_frames == "_reproduce":
        p = L.reproduce_dir(sample) / "real_480sq" / f"view{view}.mp4"
        if p.exists():
            return load_reference(sample, view, protocol), "_reproduce/real_480sq"
    W, H = settings["eval_size"]
    if settings["decoder"] == "opencv":
        fr = L.read_frames_opencv(sample.video_path(view), protocol.window_indices)
        if (fr.shape[2], fr.shape[1]) != (W, H):
            m = sample.image_map(view, (W, H), settings["resize_mode"])
            fr = np.stack([warp_image(f, m) for f in fr])
        return fr, "video (opencv)"
    fr = sample.frames(view, protocol.window_indices, size=(W, H), mode=settings["resize_mode"])
    return fr, "video"


def _fraction(x) -> Fraction:
    """A frame rate as an exact fraction (29.97 -> 2997/100), so frame times are free of float rounding."""
    return Fraction(float(x)).limit_denominator(10 ** 6)


def _frames_needed(protocol: Protocol, fps_gt: float, meta: dict) -> int:
    """How many frames from the start of a prediction video the window needs (decoding stops there). The time of the
    last window frame is computed exactly (rational frame rates): a float product such as 41 / 20 * 30 =
    61.49999999999999 must not lose the frame a tie rounds up to."""
    future = meta.get("first_frame", "ref") == "future"
    n = len(protocol.window_indices)
    if meta.get("fps") is None:
        return n - 1 if future else n
    t_last = (protocol.window_indices[-1] - protocol.ref_frame) * _fraction(meta["fps"]) / _fraction(fps_gt)
    return math.floor(t_last) + 2 - (1 if future else 0)


def _decode_head(path: Path, decoder: str, limit: int) -> np.ndarray:
    """The first `limit` frames (fewer if the video is shorter) as RGB uint8 (n, H, W, 3)."""
    if decoder == "opencv":
        return L.read_frames_opencv(path, limit=limit)
    from itertools import islice

    from ..io.video import iter_frames
    frames = list(islice(iter_frames(path), int(limit)))
    if not frames:
        return np.zeros((0, 0, 0, 3), np.uint8)
    return np.stack(frames)


def load_prediction_frames(pred_root, sample, view: int, protocol: Protocol, settings: dict,
                           ref_frame: np.ndarray):
    """The prediction of `view` on the window and eval grid -> (frames (n, H, W, 3) uint8, info). Like
    predictions.load_prediction, plus the decoder choice, decoding only the frames the window needs, and with
    first_frame="future" the video is mapped to the eval grid BEFORE the real conditioning frame (eval grid) is
    prepended. Raises ValueError for unusable predictions (bad meta.json, undecodable or too short video)."""
    path = prediction_dir(pred_root, sample) / f"view{view}.mp4"
    meta = prediction_meta(pred_root, sample)
    need = _frames_needed(protocol, sample.fps, meta)
    try:
        fr = _decode_head(path, settings["decoder"], need)
    except Exception as e:
        raise ValueError(f"cannot decode {path.name}: {type(e).__name__}: {e}") from e
    if fr.ndim != 4 or len(fr) == 0 or fr.shape[-1] != 3:
        raise ValueError(f"{path.name}: no RGB frame decoded")
    info = {"size": [int(fr.shape[2]), int(fr.shape[1])], "n_decoded": int(len(fr))}
    if len(fr) < need and meta.get("fps") is None:
        raise ValueError(f"{path.name} has {len(fr)} frames, the window needs {need} "
                         f"({'the future frames' if meta.get('first_frame') == 'future' else 't_ref + horizon'})")
    W, H = settings["eval_size"]
    mode = settings["resize_mode"]

    def to_grid(x):
        if (x.shape[2], x.shape[1]) == (W, H):
            return x
        m = image_map((x.shape[2], x.shape[1]), (W, H), mode)
        return np.stack([warp_image(f, m) for f in x])

    if meta.get("first_frame", "ref") == "future":
        fr = to_grid(fr)
    fr = _resample(fr, protocol, sample.fps, meta, ref_frame)
    return to_grid(fr), info


def prediction_stamp(pred_root, sample) -> str | None:
    """Fingerprint of the prediction files of `sample`: a digest of the name, size and modification time (ns) of every
    view*.mp4 and of meta.json in <pred_root>/<Subset>/<id>/ (None when the folder does not exist). Every model record
    stores it ("pred_stamp"); a record is reused only while it is unchanged, so predictions regenerated in place are
    scored again."""
    d = prediction_dir(pred_root, sample)
    if not d.is_dir():
        return None
    h = hashlib.sha1()
    for p in sorted(d.glob("view*.mp4")) + [d / "meta.json"]:
        try:
            st = p.stat()
        except OSError:
            continue
        h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def _root_str(pred_root) -> str:
    """A prediction root as stored in run directories and records ("pred_root"): the resolved absolute path."""
    return str(Path(pred_root).resolve())


def _prediction_ident(pred_root, sample) -> dict:
    """{"pred_root", "pred_stamp"}: the predictions a model record is scored from (see `_load_old`)."""
    return {"pred_root": _root_str(pred_root), "pred_stamp": prediction_stamp(pred_root, sample)}


def resolve_device(device) -> str:
    """'auto' (or None) -> 'cuda' when PyTorch sees a GPU, else 'cpu'; any other value is returned unchanged."""
    if device is None or str(device) == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


MET3R_MODULES = ("torch", "pytorch3d", "featup", "met3r")
LIFTER_MODULES = {"mvtracker": ("torch", "moge")}


def missing_dependencies(metrics, lpips: bool = True, ssim: bool = True, models=None) -> list:
    """One message per metric whose optional packages are not installed (with the pip extra and the import names that
    are missing): SSIM -> scikit-image, LPIPS -> torch + lpips, FVD -> torch + cd-fvd, MEt3R -> torch, pytorch3d,
    featup, met3r (MET3R_MODULES), delta -> torch (+ moge for the MVTracker lifter, LIFTER_MODULES). Probed with
    importlib.util.find_spec, before anything is written; the weights, MASt3R and the MVTracker torch.hub repo are
    checked when the models load. models: the Models instance the run uses: only the models it does not hold yet are
    checked, and its lifter_name picks the delta packages (None: all, MVTracker lifter; another object: none, it
    brings its own). [] when nothing is missing."""
    def unloaded(attr):
        return models is None or (isinstance(models, Models) and getattr(models, attr, None) is None)

    def absent(names) -> str:
        return ", ".join(n for n in names if not _has_module(n))

    out = []
    if "appearance" in metrics and ssim and not _has_module("skimage"):
        out.append("SSIM needs scikit-image: pip install 'vpm-worldbench[metrics]' (or --no-ssim)")
    if "appearance" in metrics and lpips and unloaded("_lpips") and not (_has_module("torch") and _has_module("lpips")):
        out.append("LPIPS needs torch and lpips: pip install 'vpm-worldbench[metrics]' (or --no-lpips)")
    if "fvd" in metrics and unloaded("_i3d") and not (_has_module("torch") and _has_module("cdfvd")):
        out.append("FVD needs torch and cd-fvd: pip install 'vpm-worldbench[fvd]'")
    if "met3r" in metrics and unloaded("_met3r"):
        miss = absent(MET3R_MODULES)
        if miss:
            out.append(f"MEt3R needs torch and the official MEt3R stack (not installed: {miss}): pip install "
                       f"'vpm-worldbench[met3r]', then the installation recipe in the docstring of "
                       f"vpm_worldbench/metrics/met3r.py")
    if "delta" in metrics and unloaded("_lifter"):
        lifter = models.lifter_name if isinstance(models, Models) else "mvtracker"
        miss = absent(LIFTER_MODULES.get(lifter, ("torch",)))
        if miss and lifter == "mvtracker":
            out.append(f"delta needs torch, MVTracker and MoGe-2 (not installed: {miss}): pip install "
                       f"'vpm-worldbench[track]', then the installation recipe in the docstring of "
                       f"vpm_worldbench/lifters/mvtracker.py")
        elif miss:
            out.append(f"delta (lifter {lifter!r}) needs torch: pip install 'vpm-worldbench[track]'")
    return out


class Models:
    """The heavy models of the metrics, each loaded on first use and kept (one instance per process is enough).

    device                torch device for all models ('auto': cuda when available, else cpu)
    lpips_net             LPIPS backbone ("alex" = benchmark)
    i3d_weights           local i3d_pretrained_400.pt (None: cd-fvd's packaged / downloaded copy)
    met3r_kwargs          extra MEt3R(...) arguments (featup_repo, backbone_weights, tf32, raster)
    lifter / lifter_kwargs  registered lifter name (protocol.lifter) and its constructor arguments
                          (MVTrackerLifter: hub_repo, moge, ...)
    Any attribute (lpips, i3d, met3r, lifter) may be set to a ready object (tests, custom models)."""

    def __init__(self, device: str = "auto", lpips_net: str = "alex", i3d_weights=None, met3r_kwargs=None,
                 lifter: str = "mvtracker", lifter_kwargs=None):
        self.device = resolve_device(device)
        self.lpips_net = lpips_net
        self.i3d_weights = i3d_weights
        self.met3r_kwargs = dict(met3r_kwargs or {})
        self.lifter_name = lifter
        self.lifter_kwargs = dict(lifter_kwargs or {})
        self._lpips = self._i3d = self._met3r = self._lifter = None
        self.load_seconds = {}
        self._identity = {}

    def _timed(self, name, fn):
        t0 = time.time()
        obj = fn()
        self.load_seconds[name] = round(time.time() - t0, 1)
        return obj

    @property
    def lpips(self):
        if self._lpips is None:
            from ..metrics.appearance import LPIPS
            self._lpips = self._timed("lpips", lambda: LPIPS(device=self.device, net=self.lpips_net))
        return self._lpips

    @lpips.setter
    def lpips(self, v):
        self._lpips = v

    @property
    def i3d(self):
        if self._i3d is None:
            from ..metrics.fvd import I3D
            self._i3d = self._timed("i3d", lambda: I3D(device=self.device, weights=self.i3d_weights))
        return self._i3d

    @i3d.setter
    def i3d(self, v):
        self._i3d = v

    @property
    def met3r(self):
        if self._met3r is None:
            from ..metrics.met3r import MEt3R
            kw = {"feature_chunk": DEFAULT_MET3R_CHUNK, **self.met3r_kwargs}
            self._met3r = self._timed("met3r", lambda: MEt3R(device=self.device, **kw))
        return self._met3r

    @met3r.setter
    def met3r(self, v):
        self._met3r = v

    @property
    def lifter(self):
        if self._lifter is None:
            from ..lifters.base import get_lifter

            def build():
                lf = get_lifter(self.lifter_name, device=self.device, **self.lifter_kwargs)
                if hasattr(lf, "load"):
                    lf.load()
                return lf
            self._lifter = self._timed("lifter", build)
        return self._lifter

    @lifter.setter
    def lifter(self, v):
        self._lifter = v

    def load(self, metrics=METRICS, lpips: bool = True):
        """Load what `metrics` need now, in a safe order (the official MEt3R stack last: its import puts the
        MASt3R / DUSt3R checkouts on sys.path)."""
        if "delta" in metrics:
            self.lifter
        if "fvd" in metrics:
            self.i3d
        if "appearance" in metrics and lpips:
            self.lpips
        if "met3r" in metrics:
            self.met3r
        return self

    def identity(self, metric: str) -> dict:
        """What the values of `metric` depend on besides the frames and the protocol: weight fingerprints
        (`weights_fingerprint`, or the I3D file md5) and numeric options. Paths are kept under "paths" and are not
        compared by `summarize`. Computed once per process (hashing the weights takes a few seconds)."""
        if metric in self._identity:
            return self._identity[metric]
        if metric == "appearance":
            lp = self._lpips
            out = {"lpips_net": self.lpips_net,
                   "lpips_weights": weights_fingerprint(getattr(lp, "model", None)) if lp is not None else None}
        elif metric == "fvd":
            info = dict(getattr(self.i3d, "info", {}) or {})
            out = {"model": info.get("model", type(self.i3d).__name__), "weights_md5": info.get("weights_md5"),
                   "batch": info.get("batch"), "paths": {"weights": info.get("weights")}}
        elif metric == "met3r":
            m = self.met3r
            out = {k: getattr(m, k, None) for k in ("img_size", "raster", "tf32", "backbone")}
            out["weights"] = weights_fingerprint(getattr(m, "model", None))
            fs = getattr(m, "featup_source", None)
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                fs = {"repo": str(fs[0]), "source": str(fs[1])}
            out["paths"] = {"featup": None if fs is None else fs if isinstance(fs, dict) else str(fs),
                            **{k: v for k, v in self.met3r_kwargs.items() if k in ("featup_repo", "backbone_weights")}}
        elif metric == "delta":
            lf = self.lifter
            out = {"lifter": getattr(lf, "name", type(lf).__name__)}
            for k in ("hub_model", "moge_resolution_level", "moge_fp16", "moge_batch"):
                if hasattr(lf, k):
                    out[k] = getattr(lf, k)
            for k, attr in (("tracker_weights", "_tracker"), ("moge_weights", "_moge")):
                if getattr(lf, attr, None) is not None:
                    out[k] = weights_fingerprint(getattr(lf, attr))
            if callable(getattr(lf, "identity", None)):
                out.update(lf.identity())
            paths = {k: str(getattr(lf, a)) for k, a in (("hub_repo", "hub_repo"), ("moge", "moge_weights"))
                     if hasattr(lf, a)}
            if "hub_repo" in paths:
                paths["hub_commit"] = _hub_commit(paths["hub_repo"])
            out["paths"] = paths
        else:
            raise ValueError(metric)
        self._identity[metric] = jsonable(out)
        return self._identity[metric]


def weights_fingerprint(module) -> str | None:
    """sha1 (16 hex) over the names, dtypes, shapes and bytes of every parameter and buffer of a torch module (also
    non-persistent buffers: the official MEt3R turns its frozen weights into those, so its state_dict is empty), or
    None when `module` is not a torch module or holds no tensor. Identifies the loaded weights independently of the
    file they came from."""
    if module is None or not callable(getattr(module, "named_parameters", None)):
        return None
    import torch
    tensors = dict(module.named_buffers())
    tensors.update(module.named_parameters())
    if not tensors:
        return None
    h = hashlib.sha1()
    for name in sorted(tensors):
        x = tensors[name].detach()
        h.update(f"{name}|{x.dtype}|{tuple(x.shape)}|".encode())
        x = x.contiguous().reshape(-1).cpu()
        if x.numel():
            h.update(x.view(torch.uint8).numpy())
    return h.hexdigest()[:16]


def _git_head(path) -> str | None:
    """The commit of a local git checkout (None when `path` is not one)."""
    g = Path(os.path.expanduser(str(path))) / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            p = g / ref
            if p.exists():
                return p.read_text().strip()
            for line in (g / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
            return None
        return head
    except OSError:
        return None


def _hub_commit(repo) -> str | None:
    """The commit behind a torch.hub repo: the HEAD of a local clone, or the ref of a GitHub spec pinned to a full
    commit ('owner/repo:<40 hex>'). None for a branch, a tag or the default branch (torch.hub keeps no commit)."""
    head = _git_head(repo)
    if head is not None or os.path.isdir(os.path.expanduser(str(repo))):
        return head
    ref = str(repo).rpartition(":")[2] if ":" in str(repo) else ""
    return ref if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref) else None


_DATA_VERSION: dict = {}


def data_version(root) -> str | None:
    """'<schema> <version>' of <root>/index.json (None without an index); cached per root."""
    key = str(root)
    if key not in _DATA_VERSION:
        try:
            idx = read_json(Path(root) / "index.json")
            _DATA_VERSION[key] = f"{idx.get('schema')} {idx.get('version')}"
        except (OSError, json.JSONDecodeError):
            _DATA_VERSION[key] = None
    return _DATA_VERSION[key]


_MODEL_CACHE: dict = {}


def get_models(device="auto", **kwargs) -> Models:
    """A process-wide Models instance per configuration (repeated evaluate() calls reuse the loaded models)."""
    device = resolve_device(device)
    key = json.dumps({"device": device, **kwargs}, sort_keys=True, default=str)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = Models(device=device, **kwargs)
    return _MODEL_CACHE[key]


def torch_state() -> dict:
    """Process-wide float32 precision flags (TF32 use of cuBLAS / cuDNN, autotuning) at scoring time."""
    try:
        import torch
    except ImportError:
        return {}
    b = torch.backends
    st = {"torch": torch.__version__, "cudnn_benchmark": bool(b.cudnn.benchmark),
          "cudnn_deterministic": bool(b.cudnn.deterministic)}
    for k, get in (("matmul_precision", torch.get_float32_matmul_precision),
                   ("cuda_matmul_allow_tf32", lambda: bool(b.cuda.matmul.allow_tf32)),
                   ("cudnn_allow_tf32", lambda: bool(b.cudnn.allow_tf32))):
        try:
            st[k] = get()
        except RuntimeError:
            st[k] = None
    if torch.cuda.is_available():
        st["gpu"] = torch.cuda.get_device_name()
    return st


class _Stage:
    """Seconds and peak GPU memory (GiB, torch allocator) of named stages."""

    def __init__(self, device: str):
        self.seconds, self.gib = {}, {}
        self.cuda = False
        try:
            import torch
            self.torch = torch
            self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        except ImportError:
            self.torch = None

    @contextlib.contextmanager
    def __call__(self, name: str):
        if self.cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            yield
        finally:
            if self.cuda:
                self.torch.cuda.synchronize()
                self.gib[name] = round(self.torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            self.seconds[name] = round(self.seconds.get(name, 0.0) + time.time() - t0, 2)


def _runtime(rec: dict, seconds: dict, gib: dict) -> dict:
    """The "runtime" section of a record: facts of the run, not results (stage seconds, peak GPU memory per stage in
    GiB, float32 precision flags, host, finish time). Merged with the record's previous section when a record is
    extended. Two evaluations of the same predictions give identical records apart from this section."""
    old = rec.get("runtime") or {}
    for k in ("timing", "gpu_peak_gib", "torch_state"):
        rec.pop(k, None)
    return {"seconds": {**(old.get("seconds") or {}), **seconds},
            "gpu_peak_gib": {**(old.get("gpu_peak_gib") or {}), **gib}, "torch_state": torch_state(),
            "host": socket.gethostname(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


_OOM_PATTERNS = ("out of memory", "cusolverdncreate", "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
                 "cufft_alloc_failed")


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return type(e).__name__ in ("OutOfMemoryError",) or any(k in msg for k in _OOM_PATTERNS)


def _retry_oom(fn):
    """fn(); after a CUDA out-of-memory failure (or a CUDA library that cannot allocate its handle) the allocator
    cache is released and fn is run once more. The values do not depend on it (no state is kept between the tries)."""
    try:
        return fn()
    except Exception as e:
        if not _is_oom(e):
            raise
        _empty_cache()
        return fn()


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _psnr(a, b) -> float:
    from ..metrics.appearance import psnr_frames
    return float(psnr_frames(a[None], b[None])[0])


def appearance_metrics(ref: np.ndarray, pred: np.ndarray, lpips=None, ssim: bool = True) -> dict:
    """PSNR / SSIM / LPIPS per view over positions 1.. (the conditioning frame excluded); clip values = mean over
    views. ref / pred (V, n, H, W, 3) uint8. SSIM of the views runs in threads."""
    from ..metrics.appearance import appearance, ssim_frames
    V = len(ref)
    per = []
    with ThreadPoolExecutor(max_workers=max(1, min(V, os.cpu_count() or 1))) as ex:
        futs = [ex.submit(ssim_frames, ref[v, 1:], pred[v, 1:]) for v in range(V)] if ssim else []
        for v in range(V):
            per.append(appearance(ref[v, 1:], pred[v, 1:], lpips=lpips, ssim=False))
        for v, f in enumerate(futs):
            s = f.result()
            per[v]["ssim"], per[v]["ssim_per_frame"] = float(np.mean(s)), s.tolist()
    out = {"n_frames": int(ref.shape[1] - 1), "per_view": {}}
    for key in ("psnr", "ssim", "lpips"):
        vals = [x[key] for x in per]
        out[key] = None if any(v is None for v in vals) else float(np.mean(vals))
        out["per_view"][key] = vals
    out["per_frame"] = {key: [x[f"{key}_per_frame"] for x in per] for key in ("psnr", "ssim", "lpips")}
    return out


def fvd_features(frames: np.ndarray, i3d, settings: dict):
    """I3D features of the FVD windows of every view: frames (V, n, H, W, 3) -> (feats (V*k, 400) float64, starts),
    or (None, None) when no complete window fits after the conditioning frame."""
    from ..metrics.fvd import fvd_windows, window_starts
    win, nw = settings["fvd_window"], settings["fvd_windows"]
    try:
        starts = window_starts(frames.shape[1], win, nw, 1)
    except ValueError:
        return None, None
    wins = np.concatenate([fvd_windows(f, window=win, n_windows=nw, start=1) for f in frames])
    return np.asarray(i3d.features(wins), np.float64), starts


def _fvd_entry(feats, starts, rel: str, settings: dict, i3d) -> dict:
    if feats is None:
        return {"skipped": f"no complete {settings['fvd_window']}-frame window after the conditioning frame"
                           + (f" ({settings['fvd_windows']} requested)" if settings["fvd_windows"] else "")}
    return {"file": rel, "n_windows": int(len(feats)), "starts": [int(x) for x in starts],
            "window": settings["fvd_window"], "i3d": getattr(i3d, "info", {}), "torch_state": torch_state()}


def _save_feats(path: Path, feats: np.ndarray, views, starts, window):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, feats=feats.astype(np.float32), views=np.asarray(views, np.int32),
             starts=np.asarray(starts, np.int32), window=np.int32(window))
    os.replace(tmp, path)


def met3r_cross(frames: np.ndarray, scorer, settings: dict, batch: int, feature_chunk: int) -> dict:
    """Cross-view MEt3R at settings['met3r_positions'] (position 0 stored, not scored); halves the batch on CUDA OOM."""
    from ..metrics.met3r import cross_view
    pos = settings["met3r_positions"]
    if frames.shape[0] < 2:
        return {"skipped": "fewer than 2 views"}
    if len(pos) < 2:
        return {"skipped": "no scored MEt3R position in the window"}
    if hasattr(scorer, "feature_chunk"):
        scorer.feature_chunk = int(feature_chunk)
    b = int(batch)
    while True:
        try:
            r = cross_view(scorer, frames, pos, aspect=settings["met3r_aspect"], size=settings["met3r_size"],
                           batch=b, temporal=settings["met3r_temporal"])
            break
        except Exception as e:
            if not _is_oom(e) or b <= 1:
                raise
            _empty_cache()
            b = max(1, b // 2)
    rows = np.asarray(r["cross"], np.float64)
    scored = [k for k, q in enumerate(pos) if q != 0]
    r["scored_rows"] = scored
    r["mean"] = float(rows[scored].mean())
    r["batch"], r["feature_chunk"] = b, int(feature_chunk)
    return r


def score_tracks(case: SampleCase, xyz, vis) -> dict:
    """track3d_metrics (+ allframes ADE / MTE / FDE) per point group of a lifted (or frozen) prediction.
    xyz (n, N, 3) world metres (non-finite = failure), vis (n, N) bool or None (always visible)."""
    from ..metrics.tracking import ade, fde, moving_delta, mte, track3d_metrics
    s = case.settings
    pred = np.array(xyz, np.float64, copy=True)
    bad = ~np.isfinite(pred).all(-1)
    pred[bad] = np.inf
    q = np.zeros(case.n_queries, np.int64)
    groups = {}
    for g, m in case.groups.items():
        if not m.any():
            groups[g] = {"n_queries": 0}
            continue
        r = track3d_metrics(case.gt[:, m], case.gt_vis[:, m], pred[:, m], None if vis is None else vis[:, m], q[m],
                            s["ladder_m"], s["survival_m"], static_m=s["static_m"], dynamic_m=s["dynamic_m"])
        r.update(moving_delta(case.gt[:, m], case.gt_vis[:, m], pred[:, m], q[m], s["ladder_m"]))
        with np.errstate(invalid="ignore"):
            r["allframes/ade"] = ade(case.gt[:, m], pred[:, m])
            r["allframes/mte"] = mte(case.gt[:, m], pred[:, m])
            r["allframes/fde"] = fde(case.gt[:, m], pred[:, m])
        r["n_queries"] = int(m.sum())
        groups[g] = r
    return {"groups": groups, "nonfinite_frac": float(bad.mean()) if bad.size else 0.0}


def lift(case: SampleCase, frames_eval: np.ndarray, lifter, anchor) -> tuple:
    """Run the lifter on (V, n, H, W, 3) eval-grid frames -> (xyz (n, N, 3), vis (n, N), info)."""
    from ..lifters.base import DepthAnchoredLifter
    s = case.settings
    frames = case.to_track_grid(frames_eval)
    if hasattr(lifter, "rgb_scale_quirk"):
        lifter.rgb_scale_quirk = s["rgb_scale_quirk"]
    if hasattr(lifter, "plain_K_scaling"):
        lifter.plain_K_scaling = s["plain_K_scaling"]
    K = case.K_window()
    if isinstance(lifter, DepthAnchoredLifter):
        xyz, vis, info = lifter.track(frames, K, case.w2c, case.queries, anchor, track_size=tuple(s["track_size"]),
                                      fov_x=case.fov_x, world_transform=case.world_transform, return_info=True)
    else:
        xyz, vis = lifter.track(frames, K, case.w2c, case.queries, anchor)
        info = {"lifter": getattr(lifter, "name", type(lifter).__name__)}
    return np.asarray(xyz), (None if vis is None else np.asarray(vis, bool)), info


def frozen_tracks(case: SampleCase) -> np.ndarray:
    """The frozen-points reference prediction: the query positions at every window frame."""
    return np.broadcast_to(case.gt[0][None], case.gt.shape).copy()


class ReferenceStore:
    """<reference_dir>/<hash>/: reference records, real FVD features, scene-normalisation depth, locks."""

    def __init__(self, root, h: str):
        self.dir = Path(root) / h
        self.hash = h

    def record_path(self, sample) -> Path:
        return self.dir / sample.subset / f"{sample.id}.json"

    def fvd_rel(self, sample) -> str:
        return f"_fvd/{sample.subset}/{sample.id}.npz"

    def norm_path(self, sample) -> Path:
        return self.dir / "_norm" / sample.subset / f"{sample.id}.npz"

    def load(self, sample) -> dict | None:
        p = self.record_path(sample)
        if not p.exists():
            return None
        try:
            r = read_json(p)
        except json.JSONDecodeError:
            return None
        return r if r.get("protocol_hash") == self.hash else None

    def lock(self, sample) -> Path:
        return self.dir / sample.subset / f".{sample.id}.lock"

    def acquire(self, sample) -> bool:
        lk = self.lock(sample)
        lk.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time():.0f}".encode())
                os.close(fd)
                return True
            except FileExistsError:
                if not self._stale(lk):
                    return False
                with contextlib.suppress(FileNotFoundError):
                    lk.unlink()
        return False

    @staticmethod
    def _stale(lk: Path) -> bool:
        try:
            host, pid, t = lk.read_text().split()
            age = time.time() - float(t)
        except (FileNotFoundError, ValueError):
            return True
        if host == socket.gethostname():
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        return age > LOCK_STALE_S

    def release(self, sample):
        with contextlib.suppress(FileNotFoundError):
            self.lock(sample).unlink()

    def anchor(self, case: SampleCase, lifter, real_frames=None):
        """The sample's anchor with `norm_depth` (identical for every source): the GT depth for 'gt_depth', else the
        anchored depth of the real video's t_ref (cached in _norm/, computed from `real_frames` (V, n, H, W, 3) eval
        grid, or loaded on demand)."""
        a = case.base_anchor()
        if a.norm_depth is not None:
            return a
        p = self.norm_path(case.sample)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                return replace(a, norm_depth=np.asarray(z["norm_depth"], np.float32))
        if real_frames is None:
            real_frames = np.stack([load_reference_frames(case.sample, v, case.protocol, case.settings)[0][:1]
                                    for v in case.views])
        from ..lifters.base import DepthAnchoredLifter
        if not isinstance(lifter, DepthAnchoredLifter):
            return a
        s = case.settings
        if hasattr(lifter, "rgb_scale_quirk"):
            lifter.rgb_scale_quirk = s["rgb_scale_quirk"]
        fr = case.to_track_grid(np.asarray(real_frames)[:, :1])
        a2 = lifter.prepare_anchor(a, fr, case.K, case.w2c[:, :1], track_size=tuple(s["track_size"]),
                                   fov_x=case.fov_x)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, norm_depth=np.asarray(a2.norm_depth, np.float32))
        os.replace(tmp, p)
        return a2


def _sample_attrs(sample) -> dict:
    out = {}
    for k, v in sample.attributes.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = list(v)
    return out


def _record_head(sample, protocol: Protocol, h: str, task: str) -> dict:
    """The fields every record starts with (also error records: summarize's attribute splits need them)."""
    return {"task": task, "uid": sample.uid, "subset": sample.subset, "id": sample.id, "protocol": protocol.name,
            "protocol_hash": h, "attributes": _sample_attrs(sample), "flags": list(sample.flags),
            "data_version": data_version(sample.root)}


def _model_identities(models, metrics) -> dict:
    """{metric: Models.identity(metric)} of the computed metrics ({} for model objects without `identity`)."""
    fn = getattr(models, "identity", None)
    if not callable(fn):
        return {}
    out = {}
    for m in metrics:
        try:
            out[m] = fn(m)
        except Exception as e:
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _base_record(sample, case: SampleCase, protocol: Protocol, h: str, task: str) -> dict:
    rec = _record_head(sample, protocol, h, task)
    if case.ok:
        rec.update(views=case.views, view_names=[sample.view_names[v] for v in case.views],
                   window=case.settings["window"], n_window=int(len(case.window)),
                   eval_size=case.settings["eval_size"], track_size=case.settings["track_size"])
    return rec


def _metrics_done(rec: dict | None) -> set:
    if not rec:
        return set()
    return set(rec.get("metrics_done", []))


def _parse_metrics(metrics) -> tuple:
    """'appearance,fvd' / ['appearance', 'fvd,met3r'] / None (all) -> the requested metrics, in METRICS order."""
    from ..utils import parse_list
    out = METRICS if metrics is None else tuple(parse_list(metrics) or ())
    bad = [m for m in out if m not in METRICS]
    if bad:
        raise ValueError(f"unknown metrics {bad}; choose from {METRICS}")
    if not out:
        raise ValueError(f"no metric requested; choose from {METRICS}")
    return tuple(m for m in METRICS if m in out)


def parse_shard(shard) -> tuple:
    """'k/n' or (k, n) -> (k, n); ValueError unless 0 <= k < n."""
    try:
        k, n = shard if isinstance(shard, (tuple, list)) else (int(v) for v in str(shard).split("/"))
        k, n = int(k), int(n)
    except (TypeError, ValueError):
        k = n = -1
    if not (n >= 1 and 0 <= k < n):
        raise ValueError(f"shard must be 'k/n' with 0 <= k < n, got {shard!r}")
    return k, n


def canonical_ids(ids):
    """Sample ids / uids with the subset part of 'Subset/id' entries resolved from its aliases ('sr/c001' ->
    'Simul-Robot/c001'); entries without a known subset part are kept as given. None -> None."""
    if ids is None:
        return None
    out = []
    for i in ids:
        i = str(i)
        if "/" in i:
            sub, sid = i.split("/", 1)
            with contextlib.suppress(KeyError):
                i = f"{canonical_subset(sub)}/{sid}"
        out.append(i)
    return out


def iter_samples(bench, subsets=None, ids=None, clip_filter=None):
    """Samples of `bench` in the given subsets (all by default), restricted to `ids` (ids or uids; subset aliases in
    uids are resolved) and filters."""
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    ids = canonical_ids(ids)
    for sub in subs:
        yield from bench.samples(sub, ids=ids, **(clip_filter or {}))


class RunDirError(UsageError, ValueError):
    """A run directory holds the results of another protocol hash or of another prediction root (a usage error:
    `vpmwb` reports it on one line with exit status 2; also a ValueError, as before)."""


def _write_run_protocol(path: Path, protocol: Protocol, h: str, track_size, decoder, overwrite: bool,
                        pred_root: str | None = None):
    """Write <dir>/protocol.json of a run directory (pred_root: its resolved prediction root) or of a reference
    directory (pred_root None). A directory holds one protocol hash (another one raises RunDirError) and a run directory
    one prediction root: another root raises RunDirError unless overwrite, which records the new root (the selected
    samples are then scored afresh; `summarize` excludes records of the previous root). A protocol.json written before
    prediction roots were recorded gets the root added (the rest of it is kept)."""
    info = {"task": TASK, "protocol": protocol.to_dict(), "hash": h, "track_size": track_size, "decoder": decoder,
            "settings": {s: subset_settings(protocol, s, track_size, decoder) for s in SUBSETS}}
    if pred_root is not None:
        info["pred_root"] = pred_root
    if path.exists():
        try:
            old = read_json(path)
        except json.JSONDecodeError:
            old = {}
        if old.get("hash") not in (None, h):
            raise RunDirError(f"{path.parent} holds results of protocol hash {old.get('hash')}, this protocol has {h}; "
                              f"use another directory")
        old_root = old.get("pred_root")
        if pred_root is not None and old_root not in (None, pred_root) and not overwrite:
            raise RunDirError(f"{path.parent} holds the results of the predictions in {old_root}, this call scores "
                              f"{pred_root}; use another output directory, or overwrite (--overwrite) to score the "
                              f"selected samples afresh from the new root")
        if old.get("hash") == h and not overwrite:
            if pred_root is not None and old_root is None:
                write_json(path, {**old, "pred_root": pred_root})
            return
    write_json(path, info)


def _write_selection(run_dir: Path, h: str, samples, subsets, ids, clip_filter) -> Path | None:
    """<run_dir>/_selection/<hash>_<digest>.json: the samples this evaluate() call was asked for (all shards, before
    sharding), with their attributes and flags. `summarize` takes the union over the files of a run as the expected
    sample set, so samples that no shard reached are reported ("not evaluated") instead of silently missing. Every
    shard writes the same content (one file per selection: no read-modify-write race)."""
    uids = sorted(s.uid for s in samples)
    digest = hashlib.sha1("\n".join(uids).encode()).hexdigest()[:12]
    path = Path(run_dir) / SELECTION_DIR / f"{h}_{digest}.json"
    if path.exists():
        return path
    info = {"task": TASK, "protocol_hash": h, "n_samples": len(uids), "subsets": subsets, "ids": ids,
            "clip_filter": clip_filter or {},
            "samples": {s.uid: {"attributes": _sample_attrs(s), "flags": list(s.flags)} for s in samples}}
    try:
        write_json(path, info)
    except OSError as e:
        if not path.exists():
            raise
        print(f"[generation] note: selection file written concurrently ({e})", flush=True)
    return path


def _runs_of(pred_root, out_dir) -> list:
    """[(name, pred_root, run_dir)] from a prediction root (one model, run_dir = out_dir) or {name: pred_root}
    (several models, run_dir = out_dir/<name>); [] for None (reference rows only)."""
    if pred_root is None:
        return []
    if out_dir is None:
        raise ValueError("out_dir is required to score predictions")
    if isinstance(pred_root, dict):
        runs = [(str(n), Path(pr), Path(out_dir) / str(n)) for n, pr in pred_root.items()]
        for n, _, _ in runs:
            if n.startswith("_") or n in ("real", "summary"):
                raise ValueError(f"model name {n!r} is reserved")
        return runs
    return [(Path(out_dir).name, Path(pred_root), Path(out_dir))]


def evaluate(bench_root, pred_root, out_dir, protocol=None, subsets=None, ids=None, metrics=METRICS,
             device: str = "auto", shard="0/1", reference_dir=None, overwrite: bool = False, lpips: bool = True,
             ssim: bool = True, met3r_batch: int | None = None, met3r_feature_chunk: int | None = None, *,
             references: bool = True, overwrite_reference: bool = False, track_size=None, decoder: str | None = None,
             lifter_kwargs=None, met3r_kwargs=None, i3d_weights=None, save_tracks: bool = False, models=None,
             raise_errors: bool = False, verbose: bool = True) -> dict:
    """Score the predictions in `pred_root` (layout in the module docstring); write per-sample records.

    bench_root      VP-WorldBench root (or a Benchmark)
    pred_root       prediction root of one model; or {name: prediction root} to score several models in one pass (the
                    real frames are decoded and the reference rows computed once per sample; run dirs out_dir/<name>,
                    reference rows in out_dir/_reference unless reference_dir is given); or None to compute only the
                    reference rows (then reference_dir or out_dir is required)
    out_dir         run directory of this model (<out_dir>/<Subset>/<id>.json, _fvd/, protocol.json), or the root of
                    the run directories when pred_root is a dict
    protocol        Protocol, preset name or file ('vpm-v1' default, 'legacy-prerelease')
    subsets / ids   restrict the samples (ids may be ids or uids); protocol.clip_filter applies too
    metrics         subset of ('appearance', 'fvd', 'met3r', 'delta')
    device          torch device of the metric models: 'auto' (cuda when available, else cpu), 'cuda', 'cpu', ...
    shard           'k/n': this process takes every n-th sample starting at k (order: subsets, then index order), e.g.
                    one shard per GPU
    reference_dir   shared reference directory; reference rows are written to <reference_dir>/<protocol hash>/.
                    Default: _reference next to the run directory (<out_dir>/../_reference for one model,
                    <out_dir>/_reference for several, <out_dir>/_reference for reference-only calls)
    overwrite       recompute the requested metrics of existing model records (overwrite_reference: same for the
                    reference rows); metrics not requested are kept. A record scored from other predictions (another
                    resolved root, or files changed since: `prediction_stamp`) is always scored afresh, without the
                    metrics of the old predictions; a run directory of another prediction root raises RunDirError
                    unless overwrite (which records the new root)
    lpips / ssim    compute LPIPS / SSIM (PSNR always)
    met3r_batch, met3r_feature_chunk  MEt3R batching (None: 16 / 8, or the pre-release 64 / 32 in legacy mode)
    references      also compute missing reference rows of the samples of this shard (lock files avoid duplicate work
                    by concurrent processes; a sample locked by a live process is left to it)
    track_size      (W, H) or {subset: (W, H)}: tracking grid (default: legacy recipe, else the eval grid)
    decoder         'pyav' (default) | 'opencv' (legacy default)
    lifter_kwargs   e.g. {"hub_repo": "/path/mvtracker", "moge": "/path/model.pt"}; met3r_kwargs e.g.
                    {"featup_repo": ..., "backbone_weights": ...}; i3d_weights: local i3d_pretrained_400.pt
    save_tracks     also write the lifted tracks (<run_dir>/_tracks/, float32)
    models          a Models instance (default: one per configuration and process, see get_models)
    raise_errors    re-raise exceptions instead of recording status "error"
    Returns {"counts": {subset: {status: n}} (one model) or {name: {subset: {status: n}}} (several), "reference":
    {computed, cached, busy, error}, "n_selected" (samples selected, all shards), "seconds", "protocol_hash",
    "model_load_seconds"[, "unmatched_ids"]}. Statuses:
    ok, error, missing, invalid, skipped (not evaluable), cached (nothing to do). A failure outside the per-metric
    handlers (e.g. unreadable sample data) is recorded as status "error" and the run continues; a package the
    requested metrics need that is not installed (`missing_dependencies`, checked before the first sample) or a failure
    to load the metric models raises ModelLoadError; a run directory of another protocol hash or prediction root
    raises RunDirError. Appearance metrics need no reference rows: without FVD / MEt3R / delta no reference directory
    is written."""
    from ..data.benchmark import Benchmark
    t_start = time.time()
    p = resolve_protocol(protocol)
    bench = bench_root if isinstance(bench_root, Benchmark) else Benchmark(bench_root)
    metrics = _parse_metrics(metrics)
    device = resolve_device(device)
    ids = canonical_ids(ids)
    h = protocol_hash(p, track_size, decoder)
    legacy = L.is_legacy(p)
    m3_batch = met3r_batch or (L.MET3R_LEGACY["batch"] if legacy else DEFAULT_MET3R_BATCH)
    m3_chunk = met3r_feature_chunk or (L.MET3R_LEGACY["feature_chunk"] if legacy else DEFAULT_MET3R_CHUNK)
    runs = _runs_of(pred_root, out_dir)
    if reference_dir is None and out_dir is None:
        raise ValueError("reference-only mode (pred_root None) needs reference_dir or out_dir")
    references = references or not runs
    ref_metrics = [m for m in metrics if m != "appearance"]
    if models is None:
        models = get_models(device, i3d_weights=i3d_weights, met3r_kwargs=met3r_kwargs, lifter=p.lifter,
                            lifter_kwargs=lifter_kwargs)
    missing = missing_dependencies(metrics if runs else ref_metrics, lpips, ssim, models)
    if missing:
        raise ModelLoadError("the requested metrics need packages that are not installed: " + "; ".join(missing))
    if reference_dir is not None:
        ref_root = Path(reference_dir)
    elif runs and not isinstance(pred_root, dict):
        ref_root = Path(out_dir).resolve().parent / "_reference"
    else:
        ref_root = Path(out_dir) / "_reference"
    store = ReferenceStore(ref_root, h)
    for _, pr, run_dir in runs:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_protocol(run_dir / "protocol.json", p, h, track_size, decoder, overwrite, _root_str(pr))
    if references and ref_metrics:
        _write_run_protocol(store.dir / "protocol.json", p, h, track_size, decoder, False)
    k, n = parse_shard(shard)
    samples = list(iter_samples(bench, subsets, ids, p.clip_filter))
    for _, _, run_dir in runs:
        _write_selection(run_dir, h, samples, None if not subsets else [canonical_subset(s) for s in subsets],
                         None if not ids else list(ids), p.clip_filter)
    unmatched = []
    if ids:
        seen = {s.id for s in samples} | {s.uid for s in samples}
        unmatched = [i for i in ids if i not in seen]
        if unmatched and verbose:
            print(f"[generation] warning: {len(unmatched)} requested id(s) match no sample of the selected subsets "
                  f"(or are excluded by clip_filter): {unmatched[:10]}", flush=True)
    mine = samples[k::n]
    counts = {name: {} for name, _, _ in runs}
    ref_counts = {"computed": 0, "cached": 0, "busy": 0, "error": 0}
    if verbose:
        print(f"[generation] protocol {p.name} ({h}), shard {k}/{n}: {len(mine)} of {len(samples)} samples, "
              f"models {[r[0] for r in runs] or '(reference only)'}, metrics {','.join(metrics)}, decoder "
              f"{decoder or ('opencv' if legacy else 'pyav')}, reference dir "
              f"{store.dir if ref_metrics else '- (appearance metrics need none)'}", flush=True)
    for s in mine:
        try:
            st = _evaluate_one(s, p, h, runs, store, models, metrics, device, overwrite, overwrite_reference,
                               references, lpips, ssim, m3_batch, m3_chunk, track_size, decoder, save_tracks,
                               raise_errors, verbose, ref_counts)
        except ModelLoadError:
            raise
        except Exception as e:
            if raise_errors:
                raise
            st = _sample_error(s, p, h, runs, metrics, e, verbose)
        for name, status in st.items():
            c = counts[name].setdefault(s.subset, {})
            c[status] = c.get(status, 0) + 1
    res = {"counts": counts[runs[0][0]] if len(runs) == 1 and not isinstance(pred_root, dict) else counts,
           "reference": ref_counts, "n_selected": len(samples), "seconds": round(time.time() - t_start, 1),
           "protocol_hash": h, "model_load_seconds": dict(getattr(models, "load_seconds", {}))}
    if unmatched:
        res["unmatched_ids"] = unmatched
    if verbose:
        print(f"[generation] done: {json.dumps(res)}", flush=True)
    return res


def _per_view(fn, items) -> list:
    """[fn(x) for x in items], one thread per view (video decoding and the YUV->RGB conversion release the GIL);
    exceptions propagate."""
    items = list(items)
    if len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        return list(ex.map(fn, items))


def _load_old(rec_path: Path, h: str, ident: dict | None = None):
    """The existing record if it can be extended: same protocol hash, status ok / error, and scored from the same
    predictions (`ident` = `_prediction_ident`: the same resolved root and unchanged files). Else None: the record is
    computed afresh. Records written before "pred_root" / "pred_stamp" were stored are reused as they are."""
    if not rec_path.exists():
        return None
    try:
        old = read_json(rec_path)
    except json.JSONDecodeError:
        return None
    if old.get("protocol_hash") != h or old.get("status") not in ("ok", "error"):
        return None
    for k, v in (ident or {}).items():
        if k in old and old[k] != v:
            return None
    return old


class ModelLoadError(UsageError, RuntimeError):
    """The metric models could not be loaded (missing weights / packages): evaluate() stops instead of recording an
    error for every sample. A setup error: `vpmwb` reports it on one line with exit status 2."""


def _sample_error(sample, p, h, runs, metrics, e: BaseException, verbose: bool) -> dict:
    """Record a failure outside the per-metric handlers (unreadable sample data, reference frames, ...) as status
    "error" in every run whose record is not complete; the next evaluate() retries it. Returns {name: status}."""
    reason = f"{type(e).__name__}: {e}"
    tb = traceback.format_exc(limit=8)
    try:
        head = _record_head(sample, p, h, TASK)
    except Exception:
        head = {"task": TASK, "uid": f"{sample.subset}/{sample.id}", "subset": sample.subset, "id": sample.id,
                "protocol": p.name, "protocol_hash": h}
    out = {}
    for name, pred_root, run_dir in runs:
        rec_path = run_dir / sample.subset / f"{sample.id}.json"
        ident = _prediction_ident(pred_root, sample)
        old = _load_old(rec_path, h, ident)
        if old is not None and old.get("status") == "ok" and set(metrics) <= _metrics_done(old):
            out[name] = "cached"
            continue
        rec = dict(old) if old is not None else {**head, "metrics_done": []}
        for k, v in head.items():
            rec.setdefault(k, v)
        rec.update(status="error", reason=reason, traceback=tb, **ident)
        write_json(rec_path, rec)
        out[name] = "error"
    if verbose:
        print(f"[generation] {sample.uid}: error ({reason[:300]})", flush=True)
    return out


def _evaluate_one(sample, p, h, runs, store, models, metrics, device, overwrite, overwrite_reference, references,
                  lpips, ssim, m3_batch, m3_chunk, track_size, decoder, save_tracks, raise_errors, verbose,
                  ref_counts) -> dict:
    """One sample: the reference rows (when missing) and the record of every run [(name, pred_root, run_dir)]; the
    real frames are decoded once. Returns {name: status}."""
    t0 = time.time()
    settings = subset_settings(p, sample.subset, track_size, decoder)
    case = SampleCase(sample, p, settings)
    if not case.ok:
        for _, pred_root, run_dir in runs:
            rec_path = run_dir / sample.subset / f"{sample.id}.json"
            if overwrite or not rec_path.exists():
                rec = _base_record(sample, case, p, h, TASK)
                rec.update(status="skipped", reason=case.reason, pred_root=_root_str(pred_root))
                write_json(rec_path, rec)
        if verbose:
            print(f"[generation] {sample.uid}: skipped ({case.reason})", flush=True)
        return {name: "skipped" for name, _, _ in runs}

    plans = []
    for name, pred_root, run_dir in runs:
        rec_path = run_dir / sample.subset / f"{sample.id}.json"
        ident = _prediction_ident(pred_root, sample)
        old = _load_old(rec_path, h, ident)
        done = set() if overwrite else _metrics_done(old)
        plans.append({"name": name, "pred_root": pred_root, "run_dir": run_dir, "rec_path": rec_path, "old": old,
                      "todo": [m for m in metrics if m not in done], "status": None, "rec": None, "ident": ident})
    ref = store.load(sample)
    ref_todo = []
    if references:
        rdone = _metrics_done(ref) if ref and ref.get("status") in ("ok", "error") and not overwrite_reference \
            else set()
        ref_todo = [m for m in metrics if m != "appearance" and m not in rdone]
    if not any(pl["todo"] for pl in plans) and not ref_todo:
        if not runs:
            ref_counts["cached"] += 1
        return {pl["name"]: "cached" for pl in plans}
    have_lock = False
    if ref_todo:
        have_lock = store.acquire(sample)
        if not have_lock:
            ref_counts["busy"] += 1
            ref_todo = []
    stage = _Stage(device)
    try:
        for pl in plans:
            if pl["todo"] and not has_prediction(pl["pred_root"], sample, case.views):
                d = prediction_dir(pl["pred_root"], sample)
                miss = [f"view{v}.mp4" for v in case.views if not (d / f"view{v}.mp4").exists()]
                rec = _base_record(sample, case, p, h, TASK)
                rec.update(status="missing", reason=f"missing {', '.join(miss)} in {d}", **pl["ident"])
                write_json(pl["rec_path"], rec)
                pl.update(todo=[], status="missing", rec=rec)
        work = [pl for pl in plans if pl["todo"]]
        if work or ref_todo:
            if hasattr(models, "load"):
                try:
                    models.load(tuple(set(ref_todo).union(*[pl["todo"] for pl in work])), lpips)
                except Exception as e:
                    raise ModelLoadError(f"loading the metric models failed: {type(e).__name__}: {e}") from e
            with stage("decode_reference"):
                loaded = _per_view(lambda v: load_reference_frames(sample, v, p, settings), case.views)
                ref_frames = np.stack([f for f, _ in loaded])
                ref_source = loaded[0][1]
            if ref_todo:
                ref = _reference_rows(sample, case, settings, p, h, store, models, ref, ref_frames, ref_source,
                                      ref_todo, m3_batch, m3_chunk, stage, raise_errors)
                ref_counts["computed" if ref.get("status") == "ok" else "error"] += 1
            for pl in work:
                _model_record(pl, sample, case, settings, p, h, store, models, ref_frames, ref_source, lpips, ssim,
                              m3_batch, m3_chunk, save_tracks, raise_errors, device,
                              shared={"decode_reference": stage.seconds.get("decode_reference", 0.0)})
        if verbose:
            _log(sample, plans, ref if ref_todo else None, stage, t0)
        return {pl["name"]: pl["status"] or "cached" for pl in plans}
    finally:
        if have_lock:
            store.release(sample)


def _model_record(pl, sample, case, settings, p, h, store, models, ref_frames, ref_source, lpips, ssim, m3_batch,
                  m3_chunk, save_tracks, raise_errors, device, shared=None):
    """Load one model's prediction, compute its missing metrics, write its record (pl is updated in place)."""
    t0 = time.time()
    stage = _Stage(device)
    try:
        with stage("decode_prediction"):
            got = _per_view(lambda iv: load_prediction_frames(pl["pred_root"], sample, iv[1], p, settings,
                                                              ref_frames[iv[0]][0]), list(enumerate(case.views)))
        pred_frames = np.stack([f for f, _ in got])
    except ValueError as e:
        rec = _base_record(sample, case, p, h, TASK)
        rec.update(status="invalid", reason=str(e), **pl["ident"])
        write_json(pl["rec_path"], rec)
        pl.update(status="invalid", rec=rec)
        return
    rec = dict(pl["old"]) if pl["old"] is not None else _base_record(sample, case, p, h, TASK)
    rec.update({k: v for k, v in _base_record(sample, case, p, h, TASK).items() if k not in rec})
    rec.pop("reason", None)
    rec.pop("traceback", None)
    rec.update(pl["ident"])
    rec["prediction"] = {"dir": str(prediction_dir(pl["pred_root"], sample)),
                         "meta": prediction_meta(pl["pred_root"], sample), "views": [i for _, i in got]}
    rec["reference_source"] = ref_source
    rec["decoder"] = settings["decoder"]
    c0 = [_psnr(ref_frames[i][0], pred_frames[i][0]) for i in range(len(case.views))]
    rec["cond_psnr"] = c0
    rec["cond_psnr_ok"] = bool(min(c0) >= COND_PSNR_MIN_DB)
    errors = {}
    for m in pl["todo"]:
        try:
            with stage(m):
                rec[m] = _retry_oom(lambda: _model_metric(m, sample, case, settings, store, models, pred_frames,
                                                          ref_frames, pl["run_dir"], lpips, ssim, m3_batch, m3_chunk,
                                                          save_tracks))
            if m == "met3r":
                _empty_cache()
        except Exception as e:
            if raise_errors:
                raise
            errors[m] = f"{type(e).__name__}: {e}"
            rec[m] = {"error": errors[m], "traceback": traceback.format_exc(limit=6)}
            if _is_oom(e):
                _empty_cache()
    done = (_metrics_done(rec) | set(pl["todo"])) - set(errors)
    rec["metrics_done"] = [m for m in METRICS if m in done]
    rec["models"] = {**(rec.get("models") or {}),
                     **_model_identities(models, [m for m in pl["todo"] if m not in errors])}
    rec["status"] = "error" if errors else "ok"
    if errors:
        rec["reason"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
    rec["runtime"] = _runtime(rec, {**(shared or {}), **stage.seconds, "total": round(time.time() - t0, 2)}, stage.gib)
    write_json(pl["rec_path"], rec)
    pl.update(status=rec["status"], rec=rec)


def _model_metric(m, sample, case, settings, store, models, pred, ref, out, lpips, ssim, m3_batch, m3_chunk,
                  save_tracks):
    if m == "appearance":
        return appearance_metrics(ref, pred, models.lpips if lpips else None, ssim)
    if m == "fvd":
        feats, starts = fvd_features(pred, models.i3d, settings)
        rel = f"_fvd/{sample.subset}/{sample.id}.npz"
        if feats is not None:
            _save_feats(out / rel, feats, case.views, starts, settings["fvd_window"])
        return _fvd_entry(feats, starts, rel, settings, models.i3d)
    if m == "met3r":
        return met3r_cross(pred, models.met3r, settings, m3_batch, m3_chunk)
    if m == "delta":
        if case.n_queries == 0:
            return {"main_group": case.main_group, "n_queries": 0, "groups": None,
                    "skipped": f"no query point at frame {case.t_ref}"}
        anchor = store.anchor(case, models.lifter, ref)
        xyz, vis, info = lift(case, pred, models.lifter, anchor)
        res = {"main_group": case.main_group, "n_queries": case.n_queries, **score_tracks(case, xyz, vis),
               "lifter": info, "depth_source": case.depth_source}
        if save_tracks:
            rel = f"_tracks/{sample.subset}/{sample.id}.npz"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / rel, xyz=np.asarray(xyz, np.float32),
                                vis=np.zeros((0,), bool) if vis is None else vis,
                                query_index=case.query_index.astype(np.int32))
            res["tracks_file"] = rel
        return res
    raise ValueError(m)


def _reference_rows(sample, case, settings, p, h, store, models, old, ref_frames, ref_source, todo, m3_batch,
                    m3_chunk, stage, raise_errors) -> dict:
    """Compute the missing reference rows of one sample (real FVD features, real MEt3R, real / frozen delta)."""
    rec = dict(old) if old and old.get("status") in ("ok", "error") else \
        _base_record(sample, case, p, h, REFERENCE_TASK)
    rec.pop("reason", None)
    rec.pop("traceback", None)
    rec.update(settings=settings, reference_source=ref_source, decoder=settings["decoder"])
    errors = {}

    def compute(m):
        if m == "fvd":
            feats, starts = fvd_features(ref_frames, models.i3d, settings)
            rel = store.fvd_rel(sample)
            if feats is not None:
                _save_feats(store.dir / rel, feats, case.views, starts, settings["fvd_window"])
            return _fvd_entry(feats, starts, rel, settings, models.i3d)
        if m == "met3r":
            return met3r_cross(ref_frames, models.met3r, settings, m3_batch, m3_chunk)
        d = {"main_group": case.main_group, "n_queries": case.n_queries}
        if case.n_queries == 0:
            d.update(real=None, frozen=None, skipped=f"no query point at frame {case.t_ref}")
        else:
            anchor = store.anchor(case, models.lifter, ref_frames)
            xyz, vis, info = lift(case, ref_frames, models.lifter, anchor)
            d["real"] = {**score_tracks(case, xyz, vis), "lifter": info}
            d["frozen"] = score_tracks(case, frozen_tracks(case), None)
            d["depth_source"] = case.depth_source
            d["query_index"] = case.query_index.tolist()
        return d

    for m in todo:
        try:
            with stage(f"reference_{m}"):
                rec[m] = _retry_oom(lambda: compute(m))
            if m == "met3r":
                _empty_cache()
        except Exception as e:
            if raise_errors:
                raise
            errors[m] = f"{type(e).__name__}: {e}"
            rec[m] = {"error": errors[m], "traceback": traceback.format_exc(limit=6)}
            if _is_oom(e):
                _empty_cache()
    done = (_metrics_done(rec) | set(todo)) - set(errors)
    rec["metrics_done"] = [m for m in METRICS if m in done]
    rec["models"] = {**(rec.get("models") or {}), **_model_identities(models, [m for m in todo if m not in errors])}
    for k, v in _record_head(sample, p, h, REFERENCE_TASK).items():
        rec.setdefault(k, v)
    rec["status"] = "ok" if not errors else "error"
    if errors:
        rec["reason"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
    rec["runtime"] = _runtime(rec, {k: v for k, v in stage.seconds.items() if k.startswith(("reference", "decode_ref"))},
                              {k: v for k, v in stage.gib.items() if k.startswith("reference")})
    write_json(store.record_path(sample), rec)
    return rec


def _log(sample, plans, ref, stage, t0):
    bits = [f"[generation] {sample.uid}:"]
    peak = max(stage.gib.values(), default=0.0)
    for pl in plans:
        rec, st = pl.get("rec"), pl.get("status") or "cached"
        txt = [f"{pl['name']} {st}"]
        if rec is not None and st in ("ok", "error"):
            a = rec.get("appearance") or {}
            if a.get("psnr") is not None:
                txt.append(f"PSNR {a['psnr']:.2f}")
            m = rec.get("met3r") or {}
            if m.get("mean") is not None:
                txt.append(f"MEt3R {m['mean']:.4f}")
            d = rec.get("delta") or {}
            g = (d.get("groups") or {}).get(d.get("main_group")) or {}
            if g.get("any/delta_avg") is not None:
                txt.append(f"delta {g['any/delta_avg']:.1f}")
            peak = max([peak, *((rec.get("runtime") or {}).get("gpu_peak_gib") or {}).values()])
        if rec is not None and rec.get("reason"):
            txt.append(f"({rec['reason'][:160]})")
        bits.append(" ".join(txt) + ";")
    if ref is not None:
        d = ref.get("delta") or {}
        g = ((d.get("real") or {}).get("groups") or {}).get(d.get("main_group")) or {}
        m = ref.get("met3r") or {}
        bits.append(f"reference {ref.get('status')}" + (f" real MEt3R {m['mean']:.4f}" if m.get("mean") is not None
                                                       else "")
                    + (f" real delta {g['any/delta_avg']:.1f}" if g.get("any/delta_avg") is not None else "") + ";")
    bits.append(f"({time.time() - t0:.1f} s, peak {peak:.1f} GiB)")
    print(" ".join(bits), flush=True)


def summarize(run_dirs: dict, reference_dir=None, protocol=None, out=None, splits=()) -> dict:
    """Tables over several models (+ Real video / Frozen points rows): see vpm_worldbench.report.summarize_generation.
    run_dirs {name: out_dir of evaluate()}; out: directory for summary.json / summary.md."""
    return summarize_generation(run_dirs, out=out, reference_dir=reference_dir, protocol=protocol, splits=splits)


def _parse_track_size(items):
    """['WxH'] -> (W, H); ['Subset=WxH', ...] -> {subset: (W, H)}; [] / None -> None. ValueError when malformed."""
    if not items:
        return None
    out = {}
    for it in items:
        sub, wh = it.split("=", 1) if "=" in it else (None, it)
        try:
            w, hh = (int(x) for x in wh.lower().split("x"))
        except ValueError:
            w = hh = 0
        if w <= 0 or hh <= 0:
            raise ValueError(f"--track-size must be [Subset=]WxH with positive sizes (e.g. 480x480), got {it!r}")
        if sub is None:
            return (w, hh)
        try:
            out[canonical_subset(sub)] = (w, hh)
        except KeyError as e:
            raise ValueError(f"--track-size {it!r}: {e.args[0] if e.args else e}") from None
    return out


def _parse_named(items, what: str) -> dict:
    """['name=dir', ...] -> {name: dir} (names unique and non-empty, dirs non-empty); ValueError otherwise."""
    out = {}
    for it in items:
        n, eq, d = it.partition("=")
        if not eq or not d:
            raise ValueError(f"{what}: expected NAME=DIR, got {it!r}")
        if not n or n in out:
            raise ValueError(f"{what}: empty or repeated name in {it!r}")
        out[n] = d
    return out


SUBCOMMANDS = ("export", "evaluate", "reference", "summarize")
FAILED_STATUSES = ("error", "missing", "invalid")


def failed_runs(res: dict, named: bool) -> list:
    """['<model>: error 3, missing 1', ...]: the runs of an evaluate() result whose evaluated samples ALL failed (status
    error / missing / invalid; cached and not evaluable samples do not count); for a reference-only call (no run) the
    reference rows when every computed row failed. named: res["counts"] is {name: {subset: counts}}."""
    out = []
    per_run = (res.get("counts") or {}) if named else ({"": res["counts"]} if res.get("counts") else {})
    for name, subs in per_run.items():
        tot = {}
        for c in subs.values():
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
        tried = {k: v for k, v in tot.items() if k not in ("cached", "skipped")}
        if tried and set(tried) <= set(FAILED_STATUSES):
            out.append((f"{name}: " if name else "") + ", ".join(f"{k} {v}" for k, v in sorted(tried.items())))
    ref = res.get("reference") or {}
    if not per_run and ref.get("error") and not ref.get("computed"):
        out.append(f"reference rows: error {ref['error']}")
    return out


def _one_line(e: BaseException) -> str:
    return " ".join(str(e).split())


def _traceback_requested() -> bool:
    """VPMWB_TRACEBACK is set to anything but '' / '0': errors are re-raised (the rule of vpm_worldbench.cli.main)."""
    return os.environ.get("VPMWB_TRACEBACK", "") not in ("", "0")


def _error_exit(e: BaseException) -> int:
    """A setup error as one line `vpmwb: error: ...` on stderr (as vpm_worldbench.cli.main prints it); returns the exit
    status 2. Call it from an `except` block after re-raising when `_traceback_requested()`."""
    msg = _one_line(e) or type(e).__name__
    if isinstance(e, ModuleNotFoundError) and "pip install" not in msg:
        msg += " (an optional package: see the extras in the README, section Install)"
    print(f"vpmwb: error: {msg}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    """`vpmwb generation {export, evaluate (alias eval), reference, summarize} ...` (also
    `python -m vpm_worldbench.tasks.generation ...`). `export` is vpm_worldbench.inputs.main."""
    import argparse

    from ..utils import parse_list, parse_subsets
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        from .. import inputs
        return int(inputs.main(argv[1:]) or 0)
    ap = argparse.ArgumentParser(prog="vpmwb generation", description="the generation task: export the model inputs, "
                                 "score generated multi-view videos (PSNR / SSIM / LPIPS, FVD, cross-view MEt3R, δ), tabulate "
                                 "models (docs/PROTOCOL.md, docs/METRICS.md)")
    sp = ap.add_subparsers(dest="cmd", required=True, metavar="{export,evaluate,reference,summarize}")
    sp.add_parser("export", add_help=False, help="write the model inputs: context frames, prompts, cameras "
                                                 "(`vpmwb generation export --help`)")
    parsers = {}
    for name in ("evaluate", "reference"):
        q = sp.add_parser(name, aliases=["eval"] if name == "evaluate" else [],
                          help="score predictions (and compute missing reference rows)" if name == "evaluate" else
                          "compute only the reference rows (real video, frozen points) into --reference-dir")
        parsers[name] = q
        q.add_argument("--root", required=True, help="VP-WorldBench data root")
        if name == "evaluate":
            q.add_argument("--pred", required=True, nargs="+", action="extend", metavar="ROOT|NAME=ROOT",
                           help="prediction root (<pred>/<Subset>/<id>/view{k}.mp4); or NAME=ROOT pairs (repeatable) to "
                                "score several models in one pass, into run directories <out>/NAME")
            q.add_argument("--out", required=True, help="run directory of the model (with NAME=ROOT: the parent of the "
                                                        "run directories)")
            q.add_argument("--overwrite", action="store_true",
                           help="recompute the requested metrics of existing records (also: accept a run directory "
                                "of another prediction root and score the selected samples afresh from this one)")
            q.add_argument("--no-references", action="store_true", help="do not compute missing reference rows")
            q.add_argument("--no-lpips", action="store_true")
            q.add_argument("--no-ssim", action="store_true")
            q.add_argument("--save-tracks", action="store_true", help="also write the lifted 3D tracks (_tracks/)")
        q.add_argument("--reference-dir", default=None, required=(name == "reference"),
                       help="shared reference directory, rows go to <dir>/<protocol hash>/ (default: _reference next "
                            "to the run directory, i.e. <out>/../_reference; <out>/_reference with NAME=ROOT)")
        q.add_argument("--overwrite-reference", action="store_true", help="recompute existing reference rows")
        q.add_argument("--protocol", default="vpm-v1", help="vpm-v1 | legacy-prerelease | <yaml/json file>")
        q.add_argument("--subsets", nargs="*", default=None, help="subset names (space or comma separated; default all)")
        q.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (space or comma separated)")
        q.add_argument("--metrics", nargs="*", default=None, help=f"subset of {','.join(METRICS)} (default all)")
        q.add_argument("--shard", default="0/1", help="k/n: this process takes every n-th sample from the k-th")
        q.add_argument("--device", default="auto", help="torch device of the metric models: auto (cuda when PyTorch "
                                                        "sees a GPU, else cpu; default) | cuda | cuda:N | cpu")
        q.add_argument("--met3r-batch", type=int, default=None, help=f"ordered pairs per MASt3R forward (default "
                                                                      f"{DEFAULT_MET3R_BATCH}; legacy {L.MET3R_LEGACY['batch']})")
        q.add_argument("--met3r-feature-chunk", type=int, default=None, help=f"images per FeatUp call (default "
                                                                              f"{DEFAULT_MET3R_CHUNK})")
        q.add_argument("--decoder", choices=L.DECODERS, default=None, help="pyav (default) | opencv (legacy default)")
        q.add_argument("--track-size", action="append", default=[], metavar="[Subset=]WxH",
                       help="grid of depth estimation and tracking (default: the eval grid; repeatable)")
        q.add_argument("--mvtracker-repo", default=None, help="MVTracker torch.hub repo or local clone")
        q.add_argument("--moge", default=None, help="MoGe-2 HF repo id or local model.pt")
        q.add_argument("--featup-repo", default=None, help="local FeatUp checkout for MEt3R")
        q.add_argument("--mast3r-weights", default=None, help="local MASt3R weights for MEt3R")
        q.add_argument("--i3d-weights", default=None, help="local i3d_pretrained_400.pt")
        q.add_argument("--raise-errors", action="store_true",
                       help="stop at the first failing sample or metric with its exception instead of recording "
                            "status error (debugging; the traceback of a setup error: VPMWB_TRACEBACK=1)")
        q.add_argument("--quiet", action="store_true", help="no per-sample log lines")
    q = sp.add_parser("summarize", help="paper-style tables of one or more runs on the paired clip set")
    parsers["summarize"] = q
    q.add_argument("--run", "--runs", dest="run", nargs="+", action="extend", required=True, metavar="NAME=DIR",
                   help="model name and run directory of evaluate (repeatable)")
    q.add_argument("--reference-dir", default=None, help="the reference directory given to evaluate (default: "
                                                         "<run>/_reference or <run>/../_reference)")
    q.add_argument("--protocol", default=None, help="check the runs against this protocol; its fvd_cov is used "
                                                    "(default: the runs' protocol.json)")
    q.add_argument("--out", required=True, help="output directory (summary.json, summary.md)")
    q.add_argument("--split", nargs="+", action="extend", default=[], metavar="SPEC",
                   help="extra tables on a clip subset: key=value, key!=value, flag:NAME, !flag:NAME (repeatable)")
    q.add_argument("--require-complete", action="store_true",
                   help="exit with status 3 (after writing the summary) when an expected sample of a run has no "
                        "record, or the summary has warnings (e.g. weights that differ between records)")
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        try:
            runs = _parse_named(a.run, "--run")
        except ValueError as e:
            parsers["summarize"].error(_one_line(e))
        try:
            summ = summarize(runs, reference_dir=a.reference_dir, protocol=a.protocol, out=a.out, splits=a.split)
        except (OSError, ValueError, KeyError, UsageError) as e:
            if _traceback_requested():
                raise
            return _error_exit(e)
        print((Path(a.out) / "summary.md").read_text())
        if a.require_complete and (summ.get("n_not_evaluated") or summ.get("warnings")):
            print(f"[generation] incomplete: {summ.get('n_not_evaluated', 0)} expected sample record(s) missing, "
                  f"{len(summ.get('warnings') or [])} warning(s)", file=sys.stderr)
            return 3
        return 0
    lk, mk = {}, {}
    if a.mvtracker_repo:
        lk["hub_repo"] = a.mvtracker_repo
    if a.moge:
        lk["moge"] = a.moge
    if a.featup_repo:
        mk["featup_repo"] = a.featup_repo
    if a.mast3r_weights:
        mk["backbone_weights"] = a.mast3r_weights
    is_eval = a.cmd in ("evaluate", "eval")
    q = parsers["evaluate" if is_eval else "reference"]
    pred = None
    if is_eval:
        named = [x for x in a.pred if "=" in x.split("/", 1)[0]]
        if named and len(named) != len(a.pred):
            q.error("--pred: give either one prediction root or NAME=ROOT pairs")
        if named:
            try:
                pred = _parse_named(named, "--pred")
            except ValueError as e:
                q.error(_one_line(e))
        elif len(a.pred) == 1:
            pred = a.pred[0]
        else:
            q.error("--pred: several roots need names (NAME=ROOT)")
    if not Path(a.root).is_dir():
        q.error(f"--root {a.root}: not a directory (the VP-WorldBench data root)")
    try:
        protocol = resolve_protocol(a.protocol)
    except Exception as e:
        from .. import protocol as P
        q.error(f"--protocol {a.protocol}: {type(e).__name__}: {_one_line(e)} (expected a preset "
                f"{' | '.join(sorted(getattr(P, 'PRESETS', {'vpm-v1': None})))} or a yaml / json file)")
    try:
        subsets = parse_subsets(a.subsets)
    except KeyError as e:
        q.error(f"--subsets: {e.args[0] if e.args else e}")
    try:
        metrics = _parse_metrics(parse_list(a.metrics) or METRICS)
        parse_shard(a.shard)
        track_size = _parse_track_size(a.track_size)
        protocol_hash(protocol, track_size, a.decoder)
        _runs_of(pred, a.out if is_eval else None)
    except (ValueError, KeyError) as e:
        q.error(_one_line(e))
    try:
        res = evaluate(a.root, pred, a.out if is_eval else None, protocol, subsets=subsets, ids=parse_list(a.ids),
                       metrics=metrics, device=a.device, shard=a.shard, reference_dir=a.reference_dir,
                       overwrite=is_eval and a.overwrite, lpips=not (is_eval and a.no_lpips),
                       ssim=not (is_eval and a.no_ssim), met3r_batch=a.met3r_batch,
                       met3r_feature_chunk=a.met3r_feature_chunk, references=not (is_eval and a.no_references),
                       overwrite_reference=a.overwrite_reference, track_size=track_size, decoder=a.decoder,
                       lifter_kwargs=lk, met3r_kwargs=mk, i3d_weights=a.i3d_weights,
                       save_tracks=is_eval and a.save_tracks, raise_errors=a.raise_errors, verbose=not a.quiet)
    except (ImportError, UsageError) as e:
        if _traceback_requested():
            raise
        return _error_exit(e)
    print(json.dumps(res))
    if res.get("n_selected") == 0:
        print(f"error: {TASK}: no sample matches the selection (--subsets / --ids / clip_filter)", file=sys.stderr)
        return 1
    failed = failed_runs(res, isinstance(pred, dict))
    if failed:
        print(f"error: {TASK}: every evaluated sample failed ({'; '.join(failed)}); the per-sample \"reason\" is "
              f"in the records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from ..cli import main as _cli_main
    raise SystemExit(_cli_main(["generation"] + sys.argv[1:]))
