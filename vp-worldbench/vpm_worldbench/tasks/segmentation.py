"""Task ``segmentation``: promptable video object segmentation on the multi-view clips.

Task
----
Given the frames of a time window of one view and a prompt for ONE target object at the prompt frame t_p (k positive
clicks, a box, the mask, and/or a text description), predict a binary mask of the target in every frame of the window.
Each view of a sample is an independent video (the same target in all views). Two ground truths are scored:

    (a) dense   Real-General only: the SAM2.1-L actor masks of labels/human/masks.npz (PSEUDO-GT: video masks of the
                whole-body pipeline, checked against the visual hull of the other rig views). Frames / views whose
                ``mask_ok`` is False (or where SAM2 did not run) are not scored; the black undistortion border
                (``valid_px`` False) is DAVIS "void". Metrics J, F, J&F (`metrics.segmentation.j_and_f`, DAVIS
                definitions, boundary tolerance 0.008 x image diagonal = 11 px at 1024x768).
    (b) sparse  every subset: GT 3D track points projected into the view give point labels. Positives = visible points
                of the target; negatives = visible points of other tracked entities and (where GT depth exists)
                background pixels whose 3D point is far from every target point. The predicted mask is read at each
                labelled pixel; precision / recall / IoU / F1 / accuracy of the point labels
                (`metrics.segmentation.sample_mask` / `point_label_metrics` counting, pooled over views and frames).

Targets (``SegProtocol.targets``; default in DEFAULT_TARGETS)
    Simul-Robot    "primary_object": points of the vendor primary object (tracks3d ``is_primary``; the object the robot
                   picks and places, attributes.primary_object). Negative points = the other tracked objects + the
                   robot arms/hands. Alternative "robot": the robot arm + hand points (torso/legs are not tracked, so
                   no background negatives are used for it).
    Simul-General  "target": all points (the single tracked pedestrian / vehicle). Negatives = background pixels only.
    Real-Robot     "active_arm": arm + hand points of the side that acts first in the task instruction (single-hand
                   tasks: that hand; bimanual: the hand of the first step). Negatives = the other arm + hand.
                   Alternatives "left_arm" / "right_arm", and "arms" (both arms + hands; then there is no negative
                   label at all - no depth, no other tracked entity - so only recall is meaningful). No depth -> no
                   background negatives; visibility models robot self-occlusion only (scene objects never occlude), so
                   a few positives may sit on objects.
    Real-General   "person": the actor = whole-body MHR surface points (kind human_body) + orhtrack entities that lie
                   on the actor. An orhtrack entity (co-moving point group, kind dynamic_surface) is "person" when its
                   median 3D distance to the nearest valid whole-body point is <= 0.12 m AND >= 85 % of its visible
                   points fall inside the SAM2 actor mask (mask_ok frames, every 10th frame); "other" (negatives) when
                   the distance is >= 0.25 m AND <= 15 % fall inside; otherwise ambiguous (not labelled). Entities
                   with < 50 mask observations use the 3D rule alone with 0.08 m / 0.30 m. Pseudo-GT.

Point labels (sparse GT; per view and scored frame t)
    candidates  points with ``visible[t, :, view]`` that project inside the image (uv from `geometry.project` with the
                stored K / w2c: pixel centres at integers). Where GT depth exists the label is dropped when the
                depth at the point's pixel (floor(u + 0.5), floor(v + 0.5)) disagrees with the point depth by more than
                max(abs, rel * z) (Simul-Robot 2 cm / 1 %, Simul-General 5 cm / 2 %, Real-General 10 cm; contact and
                silhouette ambiguity of the visibility flags). Simul-* pixels without depth (sky) reject the label;
                Real-General holes keep it.
    background  (protocol.background: Simul-* and Real-General by default; Real-General uses its reconstruction depth
                because most of its clips have no other non-person tracked entity, so its point labels would have no
                negatives; turn it off with ``background={"Real-General": False}``) pixel centres of a regular grid
                (stride ``bg_stride_px`` native pixels) whose depth-unprojected 3D point is at least ``margin``
                away from every valid target point at t; margin = max(bg_margin_m, 2 x the 95th percentile
                nearest-neighbour spacing of the target points at t_p). Simul-*: exact rendered depth, sky pixels (no
                depth) are background.
                Real-General: reconstruction depth (PSEUDO-GT), holes are skipped and pixels inside the 2D box of the
                projected target points dilated by 5 % of the image diagonal are excluded (mixed silhouette pixels).
                A frame whose target has < 10 valid 3D points gets no background negatives.
    sampling    per view and frame at most ``max_pos`` positives and ``max_neg`` negatives (``neg_point_share`` of the
                negatives from labelled points, the rest background, each backfilled by the other), chosen with a
                seeded RNG (seed, uid, view, frame) -> roughly balanced classes: an all-ones mask scores ~50 %
                precision / IoU, an all-zeros mask 0 % recall. Counts are pooled over the scored frames and views of a
                sample. ``fpr_points`` / ``fpr_background`` = share of negative track points / background pixels
                predicted as target.

Protocol (`SegProtocol`, same conventions as `vpm_worldbench.protocol.Protocol`)
    window      GT frames ``t_p + j * frame_step``, j = 0..horizon (t_p = ``ref_frame`` = 16, horizon 64 -> 16..80);
                position 0 = the prompt frame (not scored); scored = positions 1..horizon. ``horizon=None`` = to the end
                of the clip. A sample is evaluable iff the window lies inside the clip; with the defaults every sample
                of every subset is.
    views       all views (or ``views``); a view is scored only if it has a prompt (>= 1 positive prompt point at t_p).
    dense grid  native 1024x768 (``dense_size=(W, H)`` resamples GT, void and prediction with nearest neighbour).
    per subset  ``targets`` / ``background`` / ``bg_margin_m`` are dicts keyed by subset; aliases ("rr", "real_robot",
                ...) are canonicalised (`constants.canonical_subset`), unknown subsets raise KeyError.
    files       `get_protocol` loads a JSON / YAML file of fields; a file without ``name`` is named after the file
                (its stem). Command-line window overrides add "+field=value" to the name (`common.override_protocol`).

Inputs (`sample_inputs` in Python, `export_inputs` on disk; the GT beyond the prompt is never exported)
    ``<out>/<Subset>/<id>/inputs.json``: target name + text description, prompt frame, window frame indices, fps and
    per view: ``video`` (path relative to the benchmark root; decode the 0-based frames ``frames``), ``size`` [W, H]
    and the prompt at the prompt frame in native pixels (pixel centres at integers):
        points  up to 5 positive clicks [[u, v], ...] in farthest-point order (the k-click prompt = the first k;
                protocol default k = ``n_points`` = 3); sampled deterministically from the visible GT track points of
                the target (Real-General: those inside the GT mask when it is usable; mask pixels if no point is).
        box     [x0, y0, x1, y1] inclusive pixel bounds: of the GT mask (Real-General, mask usable) or of the prompt
                candidate points (other subsets; a modal box of the visible target points, approximate).
        mask    ``prompt/view{k}.png`` (uint8 0/255, native size): the GT actor mask (Real-General only, when usable).
    ``frames="png"`` additionally writes ``frames/view{k}/frame_{t:04d}.png`` for the window. ``manifest.jsonl``: one
    line per sample. Resumable (existing inputs.json are kept unless ``overwrite``); an out dir written with another
    protocol / frame option (``export.json``) is refused unless ``overwrite``.

Prediction file (one per sample and view): ``<pred_root>/<Subset>/<id>/view{k}.npz``
    ``masks``  (n, H, W) bool (uint8 / int: != 0, float: > 0.5), n = window length, row j = GT frame
               ``window[j]`` (row 0 = prompt frame, ignored). Or ``masks_packbits`` (n, H, ceil(W/8)) uint8 =
               np.packbits(masks, axis=-1) plus ``width`` (W). Optional ``frames`` (n,) int: the GT frame index of
               every row (any subset / order; scored frames without a row count as empty masks).
    Any resolution with the full field of view of the stored video: masks are compared with nearest-neighbour
    resampling with pixel centres (dense: the prediction is resized to the GT grid, source pixel
    floor((x + 0.5) * w / W); sparse: each label (u, v) is read at ((u + 0.5) * w / W - 0.5, ...) of the prediction).
    Checked (`check_prediction`; a failing view, like an unreadable file, makes the sample status "invalid" with the
    reason): the mask aspect ratio must be within 2 % of the video's (catches (n, W, H) instead of (n, H, W)),
    ``frames`` must not list a frame twice, and without ``frames`` there must be >= one row per window frame (extra
    rows and frames outside the window are ignored and counted as ``n_extra_rows``). Scored frames without a row count
    as empty masks (``n_missing_frames`` = distinct such frames per view).
    Views of a sample without a file count as empty masks when at least one view was submitted (so every prompted view
    must be predicted, also when its prompt has no mask: use its points or box); a sample without any file is
    "missing" (counted, not scored). `write_prediction` writes a valid (packed) file.

Metrics (per sample: mean over its scored views for dense; pooled counts for sparse; then mean ± s.e. over samples
    per subset) dense ``J``, ``F``, ``J&F``, ``J_recall``, ``F_recall``, ``J_decay``, ``F_decay`` (percent; DAVIS per
    view sequence over the scored ok frames); sparse ``precision``, ``recall``, ``iou``, ``f1``, ``accuracy``,
    ``fpr_points``, ``fpr_background`` (percent; NaN where undefined, e.g. precision of an all-empty prediction or
    fpr_points without negative points; NaN samples are left out of that metric's mean and its n is reported), and
    counts. ``summary.json`` also has ``sparse_pooled`` (counts pooled over all samples of the subset).
    GT check (Real-General): on mask_ok frames 99 % of the visible whole-body points fall inside the actor mask
    (1st percentile of frames 91 %; 2 of ~10.8k frames below 80 %).

Ground truth per subset
    Simul-Robot    exact synthetic tracks + rendered depth (sparse).
    Simul-General  exact synthetic tracks + rendered depth (sparse; background negatives only).
    Real-Robot     forward kinematics + calibration (~2 px; ego extrinsic is a manual fit), self-occlusion-only
                   visibility (sparse; the other arm as negatives).
    Real-General   PSEUDO-GT throughout: SAM2.1 actor masks (dense), whole-body + orhtrack tracks and the
                   reconstruction depth (sparse).

Baselines (`run_baseline` / BASELINES)
    oracle        Real-General: the GT actor masks where they are usable (ran and mask_ok), the label raster in the
                  other frames / views (dense J&F = 100; the sparse score then measures the agreement of the two
                  pseudo-GTs, mostly on the usable frames); other subsets: ``label_raster``.
    label_raster  the positive point labels rasterised as single pixels (sparse ~100 %; checks the pipeline).
    prompt_static the prompt held fixed: the GT mask at t_p (Real-General) or the filled prompt box, repeated (the
                  zero-motion baseline).
    empty / full  all False / all True.

Results: ``evaluate`` writes ``<out>/<Subset>/<id>.json`` per sample (status ok / skipped / missing / invalid / error;
`common.STATUSES`) and calls ``summarize`` -> ``<out>/summary.json`` and ``<out>/summary.md`` (mean ± s.e. over
samples per subset; s.e. = std(ddof=1)/sqrt(n); `common.mean_se`, as in every task). NaN is written as null.
``run.json`` records the prediction folder and protocol: another run into the same out dir is refused
(`common.RunDirError`) unless ``overwrite`` (which first deletes the old per-sample results); ``summary.json`` records
the prediction folder as ``pred_root`` and counts the failed samples per subset (``n_invalid`` / ``n_error``).

    from vpm_worldbench import Benchmark
    from vpm_worldbench.tasks import segmentation as seg
    bench = Benchmark("/data/VP-WorldBench")
    p = seg.SegProtocol()
    for s in bench.samples("Real-General"):
        inp = seg.sample_inputs(s, p)                     # prompts per view
        for k, v in inp["views"].items():
            if v["status"] != "ok":                       # target not visible at the prompt frame: not scored
                continue
            frames = s.frames(int(k[4:]), inp["frames"])  # (n, H, W, 3)
            masks = my_vos(frames, v["prompt"]["points"][:p.n_points])
            seg.write_prediction("preds", s, int(k[4:]), masks)
    seg.evaluate(bench, "preds", "results/seg", p)

    vpmwb segmentation export   --root R --out inputs/seg [--frames png] [--subsets rg sr] [--overwrite]
    vpmwb segmentation baseline --root R --pred preds/static --kind prompt_static
    vpmwb segmentation evaluate --root R --pred preds/static --out results/static [--labels-cache cache/] [--jobs 16]
    vpmwb segmentation summarize --out results/static
(also `python -m vpm_worldbench.tasks.segmentation ...`; --subsets / --ids take comma- and/or space-separated
values, subset aliases accepted, also in uids; --ref-frame / --horizon (<= 0: to the clip end) / --frame-step
override the window). In Python, ``run_baseline(bench, pred_root, kind="empty", ...)`` as in every task. Exit status:
1 when the selection matches no sample or no evaluable sample is scored (every one missing / invalid / error); 2 for
a usage or setup error, e.g. a malformed --dense-size or an out dir of another run without --overwrite (one line).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import SUBSETS, canonical_subset
from ..geometry import inside, project
from ..metrics.segmentation import j_and_f, point_label_metrics, sample_mask
from .common import (InvalidPrediction, baseline_args, canonical_status, check_out_stamp, cli_protocol, cli_size,
                     cli_subsets, evaluation_exit_status, file_protocol_fields, override_protocol, read_npz,
                     run_pred_root, run_status, select_samples, selection_exit_status)
from .common import mean_se as _common_mean_se

TASK = "segmentation"
SCHEMA = "vpm-worldbench/segmentation/1.0"

DEFAULT_TARGETS = {"Simul-Robot": "primary_object", "Simul-General": "target", "Real-Robot": "active_arm",
                   "Real-General": "person"}
TARGET_CHOICES = {"Simul-Robot": ("primary_object", "robot"), "Simul-General": ("target",),
                  "Real-Robot": ("active_arm", "left_arm", "right_arm", "arms"), "Real-General": ("person",)}
DEFAULT_BACKGROUND = {"Simul-Robot": True, "Simul-General": True, "Real-Robot": False, "Real-General": True}
DEFAULT_BG_MARGIN_M = {"Simul-Robot": 0.03, "Simul-General": 0.30, "Real-Robot": 0.30, "Real-General": 0.30}
BG_EXCLUSION_FRAC = {"Simul-Robot": 0.0, "Simul-General": 0.0, "Real-Robot": 0.0, "Real-General": 0.05}
DEPTH_CHECK = {"Simul-Robot": (0.02, 0.01), "Simul-General": (0.05, 0.02), "Real-Robot": None,
               "Real-General": (0.10, 0.0)}
SKY_IS_BACKGROUND = {"Simul-Robot": True, "Simul-General": True, "Real-Robot": False, "Real-General": False}
ENTITY_PERSON_MAX_M, ENTITY_OTHER_MIN_M = 0.12, 0.25
ENTITY_PERSON_MIN_IN, ENTITY_OTHER_MAX_IN = 0.85, 0.15
ENTITY_3D_ONLY = (0.08, 0.30)
ENTITY_MIN_MASK_OBS = 50

BG_MIN_TARGET_POINTS = 10
SRC_POS, SRC_NEG_POINT, SRC_BACKGROUND = 0, 1, 2
DENSE_KEYS = ("J", "F", "J&F", "J_recall", "F_recall", "J_decay", "F_decay")
SPARSE_KEYS = ("precision", "recall", "iou", "f1", "accuracy", "fpr_points", "fpr_background")
_NAN = float("nan")


@dataclass
class SegProtocol:
    name: str = "vpm-seg-v1"
    ref_frame: int = 16
    horizon: Optional[int] = 64
    frame_step: int = 1
    views: Optional[list] = None
    targets: dict = field(default_factory=dict)
    prompt: str = "points"
    n_points: int = 3
    n_points_export: int = 5
    min_prompt_points: int = 3
    min_prompt_area_px: int = 64
    dense_void_border: bool = True
    dense_size: Optional[tuple] = None
    max_pos: int = 64
    max_neg: int = 64
    neg_point_share: float = 0.5
    sparse_frame_step: int = 2
    background: dict = field(default_factory=dict)
    bg_stride_px: int = 32
    bg_margin_m: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        if self.ref_frame < 0:
            raise ValueError("ref_frame must be >= 0")
        if self.frame_step < 1 or self.sparse_frame_step < 1:
            raise ValueError("frame_step and sparse_frame_step must be >= 1")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("horizon must be >= 1 (or None)")
        if self.prompt not in ("points", "box", "mask", "text"):
            raise ValueError("prompt must be points | box | mask | text")
        if self.dense_size is not None:
            try:
                self.dense_size = tuple(int(x) for x in self.dense_size)
                ok = len(self.dense_size) == 2 and min(self.dense_size) >= 1
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"dense_size must be (W, H) with W, H >= 1 (or None), got {self.dense_size!r}")
        for name in ("targets", "background", "bg_margin_m"):
            d = {}
            for k, v in dict(getattr(self, name) or {}).items():
                c = canonical_subset(k)
                if c in d:
                    raise ValueError(f"{name}: subset {c!r} given twice")
                d[c] = v
            setattr(self, name, d)
        for sub, t in self.targets.items():
            if t not in TARGET_CHOICES[sub]:
                raise ValueError(f"target {t!r} not available for {sub}: {TARGET_CHOICES[sub]}")

    def target_for(self, subset: str) -> str:
        subset = canonical_subset(subset)
        return self.targets.get(subset, DEFAULT_TARGETS[subset])

    def background_for(self, subset: str) -> bool:
        subset = canonical_subset(subset)
        return bool(self.background.get(subset, DEFAULT_BACKGROUND[subset]))

    def bg_margin_for(self, subset: str) -> float:
        subset = canonical_subset(subset)
        return float(self.bg_margin_m.get(subset, DEFAULT_BG_MARGIN_M[subset]))

    def window(self, sample) -> list:
        """GT frame indices of the window: [t_p, t_p + step, ...]; position 0 = prompt frame."""
        if self.horizon is None:
            n = (sample.n_frames - 1 - self.ref_frame) // self.frame_step
        else:
            n = self.horizon
        return [self.ref_frame + j * self.frame_step for j in range(n + 1)]

    def evaluable(self, sample) -> tuple:
        if self.ref_frame < 0:
            return False, "ref_frame < 0"
        w = self.window(sample)
        if len(w) < 2:
            return False, "window has no scored frame"
        if w[-1] > sample.n_frames - 1:
            return False, f"window ends at {w[-1]} > last frame {sample.n_frames - 1}"
        if self.views is not None and max(self.views) >= sample.n_views:
            return False, f"view {max(self.views)} >= n_views {sample.n_views}"
        return True, ""

    def views_for(self, sample) -> list:
        return list(range(sample.n_views)) if self.views is None else [int(v) for v in self.views]

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["dense_size"] is not None:
            d["dense_size"] = list(d["dense_size"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SegProtocol":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown segmentation protocol fields {sorted(unknown)}")
        return cls(**d)


def get_protocol(spec=None) -> SegProtocol:
    """None -> defaults; a SegProtocol; a dict of fields; or a path to a .json / .yaml file (a file without "name" is
    named after the file)."""
    if spec is None:
        return SegProtocol()
    if isinstance(spec, SegProtocol):
        return spec
    if isinstance(spec, dict):
        return SegProtocol.from_dict(spec)
    p = Path(spec)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return SegProtocol.from_dict(file_protocol_fields(yaml.safe_load(text), p))
    return SegProtocol.from_dict(file_protocol_fields(json.loads(text), p))


def _rng(*keys) -> np.random.Generator:
    """Deterministic generator from arbitrary keys (sha256, independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def display_name(object_id: str) -> str:
    """Human-readable name of a Simul-Robot object id ('small_yellow_cube' -> 'small yellow cube')."""
    special = {"square_big_box": "large square box", "red_rimmed_plate": "red-rimmed plate",
               "rubiks_puzzle_cube": "Rubik's puzzle cube", "bbq_sauce_bottle": "BBQ sauce bottle"}
    return special.get(object_id, object_id.replace("_", " "))


def _centre_index(n_dst: int, n_src: int) -> np.ndarray:
    """Source index of every target pixel: floor((i + 0.5) * n_src / n_dst) in exact integer arithmetic (the
    pixel-centre rule of `geometry.warp_image(interp="nearest")`)."""
    return ((2 * np.arange(n_dst, dtype=np.int64) + 1) * n_src) // (2 * n_dst)


def resize_mask_nearest(mask: np.ndarray, size) -> np.ndarray:
    """Nearest-neighbour resampling of a mask (H, W) or (n, H, W) to size = (W', H') with pixel centres: target pixel
    x takes source pixel floor((x + 0.5) * W / W') (exact integer arithmetic, the rule of
    `geometry.warp_image(interp="nearest")`)."""
    m = np.asarray(mask)
    W2, H2 = int(size[0]), int(size[1])
    H, W = m.shape[-2:]
    if (W, H) == (W2, H2):
        return m
    return m[..., _centre_index(H2, H), :][..., _centre_index(W2, W)]


def _as_bool(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == bool:
        return a
    if np.issubdtype(a.dtype, np.floating):
        with np.errstate(invalid="ignore"):
            return a > 0.5
    return a != 0


def _unpack(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=int(width)).astype(bool)


def _fps(points: np.ndarray, k: int, start: int) -> list:
    """Farthest-point order (indices) of points (M, 2), starting with index `start`."""
    if len(points) == 0 or k <= 0:
        return []
    chosen = [int(start)]
    d = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(k, len(points)):
        i = int(np.argmax(d))
        if d[i] <= 0:
            break
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return chosen


def _nearest_to_median(uv: np.ndarray) -> int:
    med = np.median(uv, axis=0)
    return int(np.argmin(np.linalg.norm(uv - med, axis=1)))


def _pixel(uv: np.ndarray, size):
    """OpenCV pixel-centre nearest pixel (iu, iv) clipped to the grid."""
    W, H = size
    iu = np.clip(np.floor(uv[..., 0] + 0.5).astype(np.int64), 0, W - 1)
    iv = np.clip(np.floor(uv[..., 1] + 0.5).astype(np.int64), 0, H - 1)
    return iu, iv


class ActorMasks:
    """Real-General SAM2.1 actor masks (labels/human/masks.npz, PSEUDO-GT) on the undistorted 1024x768 video grid.
    Packed with np.packbits along x (bit order big); unpacked lazily per (frame, view)."""

    def __init__(self, sample):
        with np.load(sample.label_path("masks"), allow_pickle=False) as z:
            self.actor = z["actor"]
            self.other = z["other"]
            self.ran = z["ran"].astype(bool)
            self.mask_ok = z["mask_ok"].astype(bool)
            self.width, self.height = (int(x) for x in z["image_size"])
            self._valid_px = z["valid_px"]
        for k in range(self.actor.shape[1]):
            if tuple(sample.image_size(k)) != (self.width, self.height):
                raise ValueError(f"{sample.uid}: mask grid {self.width}x{self.height} != view{k} video size")

    def usable(self, t: int, v: int) -> bool:
        return bool(self.ran[t, v] and self.mask_ok[t, v])

    def get(self, t: int, v: int) -> np.ndarray:
        """(H, W) bool actor mask of frame t, view v."""
        return _unpack(self.actor[t, v], self.width)

    def valid_px(self, v: int) -> np.ndarray:
        return _unpack(self._valid_px[v], self.width)


def load_actor_masks(sample) -> ActorMasks | None:
    """The Real-General actor masks of a sample, or None when it has no labels/human/masks.npz."""
    rel = sample.meta["files"].get("labels", {}).get("masks")
    if sample.subset != "Real-General" or rel is None or not (sample.dir / rel).exists():
        return None
    return ActorMasks(sample)


def _rr_active_side(sample) -> int:
    hand = str(sample.attributes.get("hand", "")).lower()
    if hand in ("left", "right"):
        return 0 if hand == "left" else 1
    steps = (sample.captions.get("structured") or {}).get("steps") or []
    if steps and steps[0].get("hands"):
        return 0 if steps[0]["hands"][0] == "left" else 1
    raise ValueError(f"{sample.uid}: cannot resolve the active arm")


def rg_entity_labels(sample, masks: ActorMasks | None = None) -> dict:
    """Classify the orhtrack entities of a Real-General sample as person / other / ambiguous (module docstring).
    Returns {entity_id: {"label", "d_med_m", "in_actor", "n_obs", "n_points"}}."""
    from scipy.spatial import cKDTree
    tr = sample.tracks
    ent = tr.extra.get("entity")
    if ent is None:
        return {}
    hb = tr.kind == list(map(str, tr.kind_names)).index("human_body")
    if masks is None:
        masks = load_actor_masks(sample)
    T = sample.n_frames
    out = {}
    uv_cache = {}
    for e in sorted(int(x) for x in np.unique(ent[ent >= 0])):
        m = ent == e
        ds = []
        for t in range(0, T, 5):
            hv = hb & tr.valid[t]
            ev = m & tr.valid[t]
            if hv.sum() < 20 or not ev.any():
                continue
            d, _ = cKDTree(tr.xyz[t, hv]).query(tr.xyz[t, ev])
            ds.append(d)
        d_med = float(np.median(np.concatenate(ds))) if ds else _NAN
        n_in = n_obs = 0
        for v in range(sample.n_views if masks is not None else 0):
            for t in range(0, T, 10):
                if not masks.usable(t, v):
                    continue
                sel = m & tr.visible[t, :, v]
                if not sel.any():
                    continue
                key = (t, v)
                if key not in uv_cache:
                    uv_cache[key] = (project(tr.xyz[t], sample.cameras.K[v], sample.cameras.w2c[v, t])[0],
                                     masks.get(t, v))
                uv, A = uv_cache[key]
                val, ins = sample_mask(A, uv[sel])
                n_in += int((val & ins).sum())
                n_obs += int(ins.sum())
        frac = n_in / n_obs if n_obs else _NAN
        if n_obs >= ENTITY_MIN_MASK_OBS and np.isfinite(d_med):
            if d_med <= ENTITY_PERSON_MAX_M and frac >= ENTITY_PERSON_MIN_IN:
                lab = "person"
            elif d_med >= ENTITY_OTHER_MIN_M and frac <= ENTITY_OTHER_MAX_IN:
                lab = "other"
            else:
                lab = "ambiguous"
        elif np.isfinite(d_med):
            lab = "person" if d_med <= ENTITY_3D_ONLY[0] else ("other" if d_med >= ENTITY_3D_ONLY[1] else "ambiguous")
        else:
            lab = "ambiguous"
        out[e] = {"label": lab, "d_med_m": d_med, "in_actor": frac, "n_obs": n_obs, "n_points": int(m.sum())}
    return out


def resolve_target(sample, protocol: SegProtocol | None = None, masks: ActorMasks | None = None) -> dict:
    """The target of a sample: {"name", "text", "pos" (N,) bool target points, "neg" (N,) bool labelled non-target
    points, "background" bool, "info"}. Points neither pos nor neg are never labelled."""
    p = protocol or SegProtocol()
    sub = sample.subset
    name = p.target_for(sub)
    tr = sample.tracks
    N = tr.n_points
    groups = [str(g) for g in tr.group_names]
    info = {}
    bg = p.background_for(sub)
    if sub == "Simul-Robot":
        if name == "primary_object":
            pos = np.asarray(tr.extra["is_primary"], bool)
            prim = str(sample.attributes.get("primary_object", ""))
            text = f"the {display_name(prim)}" if prim else "the object the robot picks up"
            info["object"] = prim
        else:
            pos = tr.group == groups.index("robot")
            text = "the robot's arms and hands"
            bg = False
        neg = ~pos
    elif sub == "Simul-General":
        pos = np.ones(N, bool)
        neg = np.zeros(N, bool)
        cat = str(sample.attributes.get("category", "target"))
        text = "the pedestrian followed by the cameras" if cat == "pedestrian" else "the vehicle followed by the cameras"
    elif sub == "Real-Robot":
        if name == "arms":
            pos = np.ones(N, bool)
            neg = np.zeros(N, bool)
            text = "the robot's arms and hands"
        else:
            side = _rr_active_side(sample) if name == "active_arm" else (0 if name == "left_arm" else 1)
            pos = tr.side == side
            neg = tr.side == 1 - side
            text = f"the robot's {'left' if side == 0 else 'right'} arm and hand"
            info["side"] = "left" if side == 0 else "right"
    elif sub == "Real-General":
        hb = tr.kind == list(map(str, tr.kind_names)).index("human_body")
        ent = tr.extra.get("entity", np.full(N, -1))
        labels = rg_entity_labels(sample, masks)
        pos = hb.copy()
        neg = np.zeros(N, bool)
        for e, r in labels.items():
            if r["label"] == "person":
                pos |= ent == e
            elif r["label"] == "other":
                neg |= ent == e
        text = "the main person (the actor performing the motion)"
        info["entities"] = {str(e): r for e, r in labels.items()}
    else:
        raise KeyError(sub)
    if not pos.any():
        raise ValueError(f"{sample.uid}: target {name!r} has no points")
    return {"name": name, "text": text, "pos": pos, "neg": neg & ~pos, "background": bool(bg) and sample.has_depth,
            "info": info}


def _margin(sample, target, protocol) -> float:
    """Background margin (m): max(base, 2 x p95 nearest-neighbour spacing of the target points at t_p)."""
    from scipy.spatial import cKDTree
    tr = sample.tracks
    t = min(max(protocol.ref_frame, 0), sample.n_frames - 1)
    pts = tr.xyz[t, target["pos"] & tr.valid[t]]
    base = protocol.bg_margin_for(sample.subset)
    if len(pts) < 3:
        return base
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(max(base, 2.0 * np.percentile(d[:, 1], 95)))


class _Frame:
    """Per (view, frame) projections / depth shared by prompts and labels."""

    def __init__(self, sample, target, view: int, t: int, need_depth: bool):
        tr = sample.tracks
        self.size = sample.image_size(view)
        self.K = sample.cameras.K[view]
        self.w2c = sample.cameras.w2c[view, t]
        uv, z = project(tr.xyz[t], self.K, self.w2c)
        self.uv, self.z = uv, z
        ok = tr.visible[t, :, view] & tr.valid[t] & (z > 0) & inside(uv, self.size)
        self.depth = sample.depth(view, [t])[0] if (need_depth and sample.has_depth) else None
        tol = DEPTH_CHECK[sample.subset]
        if self.depth is not None and tol is not None:
            iu, iv = _pixel(np.where(np.isfinite(uv), uv, 0.0), self.size)
            D = self.depth[iv, iu]
            zz = np.where(np.isfinite(z), z, 0.0)
            bad = np.abs(D - zz) > np.maximum(tol[0], tol[1] * zz)
            if SKY_IS_BACKGROUND[sample.subset]:
                bad |= D <= 0
            else:
                bad &= D > 0
            ok &= ~bad
        self.labelled = ok
        self.pos = ok & target["pos"]
        self.neg = ok & target["neg"]


def _background(sample, target, fr: _Frame, t: int, margin: float, stride: int) -> np.ndarray:
    """Background pixel candidates (M, 2) float (pixel centres of a stride-`stride` grid) for one view / frame."""
    from scipy.spatial import cKDTree
    if fr.depth is None:
        return np.zeros((0, 2))
    W, H = fr.size
    tr = sample.tracks
    us = np.arange(stride // 2, W, stride)
    vs = np.arange(stride // 2, H, stride)
    U, Vv = np.meshgrid(us, vs)
    U, Vv = U.ravel(), Vv.ravel()
    D = fr.depth[Vv, U].astype(np.float64)
    tgt = tr.xyz[t, target["pos"] & tr.valid[t]].astype(np.float64)
    if len(tgt) < BG_MIN_TARGET_POINTS:
        return np.zeros((0, 2))
    keep = np.zeros(len(U), bool)
    has = D > 0
    if has.any():
        Kinv = np.linalg.inv(fr.K)
        rays = np.stack([U[has], Vv[has], np.ones(has.sum())], 0).astype(np.float64)
        Xc = (Kinv @ rays) * D[has]
        c2w = np.linalg.inv(fr.w2c)
        Xw = (c2w[:3, :3] @ Xc).T + c2w[:3, 3]
        d, _ = cKDTree(tgt).query(Xw)
        keep[has] = d >= margin
    if SKY_IS_BACKGROUND[sample.subset]:
        keep |= ~has
    frac = BG_EXCLUSION_FRAC[sample.subset]
    if frac > 0:
        uv_t, z_t = project(tgt, fr.K, fr.w2c)
        okp = (z_t > 0) & np.isfinite(uv_t).all(-1)
        if okp.any():
            pad = frac * math.hypot(W, H)
            lo = uv_t[okp].min(0) - pad
            hi = uv_t[okp].max(0) + pad
            inbox = (U >= lo[0]) & (U <= hi[0]) & (Vv >= lo[1]) & (Vv <= hi[1])
            keep &= ~inbox
    return np.stack([U[keep], Vv[keep]], -1).astype(np.float64)


def sparse_labels(sample, protocol: SegProtocol | None = None, target: dict | None = None) -> dict:
    """Point labels of every scored frame (positions 1.. of the window, every ``sparse_frame_step``) and view.

    Returns {"views": {k: {"frame" (L,) int32 GT frame, "uv" (L, 2) float64 native pixels, "label" (L,) bool,
    "source" (L,) int8 (0 target point, 1 non-target point, 2 background pixel)}}, "margin_m", "target", "info"}.
    """
    p = protocol or SegProtocol()
    target = target or resolve_target(sample, p)
    margin = _margin(sample, target, p) if target["background"] else _NAN
    win = p.window(sample)
    frames = win[1::p.sparse_frame_step] if p.sparse_frame_step > 1 else win[1:]
    need_depth = target["background"] or DEPTH_CHECK[sample.subset] is not None
    out = {}
    for k in p.views_for(sample):
        cols = {"frame": [], "uv": [], "label": [], "source": []}
        for t in frames:
            fr = _Frame(sample, target, k, t, need_depth)
            rng = _rng(p.seed, sample.uid, k, t)
            pi = rng.permutation(np.nonzero(fr.pos)[0])[: p.max_pos]
            ni = rng.permutation(np.nonzero(fr.neg)[0])
            bgc = (_background(sample, target, fr, t, margin, int(p.bg_stride_px)) if target["background"]
                   else np.zeros((0, 2)))
            bgc = bgc[rng.permutation(len(bgc))]
            k_pts = min(len(ni), int(round(p.max_neg * p.neg_point_share)))
            k_bg = min(len(bgc), p.max_neg - k_pts)
            k_pts = min(len(ni), p.max_neg - k_bg)
            uv = np.concatenate([fr.uv[pi], fr.uv[ni[:k_pts]], bgc[:k_bg]], 0)
            lab = np.concatenate([np.ones(len(pi), bool), np.zeros(k_pts + k_bg, bool)])
            src = np.concatenate([np.full(len(pi), SRC_POS), np.full(k_pts, SRC_NEG_POINT),
                                  np.full(k_bg, SRC_BACKGROUND)]).astype(np.int8)
            cols["frame"].append(np.full(len(uv), t, np.int32))
            cols["uv"].append(uv.reshape(-1, 2))
            cols["label"].append(lab)
            cols["source"].append(src)
        out[k] = {c: (np.concatenate(v) if v else np.zeros((0, 2) if c == "uv" else 0)) for c, v in cols.items()}
        out[k]["frame"] = out[k]["frame"].astype(np.int32)
        out[k]["source"] = out[k]["source"].astype(np.int8)
        out[k]["label"] = out[k]["label"].astype(bool)
    return {"views": out, "margin_m": margin, "target": target["name"], "info": target["info"]}


def _labels_cfg_hash(sample, protocol: SegProtocol) -> str:
    keys = ("ref_frame", "horizon", "frame_step", "views", "max_pos", "max_neg", "neg_point_share",
            "sparse_frame_step", "bg_stride_px", "seed")
    d = {k: getattr(protocol, k) for k in keys}
    d.update(target=protocol.target_for(sample.subset), bg=protocol.background_for(sample.subset),
             margin=protocol.bg_margin_for(sample.subset), schema=SCHEMA,
             data=sample.meta.get("files_sha256", {}).get(sample.meta["files"]["tracks3d"]))
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_sparse_labels(sample, protocol: SegProtocol, cache_dir=None, target: dict | None = None) -> dict:
    """`sparse_labels` with an optional npz cache ``<cache_dir>/<Subset>/<id>.npz`` (keyed by the protocol)."""
    if cache_dir is None:
        return sparse_labels(sample, protocol, target)
    path = Path(cache_dir) / sample.subset / f"{sample.id}.npz"
    h = _labels_cfg_hash(sample, protocol)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["cfg_hash"]) == h:
                    meta = json.loads(str(z["meta"]))
                    views = {}
                    for k in meta["views"]:
                        views[int(k)] = {c: z[f"v{k}_{c}"] for c in ("frame", "uv", "label", "source")}
                    return {"views": views, "margin_m": meta["margin_m"], "target": meta["target"],
                            "info": meta["info"]}
        except Exception:
            pass
    lab = sparse_labels(sample, protocol, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"v{k}_{c}": v[c] for k, v in lab["views"].items() for c in ("frame", "uv", "label", "source")}
    meta = {"views": [int(k) for k in lab["views"]], "margin_m": lab["margin_m"], "target": lab["target"],
            "info": jsonable(lab["info"])}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, cfg_hash=np.array(h), meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return lab


def sample_prompts(sample, protocol: SegProtocol | None = None, target: dict | None = None,
                   masks: ActorMasks | None = None) -> dict:
    """Prompts at the prompt frame for every view: {k: {"ok", "points" (M, 2) float, "points_source", "box"
    [x0, y0, x1, y1] | None, "box_source", "mask" (H, W) bool | None, "dense_ok", "reason"}} (native pixels)."""
    p = protocol or SegProtocol()
    t = p.ref_frame
    rg = sample.subset == "Real-General"
    if rg and masks is None:
        masks = load_actor_masks(sample)
    target = target or resolve_target(sample, p, masks)
    out = {}
    for k in p.views_for(sample):
        fr = _Frame(sample, target, k, t, DEPTH_CHECK[sample.subset] is not None)
        cand = fr.uv[fr.pos]
        gt = None
        dense_ok = False
        if rg and masks is not None and masks.usable(t, k):
            gt = masks.get(t, k)
            dense_ok = int(gt.sum()) >= p.min_prompt_area_px
            if not dense_ok:
                gt = None
        if gt is not None and len(cand):
            val, _ = sample_mask(gt, cand)
            cand = cand[val]
        if len(cand) < p.min_prompt_points:
            cand = cand[:0]
        rec = {"ok": False, "points": np.zeros((0, 2)), "points_source": None, "box": None, "box_source": None,
               "mask": gt, "dense_ok": dense_ok, "reason": ""}
        if len(cand):
            order = _fps(cand, p.n_points_export, _nearest_to_median(cand))
            rec["points"] = cand[order]
            rec["points_source"] = "track_points"
        elif gt is not None:
            import cv2
            dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
            vv, uu = np.nonzero(gt)
            sub = np.stack([uu, vv], -1).astype(np.float64)[::4]
            start = int(np.argmax(dist[vv, uu][::4]))
            rec["points"] = sub[_fps(sub, p.n_points_export, start)]
            rec["points_source"] = "mask_pixels"
        if gt is not None:
            vv, uu = np.nonzero(gt)
            rec["box"] = [int(uu.min()), int(vv.min()), int(uu.max()), int(vv.max())]
            rec["box_source"] = "gt_mask"
        elif len(cand):
            iu, iv = _pixel(cand, fr.size)
            rec["box"] = [int(iu.min()), int(iv.min()), int(iu.max()), int(iv.max())]
            rec["box_source"] = "track_points"
        rec["ok"] = len(rec["points"]) > 0
        if not rec["ok"]:
            rec["reason"] = f"fewer than {p.min_prompt_points} visible target points at the prompt frame"
        out[k] = rec
    return out


def sample_inputs(sample, protocol: SegProtocol | None = None) -> dict:
    """Everything a model receives for one sample (no GT beyond the prompts; pixel coordinates are native). The
    inputs.json content plus ``views[name]["mask_array"]``: the (H, W) bool prompt mask or None (in memory only;
    `export_inputs` writes it as prompt/view{k}.png and drops the array)."""
    p = protocol or SegProtocol()
    masks = load_actor_masks(sample)
    target = resolve_target(sample, p, masks)
    pr = sample_prompts(sample, p, target, masks)
    views = {}
    for k, r in pr.items():
        views[f"view{k}"] = {
            "video": f"{sample.subset}/{sample.id}/{sample.meta['views'][k]['video']}",
            "size": list(sample.image_size(k)), "status": "ok" if r["ok"] else "no_prompt",
            "prompt": {"points": np.round(r["points"], 3).tolist(), "points_source": r["points_source"],
                       "box": r["box"], "box_source": r["box_source"],
                       "mask": None if r["mask"] is None else f"prompt/view{k}.png"},
            "mask_array": r["mask"]}
    return {"schema": SCHEMA, "task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id,
            "target": {"name": target["name"], "text": target["text"]}, "prompt_frame": p.ref_frame,
            "frames": p.window(sample), "fps": sample.fps,
            "tracks": ["dense", "sparse"] if masks is not None else ["sparse"],
            "views": views, "protocol": p.to_dict()}


def _export_one(args) -> dict:
    root, subset, sid, out_dir, pdict, frames_mode, overwrite = args
    import cv2
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = SegProtocol.from_dict(pdict)
    ok, why = p.evaluable(s)
    if not ok:
        return {"uid": s.uid, "status": "skipped", "reason": why}
    done = Path(out_dir) / s.subset / s.id / "inputs.json"
    if done.exists() and not overwrite:
        prev = json.loads(done.read_text())
        return {"uid": s.uid, "status": "ok", "reused": True,
                "n_views_with_prompt": sum(v["status"] == "ok" for v in prev["views"].values())}
    try:
        inp = sample_inputs(s, p)
    except ValueError as e:
        return {"uid": s.uid, "status": "skipped", "reason": str(e)}
    d = Path(out_dir) / s.subset / s.id
    d.mkdir(parents=True, exist_ok=True)
    for name, v in inp["views"].items():
        m = v.pop("mask_array")
        if m is not None:
            (d / "prompt").mkdir(exist_ok=True)
            cv2.imwrite(str(d / v["prompt"]["mask"]), m.astype(np.uint8) * 255)
        if frames_mode == "png":
            k = int(name[4:])
            fd = d / "frames" / name
            fd.mkdir(parents=True, exist_ok=True)
            for t, f in zip(inp["frames"], s.frames(k, inp["frames"])):
                cv2.imwrite(str(fd / f"frame_{t:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    write_json(d / "inputs.json", inp)
    return {"uid": s.uid, "status": "ok", "n_views_with_prompt": sum(v["status"] == "ok" for v in inp["views"].values())}


def export_inputs(bench, out_dir, protocol: SegProtocol | None = None, subsets=None, ids=None, frames: str = "none",
                  jobs: int = 1, overwrite: bool = False) -> dict:
    """Write ``<out_dir>/<Subset>/<id>/inputs.json`` (+ prompt masks, optional frames) and ``manifest.jsonl``.
    Resumable: samples whose inputs.json exists are kept unless ``overwrite``; an out dir written with another
    protocol or frame option is refused (`common.RunDirError`) unless ``overwrite`` (then every sample is
    rewritten)."""
    if frames not in ("none", "png"):
        raise ValueError("frames must be 'none' or 'png'")
    p = protocol or SegProtocol()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_out_stamp(out_dir, "export.json", {"task": TASK, "protocol": p.to_dict(), "frames": frames}, overwrite, TASK)
    work = [(str(bench.root), s.subset, s.id, str(out_dir), p.to_dict(), frames, overwrite)
            for s in iter_samples(bench, subsets, ids)]
    res = _map(_export_one, work, jobs)
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    write_json(out_dir / "protocol.json", p.to_dict())
    write_json(out_dir / "export.json", {"task": TASK, "protocol": p.to_dict(), "frames": frames})
    return {"exported": sum(r["status"] == "ok" for r in res), "reused": sum(bool(r.get("reused")) for r in res),
            "skipped": sum(r["status"] != "ok" for r in res)}


def prediction_path(pred_root, sample, view: int) -> Path:
    return Path(pred_root) / sample.subset / sample.id / f"view{view}.npz"


def write_prediction(pred_root, sample, view: int, masks, frames=None, packbits: bool = True) -> Path:
    """Write masks (n, H, W) (bool-like) for one view; rows = window frames unless `frames` (n,) GT indices given."""
    m = _as_bool(masks)
    if m.ndim != 3:
        raise ValueError(f"masks must be (n, H, W), got {m.shape}")
    path = prediction_path(pred_root, sample, view)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {"masks_packbits": np.packbits(m, axis=-1), "width": np.int32(m.shape[-1])} if packbits else {"masks": m}
    if frames is not None:
        arr["frames"] = np.asarray(frames, np.int32)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(path)
    return path


def load_prediction(path) -> tuple:
    """-> (masks (n, H, W) bool, frames (n,) int or None). Raises InvalidPrediction (a ValueError) when the file is
    unreadable or malformed."""
    names, z = read_npz(path, ("masks", "masks_packbits", "width", "frames"))
    try:
        if "masks" in z:
            m = _as_bool(z["masks"])
        elif "masks_packbits" in z:
            if "width" not in z:
                raise InvalidPrediction(f"{path}: masks_packbits needs 'width'")
            m = _unpack(z["masks_packbits"], int(z["width"]))
        else:
            raise InvalidPrediction(f"{path}: no 'masks' or 'masks_packbits' (has {names})")
        frames = z["frames"].astype(np.int64) if "frames" in z else None
    except (TypeError, ValueError) as e:
        if isinstance(e, InvalidPrediction):
            raise
        raise InvalidPrediction(f"{path}: {type(e).__name__}: {e}") from e
    if m.ndim != 3:
        raise InvalidPrediction(f"{path}: masks must be (n, H, W), got {m.shape}")
    if frames is not None and (frames.ndim != 1 or len(frames) != len(m)):
        raise InvalidPrediction(f"{path}: 'frames' of shape {frames.shape} for {len(m)} masks")
    return m, frames


ASPECT_TOL = 0.02


def check_prediction(masks: np.ndarray, frames, size, window, name: str = "prediction") -> dict:
    """Validate a loaded prediction of one view against the video size (W, H) and the window. Raises
    InvalidPrediction (a ValueError) for:
    an aspect ratio more than ASPECT_TOL off the video's (e.g. H and W swapped: masks must cover the full field of view
    of the stored video), duplicate entries in ``frames``, or fewer rows than window frames without ``frames``.
    Returns {"n_extra_rows": rows that are not window frames (ignored)}."""
    n, h, w = masks.shape
    W, H = (int(x) for x in size)
    if h < 1 or w < 1:
        raise InvalidPrediction(f"{name}: empty masks {masks.shape}")
    if abs((w / h) / (W / H) - 1.0) > ASPECT_TOL:
        hint = " (H and W swapped?)" if abs((h / w) / (W / H) - 1.0) <= ASPECT_TOL else ""
        raise InvalidPrediction(f"{name}: masks are {w}x{h} (W x H), aspect {w / h:.3f}, but the video is {W}x{H}, "
                                f"aspect {W / H:.3f}{hint}; masks must cover the full field of view of the video")
    win = {int(t) for t in window}
    if frames is None:
        if n < len(window):
            raise InvalidPrediction(f"{name}: {n} masks, the window has {len(window)} frames (or give 'frames')")
        return {"n_extra_rows": int(n - len(window))}
    fl = [int(t) for t in frames]
    if len(set(fl)) != len(fl):
        dup = sorted({t for t in fl if fl.count(t) > 1})[:5]
        raise InvalidPrediction(f"{name}: 'frames' lists frames more than once (e.g. {dup})")
    return {"n_extra_rows": sum(1 for t in fl if t not in win)}


class _PredView:
    """Frame lookup of one (validated) prediction file; counts the distinct scored frames without a row."""

    def __init__(self, masks, frames, window):
        self.masks = masks
        self.shape = masks.shape[1:]
        if frames is None:
            self.index = {int(t): j for j, t in enumerate(window)}
        else:
            self.index = {int(t): j for j, t in enumerate(frames)}
        self.missing = set()

    @property
    def n_missing_frames(self) -> int:
        return len(self.missing)

    def get(self, t: int) -> np.ndarray:
        j = self.index.get(int(t))
        if j is None:
            self.missing.add(int(t))
            return np.zeros(self.shape, bool)
        return self.masks[j]


def _ratios(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Same definitions as metrics.segmentation.point_label_metrics (percent; NaN where undefined)."""
    r = lambda a, b: 100.0 * a / b if b > 0 else _NAN
    prec, rec = r(tp, tp + fp), r(tp, tp + fn)
    if np.isnan(prec) or np.isnan(rec):
        f1 = _NAN
    else:
        f1 = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "iou": r(tp, tp + fp + fn) if tp + fp + fn > 0 else 100.0, "f1": f1,
            "accuracy": r(tp + tn, tp + fp + fn + tn)}


def _score_sparse_view(pv: _PredView, lab: dict, size) -> dict:
    """Counts of one view: pred masks read at the labels (mapped to the prediction grid)."""
    W, H = size
    out = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_points": 0, "n_neg_points": 0, "fp_background": 0,
           "n_background": 0, "n_pos": 0}
    if len(lab["frame"]) == 0:
        return out
    h, w = pv.shape
    for t in np.unique(lab["frame"]):
        sel = lab["frame"] == t
        uv = lab["uv"][sel].astype(np.float64)
        uvp = np.stack([(uv[:, 0] + 0.5) * w / W - 0.5, (uv[:, 1] + 0.5) * h / H - 0.5], -1)
        m = pv.get(int(t))
        y = lab["label"][sel]
        src = lab["source"][sel]
        res = point_label_metrics(m, uvp, y)
        for c in ("tp", "fp", "fn", "tn"):
            out[c] += res[c]
        val, _ = sample_mask(m, uvp)
        out["n_pos"] += int(y.sum())
        out["fp_points"] += int((val & (src == SRC_NEG_POINT)).sum())
        out["n_neg_points"] += int((src == SRC_NEG_POINT).sum())
        out["fp_background"] += int((val & (src == SRC_BACKGROUND)).sum())
        out["n_background"] += int((src == SRC_BACKGROUND).sum())
    return out


def _sparse_summary(c: dict) -> dict:
    d = _ratios(c["tp"], c["fp"], c["fn"], c["tn"])
    d["fpr_points"] = 100.0 * c["fp_points"] / c["n_neg_points"] if c["n_neg_points"] else _NAN
    d["fpr_background"] = 100.0 * c["fp_background"] / c["n_background"] if c["n_background"] else _NAN
    d.update({k: int(v) for k, v in c.items()})
    return d


def evaluate_sample(sample, pred_root, protocol: SegProtocol | None = None, labels_cache=None) -> dict:
    """Score one sample (both tracks where available). Returns the per-sample record."""
    p = protocol or SegProtocol()
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id}
    ok, why = p.evaluable(sample)
    if not ok:
        return {**rec, "status": "skipped", "reason": why}
    views = p.views_for(sample)
    masks = load_actor_masks(sample)
    rg = masks is not None
    try:
        target = resolve_target(sample, p, masks)
    except ValueError as e:
        return {**rec, "status": "skipped", "reason": str(e)}
    prompts = sample_prompts(sample, p, target, masks)
    if not any(prompts[k]["ok"] for k in views):
        return {**rec, "status": "skipped", "reason": "no view with a prompt (target not visible at the prompt frame)"}
    files = {k: prediction_path(pred_root, sample, k) for k in views}
    if not any(files[k].exists() for k in views if prompts[k]["ok"]):
        return {**rec, "status": "missing", "reason": "no prediction file"}
    labels = cached_sparse_labels(sample, p, labels_cache, target)
    win = p.window(sample)
    scored = win[1:]
    rec.update(target={"name": target["name"], "text": target["text"]}, prompt_frame=p.ref_frame,
               window=[win[0], win[-1]], frame_step=p.frame_step, margin_m=labels["margin_m"])
    per_view, dense_rows = {}, []
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fp_points": 0, "n_neg_points": 0, "fp_background": 0,
             "n_background": 0, "n_pos": 0}
    n_missing_views = 0
    for k in views:
        vr = {}
        if not prompts[k]["ok"]:
            per_view[f"view{k}"] = {"status": "no_prompt"}
            continue
        if files[k].exists():
            m, fr = load_prediction(files[k])
            vr.update(check_prediction(m, fr, sample.image_size(k), win, f"{sample.uid} view{k}"))
            pv = _PredView(m, fr, win)
        else:
            n_missing_views += 1
            pv = _PredView(np.zeros((len(win),) + tuple(sample.image_size(k))[::-1], bool), None, win)
            vr["missing_file"] = True
        c = _score_sparse_view(pv, labels["views"][k], sample.image_size(k))
        for key in total:
            total[key] += c[key]
        vr["sparse"] = _sparse_summary(c)
        if rg and prompts[k]["dense_ok"]:
            okf = [t for t in scored if masks.usable(t, k)]
            if okf:
                Wn, Hn = sample.image_size(k)
                grid = p.dense_size or (Wn, Hn)
                P = np.stack([resize_mask_nearest(pv.get(t), grid) for t in okf])
                G = np.stack([resize_mask_nearest(masks.get(t, k), grid) for t in okf])
                void = ~resize_mask_nearest(masks.valid_px(k), grid) if p.dense_void_border else None
                d = j_and_f(P, G, void=void)
                d["frames"] = [int(t) for t in okf]
                vr["dense"] = d
                dense_rows.append(d)
            else:
                vr["dense"] = None
                vr["dense_reason"] = "no usable GT mask in the scored frames"
        vr["n_missing_frames"] = pv.n_missing_frames
        vr["status"] = "ok"
        per_view[f"view{k}"] = vr
    n_scored = sum(1 for v in per_view.values() if v.get("status") == "ok")
    if n_scored == 0:
        return {**rec, "status": "skipped", "reason": "no view with a prompt", "views": per_view}
    rec["status"] = "ok"
    rec["n_views_scored"] = n_scored
    rec["n_views_missing_file"] = n_missing_views
    rec["sparse"] = _sparse_summary(total) if total["tp"] + total["fp"] + total["fn"] + total["tn"] else None
    if dense_rows:
        rec["dense"] = {key: float(np.mean([d[key] for d in dense_rows])) for key in DENSE_KEYS}
        rec["dense"]["n_views"] = len(dense_rows)
        rec["dense"]["n_frames"] = int(sum(d["n_frames"] for d in dense_rows))
    else:
        rec["dense"] = None
    rec["views"] = per_view
    rec["target_info"] = target["info"]
    return rec


def _eval_one(args) -> dict:
    root, subset, sid, pred_root, pdict, cache = args
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    try:
        return evaluate_sample(s, pred_root, SegProtocol.from_dict(pdict), cache)
    except InvalidPrediction as e:
        return {"task": TASK, "uid": s.uid, "subset": subset, "id": sid, "status": "invalid", "reason": str(e)}
    except Exception as e:
        import traceback
        return {"task": TASK, "uid": s.uid, "subset": subset, "id": sid, "status": "error",
                "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=4)}


def evaluate(bench, pred_root, out_dir, protocol: SegProtocol | None = None, subsets=None, ids=None,
             labels_cache=None, jobs: int = 1, on_missing: str = "skip", overwrite: bool = False) -> dict:
    """Score every sample: writes ``<out_dir>/<Subset>/<id>.json`` and the summary (`summarize`), returns the summary.
    labels_cache: optional directory for the GT point labels (reused across models). on_missing: "skip" | "error".
    ``run.json`` records the prediction folder and protocol: evaluating another run into the same out dir is refused
    (`common.RunDirError`) unless ``overwrite``, which first deletes the old per-sample results (re-evaluating the same
    run rewrites the selected samples). Statuses: ok / skipped / missing / invalid (unreadable or malformed prediction)
    / error. Warns about failed samples; the returned summary also has "evaluated" (`common.run_status` of the samples
    selected by this call; not written to summary.json)."""
    p = protocol or SegProtocol()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {"task": TASK, "protocol": p.to_dict(), "pred_root": str(Path(pred_root).resolve())}
    if check_out_stamp(out_dir, "run.json", run, overwrite, TASK) or overwrite:
        _clear_results(out_dir)
    work = [(str(bench.root), s.subset, s.id, str(pred_root), p.to_dict(), None if labels_cache is None else
             str(labels_cache)) for s in iter_samples(bench, subsets, ids)]
    res = _map(_eval_one, work, jobs)
    for r in res:
        if r["status"] == "missing" and on_missing == "error":
            raise FileNotFoundError(f"no prediction for {r['uid']}")
        write_json(out_dir / r["subset"] / f"{r['id']}.json", r)
    write_json(out_dir / "protocol.json", p.to_dict())
    write_json(out_dir / "run.json", run)
    bad = [r for r in res if r["status"] in ("invalid", "error")]
    if bad:
        warnings.warn(f"{TASK}: {len(bad)} sample(s) failed (status invalid / error), e.g. {bad[0]['uid']}: "
                      f"{str(bad[0].get('reason') or '')[:300]}")
    summary = summarize(out_dir)
    unmatched = bench.unmatched_ids(ids, [canonical_subset(s) for s in subsets] if subsets else None) if ids else []
    summary["evaluated"] = run_status([r["status"] for r in res], unmatched)
    return summary


def _clear_results(out_dir: Path):
    """Delete the per-sample results and summaries of this task in out_dir."""
    for sub in SUBSETS:
        d = out_dir / sub
        if d.is_dir():
            for f in d.glob("*.json"):
                try:
                    if json.loads(f.read_text()).get("task") == TASK:
                        f.unlink()
                except (ValueError, OSError):
                    pass
    for name in ("summary.json", "summary.md"):
        if (out_dir / name).exists():
            (out_dir / name).unlink()


BASELINES = ("oracle", "label_raster", "prompt_static", "empty", "full")


def baseline_masks(sample, view: int, kind: str, protocol: SegProtocol | None = None, target=None, masks=None,
                   prompts=None, labels=None) -> np.ndarray | None:
    """(n_window, H, W) bool masks of a reference baseline for one view (None if the view has no prompt)."""
    p = protocol or SegProtocol()
    if kind not in BASELINES:
        raise ValueError(f"baseline {kind!r} not in {BASELINES}")
    win = p.window(sample)
    W, H = sample.image_size(view)
    if masks is None:
        masks = load_actor_masks(sample)
    rg = masks is not None
    target = target or resolve_target(sample, p, masks)
    prompts = prompts or sample_prompts(sample, p, target, masks)
    pr = prompts[view]
    if not pr["ok"]:
        return None
    if kind == "empty":
        return np.zeros((len(win), H, W), bool)
    if kind == "full":
        return np.ones((len(win), H, W), bool)
    if kind in ("oracle", "label_raster"):
        labels = labels or sparse_labels(sample, p, target)
        lab = labels["views"][view]
        out = np.zeros((len(win), H, W), bool)
        pos = lab["label"]
        idx = {t: j for j, t in enumerate(win)}
        iu, iv = _pixel(lab["uv"][pos], (W, H))
        for t, u, v in zip(lab["frame"][pos], iu, iv):
            out[idx[int(t)], v, u] = True
        if kind == "oracle" and rg:
            for j, t in enumerate(win):
                if masks.usable(t, view):
                    out[j] = masks.get(t, view)
        return out
    if pr["mask"] is not None:
        m0 = pr["mask"]
    else:
        m0 = np.zeros((H, W), bool)
        x0, y0, x1, y1 = pr["box"]
        m0[y0:y1 + 1, x0:x1 + 1] = True
    return np.repeat(m0[None], len(win), 0)


def _baseline_one(args) -> dict:
    root, subset, sid, pred_root, kind, pdict = args
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = SegProtocol.from_dict(pdict)
    ok, why = p.evaluable(s)
    if not ok:
        return {"uid": s.uid, "status": "skipped", "reason": why}
    masks = load_actor_masks(s)
    try:
        target = resolve_target(s, p, masks)
    except ValueError as e:
        return {"uid": s.uid, "status": "skipped", "reason": str(e)}
    prompts = sample_prompts(s, p, target, masks)
    labels = sparse_labels(s, p, target) if kind in ("oracle", "label_raster") else None
    n = 0
    for k in p.views_for(s):
        m = baseline_masks(s, k, kind, p, target, masks, prompts, labels)
        if m is not None:
            write_prediction(pred_root, s, k, m)
            n += 1
    return {"uid": s.uid, "status": "ok", "n_views": n}


def run_baseline(bench, pred_root=None, kind: str = "prompt_static", protocol: SegProtocol | None = None,
                 subsets=None, ids=None, jobs: int = 1) -> dict:
    """Write the predictions of a reference baseline (BASELINES) for every evaluable sample. The call form of every
    task: run_baseline(bench, pred_root, kind="empty", ...) (run_baseline("empty", bench, pred_root, ...) is accepted
    too)."""
    bench, pred_root, kind = baseline_args(bench, pred_root, kind, {})
    if kind not in BASELINES:
        raise ValueError(f"baseline {kind!r} not in {BASELINES}")
    p = protocol or SegProtocol()
    work = [(str(bench.root), s.subset, s.id, str(pred_root), kind, p.to_dict())
            for s in iter_samples(bench, subsets, ids)]
    res = _map(_baseline_one, work, jobs)
    write_json(Path(pred_root) / "meta_baseline.json", {"task": TASK, "baseline": kind, "protocol": p.to_dict()})
    return {"written": sum(r["status"] == "ok" for r in res), "skipped": sum(r["status"] != "ok" for r in res)}


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v


def mean_se(values) -> dict:
    """{"mean", "se", "n", "n_undefined", "n_inf"} over the finite values (s.e. = std(ddof=1)/sqrt(n); NaN for
    n < 2): the rule of every task (`common.mean_se`); NaN / None (undefined) and +-inf values are counted, not
    averaged."""
    return _common_mean_se([_num(x) for x in values])


def fmt_ms(x: dict, nd: int = 1) -> str:
    if x is None or not np.isfinite(_num(x.get("mean"))):
        return "-"
    se = _num(x.get("se"))
    return f"{x['mean']:.{nd}f} ± {se:.{nd}f}" if np.isfinite(se) else f"{x['mean']:.{nd}f}"


def summarize(out_dir) -> dict:
    """Aggregate ``<out_dir>/<Subset>/<id>.json`` into ``summary.json`` / ``summary.md`` (mean ± s.e. over samples per
    subset, plus pooled sparse counts). Per subset: ``status`` (counts; `common.STATUSES` names, earlier records'
    names mapped by `common.canonical_status`), ``n_ok``, ``n_invalid`` / ``n_error`` (failed samples); ``pred_root`` =
    the prediction root of the run in ``run.json`` (None without it)."""
    out_dir = Path(out_dir)
    proto = None
    pp = out_dir / "protocol.json"
    if pp.exists():
        proto = json.loads(pp.read_text())
    summary = {"task": TASK, "protocol": proto, "pred_root": run_pred_root(out_dir), "subsets": {}}
    for sub in SUBSETS:
        d = out_dir / sub
        if not d.is_dir():
            continue
        recs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        recs = [r for r in recs if r.get("task") == TASK]
        if not recs:
            continue
        status = {}
        for r in recs:
            st = canonical_status(r["status"])
            status[st] = status.get(st, 0) + 1
        ok = [r for r in recs if r["status"] == "ok"]
        sparse = [r["sparse"] for r in ok if r.get("sparse")]
        dense = [r["dense"] for r in ok if r.get("dense")]
        pooled = None
        if sparse:
            c = {k: int(sum(s[k] for s in sparse)) for k in ("tp", "fp", "fn", "tn", "fp_points", "n_neg_points",
                                                              "fp_background", "n_background", "n_pos")}
            pooled = _sparse_summary(c)
        summary["subsets"][sub] = {
            "status": status, "n_samples": len(recs), "n_ok": len(ok), "n_invalid": status.get("invalid", 0),
            "n_error": status.get("error", 0),
            "target": sorted({(r.get("target") or {}).get("name") for r in ok if r.get("target")}),
            "sparse": {k: mean_se(s[k] for s in sparse) for k in SPARSE_KEYS} if sparse else None,
            "sparse_pooled": pooled,
            "dense": {k: mean_se(x[k] for x in dense) for k in DENSE_KEYS} if dense else None,
            "n_views_missing_file": int(sum(r.get("n_views_missing_file", 0) for r in ok)),
        }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_summary_md(summary))
    return summary


def _summary_md(summary: dict) -> str:
    lines = ["# segmentation", ""]
    p = summary.get("protocol") or {}
    if p:
        lines += [f"prompt frame {p.get('ref_frame')}, horizon {p.get('horizon')}, step {p.get('frame_step')}, "
                  f"prompt '{p.get('prompt')}' (k = {p.get('n_points')})", ""]
    lines += ["## sparse point labels (mean ± s.e. over samples, percent)", "",
              "| subset | target | n | precision | recall | IoU | F1 | FPR points | FPR background |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sub, s in summary["subsets"].items():
        sp = s.get("sparse")
        if not sp:
            continue
        lines.append(f"| {sub} | {', '.join(t for t in s['target'] if t)} | {sp['iou']['n']} | "
                     + " | ".join(fmt_ms(sp[k]) for k in ("precision", "recall", "iou", "f1", "fpr_points",
                                                          "fpr_background")) + " |")
    dense = [(sub, s) for sub, s in summary["subsets"].items() if s.get("dense")]
    if dense:
        lines += ["", "## dense masks (Real-General SAM2.1 actor masks, pseudo-GT; mean ± s.e. over samples)", "",
                  "| subset | n | J | F | J&F | J recall | F recall |", "|---|---|---|---|---|---|---|"]
        for sub, s in dense:
            dd = s["dense"]
            lines.append(f"| {sub} | {dd['J&F']['n']} | " + " | ".join(fmt_ms(dd[k]) for k in
                                                                       ("J", "F", "J&F", "J_recall", "F_recall")) + " |")
    lines += ["", "status: " + "; ".join(f"{sub} {s['status']}" for sub, s in summary["subsets"].items()), ""]
    return "\n".join(lines)


def iter_samples(bench, subsets=None, ids=None) -> list:
    """Samples of `subsets` (aliases accepted) restricted to `ids` (ids or uids, subset aliases accepted in uids);
    warns about requested ids that match no sample (`common.select_samples`)."""
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    return select_samples(bench, subs, ids)


def _map(fn, work, jobs: int):
    if jobs and jobs > 1 and len(work) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(fn, work, chunksize=1))
    return [fn(w) for w in work]


def jsonable(x):
    """NaN -> None, inf -> "inf"/"-inf", numpy -> python (recursively)."""
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
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return x


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path


def main(argv=None) -> int:
    """`vpmwb segmentation {export,baseline,evaluate,summarize}` (also `python -m vpm_worldbench.tasks.segmentation`).
    Shared option names: --root, --pred, --out, --protocol, --subsets, --ids, --jobs, --overwrite; window overrides
    --ref-frame / --horizon (<= 0: to the end of the clip) / --frame-step (a changed value adds "+field=value" to the
    protocol name). Exit status: 1 when the selection matches no sample or no evaluable sample is scored (every one
    missing / invalid / error); 2 for a usage error (unknown subset, --protocol neither the preset nor a file, a
    malformed --dense-size, a summarize --out that is no folder: usage + one line) or an out dir of another run
    without --overwrite (`common.RunDirError`, which propagates: `vpmwb` reports it as one line `vpmwb: error: ...`)."""
    import argparse

    from ..utils import parse_list
    ap = argparse.ArgumentParser(prog="vpmwb segmentation",
                                 description="promptable video object segmentation: export inputs, write baselines, "
                                             "evaluate, summarize")
    sp = ap.add_subparsers(dest="cmd", required=True, metavar="{export,baseline,evaluate,summarize}")
    helps = {"export": "write the model inputs (prompts, window, video paths)",
             "baseline": "write the predictions of a reference baseline (--kind)",
             "evaluate": "score predictions", "summarize": "rebuild summary.json / summary.md"}
    parsers = {}
    for name in ("export", "baseline", "evaluate", "summarize"):
        q = parsers[name] = sp.add_parser(name, help=helps[name], description=helps[name])
        if name != "summarize":
            q.add_argument("--root", required=True, help="VP-WorldBench data root")
            q.add_argument("--protocol", default=None,
                           help=f"'{SegProtocol().name}' (default) or a JSON / YAML file of SegProtocol fields (a file "
                                "without 'name' is named after the file)")
            q.add_argument("--subsets", nargs="*", default=None, help="subset names or aliases (space / comma separated)")
            q.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (space / comma separated)")
            q.add_argument("--jobs", type=int, default=1, help="worker processes")
            q.add_argument("--ref-frame", type=int, default=None, help="override the prompt frame")
            q.add_argument("--horizon", type=int, default=None, help="override the horizon (<= 0: to the clip end)")
            q.add_argument("--frame-step", type=int, default=None, help="override the window frame step")
        if name in ("baseline", "evaluate"):
            q.add_argument("--pred", required=True, help="prediction root")
        if name in ("export", "evaluate", "summarize"):
            q.add_argument("--out", required=True, help="output folder")
        if name in ("export", "evaluate"):
            q.add_argument("--overwrite", action="store_true",
                           help="redo existing samples / replace an out dir written by another run")
        if name == "export":
            q.add_argument("--frames", default="none", choices=("none", "png"), help="also write the window frames")
        elif name == "baseline":
            q.add_argument("--kind", default="prompt_static", choices=BASELINES)
        elif name == "evaluate":
            q.add_argument("--labels-cache", default=None, help="folder caching the GT point labels across runs")
            q.add_argument("--dense-size", default=None, help="WxH grid for J / F (default native)")
            q.add_argument("--on-missing", default="skip", choices=("skip", "error"))
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        if not Path(a.out).is_dir():
            parsers["summarize"].error(f"--out {a.out}: no such directory")
        summarize(a.out)
        print((Path(a.out) / "summary.md").read_text())
        return 0
    q = parsers[a.cmd]
    if not Path(a.root).is_dir():
        q.error(f"--root {a.root}: no such directory")
    p = cli_protocol(q, a.protocol, SegProtocol, get_protocol)
    over = {}
    if a.ref_frame is not None:
        over["ref_frame"] = a.ref_frame
    if a.horizon is not None:
        over["horizon"] = None if a.horizon <= 0 else a.horizon
    if a.frame_step is not None:
        over["frame_step"] = a.frame_step
    if getattr(a, "dense_size", None):
        over["dense_size"] = cli_size(q, "--dense-size", a.dense_size)
    if over:
        try:
            p = override_protocol(p, over, SegProtocol)
        except (KeyError, TypeError, ValueError) as e:
            q.error(str(e))
    subsets, ids = cli_subsets(q, a.subsets), parse_list(a.ids)
    from ..data.benchmark import Benchmark
    bench = Benchmark(a.root)
    if a.cmd == "export":
        res = export_inputs(bench, a.out, p, subsets, ids, frames=a.frames, jobs=a.jobs, overwrite=a.overwrite)
        print(res)
        return selection_exit_status(res, TASK)
    if a.cmd == "baseline":
        res = run_baseline(bench, a.pred, kind=a.kind, protocol=p, subsets=subsets, ids=ids, jobs=a.jobs)
        print(res)
        return selection_exit_status(res, TASK)
    res = evaluate(bench, a.pred, a.out, p, subsets, ids, labels_cache=a.labels_cache, jobs=a.jobs,
                   on_missing=a.on_missing, overwrite=a.overwrite)
    print((Path(a.out) / "summary.md").read_text())
    return evaluation_exit_status(res, TASK, a.out)


if __name__ == "__main__":
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    warnings.simplefilter("default")
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
