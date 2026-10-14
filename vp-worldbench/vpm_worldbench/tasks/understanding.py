"""Task ``understanding``: video understanding on the multi-view clips. Three sub-tasks:

    (a) qa         auto-generated multiple-choice questions whose answers are derived ONLY from GT metadata, GT tracks,
                   cameras and labels (never from a model); per-subset templates + cross-subset templates.
    (b) retrieval  text <-> video retrieval with the clip captions (all subsets; Real-General also segment level,
                   exported only on request).
    (c) grounding  dense temporal grounding on Real-General: given a segment caption, find its [start, end] frames.

(a) Multiple-choice QA
----------------------
Inputs (``export_inputs`` -> ``<out>/qa/questions.jsonl``, one line per question; no answers in this file)
    ``qid`` ("<uid>#<template>#<k>"), ``uid``/``subset``/``id``, ``template``, ``question``, ``options`` (list of
    strings, labelled A, B, ... in order), ``frames`` [start, end] (inclusive GT frame range of the clip the model
    watches; full clip unless the template is about a window), ``views`` (view indices; all views of the sample),
    ``videos`` (paths relative to the benchmark root), ``fps``, ``prompt`` (a ready-made text: the view names, the
    question, the lettered options and "Answer with the option's letter."). The clip captions and task instructions are
    NEVER part of the input (only rg_segment_order shows two segment captions, as its options). The answer key
    (``answer_key.jsonl``: qid, answer letter, answer text, template, ``source`` = how the answer was derived) is NOT
    written into the export folder (a model may read that folder); ``answer_key_out`` (``--answer-key-out``) writes it
    to a path of your choice, outside that folder. `evaluate` regenerates the key from the data by default; its
    ``answer_key`` (``--answer-key``) scores a stale export with the key written alongside it.
Templates (QA_TEMPLATES; thresholds in the code; every question's ``source`` states the numbers used)
    cross-subset
      num_views         how many camera views (sample.json n_views); a SANITY check of the multi-view input handling,
                        answerable from the input itself (exempt from balancing; reported in its own row and left
                        out of the headline accuracies, SANITY_TEMPLATES)
      camera_motion     one view per sample: static / rotates in place / moves through the scene (w2c: max camera
                        centre displacement and rotation w.r.t. frame 0; ambiguous magnitudes skipped)
      target_distance   one view per sample: is the target closer to / farther from / about as far from that camera at
                        the end as at the start (centroid of the target's GT track points, Real-General: mid-hip of the
                        triangulated keypoints; mean over the first / last 5 frames; target in the frustum at both
                        ends; ratios 0.8 / 1.25 with a same-distance band 0.93-1.075, gaps skipped)
    Simul-Robot
      sr_hands          left / right / both hands (max displacement of each hand's robot_hand points from frame 0:
                        used >= 0.10 m, idle <= 0.03 m; must agree with attributes.hands_used)
      sr_moved_object   which object is picked and moved (exactly one tracked object moves > 5 cm, all others < 1 cm,
                        untracked distractors pushed <= 1 cm and none fell; distractor pool = the other objects of the
                        scene (preferred) and the subset's object vocabulary, minus names confusable with the answer)
      sr_num_moved      how many objects the robot moves (count of tracked objects moving > 5 cm; skipped if any moves
                        1-5 cm or an untracked distractor is pushed > 1 cm / fell)
      sr_first_moved    which object is moved first (>= 2 moved objects, motion onsets (> 2 cm) >= 20 frames apart;
                        distractor pool = the objects moved later (preferred), the still objects and the vocabulary)
      sr_end_state      is the primary object resting on a surface or still held at the end (attributes.flag_details:
                        supported & not held & not airborne & no clip_ends_mid_motion (borderline skipped) vs held &
                        airborne & not supported; other combinations skipped)
    Simul-General
      sg_agent_type     pedestrian / vehicle followed by the cameras (attributes.category)
      sg_vehicle_type   car / van / ambulance / fire truck / box truck / bus (CARLA blueprint -> visual type; options
                        that would also be true are excluded, e.g. no "van" for an ambulance)
      sg_views_out      in how many views the target is out of the field of view for a while (labels/target.npz
                        state_strict out_of_frustum runs >= 8 frames; views with 1-7 such frames make it ambiguous;
                        must agree with the out_of_frustum segments of labels/oos_segments.json)
      sg_distance       net displacement of the target (centroid, first -> last frame): < 10 / 10-20 / 20-40 / > 40 m
                        (within 12 % of a bin edge skipped)
    Real-Robot (the instruction is never shown to the model)
      rr_hands          left / right / both (instruction parse captions.structured.hand, cross-checked with the hand
                        tracks: used >= 0.15 m, idle <= 0.10 m)
      rr_first_hand     which hand acts first in two-step bimanual tasks (structured.steps, cross-checked: motion onset
                        (> 5 cm) of the first hand >= 15 frames before the second)
      rr_task           task family (instruction parse; plate tasks are not distractors of each other)
      rr_object         which object the robot handles (structured.objects; distractors from the other episodes'
                        objects that share no noun with this episode's objects)
    Real-General (windows: the protocol window [ref_frame, ref_frame + horizon * frame_step] = frames 16..80 and the
    following window of the same length, 80..144 (``n_windows`` = 2); the question shows only that window)
      rg_motion         which motion the person performs in the window (labels/motion_events.json; any event type
                        of RG_ANSWER_TYPES overlapping the window by >= min(half its length, 15 frames) can be the
                        answer, `finalize` picks the shown one; distractors = types with no event within the window
                        +- 8 frames, not confusable with any present type and verified absent on the triangulated
                        keypoints over the window +- 8 frames (RG_ABSENT; event thresholds in brackets): no walking:
                        pelvis stays within 0.3 m of its start; no turning: |hip-line yaw change| < 30 deg; never
                        still: pelvis speed > 0.25 m/s over every 0.5 s [0.12]; no raise: wrists >= 5 cm below the
                        shoulders [+5 cm]; no bend: torso tilt < 15 deg [28]; no squat: pelvis height > 85 % of the
                        standing height of motion_events.summary [80 %]; no reach: both wrists within 0.36 m of the
                        mid-shoulder horizontally [0.40 m]; hands never together: wrists always > 0.30 m apart
                        [0.22 m]). Every answer type can thus also be a distractor.
      rg_turn_direction which way the person turns (turn events in the window all in one direction, |yaw| >= 45 deg,
                        same sign as the keypoint hip yaw change)
      rg_body_part      which of head / left hand / right hand / left foot / right foot moves most relative to the
                        hips (keypoints3d nose / wrist / ankle path length relative to the mid-hip on a 4-frame grid,
                        which suppresses keypoint jitter; winner >= 1.5x the second and >= 0.3 m). The whole-body
                        track regions are too sparse in time (~30 % valid point-frames) for this, so the triangulated
                        keypoints (pseudo-GT, ~97 % of body keypoint-frames ok) are used.
      rg_segment_order  which of two segment captions happens first (non-adjacent, unclipped segments of >= 8 frames,
                        token-Jaccard < 0.5, at most 2 pairs per clip)
Balancing (`finalize`, per (template, subset) group; num_views and rg_segment_order exempt). A template yields for
each clip the correct option text and the POOL of every valid wrong option text; the shown options are chosen so that
the option texts alone do not give the answer away:
    1. questions with several true options (rg_motion) show the true type with the most spare distractor capacity
       (times it is a valid distractor in the group / (n_options - 1) minus times already shown as the answer);
    2. the most frequent answer values are down-sampled (hash order) until none exceeds ``max_answer_share``;
    3. distractors come only from texts that are the answer of some question of the group (a text that is never the
       answer would be a give-away); per question up to n_options - 1 of them, first the texts whose distractor count
       is furthest below (n_options - 1) x their answer count, preferring ``pref`` texts (e.g. other objects of the
       same scene), ties by hash; a question then has 2 .. n_options options;
    4. guard: while an option text with >= ``option_bias_min_count`` appearances is the answer in a share of its
       questions more than ``max_option_bias`` (0.10) away from its chance level (mean 1 / n_options), or a shown
       text is never the answer, drop the last question it answers (too often the answer) or remove it from the last
       question it is a distractor of that keeps >= 2 options (too rarely the answer);
    5. groups left with a single answer value are dropped; the correct option's position is assigned cyclically over
       the questions of the same option count sorted by a hash of the qid (balanced positions).
The "option_prior" baseline (blind leave-one-out answer rate of every option text within its group,
`option_prior_choice`) measures what is left: on the default set 29.2 % against 36.4 % chance over the 3,821 headline
questions (leave-one-out flips the answer in exactly balanced groups, so a template far below chance is a balanced
one). Answer values of the fixed-option templates (e.g. target_distance, sr_hands) can still be up to
max_answer_share frequent, which the "prior" baseline measures (41.0 %).
Generation is deterministic (seeded with sha256 of (seed, qid, ...)) and always covers every sample of a subset, so the
set of a subset never depends on the other subsets or on an ``ids`` filter (applied after generation).
Predictions ``<pred_root>/qa.jsonl``: {"qid", "response"} (free text, mapped to a letter with
`metrics.qa.extract_choice` using the option texts) or {"qid", "choice"} (letter or 0-based index). Missing or
unparseable answers are wrong (counted as n_missing / n_unparsed); predictions whose qid matches no question are
counted in summary.json meta.qa.n_predictions_unknown_qid (a sign of a stale export: score with the answer key written
by that export's ``answer_key_out``).
Metrics (`metrics.qa`): accuracy per sample (mean ± s.e. over samples), pooled accuracy, per template and macro over
templates, per subset and over all subsets. The headline numbers leave out the sanity template num_views; its accuracy
is reported apart (``sanity``). Lines of qa.jsonl that are not JSON objects with a string ``qid`` are counted
(meta.json qa.n_bad_lines), never fatal.
GT: Simul-* exact synthetic metadata / tracks; Real-Robot instruction parse + forward-kinematics tracks; Real-General
motion events (measured from the fitted whole-body model), triangulated keypoints and captions are PSEUDO-GT. The
Real-General captions were written by vision-language-model annotators and checked by model verifiers (no human;
before correction ~72 % of the segments were fully correct, ~21 % had minor and ~7 % major errors, see
Real-General/_source/CAPTIONS.md of the data); rg_segment_order only pairs non-adjacent segments, which is robust to
their boundary errors.

(b) Caption retrieval
---------------------
Per subset ("clip" level, sub-task "retrieval"): videos = the clips (full length, all views), texts = their global
captions in a seeded random order (``retrieval/<Subset>/texts.jsonl``: index, text_id, text; ``videos.jsonl``: index,
uid, frames, videos). Real-General "segment" level (sub-task "retrieval_segment", exported ONLY on request:
``tasks=(..., "retrieval_segment")`` / ``--tasks retrieval_segment``): videos = (clip, segment frame range) of the
unclipped segments of >= 8 frames, texts = the segment captions. Its videos.jsonl lists the GT frame range of every
grounding query, so a grounding method must never read it: export it into another folder than the grounding inputs
(its folder carries a README.txt saying so; `evaluate` adds a note to the summary when one prediction folder has both
grounding.jsonl and a segment-level retrieval prediction). Prediction ``<pred_root>/retrieval/<Subset>[_segment].npy``
(or .npz key ``sim``): similarity (n_videos, n_texts), higher = more similar, rows/columns in the exported index
order; NaN counts as -inf. Identical captions (after normalisation) count as correct for every clip that has them.
Metrics (`metrics.qa.bidirectional_retrieval`, ties pessimistic): text->video and video->text R@1/5/10, median /
mean rank; ``ids`` filtering is not applied (the gallery is the whole subset).

(c) Dense temporal grounding (Real-General)
-------------------------------------------
Queries (``grounding/queries.jsonl``): one per unclipped segment of >= 8 frames whose caption is not a near-duplicate
(token-Jaccard >= 0.5) of another segment of the clip; query ids are shuffled within a clip (no order leak). The model
watches the whole clip (all views) and returns ``<pred_root>/grounding.jsonl`` {"query_id", "start_frame",
"end_frame"} (inclusive GT frames; non-integers rounded half up) or {"query_id", "start_s", "end_s"} (seconds, the
half-open range [start_s, end_s) = the frames t with start_s <= t / fps < end_s, i.e. frames ceil(start_s * fps) ..
ceil(end_s * fps) - 1, the convention of captions.json; `parse_interval`); predictions are clipped to the clip,
frames 0 .. n_frames - 1. Metric: temporal IoU on inclusive frame ranges (|inter| = max(0, min(e1, e2) - max(s1, s2) + 1)); mIoU and R@1 at IoU >= 0.3 / 0.5 / 0.7 (percent); per
sample then mean ± s.e. over samples, plus pooled over queries. A missing, malformed (non-numeric / non-finite
values, missing keys, a line that is not a JSON object) or empty (end < start) prediction scores IoU 0 and is counted
(status missing / invalid / empty), never fatal. Grounding methods must not use the segment-level retrieval export
(see (b)). PSEUDO-GT: segment
boundaries and captions come from the vision-language-model annotation pipeline (motion hints from the whole-body fit,
model verifiers; minor timing errors are common, see (a) GT).

Baselines (`run_baseline`): "oracle" (answer key / GT similarity / GT segments), "random" (seeded), "prior" (QA: the
most frequent answer text of the template in the subset; retrieval: random; grounding: the centred interval of the
mean GT length), "option_prior" (QA: the blind leave-one-out option-text prior `option_prior_choice`; otherwise as
prior), "full" (grounding: the whole clip). Reference numbers of the default protocol on the v1 data (QA: pooled
accuracy over the 3,821 headline questions, chance 36.4 %; grounding: pooled mIoU over 726 queries): oracle 100 /
100; random 36.4 / 12.1; prior 41.0 / 9.8; option_prior 29.2 / 9.8; full 41.0 / 13.4; retrieval R@1 of the
non-oracle baselines is at chance (<= 3 %).

Results: ``evaluate`` scores whatever prediction files exist under pred_root and writes ``<out>/<Subset>/<id>.json``
(sections qa / retrieval / grounding), ``meta.json`` (found sub-tasks, bad-line counts, notes), ``run.json`` (the
prediction folder and protocol; another run into the same out dir is refused (`common.RunDirError`) unless
``overwrite``) and ``summary.json`` / ``summary.md`` (`summarize`; summary.json records the prediction folder as
``pred_root``). Without any prediction file nothing is scored (`NoPredictionFiles`; the command exits with status 1).

    from vpm_worldbench import Benchmark
    from vpm_worldbench.tasks import understanding as und
    bench = Benchmark("/data/VP-WorldBench")
    und.export_inputs(bench, "inputs/understanding")            # qa/, retrieval/<Subset>/, grounding/
    ... run a model, write preds/qa.jsonl, preds/retrieval/<Subset>.npy, preds/grounding.jsonl ...
    und.run_baseline(bench, "preds/prior", kind="prior")         # the call form of every task
    und.evaluate(bench, "preds", "results/understanding")

    vpmwb understanding export   --root R --out inputs/und [--tasks qa retrieval grounding] [--subsets rg]
                                 [--answer-key-out keys/und_answer_key.jsonl]
    vpmwb understanding export   --root R --out inputs/und_segment --tasks retrieval_segment
    vpmwb understanding baseline --root R --pred preds/prior --kind prior     (oracle random prior option_prior full)
    vpmwb understanding evaluate --root R --pred preds/prior --out results/prior [--answer-key ...] [--overwrite]
    vpmwb understanding summarize --out results/prior
(also `python -m vpm_worldbench.tasks.understanding ...`; --subsets / --ids take comma- and/or space-separated
values, subset aliases accepted, also in uids; a --protocol file without ``name`` is named after the file;
--ref-frame / --horizon (>= 1) / --frame-step override the window of the window templates and add "+field=value" to
the protocol name: give the same values to export, baseline and evaluate).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import SUBSETS, canonical_subset
from ..geometry import camera_center, inside, project, rotation_angle_deg
from ..metrics.qa import (LETTERS, accuracy, bidirectional_retrieval, extract_choice, normalize_text,
                          retrieval_ranks)
from .common import (baseline_args, canonical_ids, check_out_stamp, cli_protocol, cli_subsets, file_protocol_fields,
                     override_protocol, run_pred_root, select_samples)

TASK = "understanding"
QA_SCHEMA = "vpm-worldbench/qa/1.0"
_NAN = float("nan")

QA_TEMPLATES = {
    "num_views": SUBSETS,
    "camera_motion": SUBSETS,
    "target_distance": ("Simul-Robot", "Simul-General", "Real-General"),
    "sr_hands": ("Simul-Robot",), "sr_moved_object": ("Simul-Robot",), "sr_num_moved": ("Simul-Robot",),
    "sr_first_moved": ("Simul-Robot",), "sr_end_state": ("Simul-Robot",),
    "sg_agent_type": ("Simul-General",), "sg_vehicle_type": ("Simul-General",), "sg_views_out": ("Simul-General",),
    "sg_distance": ("Simul-General",),
    "rr_hands": ("Real-Robot",), "rr_first_hand": ("Real-Robot",), "rr_task": ("Real-Robot",),
    "rr_object": ("Real-Robot",),
    "rg_motion": ("Real-General",), "rg_turn_direction": ("Real-General",), "rg_body_part": ("Real-General",),
    "rg_segment_order": ("Real-General",),
}
BALANCE_EXEMPT = ("num_views", "rg_segment_order")
SANITY_TEMPLATES = ("num_views",)


@dataclass
class UnderstandingProtocol:
    name: str = "vpm-understanding-v1"
    seed: int = 0
    ref_frame: int = 16
    horizon: int = 64
    frame_step: int = 1
    n_windows: int = 2
    templates: Optional[list] = None
    n_options: int = 4
    balance: bool = True
    max_answer_share: float = 0.5
    max_option_bias: float = 0.10
    option_bias_min_count: int = 5
    max_pairs_per_clip: int = 2
    min_segment_frames: int = 8
    max_caption_jaccard: float = 0.5

    def __post_init__(self):
        if self.templates is not None:
            unknown = set(self.templates) - set(QA_TEMPLATES)
            if unknown:
                raise KeyError(f"unknown QA templates {sorted(unknown)}")
        if not 2 <= self.n_options <= 6:
            raise ValueError("n_options must be in 2..6")
        if not 0 < self.max_answer_share <= 1:
            raise ValueError("max_answer_share must be in (0, 1]")
        if not 0 < self.max_option_bias <= 1 or self.option_bias_min_count < 1:
            raise ValueError("max_option_bias must be in (0, 1] and option_bias_min_count >= 1")
        if self.n_windows < 1 or self.horizon < 1 or self.frame_step < 1:
            raise ValueError("n_windows, horizon and frame_step must be >= 1")
        if self.ref_frame < 0:
            raise ValueError("ref_frame must be >= 0")

    @property
    def window(self) -> tuple:
        return self.ref_frame, self.ref_frame + self.horizon * self.frame_step

    def windows_for(self, sample) -> list:
        """[(w0, w1), ...] inclusive GT frame ranges of the window templates for this sample."""
        L = self.horizon * self.frame_step
        out = []
        w0 = self.ref_frame
        while len(out) < self.n_windows and w0 >= 0 and w0 + L <= sample.n_frames - 1:
            out.append((w0, w0 + L))
            w0 += L
        return out

    def templates_for(self, subset: str) -> list:
        names = self.templates if self.templates is not None else list(QA_TEMPLATES)
        return [t for t in names if subset in QA_TEMPLATES[t]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UnderstandingProtocol":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown understanding protocol fields {sorted(unknown)}")
        return cls(**d)


def get_protocol(spec=None) -> UnderstandingProtocol:
    """None -> defaults; an UnderstandingProtocol; a dict of fields; or a path to a .json / .yaml file (a file without
    "name" is named after the file)."""
    if spec is None:
        return UnderstandingProtocol()
    if isinstance(spec, UnderstandingProtocol):
        return spec
    if isinstance(spec, dict):
        return UnderstandingProtocol.from_dict(spec)
    p = Path(spec)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return UnderstandingProtocol.from_dict(file_protocol_fields(yaml.safe_load(text), p))
    return UnderstandingProtocol.from_dict(file_protocol_fields(json.loads(text), p))


def _label(sample, name: str):
    """Path of an optional label file, or None when the sample does not have it."""
    rel = sample.meta["files"].get("labels", {}).get(name)
    if rel is None:
        return None
    p = sample.dir / rel
    return p if p.exists() else None


def _hash_int(*keys) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()[:8], "little")


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng(_hash_int(*keys))


_STOP = frozenset("a an the and or of to in on at with from into onto by for its it is are as then while this that "
                  "his her their person people both".split())


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOP}


def _jaccard(a: str, b: str) -> float:
    A, B = _tokens(a), _tokens(b)
    return len(A & B) / max(len(A | B), 1)


def object_name(object_id: str) -> str:
    """Display name of a Simul-Robot object id ('small_yellow_cube' -> 'small yellow cube')."""
    special = {"square_big_box": "large square box", "red_rimmed_plate": "red-rimmed plate",
               "rubiks_puzzle_cube": "Rubik's puzzle cube", "bbq_sauce_bottle": "BBQ sauce bottle"}
    return special.get(object_id, object_id.replace("_", " "))


def _confusable(a: str, b: str) -> bool:
    """Two object names are confusable when the content words of one are a subset of the other's."""
    A, B = _tokens(a), _tokens(b)
    return A <= B or B <= A


def _q(sample, template: str, k, question: str, answer: str, pool: list, source: str, frames=None,
       value=None, meta=None, pref=(), answers=None) -> dict:
    """A candidate question. ``answer`` is the correct option text and ``pool`` every valid wrong option text; the
    distractors are chosen from the pool in `finalize` (which balances how often each option text is the answer).
    ``pref``: pool texts preferred as distractors (e.g. other objects of the same scene). ``answers``: optional
    [(text, value), ...] when several options are true for the clip and one of them is shown (rg_motion); `finalize`
    picks it, the others are never options (they must not be in the pool)."""
    pool = list(dict.fromkeys(pool))
    alts = [(str(t), str(v)) for t, v in answers] if answers else [(answer, str(answer if value is None else value))]
    if any(t in pool for t, _ in alts):
        raise AssertionError(f"{sample.uid} {template}: a correct option in the distractor pool")
    return {"qid": f"{sample.uid}#{template}#{k}", "uid": sample.uid, "subset": sample.subset, "id": sample.id,
            "template": template, "question": question, "answer_text": alts[0][0], "pool": pool,
            "pref": [x for x in pref if x in pool], "answers": alts if answers else None,
            "value": alts[0][1], "frames": list(frames or [0, sample.n_frames - 1]),
            "views": list(range(sample.n_views)), "fps": sample.fps,
            "videos": [f"{sample.subset}/{sample.id}/{v['video']}" for v in sample.meta["views"]],
            "source": source, "meta": meta or {}}


def _camera_motion(w2c: np.ndarray) -> tuple:
    """(max camera-centre displacement m, max rotation deg) w.r.t. frame 0 of one view's (T, 4, 4) w2c."""
    c = camera_center(w2c)
    trans = float(np.linalg.norm(c - c[0], axis=-1).max())
    R = w2c[:, :3, :3]
    rot = float(rotation_angle_deg(R, np.broadcast_to(R[:1], R.shape)).max())
    return trans, rot


def _centroid_track(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 3) mean of the valid points per frame (NaN where none)."""
    x = np.where(valid[..., None], xyz, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x, axis=1)


def _max_disp(xyz: np.ndarray) -> np.ndarray:
    """(T,) max over points of |X_t - X_0| for points (T, M, 3) (NaN-safe)."""
    d = np.linalg.norm(xyz - xyz[:1], axis=-1)
    d = np.where(np.isfinite(d), d, 0.0)
    return d.max(axis=1) if d.ndim == 2 else d


def _tpl_num_views(s, ctx) -> list:
    V = s.n_views
    return [_q(s, "num_views", 0, "How many different camera views of the scene are provided?", str(V),
               [str(n) for n in range(1, 7) if n != V], f"sample.json n_views = {V}")]


CAMERA_OPTIONS = ("it stays static", "it only rotates (pans or tilts) without changing position",
                  "it moves through the scene (its position changes)")


def _tpl_camera_motion(s, ctx) -> list:
    rng = _rng(ctx["p"].seed, s.uid, "camera_motion")
    views = list(rng.permutation(s.n_views))
    for v in views:
        trans, rot = _camera_motion(s.cameras.w2c[v])
        if trans < 0.01 and rot < 0.5:
            ans = CAMERA_OPTIONS[0]
        elif trans < 0.05 and rot > 5.0:
            ans = CAMERA_OPTIONS[1]
        elif trans > 0.30:
            ans = CAMERA_OPTIONS[2]
        else:
            continue
        return [_q(s, "camera_motion", f"view{v}", f"How does the camera of view {v} move during the clip?", ans,
                   [o for o in CAMERA_OPTIONS if o != ans],
                   f"cameras.npz w2c view{v}: max camera-centre displacement {trans:.3f} m, max rotation {rot:.2f} deg "
                   f"w.r.t. frame 0 (static < 0.01 m & < 0.5 deg; rotates < 0.05 m & > 5 deg; moves > 0.30 m)",
                   value=ans, meta={"view": int(v)})]
    return []


DIST_OPTIONS = ("closer to the camera", "farther from the camera", "at about the same distance")


def _target_positions(s, ctx):
    """(T, 3) world position of the template target and its name, or None."""
    tr = s.tracks
    if s.subset == "Simul-Robot":
        prim = np.asarray(tr.extra["is_primary"], bool)
        if not prim.any():
            return None
        return _centroid_track(tr.xyz[:, prim], tr.valid[:, prim]), f"the {object_name(s.attributes['primary_object'])}"
    if s.subset == "Simul-General":
        cat = s.attributes.get("category")
        return _centroid_track(tr.xyz, tr.valid), ("the pedestrian" if cat == "pedestrian" else "the vehicle") + \
            " followed by the cameras"
    if s.subset == "Real-General":
        k3 = ctx["kp3d"]()
        if k3 is None:
            return None
        X = k3["xyz"].astype(np.float64)
        ok = k3["ok"][:, 11] & k3["ok"][:, 12]
        pel = 0.5 * (X[:, 11] + X[:, 12])
        pel[~ok] = np.nan
        return pel, "the main person (the one performing the motion)"
    return None


def _tpl_target_distance(s, ctx) -> list:
    got = _target_positions(s, ctx)
    if got is None:
        return []
    pos, name = got
    T = s.n_frames
    a, b = list(range(0, 5)), list(range(T - 5, T))
    if not (np.isfinite(pos[a]).all() and np.isfinite(pos[b]).all()):
        return []
    rng = _rng(ctx["p"].seed, s.uid, "target_distance")
    for v in rng.permutation(s.n_views):
        v = int(v)
        K, W = s.cameras.K[v], s.cameras.w2c[v]
        cc = camera_center(W)
        ok = True
        for t in a + b:
            uv, z = project(pos[t], K, W[t])
            ok &= bool(z > 0.1 and inside(uv[None], s.image_size(v))[0])
        if not ok:
            continue
        d0 = float(np.mean([np.linalg.norm(pos[t] - cc[t]) for t in a]))
        d1 = float(np.mean([np.linalg.norm(pos[t] - cc[t]) for t in b]))
        r = d1 / d0
        if r < 0.8:
            ans = DIST_OPTIONS[0]
        elif r > 1.25:
            ans = DIST_OPTIONS[1]
        elif 0.93 <= r <= 1.075:
            ans = DIST_OPTIONS[2]
        else:
            continue
        what = ("target point centroid (tracks3d)" if s.subset != "Real-General"
                else "mid-hip of labels/human/keypoints3d.npz")
        return [_q(s, "target_distance", f"view{v}",
                   f"At the end of the clip, compared with its start, is {name} closer to the camera of view {v}, "
                   f"farther from it, or at about the same distance?", ans, [o for o in DIST_OPTIONS if o != ans],
                   f"{what}: distance to camera view{v} {d0:.2f} m (frames 0-4) -> {d1:.2f} m (frames {T - 5}-{T - 1}), "
                   f"ratio {r:.3f} (closer < 0.8, farther > 1.25, same 0.93-1.075); inside the view at both ends",
                   value=ans, meta={"view": v, "ratio": r})]
    return []


HAND_OPTIONS = ("only the left hand", "only the right hand", "both hands")


def _sr_facts(s, ctx) -> dict:
    tr = s.tracks
    oid = np.asarray(tr.extra["object_id"]).astype(str)
    names = [n for n in dict.fromkeys(oid.tolist()) if n]
    disp, onset = {}, {}
    for n in names:
        m = oid == n
        d = _max_disp(tr.xyz[:, m])
        disp[n] = float(d.max())
        onset[n] = int(np.argmax(d > 0.02)) if (d > 0.02).any() else -1
    kinds = [str(k) for k in tr.kind_names]
    hand = tr.kind == kinds.index("robot_hand")
    hd = {}
    for side, nm in ((0, "left"), (1, "right")):
        m = hand & (tr.side == side)
        if not m.any():
            hd[nm] = _NAN
            continue
        c = _centroid_track(tr.xyz[:, m], tr.valid[:, m])
        hd[nm] = float(np.nanmax(np.linalg.norm(c - c[:1], axis=-1)))
    return {"names": names, "disp": disp, "onset": onset, "hand_disp": hd}


def _tpl_sr_hands(s, ctx) -> list:
    f = ctx["sr"]()
    hd = f["hand_disp"]
    used = {k for k, v in hd.items() if v >= 0.10}
    idle = {k for k, v in hd.items() if v <= 0.03}
    if len(used) + len(idle) != 2 or not used:
        return []
    attr = set(s.attributes.get("hands_used", []))
    if attr != used:
        return []
    ans = HAND_OPTIONS[2] if len(used) == 2 else (HAND_OPTIONS[0] if "left" in used else HAND_OPTIONS[1])
    return [_q(s, "sr_hands", 0, "Which of the robot's hands does the robot use to manipulate objects in this clip?",
               ans, [o for o in HAND_OPTIONS if o != ans],
               f"tracks3d robot_hand points: hand centroid max displacement from frame 0 left {hd['left']:.3f} m, right "
               f"{hd['right']:.3f} m (used >= 0.10 m, idle <= 0.03 m); agrees with attributes.hands_used {sorted(attr)}",
               value=ans)]


def _sr_distractor_clean(s) -> bool:
    a = s.attributes
    return float(a.get("distractor_max_pushed_cm") or 0.0) <= 1.0 and int(a.get("distractor_fell_off") or 0) == 0


def _sr_object_pool(s, ctx, answer_id: str) -> tuple:
    """(pool, pref) of distractor names: the other tracked objects of the scene (preferred) and the subset's object
    vocabulary, minus names confusable with the answer."""
    ans = object_name(answer_id)
    names = ctx["sr"]()["names"]
    scene = [object_name(x) for x in names if x != answer_id]
    scene = [x for x in scene if not _confusable(x, ans)]
    vocab = [object_name(x) for x in ctx["vocab"]["Simul-Robot"] if x not in set(names) | {answer_id}]
    vocab = [x for x in vocab if not _confusable(x, ans)]
    return scene + vocab, scene


def _tpl_sr_moved_object(s, ctx) -> list:
    f = ctx["sr"]()
    moved = [n for n in f["names"] if f["disp"][n] > 0.05]
    still = [n for n in f["names"] if f["disp"][n] < 0.01]
    if len(moved) != 1 or len(moved) + len(still) != len(f["names"]) or not _sr_distractor_clean(s):
        return []
    m = moved[0]
    pool, pref = _sr_object_pool(s, ctx, m)
    return [_q(s, "sr_moved_object", 0, "Which object does the robot pick up and move in this clip?", object_name(m),
               pool, f"tracks3d object points: '{m}' max point displacement {f['disp'][m]:.3f} m (> 0.05 m); other "
               f"tracked objects < 0.01 m; untracked distractors pushed <= 1 cm, none fell", value=object_name(m),
               pref=pref)]


def _tpl_sr_num_moved(s, ctx) -> list:
    f = ctx["sr"]()
    moved = [n for n in f["names"] if f["disp"][n] > 0.05]
    still = [n for n in f["names"] if f["disp"][n] < 0.01]
    if len(moved) + len(still) != len(f["names"]) or not moved or not _sr_distractor_clean(s):
        return []
    n = len(moved)
    return [_q(s, "sr_num_moved", 0, "How many objects does the robot move in this clip?", str(n),
               [str(x) for x in range(1, 6) if x != n],
               "tracks3d object points: max point displacements " + ", ".join(
                   f"{k} {f['disp'][k]:.3f} m" for k in f["names"]) + " (moved > 0.05 m, still < 0.01 m); untracked "
               "distractors pushed <= 1 cm, none fell", value=str(n))]


def _tpl_sr_first_moved(s, ctx) -> list:
    f = ctx["sr"]()
    moved = sorted([n for n in f["names"] if f["disp"][n] > 0.05], key=lambda n: f["onset"][n])
    if len(moved) < 2 or f["onset"][moved[1]] - f["onset"][moved[0]] < 20:
        return []
    first = moved[0]
    ans = object_name(first)
    others = [object_name(n) for n in moved[1:] if not _confusable(object_name(n), ans)]
    if not others:
        return []
    still = [object_name(n) for n in f["names"] if f["disp"][n] < 0.01]
    still = [x for x in still if not _confusable(x, ans)]
    vocab = [object_name(x) for x in ctx["vocab"]["Simul-Robot"] if x not in set(f["names"])]
    vocab = [x for x in vocab if not _confusable(x, ans)]
    return [_q(s, "sr_first_moved", 0, "Which of these objects does the robot move first?", ans, others + still + vocab,
               "tracks3d object points: motion onset (first frame with a point > 0.02 m from frame 0) " + ", ".join(
                   f"{n} {f['onset'][n]}" for n in moved) + " (gap >= 20 frames); distractors: the objects moved "
               "later (preferred), still objects and objects of other scenes", value=ans, pref=others)]


END_OPTIONS = ("resting on a surface, released by the robot", "still held in the robot's hand")


def _tpl_sr_end_state(s, ctx) -> list:
    fd = s.attributes.get("flag_details") or {}
    flags = {f.split(":")[0] for f in s.flags}
    sup, held, air = fd.get("primary_end_supported"), fd.get("primary_end_held"), fd.get("primary_end_airborne")
    if sup and not held and not air and "clip_ends_mid_motion" not in flags and not fd.get(
            "clip_ends_mid_motion_borderline"):
        ans = END_OPTIONS[0]
    elif held and air and not sup:
        ans = END_OPTIONS[1]
    else:
        return []
    obj = object_name(s.attributes["primary_object"])
    return [_q(s, "sr_end_state", 0, f"At the end of the clip, where is the {obj}?", ans,
               [o for o in END_OPTIONS if o != ans],
               f"attributes.flag_details: supported={sup}, held={held}, airborne={air}, clip_ends_mid_motion="
               f"{'clip_ends_mid_motion' in flags}, borderline={fd.get('clip_ends_mid_motion_borderline')} "
               f"(min z {fd.get('primary_end_min_z_m')} m, hand distance {fd.get('primary_end_hand_dist_m')} m)",
               value=ans)]


VEHICLE_TYPES = {"vehicle.fuso.mitsubishi": "bus", "vehicle.ambulance.ford": "ambulance",
                 "vehicle.firetruck.actors": "fire truck", "vehicle.carlacola.actors": "box truck",
                 "vehicle.sprinter.mercedes": "van"}
VEHICLE_OPTIONS = ("car", "van", "ambulance", "fire truck", "box truck", "bus")
VEHICLE_ALSO_TRUE = {"ambulance": {"van"}, "bus": {"van"}, "fire truck": {"box truck"}, "box truck": {"fire truck"}}


def _sg_target_name(s) -> str:
    return "pedestrian" if s.attributes.get("category") == "pedestrian" else "vehicle"


def _sg_ok(s) -> bool:
    return "quality_ok_false" not in {f.split(":")[0] for f in s.flags}


def _tpl_sg_agent_type(s, ctx) -> list:
    if not _sg_ok(s):
        return []
    cat = s.attributes.get("category")
    if cat not in ("pedestrian", "vehicle"):
        return []
    ans = "a pedestrian" if cat == "pedestrian" else "a vehicle"
    return [_q(s, "sg_agent_type", 0, "What kind of road user do the cameras follow in this clip?", ans,
               ["a vehicle" if cat == "pedestrian" else "a pedestrian"], f"attributes.category = {cat!r}", value=ans)]


def _tpl_sg_vehicle_type(s, ctx) -> list:
    if not _sg_ok(s) or s.attributes.get("category") != "vehicle":
        return []
    bp = s.attributes.get("blueprint", "")
    typ = VEHICLE_TYPES.get(bp, "car" if s.attributes.get("subcategory") == "car" else None)
    if typ is None:
        return []
    pool = [o for o in VEHICLE_OPTIONS if o != typ and o not in VEHICLE_ALSO_TRUE.get(typ, set())]
    return [_q(s, "sg_vehicle_type", 0, "What type of vehicle do the cameras follow in this clip?", typ, pool,
               f"attributes.blueprint = {bp!r} (subcategory {s.attributes.get('subcategory')!r}) -> {typ!r}", value=typ)]


def _runs(mask: np.ndarray) -> list:
    """Lengths of the True runs of a 1-D bool array."""
    out, n = [], 0
    for x in mask:
        if x:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _tpl_sg_views_out(s, ctx) -> list:
    if not _sg_ok(s) or _label(s, "target") is None:
        return []
    with np.load(_label(s, "target"), allow_pickle=False) as z:
        st = z["state_strict"]
        names = [str(x) for x in z["state_names"]]
    oof = names.index("out_of_frustum")
    vis = names.index("visible")
    out_views = set()
    per = []
    for v in range(s.n_views):
        runs = _runs(st[:, v] == oof)
        nvis = int((st[:, v] == vis).sum())
        if runs and max(runs) >= 8 and nvis >= 8:
            out_views.add(f"view{v}")
            per.append(f"view{v} out-of-frustum run {max(runs)} frames")
        elif not runs:
            per.append(f"view{v} never out of frustum")
        else:
            return []
    V = s.n_views
    n_out = len(out_views)
    if _label(s, "oos_segments") is not None:
        oos = json.loads(_label(s, "oos_segments").read_text())
        if {g["view"] for g in oos.get("segments", []) if g.get("kind") == "out_of_frustum"} != out_views:
            return []
        per.append("agrees with labels/oos_segments.json")
    pool = [str(x) for x in range(V + 1) if x != n_out]
    name = _sg_target_name(s)
    return [_q(s, "sg_views_out", 0, f"In how many of the {V} camera views is the {name} out of the camera's field of "
                                     f"view for a while (at least about half a second) during the clip?", str(n_out),
               pool,
               "labels/target.npz state_strict: " + "; ".join(per) + " (counted: runs >= 8 frames)", value=str(n_out))]


SG_DIST_BINS = ((0.0, 10.0, "less than 10 m"), (10.0, 20.0, "between 10 and 20 m"), (20.0, 40.0, "between 20 and 40 m"),
                (40.0, math.inf, "more than 40 m"))


def _tpl_sg_distance(s, ctx) -> list:
    if not _sg_ok(s):
        return []
    tr = s.tracks
    c = _centroid_track(tr.xyz, tr.valid)
    if not (np.isfinite(c[0]).all() and np.isfinite(c[-1]).all()):
        return []
    d = float(np.linalg.norm(c[-1] - c[0]))
    for lo, hi, _ in SG_DIST_BINS:
        for edge in (lo, hi):
            if 0 < edge < math.inf and abs(d - edge) <= 0.12 * edge:
                return []
    ans = next(lab for lo, hi, lab in SG_DIST_BINS if lo <= d < hi)
    name = _sg_target_name(s)
    return [_q(s, "sg_distance", 0, f"Roughly how far is the {name}'s position at the end of the clip from its position "
                                    f"at the start?", ans, [lab for _, _, lab in SG_DIST_BINS if lab != ans],
               f"tracks3d target centroid: net displacement frame 0 -> {s.n_frames - 1} = {d:.2f} m (bins 10/20/40 m, "
               f"+-12 % around an edge skipped)", value=ans)]


RR_TASKS = {
    "put_on_plate": "putting an object on a plate",
    "nest_cups": "nesting one cup inside another",
    "stack_cubes": "stacking one cube on top of another",
    "plate_and_fruit": "moving a plate to the centre of the table and putting a fruit on it",
    "flip_cup": "flipping a cup over",
    "basket_and_doll": "moving a basket to the centre of the table and putting a doll into it",
    "pot_lid": "putting a lid on a pot or taking it off",
    "marker_and_cup": "holding a cup while putting a marker into it or pulling it out",
    "pull_tissue": "pulling a tissue out of a tissue box",
    "sweep_into_dustpan": "sweeping trash into a dustpan",
    "scrub_bowl": "scrubbing a bowl with a scrub pad",
}
RR_TASK_CONFUSABLE = {"put_on_plate": {"plate_and_fruit"}, "plate_and_fruit": {"put_on_plate"}}


def _rr_facts(s, ctx) -> dict:
    tr = s.tracks
    groups = [str(g) for g in tr.group_names]
    hand = tr.group == groups.index("hand")
    out = {}
    for side, nm in ((0, "left"), (1, "right")):
        m = hand & (tr.side == side)
        c = _centroid_track(tr.xyz[:, m], tr.valid[:, m])
        d = np.linalg.norm(c - c[0], axis=-1)
        d = np.where(np.isfinite(d), d, 0.0)
        out[nm] = {"disp": float(d.max()), "onset": int(np.argmax(d > 0.05)) if (d > 0.05).any() else -1}
    return out


def _tpl_rr_hands(s, ctx) -> list:
    hand = str(s.attributes.get("hand", ""))
    f = ctx["rr"]()
    used = {k for k, v in f.items() if v["disp"] >= 0.15}
    idle = {k for k, v in f.items() if v["disp"] <= 0.10}
    expect = {"left": {"left"}, "right": {"right"}, "bimanual": {"left", "right"}}.get(hand)
    if expect is None or used != expect or len(used) + len(idle) != 2:
        return []
    ans = HAND_OPTIONS[2] if len(used) == 2 else (HAND_OPTIONS[0] if "left" in used else HAND_OPTIONS[1])
    return [_q(s, "rr_hands", 0, "Which of the robot's hands does the robot use in this episode?", ans,
               [o for o in HAND_OPTIONS if o != ans],
               f"captions.structured.hand = {hand!r} (instruction parse); hand-point centroid max displacement left "
               f"{f['left']['disp']:.3f} m, right {f['right']['disp']:.3f} m (used >= 0.15 m, idle <= 0.10 m)",
               value=ans)]


def _tpl_rr_first_hand(s, ctx) -> list:
    steps = (s.captions.get("structured") or {}).get("steps") or []
    if len(steps) < 2 or len(steps[0].get("hands", [])) != 1 or len(steps[1].get("hands", [])) != 1:
        return []
    h1, h2 = steps[0]["hands"][0], steps[1]["hands"][0]
    if h1 == h2:
        return []
    f = ctx["rr"]()
    o1, o2 = f[h1]["onset"], f[h2]["onset"]
    if o1 < 0 or o2 < 0 or o2 - o1 < 15:
        return []
    ans = f"the {h1} hand"
    return [_q(s, "rr_first_hand", 0, "Which of the robot's hands starts working first in this episode?", ans,
               [f"the {h2} hand"], f"captions.structured.steps hands {[st['hands'] for st in steps]}; hand motion "
               f"onsets (> 5 cm from frame 0) {h1} frame {o1}, {h2} frame {o2} (gap >= 15 frames)", value=ans)]


def _tpl_rr_task(s, ctx) -> list:
    fam = (s.captions.get("structured") or {}).get("task_family") or s.attributes.get("task_family")
    if fam not in RR_TASKS:
        return []
    pool = [RR_TASKS[k] for k in RR_TASKS if k != fam and k not in RR_TASK_CONFUSABLE.get(fam, set())]
    return [_q(s, "rr_task", 0, "Which task does the robot perform in this episode?", RR_TASKS[fam], pool,
               f"captions.structured.task_family = {fam!r} (deterministic parse of the task instruction)", value=fam)]


def _tpl_rr_object(s, ctx) -> list:
    objs = [o["name"] for o in (s.captions.get("structured") or {}).get("objects", [])]
    if not objs:
        return []
    nouns = {o["noun"] for o in s.captions["structured"]["objects"]}
    ans = objs[0]
    vocab = ctx["vocab"]["Real-Robot"]
    pool = sorted({n for n, noun in vocab if noun not in nouns and n not in objs})
    return [_q(s, "rr_object", 0, "Which of these objects does the robot handle in this episode?", ans, pool,
               f"captions.structured.objects = {objs} (instruction parse); answer = first mentioned object; distractors "
               f"share no noun with {sorted(nouns)}", value=ans)]


RG_EVENT_TEXT = {
    "walk": "walks several steps", "turn": "turns the whole body around by a large angle",
    "stand_still": "stands still in place without stepping", "bend_forward": "bends forward at the waist",
    "lower_body": "squats, kneels or crouches down low", "arm_raised": "raises a hand above shoulder height",
    "arm_extended": "stretches an arm out to reach", "hands_together": "brings both hands together",
}
RG_ANSWER_TYPES = ("walk", "turn", "stand_still", "bend_forward", "lower_body", "arm_raised", "arm_extended",
                   "hands_together")
RG_DISTRACTOR_TYPES = RG_ANSWER_TYPES
RG_CONFUSION = ({"walk", "shift_position"}, {"bend_forward", "lower_body", "lean_sideways"},
                {"arm_raised", "arm_extended"})
RG_ABSENT = {"walk_travel_m": 0.3, "turn_yaw_deg": 30.0, "still_min_speed_mps": 0.25, "raise_wrist_m": -0.05,
             "bend_pitch_deg": 15.0, "squat_pelvis_rel": 0.85, "reach_m": 0.36, "hands_m": 0.30}
KP = {"nose": 0, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
      "l_hip": 11, "r_hip": 12, "l_ankle": 15, "r_ankle": 16}


def _rg_kin(k3, f0: int, f1: int) -> dict:
    """Keypoint kinematics on frames f0..f1 (clipped; triangulated keypoints, ok frames only): ``travel`` = max
    horizontal pelvis distance from its first position (m), ``min_speed`` = min pelvis speed over 0.5 s (8-frame)
    spans (m/s), ``yaw_change`` = max |hip-line heading change| (deg), ``yaw_net`` (deg, + = counter-clockwise from
    above), ``max_wrist_over_shoulder`` (m), ``max_pitch`` = max torso tilt from vertical (deg), ``min_pelvis_z`` (m),
    ``max_reach`` = max horizontal distance of a wrist from the mid-shoulder (m), ``min_hands`` = min wrist-to-wrist
    distance (m)."""
    T = k3["xyz"].shape[0]
    f0, f1 = max(0, f0), min(T - 1, f1)
    X = k3["xyz"][f0:f1 + 1].astype(np.float64)
    ok = k3["ok"][f0:f1 + 1]
    hips = ok[:, KP["l_hip"]] & ok[:, KP["r_hip"]]
    out = {"travel": _NAN, "min_speed": _NAN, "yaw_change": _NAN, "yaw_net": _NAN, "max_wrist_over_shoulder": _NAN,
           "max_pitch": _NAN, "min_pelvis_z": _NAN, "max_reach": _NAN, "min_hands": _NAN,
           "coverage": float(hips.mean()) if len(hips) else 0.0}
    if len(hips) == 0 or hips.mean() < 0.8:
        return out
    pel = 0.5 * (X[:, KP["l_hip"]] + X[:, KP["r_hip"]])
    good = np.nonzero(hips)[0]
    pg = pel[good][:, :2]
    out["travel"] = float(np.linalg.norm(pg - pg[0], axis=-1).max())
    out["min_pelvis_z"] = float(pel[good, 2].min())
    sh = ok[:, KP["l_shoulder"]] & ok[:, KP["r_shoulder"]] & hips
    if sh.any():
        up = 0.5 * (X[sh, KP["l_shoulder"]] + X[sh, KP["r_shoulder"]]) - pel[sh]
        cosv = up[:, 2] / np.maximum(np.linalg.norm(up, axis=-1), 1e-9)
        out["max_pitch"] = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))).max())
    sp = []
    for i in range(0, len(good) - 8):
        if good[i + 8] - good[i] == 8:
            sp.append(np.linalg.norm(pg[i + 8] - pg[i]) / (8 / 15.0))
    out["min_speed"] = float(min(sp)) if sp else _NAN
    hv = X[good][:, KP["r_hip"], :2] - X[good][:, KP["l_hip"], :2]
    yaw = np.unwrap(np.arctan2(hv[:, 1], hv[:, 0]))
    out["yaw_change"] = float(np.degrees(np.max(np.abs(yaw - yaw[0])))) if len(yaw) else _NAN
    out["yaw_net"] = float(np.degrees(yaw[-1] - yaw[0])) if len(yaw) else _NAN
    wr = []
    for w, sh in (("l_wrist", "l_shoulder"), ("r_wrist", "r_shoulder")):
        m = ok[:, KP[w]] & ok[:, KP[sh]]
        if m.any():
            wr.append(float(np.max(X[m, KP[w], 2] - X[m, KP[sh], 2])))
    out["max_wrist_over_shoulder"] = max(wr) if wr else _NAN
    shm = ok[:, KP["l_shoulder"]] & ok[:, KP["r_shoulder"]]
    mid = 0.5 * (X[:, KP["l_shoulder"]] + X[:, KP["r_shoulder"]])
    reach = []
    for w in ("l_wrist", "r_wrist"):
        m = ok[:, KP[w]] & shm
        if m.mean() < 0.8:
            reach = []
            break
        reach.append(float(np.linalg.norm((X[m, KP[w]] - mid[m])[:, :2], axis=-1).max()))
    if reach:
        out["max_reach"] = max(reach)
    hh = ok[:, KP["l_wrist"]] & ok[:, KP["r_wrist"]]
    if hh.mean() >= 0.8:
        out["min_hands"] = float(np.linalg.norm(X[hh, KP["l_wrist"]] - X[hh, KP["r_wrist"]], axis=-1).min())
    return out


def _events_in(events, w0, w1, margin=0):
    """(present types, types touching the window +- margin)."""
    present, touching = set(), set()
    for e in events:
        f0, f1 = int(e["f0"]), int(e["f1"])
        ov = min(w1, f1) - max(w0, f0) + 1
        if ov >= max(4, min(0.5 * (f1 - f0 + 1), 15)):
            present.add(e["type"])
        if min(w1 + margin, f1) - max(w0 - margin, f0) + 1 > 0:
            touching.add(e["type"])
    return present, touching


def _rg_window_phrase(w0, w1, fps) -> str:
    return "In this video segment"


def _tpl_rg_motion(s, ctx) -> list:
    out = []
    for k, (w0, w1) in enumerate(ctx["p"].windows_for(s)):
        out += _rg_motion_window(s, ctx, k, w0, w1)
    return out


def _rg_motion_window(s, ctx, k: int, w0: int, w1: int) -> list:
    ev = ctx["events"]()
    if ev is None:
        return []
    k3 = ctx["kp3d"]()
    if k3 is None:
        return []
    present, _ = _events_in(ev["events"], w0, w1)
    _, near = _events_in(ev["events"], w0, w1, margin=8)
    kin = _rg_kin(k3, w0 - 8, w1 + 8)
    answers = [t for t in RG_ANSWER_TYPES if t in present]
    if not answers:
        return []
    blocked = set(near)
    for grp in RG_CONFUSION:
        if grp & near:
            blocked |= grp
    standing = _num((ev.get("summary") or {}).get("standing_pelvis_m"))
    th = RG_ABSENT

    def lt(key, v):
        return bool(np.isfinite(kin[key]) and kin[key] < v)

    def gt(key, v):
        return bool(np.isfinite(kin[key]) and kin[key] > v)

    checks = {
        "walk": lt("travel", th["walk_travel_m"]),
        "turn": lt("yaw_change", th["turn_yaw_deg"]),
        "stand_still": gt("min_speed", th["still_min_speed_mps"]),
        "arm_raised": lt("max_wrist_over_shoulder", th["raise_wrist_m"]),
        "bend_forward": lt("max_pitch", th["bend_pitch_deg"]),
        "lower_body": bool(np.isfinite(standing) and standing > 0
                           and gt("min_pelvis_z", th["squat_pelvis_rel"] * standing)),
        "arm_extended": lt("max_reach", th["reach_m"]),
        "hands_together": gt("min_hands", th["hands_m"]),
    }
    pool = [t for t in RG_DISTRACTOR_TYPES if t not in blocked and checks.get(t, False)]
    if not pool:
        return []
    return [_q(s, "rg_motion", f"w{k}", f"{_rg_window_phrase(w0, w1, s.fps)}, which of the following does the main "
                                        f"person (the one performing the motion) do?",
               RG_EVENT_TEXT[answers[0]], [RG_EVENT_TEXT[t] for t in pool],
               f"labels/motion_events.json events in frames {w0}-{w1}: present {sorted(present)} (any present type "
               f"in {list(RG_ANSWER_TYPES)} can be the answer); distractor types have no event within +-8 frames, "
               f"are not confusable with a present type and are absent on keypoints3d (frames {w0 - 8}-{w1 + 8}): "
               f"pelvis max distance {kin['travel']:.2f} m (no walk < {th['walk_travel_m']}), min 0.5 s speed "
               f"{kin['min_speed']:.2f} m/s (never still > {th['still_min_speed_mps']}), max |yaw change| "
               f"{kin['yaw_change']:.0f} deg (no turn < {th['turn_yaw_deg']:.0f}), max wrist-over-shoulder "
               f"{kin['max_wrist_over_shoulder']:.2f} m (no raise < {th['raise_wrist_m']}), max torso tilt "
               f"{kin['max_pitch']:.0f} deg (no bend < {th['bend_pitch_deg']:.0f}), min pelvis height "
               f"{kin['min_pelvis_z']:.2f} m vs standing {standing:.2f} m (no squat > {th['squat_pelvis_rel']:.0%}), "
               f"max wrist reach from the mid-shoulder {kin['max_reach']:.2f} m (no reach < {th['reach_m']}), min "
               f"wrist distance {kin['min_hands']:.2f} m (hands never together > {th['hands_m']})",
               frames=[w0, w1], meta={"present": sorted(present)},
               answers=[(RG_EVENT_TEXT[t], t) for t in answers])]


TURN_OPTIONS = ("to the left (counter-clockwise when seen from above)", "to the right (clockwise when seen from above)")


def _tpl_rg_turn_direction(s, ctx) -> list:
    out = []
    for k, (w0, w1) in enumerate(ctx["p"].windows_for(s)):
        out += _rg_turn_window(s, ctx, k, w0, w1)
    return out


def _rg_turn_window(s, ctx, k: int, w0: int, w1: int) -> list:
    ev = ctx["events"]()
    if ev is None:
        return []
    turns = []
    for e in ev["events"]:
        if e["type"] != "turn":
            continue
        f0, f1 = int(e["f0"]), int(e["f1"])
        ov = min(w1, f1) - max(w0, f0) + 1
        if ov <= 0:
            continue
        if ov < 0.5 * (f1 - f0 + 1):
            return []
        turns.append(e)
    if not turns:
        return []
    dirs = {e.get("direction") for e in turns}
    if len(dirs) != 1 or not all(abs(float(e.get("yaw_deg", 0))) >= 45 for e in turns):
        return []
    d = dirs.pop()
    if d not in ("left", "right"):
        return []
    k3 = ctx["kp3d"]()
    if k3 is None:
        return []
    for e in turns:
        kin = _rg_kin(k3, int(e["f0"]), int(e["f1"]))
        net = kin.get("yaw_net", _NAN)
        if not np.isfinite(net) or (net > 0) != (d == "left") or abs(net) < 20:
            return []
    ans = TURN_OPTIONS[0] if d == "left" else TURN_OPTIONS[1]
    return [_q(s, "rg_turn_direction", f"w{k}", f"{_rg_window_phrase(w0, w1, s.fps)}, which way does the main person "
                                                f"(the one performing the motion) turn?",
               ans,
               [o for o in TURN_OPTIONS if o != ans],
               "labels/motion_events.json turn events " + ", ".join(
                   f"frames {e['f0']}-{e['f1']} yaw {e['yaw_deg']:.0f} deg ({e['direction']})" for e in turns)
               + " (left = counter-clockwise from above, z up); keypoints3d hip-line yaw agrees", frames=[w0, w1],
               value=ans)]


BODY_PART_STRIDE = 4
BODY_PARTS = {"the head": ("nose",), "the left hand (the person's own left)": ("l_wrist",),
              "the right hand (the person's own right)": ("r_wrist",),
              "the left foot (the person's own left)": ("l_ankle",),
              "the right foot (the person's own right)": ("r_ankle",)}


def _tpl_rg_body_part(s, ctx) -> list:
    out = []
    for k, (w0, w1) in enumerate(ctx["p"].windows_for(s)):
        out += _rg_body_part_window(s, ctx, k, w0, w1)
    return out


def _rg_body_part_window(s, ctx, k: int, w0: int, w1: int) -> list:
    k3 = ctx["kp3d"]()
    if k3 is None:
        return []
    X = k3["xyz"][w0:w1 + 1].astype(np.float64)
    ok = k3["ok"][w0:w1 + 1]
    hips = ok[:, KP["l_hip"]] & ok[:, KP["r_hip"]]
    pel = 0.5 * (X[:, KP["l_hip"]] + X[:, KP["r_hip"]])
    mot = {}
    st = BODY_PART_STRIDE
    for name, kps in BODY_PARTS.items():
        j = KP[kps[0]]
        good = ok[:, j] & hips
        pair = good[st:] & good[:-st]
        if len(pair) == 0 or pair.mean() < 0.8:
            return []
        rel = X[:, j] - pel
        step = np.linalg.norm(rel[st:] - rel[:-st], axis=-1)[pair]
        mot[name] = float(step.mean() * (len(X) - 1) / st)
    order = sorted(mot, key=lambda k: -mot[k])
    if mot[order[0]] < 1.5 * mot[order[1]] or mot[order[0]] < 0.3:
        return []
    ans = order[0]
    dis = [x for x in BODY_PARTS if x != ans]
    return [_q(s, "rg_body_part", f"w{k}", f"{_rg_window_phrase(w0, w1, s.fps)}, which body part of the main person "
                                           f"(the one performing the motion) moves the most relative to their hips?",
               ans, dis,
               "labels/human/keypoints3d.npz path length relative to the mid-hip on a 4-frame grid (frames %d-%d): "
               % (w0, w1)
               + ", ".join(f"{k.split(' (')[0]} {v:.2f} m" for k, v in mot.items()) + " (winner >= 1.5x the second)",
               frames=[w0, w1], value=ans.split(" (")[0])]


def _usable_segments(s, p: UnderstandingProtocol) -> list:
    segs = [g for g in s.captions.get("segments", []) if not g.get("clipped")
            and int(g["end_frame"]) - int(g["start_frame"]) + 1 >= p.min_segment_frames and g.get("caption")]
    out = []
    for g in segs:
        if any(h is not g and _jaccard(g["caption"], h["caption"]) >= p.max_caption_jaccard for h in segs):
            continue
        out.append(g)
    return out


def _tpl_rg_segment_order(s, ctx) -> list:
    p = ctx["p"]
    segs = _usable_segments(s, p)
    pairs = [(i, j) for i in range(len(segs)) for j in range(i + 2, len(segs))]
    pairs = [(i, j) for i, j in pairs if _jaccard(segs[i]["caption"], segs[j]["caption"]) < p.max_caption_jaccard]
    if not pairs:
        return []
    rng = _rng(p.seed, s.uid, "rg_segment_order")
    chosen, used = [], set()
    for i in rng.permutation(len(pairs)):
        a, b = pairs[int(i)]
        if a not in used and b not in used and len(chosen) < p.max_pairs_per_clip:
            chosen.append((a, b))
            used |= {a, b}
    out = []
    for k, (i, j) in enumerate(sorted(chosen)):
        a, b = segs[i], segs[j]
        out.append(_q(s, "rg_segment_order", k, "Which of these two descriptions happens first in the clip?",
                      a["caption"], [b["caption"]],
                      f"captions.json segments: frames {a['start_frame']}-{a['end_frame']} before "
                      f"{b['start_frame']}-{b['end_frame']} (non-adjacent, unclipped, token-Jaccard "
                      f"{_jaccard(a['caption'], b['caption']):.2f})", value=f"first@{k}"))
    return out


TEMPLATE_FUNCS = {
    "num_views": _tpl_num_views, "camera_motion": _tpl_camera_motion, "target_distance": _tpl_target_distance,
    "sr_hands": _tpl_sr_hands, "sr_moved_object": _tpl_sr_moved_object, "sr_num_moved": _tpl_sr_num_moved,
    "sr_first_moved": _tpl_sr_first_moved, "sr_end_state": _tpl_sr_end_state,
    "sg_agent_type": _tpl_sg_agent_type, "sg_vehicle_type": _tpl_sg_vehicle_type, "sg_views_out": _tpl_sg_views_out,
    "sg_distance": _tpl_sg_distance,
    "rr_hands": _tpl_rr_hands, "rr_first_hand": _tpl_rr_first_hand, "rr_task": _tpl_rr_task,
    "rr_object": _tpl_rr_object,
    "rg_motion": _tpl_rg_motion, "rg_turn_direction": _tpl_rg_turn_direction, "rg_body_part": _tpl_rg_body_part,
    "rg_segment_order": _tpl_rg_segment_order,
}


def _vocabulary(bench, subset: str) -> list:
    """Distractor vocabulary of a subset, from ALL its samples (independent of any ids filter)."""
    if subset == "Simul-Robot":
        names = set()
        for s in bench.samples(subset):
            p = _label(s, "objects")
            if p is not None:
                names.update(x["id"] for x in json.loads(p.read_text()).get("objects", []))
            else:
                names.update(x for x in np.asarray(s.tracks.extra.get("object_id", [])).astype(str) if x)
        return sorted(names)
    if subset == "Real-Robot":
        out = set()
        for s in bench.samples(subset):
            for o in (s.captions.get("structured") or {}).get("objects", []):
                out.add((o["name"], o["noun"]))
        return sorted(out)
    return []


class _Ctx(dict):
    pass


def _context(sample, protocol, vocab) -> dict:
    cache = {}

    def lazy(key, fn):
        def get():
            if key not in cache:
                cache[key] = fn()
            return cache[key]
        return get

    def kp3d():
        p = _label(sample, "keypoints3d")
        if p is None:
            return None
        with np.load(p, allow_pickle=False) as z:
            return {"xyz": z["xyz"], "ok": z["ok"].astype(bool)}

    def events():
        p = _label(sample, "motion_events")
        return None if p is None else json.loads(p.read_text())

    c = _Ctx(p=protocol, vocab=vocab)
    c["kp3d"] = lazy("kp3d", kp3d)
    c["events"] = lazy("events", events)
    c["sr"] = lazy("sr", lambda: _sr_facts(sample, c))
    c["rr"] = lazy("rr", lambda: _rr_facts(sample, c))
    return c


def sample_candidates(sample, protocol: UnderstandingProtocol | None = None, vocab: dict | None = None) -> list:
    """Candidate questions of one sample (before option ordering / balancing)."""
    p = protocol or UnderstandingProtocol()
    vocab = vocab or {}
    ctx = _context(sample, p, vocab)
    out = []
    for t in p.templates_for(sample.subset):
        out += TEMPLATE_FUNCS[t](sample, ctx)
    return out


def _cand_one(args) -> list:
    root, subset, sid, pdict, vocab = args
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    return sample_candidates(s, UnderstandingProtocol.from_dict(pdict), vocab)


def finalize(cands: list, protocol: UnderstandingProtocol | None = None) -> list:
    """Choose the shown answer and the distractors of every candidate and balance answers, option texts and answer
    positions per (template, subset) (module docstring, "Balancing"); returns the questions with ``options`` /
    ``answer``. Deterministic: depends only on the candidates of the group and the protocol."""
    p = protocol or UnderstandingProtocol()
    groups = {}
    for c in cands:
        groups.setdefault((c["template"], c["subset"]), []).append(c)
    out = []
    for (tpl, sub), qs in sorted(groups.items()):
        qs = sorted(qs, key=lambda q: _hash_int(p.seed, "order", q["qid"]))
        if p.balance and tpl not in BALANCE_EXEMPT:
            qs = _balance_group(qs, p)
        else:
            qs = [_plain_options(q, p) for q in qs]
        out += _place_answers(qs, p)
    return sorted(out, key=lambda q: q["qid"])


def _plain_options(q: dict, p: UnderstandingProtocol) -> dict:
    """No balancing (BALANCE_EXEMPT or balance=False): a seeded answer alternative and seeded distractors, the
    preferred ones first."""
    rng = _rng(p.seed, "distractors", q["qid"])
    pref = list(q["pref"])
    rest = [x for x in q["pool"] if x not in set(pref)]
    order = [pref[int(i)] for i in rng.permutation(len(pref))] + [rest[int(i)] for i in rng.permutation(len(rest))]
    r = dict(q)
    if q.get("answers"):
        r["answer_text"], r["value"] = q["answers"][int(rng.integers(len(q["answers"])))]
    r["distractors"] = order[: p.n_options - 1]
    return r


def _cap_values(qs: list, p: UnderstandingProtocol) -> list:
    """Down-sample the most frequent answer values (last in hash order first) until none exceeds max_answer_share."""
    vals = {}
    for q in qs:
        vals.setdefault(q["value"], []).append(q)
    if len(vals) < 2:
        return []
    while True:
        n = sum(len(v) for v in vals.values())
        big = max(vals, key=lambda k: (len(vals[k]), k))
        if len(vals[big]) <= max(1, math.floor(p.max_answer_share * n)):
            break
        vals[big].pop()
    keep = {q["qid"] for v in vals.values() for q in v}
    return [q for q in qs if q["qid"] in keep]


def option_stats(questions: list) -> dict:
    """{option text: {"n": appearances, "n_answer": times it is the answer, "chance": mean 1/n_options over its
    appearances, "answer_rate": n_answer / n}} over a list of questions (with ``options`` + ``answer_text``, or
    ``distractors`` + ``answer_text``)."""
    st = {}
    for q in questions:
        opts = q["options"] if "options" in q else [q["answer_text"]] + list(q["distractors"])
        for o in opts:
            r = st.setdefault(o, {"n": 0, "n_answer": 0, "chance": 0.0})
            r["n"] += 1
            r["chance"] += 1.0 / len(opts)
        st[q["answer_text"]]["n_answer"] += 1
    for r in st.values():
        r["chance"] /= r["n"]
        r["answer_rate"] = r["n_answer"] / r["n"]
    return st


def _option_guard(qs: list, p: UnderstandingProtocol) -> list:
    """Until every option text with >= option_bias_min_count appearances has |answer rate - chance| <=
    max_option_bias and every shown text is the answer of some question: take the worst text; if it is too often the
    answer, drop the last question (hash order) it answers; if it is too often a distractor (or never the answer),
    remove it from the last question that keeps >= 2 options (else drop that question)."""
    qs = list(qs)
    for _ in range(20 * len(qs) + 10):
        worst, wv = None, p.max_option_bias + 1e-9
        for o, r in option_stats(qs).items():
            dev = r["answer_rate"] - r["chance"]
            if r["n_answer"] == 0:
                dev = -1.0
            elif r["n"] < p.option_bias_min_count:
                continue
            if abs(dev) > wv:
                worst, wv = (o, dev), abs(dev)
        if worst is None:
            break
        o, dev = worst
        if dev > 0:
            qs.pop(max(j for j, q in enumerate(qs) if q["answer_text"] == o))
            continue
        js = [j for j, q in enumerate(qs) if o in q["distractors"]]
        jj = [j for j in js if len(qs[j]["distractors"]) > 1]
        if jj:
            q = dict(qs[jj[-1]])
            q["distractors"] = [x for x in q["distractors"] if x != o]
            qs[jj[-1]] = q
        else:
            qs.pop(js[-1])
    return qs


def _balance_group(qs: list, p: UnderstandingProtocol) -> list:
    """Balancing of one (template, subset) group (module docstring, "Balancing")."""
    n_dis = p.n_options - 1
    qs = [dict(q) for q in qs]
    avail = {}
    for q in qs:
        for o in q["pool"]:
            avail[o] = avail.get(o, 0) + 1
    A = {}
    for q in qs:
        if not q.get("answers"):
            A[q["answer_text"]] = A.get(q["answer_text"], 0) + 1
    for q in qs:
        if q.get("answers"):
            t, v = max(q["answers"], key=lambda tv: (avail.get(tv[0], 0) / n_dis - A.get(tv[0], 0),
                                                     -_hash_int(p.seed, "answer", q["qid"], tv[0])))
            q["answer_text"], q["value"] = t, v
            A[t] = A.get(t, 0) + 1
    qs = _cap_values(qs, p)
    A, D = {}, {}
    for q in qs:
        A[q["answer_text"]] = A.get(q["answer_text"], 0) + 1
    kept = []
    for q in qs:
        elig = [o for o in q["pool"] if A.get(o, 0) > 0]
        if not elig:
            continue
        pref = set(q["pref"])

        def key(o, q=q, pref=pref):
            need = A[o] * n_dis - D.get(o, 0)
            return need <= 0, o not in pref, -need, _hash_int(p.seed, "distractor", q["qid"], o)

        q["distractors"] = sorted(elig, key=key)[:n_dis]
        for o in q["distractors"]:
            D[o] = D.get(o, 0) + 1
        kept.append(q)
    kept = _option_guard(kept, p)
    return kept if len({q["value"] for q in kept}) >= 2 else []


def _place_answers(qs: list, p: UnderstandingProtocol) -> list:
    """Answer position cyclic over the questions of the same option count (hash order); distractors seeded."""
    out, counter = [], {}
    for q in qs:
        n_opt = 1 + len(q["distractors"])
        pos = counter.get(n_opt, 0) % n_opt
        counter[n_opt] = counter.get(n_opt, 0) + 1
        rng = _rng(p.seed, "options", q["qid"])
        dis = [q["distractors"][int(j)] for j in rng.permutation(len(q["distractors"]))]
        r = {k: v for k, v in q.items() if k not in ("distractors", "pool", "pref", "answers")}
        r["options"] = dis[:pos] + [q["answer_text"]] + dis[pos:]
        r["answer"] = LETTERS[pos]
        out.append(r)
    return out


def generate_questions(bench, protocol: UnderstandingProtocol | None = None, subsets=None, jobs: int = 1) -> list:
    """The QA set of the requested subsets (every sample of each subset; see the module docstring)."""
    p = protocol or UnderstandingProtocol()
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    vocab = {sub: _vocabulary(bench, sub) for sub in subs}
    work = [(str(bench.root), sub, s.id, p.to_dict(), vocab) for sub in subs for s in bench.samples(sub)]
    cands = [c for r in _map(_cand_one, work, jobs) for c in r]
    return finalize(cands, p)


def format_prompt(q: dict) -> str:
    """A plain-text prompt: the multi-view setting, the question and the lettered options."""
    views = ", ".join(f"view{v}" for v in q["views"])
    lines = [f"You are given synchronized videos of the same scene from different cameras ({views}).", q["question"],
             "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"])]
    lines.append("Answer with the option's letter.")
    return "\n".join(lines)


QUESTION_KEYS = ("qid", "uid", "subset", "id", "template", "question", "options", "frames", "views", "videos", "fps")
KEY_KEYS = ("qid", "uid", "subset", "id", "template", "answer", "answer_text", "options", "value", "source", "meta")


def retrieval_items(bench, subset: str, level: str = "clip", protocol: UnderstandingProtocol | None = None) -> dict:
    """{"videos": [{index, uid, id, frames, videos, fps}], "texts": [{index, text_id, text}], "gt" (n_texts, n_videos)
    bool, "level"} for one subset. Texts are in a seeded random order."""
    p = protocol or UnderstandingProtocol()
    subset = canonical_subset(subset)
    if level not in ("clip", "segment"):
        raise ValueError("level must be 'clip' or 'segment'")
    if level == "segment" and subset != "Real-General":
        raise ValueError("segment-level retrieval exists for Real-General only")
    videos, texts = [], []
    for s in bench.samples(subset):
        vids = [f"{s.subset}/{s.id}/{v['video']}" for v in s.meta["views"]]
        if level == "clip":
            videos.append({"uid": s.uid, "id": s.id, "frames": [0, s.n_frames - 1], "videos": vids, "fps": s.fps})
            texts.append({"text": s.captions.get("caption", s.caption), "uid": s.uid,
                          "frames": [0, s.n_frames - 1]})
        else:
            for g in _usable_segments(s, p):
                fr = [int(g["start_frame"]), int(g["end_frame"])]
                videos.append({"uid": s.uid, "id": s.id, "frames": fr, "videos": vids, "fps": s.fps})
                texts.append({"text": g["caption"], "uid": s.uid, "frames": fr})
    perm = _rng(p.seed, "retrieval", subset, level).permutation(len(texts))
    texts = [texts[int(i)] for i in perm]
    key_v = {}
    for i, v in enumerate(videos):
        key_v.setdefault((v["uid"], tuple(v["frames"])), i)
    own = np.array([key_v[(t["uid"], tuple(t["frames"]))] for t in texts], np.int64)
    gt = np.zeros((len(texts), len(videos)), bool)
    gt[np.arange(len(texts)), own] = True
    by_text = {}
    for j, t in enumerate(texts):
        by_text.setdefault(normalize_text(t["text"]), []).append(int(own[j]))
    for j, t in enumerate(texts):
        gt[j, by_text[normalize_text(t["text"])]] = True
    tag = _retrieval_tag(subset, level)
    return {"subset": subset, "level": level, "tag": tag,
            "videos": [{"index": i, **v} for i, v in enumerate(videos)],
            "texts": [{"index": j, "text_id": f"{tag}/t{j:05d}", "text": t["text"]} for j, t in enumerate(texts)],
            "gt": gt, "owner": own}


def retrieval_levels(subsets, segment: bool = False) -> list:
    """[(subset, level)]: the clip level of every subset, plus the Real-General segment level if ``segment``."""
    out = []
    for sub in subsets:
        out.append((sub, "clip"))
        if segment and sub == "Real-General":
            out.append((sub, "segment"))
    return out


def _retrieval_tag(subset: str, level: str) -> str:
    return subset if level == "clip" else f"{subset}_segment"


def grounding_queries(bench, protocol: UnderstandingProtocol | None = None, ids=None) -> list:
    """Real-General segment-caption queries with their GT [start, end] (inclusive frames)."""
    p = protocol or UnderstandingProtocol()
    out = []
    if "Real-General" not in bench.subsets():
        return out
    for s in bench.samples("Real-General", ids=ids):
        segs = _usable_segments(s, p)
        perm = _rng(p.seed, "grounding", s.uid).permutation(len(segs))
        for k, i in enumerate(perm):
            g = segs[int(i)]
            out.append({"query_id": f"{s.uid}#q{k}", "uid": s.uid, "subset": s.subset, "id": s.id,
                        "caption": g["caption"], "n_frames": s.n_frames, "fps": s.fps,
                        "videos": [f"{s.subset}/{s.id}/{v['video']}" for v in s.meta["views"]],
                        "gt": [int(g["start_frame"]), int(g["end_frame"])]})
    return out


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def temporal_iou(pred, gt) -> float:
    """IoU of two inclusive integer frame ranges [s, e] (non-integer bounds rounded half up; 0 for an invalid or
    empty prediction)."""
    try:
        s1, e1 = (_round_half_up(_finite(pred[0])), _round_half_up(_finite(pred[1])))
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        return 0.0
    s2, e2 = int(gt[0]), int(gt[1])
    if e1 < s1:
        return 0.0
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


GROUNDING_THRESHOLDS = (0.3, 0.5, 0.7)


def grounding_metrics(ious) -> dict:
    v = np.asarray(list(ious), np.float64)
    if v.size == 0:
        return {"miou": _NAN, **{f"R@{t}": _NAN for t in GROUNDING_THRESHOLDS}, "n": 0}
    out = {"miou": 100.0 * float(v.mean()), "n": int(v.size)}
    for t in GROUNDING_THRESHOLDS:
        out[f"R@{t}"] = 100.0 * float(np.mean(v >= t))
    return out


def _finite(x) -> float:
    """float(x) for a finite real number (bools, None, strings, NaN / inf raise)."""
    if x is None or isinstance(x, (bool, np.bool_, str, bytes)):
        raise TypeError(f"not a number: {x!r}")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"not finite: {x!r}")
    return v


def parse_interval(rec, fps: float, n_frames: Optional[int] = None) -> tuple:
    """A grounding prediction record -> ([start, end] inclusive integer frames, status). Frames: ``start_frame`` /
    ``end_frame`` (non-integers rounded half up). Seconds: ``start_s`` / ``end_s`` = the half-open time range
    [start_s, end_s), i.e. the frames t with start_s <= t / fps < end_s: start = ceil(start_s * fps), end =
    ceil(end_s * fps) - 1 (the convention of captions.json; 1e-6 frame tolerance for float error). With n_frames the
    range is clipped to the clip, 0 .. n_frames - 1. status: "ok"; "empty" (end < start after clipping, IoU 0);
    "invalid" (not a JSON object, missing keys, non-numeric or non-finite values; IoU 0)."""
    if not isinstance(rec, dict):
        return None, "invalid"
    try:
        if "start_frame" in rec or "end_frame" in rec:
            s0 = _round_half_up(_finite(rec["start_frame"]))
            e0 = _round_half_up(_finite(rec["end_frame"]))
        elif "start_s" in rec or "end_s" in rec:
            s0 = int(math.ceil(_finite(rec["start_s"]) * fps - 1e-6))
            e0 = int(math.ceil(_finite(rec["end_s"]) * fps - 1e-6)) - 1
        else:
            return None, "invalid"
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid"
    if n_frames is not None:
        s0, e0 = max(s0, 0), min(e0, int(n_frames) - 1)
    return [s0, e0], ("ok" if e0 >= s0 else "empty")


SUBTASKS = ("qa", "retrieval", "retrieval_segment", "grounding")
DEFAULT_SUBTASKS = ("qa", "retrieval", "grounding")
SEGMENT_WARNING = ("retrieval/Real-General_segment lists the GT frame range of every grounding query: a grounding "
                   "method must not read it (export it only for segment-level retrieval, in another folder)")


def _check_subtasks(tasks) -> tuple:
    tasks = tuple(tasks)
    unknown = set(tasks) - set(SUBTASKS)
    if unknown:
        raise ValueError(f"unknown understanding sub-tasks {sorted(unknown)}; choose from {SUBTASKS}")
    return tasks


def _filter_ids(items: list, ids) -> list:
    if ids is None:
        return items
    keep = set(ids)
    return [x for x in items if x["uid"] in keep or x["id"] in keep]


def _select_ids(bench, subs, ids):
    """`ids` with subset aliases in uids resolved; warns about ids that match no sample of `subs`
    (`common.select_samples`)."""
    if ids is None:
        return None
    select_samples(bench, subs, ids)
    return canonical_ids(ids)


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def write_answer_key(questions: list, path) -> Path:
    """Write the answer key (one JSON line per question: KEY_KEYS) of `questions` to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fk:
        for q in questions:
            fk.write(json.dumps(jsonable({k: q[k] for k in KEY_KEYS}), ensure_ascii=False) + "\n")
    return path


def export_inputs(bench, out_dir, protocol: UnderstandingProtocol | None = None, subsets=None, ids=None,
                  tasks=DEFAULT_SUBTASKS, jobs: int = 1, overwrite: bool = False, answer_key_out=None) -> dict:
    """Write ``qa/questions.jsonl``, ``retrieval/<Subset>/{videos,texts}.jsonl`` and ``grounding/queries.jsonl``
    under out_dir (module docstring). ``tasks`` selects the sub-tasks; the segment-level retrieval gallery
    ("retrieval_segment") is written only on request, since it lists the GT segments of the grounding queries. ``ids``
    filters the QA questions and grounding queries (the retrieval galleries are whole subsets). An out dir written with
    other options is refused (`common.RunDirError`) unless ``overwrite``.

    The QA answer key is never written into out_dir by default (a model may read that folder): ``answer_key_out``
    (a file path, needs the "qa" sub-task) writes it to that path, for `evaluate(answer_key=...)`; a warning is given
    when the path lies inside out_dir. A qa/answer_key.jsonl left in out_dir by an earlier version is removed when the
    questions are rewritten."""
    p = protocol or UnderstandingProtocol()
    tasks = _check_subtasks(tasks)
    if answer_key_out is not None and "qa" not in tasks:
        raise ValueError("answer_key_out needs the 'qa' sub-task")
    out_dir = Path(out_dir)
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    ids = _select_ids(bench, subs, ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = {"task": TASK, "protocol": p.to_dict(), "tasks": sorted(tasks), "subsets": subs,
             "ids": None if ids is None else sorted(ids)}
    check_out_stamp(out_dir, "export.json", stamp, overwrite, TASK)
    res = {}
    if "qa" in tasks:
        qs = _filter_ids(generate_questions(bench, p, subs, jobs), ids)
        d = out_dir / "qa"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "questions.jsonl", "w") as fq:
            for q in qs:
                rec = {k: q[k] for k in QUESTION_KEYS}
                rec["prompt"] = format_prompt(q)
                fq.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res["qa"] = {"n_questions": len(qs), "per_template": _count(qs, "template"), "per_subset": _count(qs, "subset")}
        old_key = d / "answer_key.jsonl"
        if old_key.exists() and (answer_key_out is None or Path(answer_key_out).resolve() != old_key.resolve()):
            old_key.unlink()
            warnings.warn(f"removed {old_key}: the answer key is no longer written into the inputs folder (use "
                          "answer_key_out / --answer-key-out)")
        if answer_key_out is not None:
            key = write_answer_key(qs, answer_key_out)
            res["qa"]["answer_key"] = str(key)
            if _inside(key, out_dir):
                warnings.warn(f"the answer key {key} is inside the inputs folder {out_dir}: do not give this folder "
                              "to a model")
    levels = retrieval_levels(subs, segment="retrieval_segment" in tasks)
    if "retrieval" not in tasks:
        levels = [x for x in levels if x[1] == "segment"]
    if levels:
        res["retrieval"] = {}
    for sub, level in levels:
        it = retrieval_items(bench, sub, level, p)
        d = out_dir / "retrieval" / it["tag"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "videos.jsonl", "w") as f:
            for v in it["videos"]:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        with open(d / "texts.jsonl", "w") as f:
            for t in it["texts"]:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        if level == "segment":
            (d / "README.txt").write_text(SEGMENT_WARNING + "\n")
            res["warning"] = SEGMENT_WARNING
        res["retrieval"][it["tag"]] = {"n_videos": len(it["videos"]), "n_texts": len(it["texts"])}
    if "grounding" in tasks and "Real-General" in subs:
        qs = grounding_queries(bench, p, ids=ids)
        d = out_dir / "grounding"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "queries.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps({k: v for k, v in q.items() if k != "gt"}, ensure_ascii=False) + "\n")
        res["grounding"] = {"n_queries": len(qs)}
    write_json(out_dir / "protocol.json", p.to_dict())
    write_json(out_dir / "export.json", stamp)
    return res


def _count(qs, key) -> dict:
    out = {}
    for q in qs:
        out[q[key]] = out.get(q[key], 0) + 1
    return dict(sorted(out.items()))


def _read_jsonl(path, tolerant: bool = False) -> tuple:
    """(records, n_bad_lines). tolerant: undecodable lines and non-object records are counted, not raised."""
    out, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                if not tolerant:
                    raise
                bad += 1
                continue
            if not isinstance(r, dict):
                if not tolerant:
                    raise ValueError(f"{path}: a line is not a JSON object")
                bad += 1
                continue
            out.append(r)
    return out, bad


def _load_answer_key(bench, protocol, subsets, answer_key, jobs) -> list:
    if answer_key is not None:
        key, _ = _read_jsonl(answer_key)
        return [q for q in key if q["subset"] in subsets]
    return generate_questions(bench, protocol, subsets, jobs)


def score_qa(questions: list, predictions: dict) -> list:
    """Per-question results: predictions {qid: record with "response" or "choice"}."""
    rows = []
    for q in questions:
        r = predictions.get(q["qid"])
        status, letter = "ok", None
        if r is None:
            status = "missing"
        elif r.get("choice") is not None:
            c = r["choice"]
            if isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)):
                letter = LETTERS[int(c)] if 0 <= int(c) < len(q["options"]) else None
            else:
                letter = extract_choice(str(c), len(q["options"]), q["options"])
        else:
            resp = r.get("response")
            letter = extract_choice(resp if isinstance(resp, str) else None, len(q["options"]), q["options"])
        if status == "ok" and letter is None:
            status = "unparsed"
        rows.append({"qid": q["qid"], "uid": q["uid"], "subset": q["subset"], "template": q["template"],
                     "answer": q["answer"], "pred": letter, "correct": bool(letter == q["answer"]), "status": status,
                     "sanity": q["template"] in SANITY_TEMPLATES})
    return rows


def _qa_accuracy(rows: list) -> dict:
    """Per-sample QA numbers: headline accuracy over the non-sanity questions, sanity accuracy apart."""
    main = [x["correct"] for x in rows if not x.get("sanity")]
    san = [x["correct"] for x in rows if x.get("sanity")]
    return {"accuracy": 100.0 * float(np.mean(main)) if main else _NAN, "n": len(main), "n_correct": int(sum(main)),
            "sanity_accuracy": 100.0 * float(np.mean(san)) if san else _NAN, "n_sanity": len(san)}


class NoPredictionFiles(FileNotFoundError):
    """`evaluate` found no prediction file of any sub-task under pred_root for the selected subsets: nothing is
    scored (the command line exits with status 1 and one line)."""


def _find_retrieval_pred(pred_root: Path, tag: str):
    for cand in (pred_root / "retrieval" / f"{tag}.npy", pred_root / "retrieval" / f"{tag}.npz"):
        if cand.exists():
            return cand
    return None


def _found_subtasks(pred_root: Path, subsets) -> list:
    """The sub-tasks with a prediction file under pred_root for `subsets`: qa (qa.jsonl), retrieval
    (retrieval/<Subset>[_segment].npy|npz), grounding (grounding.jsonl; with Real-General only)."""
    found = ["qa"] if (pred_root / "qa.jsonl").exists() else []
    if any(_find_retrieval_pred(pred_root, _retrieval_tag(sub, level)) is not None
           for sub, level in retrieval_levels(subsets, segment=True)):
        found.append("retrieval")
    if (pred_root / "grounding.jsonl").exists() and "Real-General" in subsets:
        found.append("grounding")
    return found


def _no_predictions(pred_root: Path) -> NoPredictionFiles:
    return NoPredictionFiles(f"no prediction file for the selected subsets under {pred_root} (qa.jsonl, "
                             "retrieval/<Subset>.npy or grounding.jsonl)")


def evaluate(bench, pred_root, out_dir, protocol: UnderstandingProtocol | None = None, subsets=None, ids=None,
             answer_key=None, jobs: int = 1, overwrite: bool = False) -> dict:
    """Score qa.jsonl / retrieval/*.npy / grounding.jsonl found under pred_root; writes per-sample JSONs and the
    summary. answer_key: optional path to an answer key written by `export_inputs(answer_key_out=...)` (default:
    regenerated from the data). An out dir holding the results of another prediction folder / protocol / answer key
    is refused (`common.RunDirError`) unless ``overwrite`` (which first deletes its per-sample results). Without any
    prediction file for the selected subsets, raises `NoPredictionFiles` before the out dir is touched."""
    p = protocol or UnderstandingProtocol()
    pred_root, out_dir = Path(pred_root), Path(out_dir)
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    if not _found_subtasks(pred_root, subs):
        raise _no_predictions(pred_root)
    ids = _select_ids(bench, subs, ids)
    keep = None if ids is None else set(ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {"task": TASK, "protocol": p.to_dict(), "pred_root": str(pred_root.resolve()),
           "answer_key": None if answer_key is None else str(Path(answer_key).resolve())}
    if check_out_stamp(out_dir, "run.json", run, overwrite, TASK) or overwrite:
        _clear_results(out_dir)

    def wanted(uid, sid):
        return keep is None or uid in keep or sid in keep

    per_sample = {}

    def rec(uid, subset, sid):
        return per_sample.setdefault(uid, {"task": TASK, "uid": uid, "subset": subset, "id": sid, "status": "ok"})

    found, notes = [], []
    qa_file = pred_root / "qa.jsonl"
    qa_meta = None
    if qa_file.exists():
        found.append("qa")
        questions = [q for q in _load_answer_key(bench, p, subs, answer_key, jobs) if wanted(q["uid"], q["id"])]
        recs, bad = _read_jsonl(qa_file, tolerant=True)
        preds = {r["qid"]: r for r in recs if isinstance(r.get("qid"), str)}
        qa_rows = score_qa(questions, preds)
        known = {q["qid"] for q in questions}
        for q, row in zip(questions, qa_rows):
            r = rec(q["uid"], q["subset"], q["id"])
            r.setdefault("qa", {"questions": []})["questions"].append(row)
        for r in per_sample.values():
            if "qa" in r:
                r["qa"].update(_qa_accuracy(r["qa"]["questions"]))
        qa_meta = {"n_questions": len(qa_rows), "n_predictions_unknown_qid": sum(1 for k in preds if k not in known),
                   "n_bad_lines": bad + sum(1 for r in recs if not isinstance(r.get("qid"), str)),
                   "answer_key": None if answer_key is None else str(answer_key)}
    retr = {}
    for sub, level in retrieval_levels(subs, segment=True):
        tag = _retrieval_tag(sub, level)
        f = _find_retrieval_pred(pred_root, tag)
        if f is None:
            continue
        if "retrieval" not in found:
            found.append("retrieval")
        it = retrieval_items(bench, sub, level, p)
        if f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as z:
                sim = z["sim"]
        else:
            sim = np.load(f, allow_pickle=False)
        sim = np.asarray(sim, np.float64)
        if sim.shape != (len(it["videos"]), len(it["texts"])):
            raise ValueError(f"{f}: similarity {sim.shape} != (n_videos, n_texts) = "
                             f"{(len(it['videos']), len(it['texts']))}")
        S = sim.T
        res = bidirectional_retrieval(S, it["gt"])
        t2v = retrieval_ranks(S, it["gt"])
        v2t = retrieval_ranks(S.T, it["gt"].T)
        res["text_to_video_se"] = _rank_se(t2v)
        res["video_to_text_se"] = _rank_se(v2t)
        res["n_nonfinite"] = int((~np.isfinite(sim)).sum())
        retr[tag] = res
        if level == "clip":
            for i, v in enumerate(it["videos"]):
                if wanted(v["uid"], v["id"]):
                    r = rec(v["uid"], sub, v["id"])
                    r["retrieval"] = {"video_to_text_rank": int(v2t[i]),
                                      "text_to_video_rank": int(min(t2v[j] for j in np.nonzero(it["owner"] == i)[0]))}
    gfile = pred_root / "grounding.jsonl"
    gmeta = None
    if gfile.exists() and "Real-General" in subs:
        found.append("grounding")
        recs, bad = _read_jsonl(gfile, tolerant=True)
        preds = {}
        for r in recs:
            if isinstance(r.get("query_id"), str):
                preds[r["query_id"]] = r
            else:
                bad += 1
        queries = grounding_queries(bench, p)
        for q in queries:
            if not wanted(q["uid"], q["id"]):
                continue
            pr = preds.get(q["query_id"])
            iv, status = parse_interval(pr, q["fps"], q["n_frames"]) if pr is not None else (None, "missing")
            iou = temporal_iou(iv, q["gt"]) if status == "ok" else 0.0
            r = rec(q["uid"], q["subset"], q["id"])
            r.setdefault("grounding", {"queries": []})["queries"].append(
                {"query_id": q["query_id"], "gt": q["gt"], "pred": iv, "iou": iou, "status": status})
        for r in per_sample.values():
            if "grounding" in r:
                r["grounding"].update(grounding_metrics(x["iou"] for x in r["grounding"]["queries"]))
        known = {q["query_id"] for q in queries}
        gmeta = {"n_bad_lines": bad, "n_predictions_unknown_query": sum(1 for k in preds if k not in known)}
        if _find_retrieval_pred(pred_root, "Real-General_segment") is not None:
            notes.append("grounding.jsonl and a segment-level retrieval prediction come from the same prediction "
                         "folder: " + SEGMENT_WARNING)
            warnings.warn(notes[-1])
    if not found:
        raise _no_predictions(pred_root)
    for uid, r in per_sample.items():
        write_json(out_dir / r["subset"] / f"{r['id']}.json", r)
    write_json(out_dir / "retrieval.json", retr)
    write_json(out_dir / "protocol.json", p.to_dict())
    write_json(out_dir / "meta.json", {"found": found, "qa": qa_meta, "grounding": gmeta, "notes": notes})
    write_json(out_dir / "run.json", run)
    return summarize(out_dir)


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
    for name in ("summary.json", "summary.md", "retrieval.json", "meta.json", "qa_meta.json"):
        if (out_dir / name).exists():
            (out_dir / name).unlink()


def _rank_se(ranks) -> dict:
    r = np.asarray(ranks)
    n = r.size
    out = {}
    for k in (1, 5, 10):
        pk = float(np.mean(r <= k))
        out[f"R@{k}"] = 100.0 * math.sqrt(pk * (1 - pk) / n) if n > 1 else _NAN
    return out


BASELINES = ("oracle", "random", "prior", "option_prior", "full")


def option_prior_choice(questions: list, seed: int = 0) -> dict:
    """Blind leave-one-out option-text prior: for every question, the option whose text is most often the answer
    among the OTHER questions of the same (template, subset) (rate (k + 0.5) / (n + 1); ties by hash). Never looks at
    the video; it measures how much the option texts alone give away. Returns {qid: letter}."""
    groups = {}
    for q in questions:
        groups.setdefault((q["template"], q["subset"]), []).append(q)
    out = {}
    for g in groups.values():
        app, ans = {}, {}
        for q in g:
            for o in q["options"]:
                app[o] = app.get(o, 0) + 1
            ans[q["answer_text"]] = ans.get(q["answer_text"], 0) + 1
        for q in g:
            a = q["answer_text"]

            def rate(o, a=a, q=q):
                n = app[o] - 1
                k = ans.get(o, 0) - int(o == a)
                return (k + 0.5) / (n + 1.0), -_hash_int(seed, "option_prior", q["qid"], o)

            best = max(range(len(q["options"])), key=lambda i: rate(q["options"][i]))
            out[q["qid"]] = LETTERS[best]
    return out


def run_baseline(bench, pred_root=None, kind: str = "prior", protocol: UnderstandingProtocol | None = None,
                 subsets=None, ids=None, jobs: int = 1, tasks=DEFAULT_SUBTASKS) -> dict:
    """Write qa.jsonl, retrieval/<tag>.npy and grounding.jsonl of a reference baseline (sub-tasks ``tasks``).
    oracle: GT; random: seeded; prior: QA the most frequent answer text per (template, subset), retrieval random,
    grounding the centred interval of the mean GT length; option_prior: QA `option_prior_choice` (blind
    leave-one-out option-text prior), otherwise as prior; full: QA as prior, grounding the whole clip, retrieval
    random. ``ids`` restricts the QA questions and grounding queries. The call form of every task:
    run_baseline(bench, pred_root, kind="prior", ...) (run_baseline("prior", bench, pred_root, ...) is accepted
    too)."""
    bench, pred_root, kind = baseline_args(bench, pred_root, kind, {})
    if kind not in BASELINES:
        raise ValueError(f"baseline {kind!r} not in {BASELINES}")
    tasks = _check_subtasks(tasks)
    p = protocol or UnderstandingProtocol()
    pred_root = Path(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    subs = [canonical_subset(s) for s in subsets] if subsets else bench.subsets()
    ids = _select_ids(bench, subs, ids)
    rng = _rng(p.seed, "baseline", kind)
    res = {}
    if "qa" in tasks:
        qs = generate_questions(bench, p, subs, jobs)
        freq = {}
        for q in qs:
            k = (q["template"], q["subset"])
            freq.setdefault(k, {})
            freq[k][q["answer_text"]] = freq[k].get(q["answer_text"], 0) + 1
        loo = option_prior_choice(qs, p.seed) if kind == "option_prior" else {}
        qs = _filter_ids(qs, ids)
        with open(pred_root / "qa.jsonl", "w") as f:
            for q in qs:
                if kind == "oracle":
                    c = q["answer"]
                elif kind == "random":
                    c = LETTERS[int(rng.integers(len(q["options"])))]
                elif kind == "option_prior":
                    c = loo[q["qid"]]
                else:
                    fr = freq[(q["template"], q["subset"])]
                    best = max(fr, key=lambda a: (fr[a], a))
                    c = LETTERS[q["options"].index(best)] if best in q["options"] else "A"
                f.write(json.dumps({"qid": q["qid"], "choice": c}) + "\n")
        res["n_questions"] = len(qs)
    levels = retrieval_levels(subs, segment="retrieval_segment" in tasks)
    if "retrieval" not in tasks:
        levels = [x for x in levels if x[1] == "segment"]
    for sub, level in levels:
        (pred_root / "retrieval").mkdir(exist_ok=True)
        it = retrieval_items(bench, sub, level, p)
        if kind == "oracle":
            sim = it["gt"].T.astype(np.float32)
        else:
            sim = rng.random((len(it["videos"]), len(it["texts"]))).astype(np.float32)
        np.save(pred_root / "retrieval" / f"{it['tag']}.npy", sim)
        res.setdefault("retrieval", []).append(it["tag"])
    if "grounding" in tasks and "Real-General" in subs:
        gq = grounding_queries(bench, p)
        mean_len = float(np.mean([q["gt"][1] - q["gt"][0] + 1 for q in gq])) if gq else 1.0
        gq = _filter_ids(gq, ids)
        with open(pred_root / "grounding.jsonl", "w") as f:
            for q in gq:
                T = q["n_frames"]
                if kind == "oracle":
                    s0, e0 = q["gt"]
                elif kind == "full":
                    s0, e0 = 0, T - 1
                elif kind in ("prior", "option_prior"):
                    L = int(round(min(mean_len, T)))
                    s0 = (T - L) // 2
                    e0 = s0 + L - 1
                else:
                    L = int(rng.integers(1, T + 1))
                    s0 = int(rng.integers(0, T - L + 1))
                    e0 = s0 + L - 1
                f.write(json.dumps({"query_id": q["query_id"], "start_frame": int(s0), "end_frame": int(e0)}) + "\n")
        res["n_queries"] = len(gq)
    write_json(pred_root / "meta_baseline.json", {"task": TASK, "baseline": kind, "tasks": list(tasks),
                                                  "protocol": p.to_dict()})
    return res


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def mean_se(values) -> dict:
    v = np.array([_num(x) for x in values], np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"mean": _NAN, "se": _NAN, "n": 0}
    return {"mean": float(v.mean()), "se": float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 else _NAN, "n": n}


def fmt_ms(x: dict, nd: int = 1) -> str:
    if x is None or not np.isfinite(_num(x.get("mean"))):
        return "-"
    se = _num(x.get("se"))
    return f"{x['mean']:.{nd}f} ± {se:.{nd}f}" if np.isfinite(se) else f"{x['mean']:.{nd}f}"


def _fmt(x, nd: int = 1) -> str:
    v = _num(x)
    return f"{v:.{nd}f}" if np.isfinite(v) else "-"


def _qa_block(rows: list) -> dict:
    """Pooled QA numbers of a set of question rows (headline = non-sanity templates)."""
    main = [x for x in rows if not x.get("sanity", x["template"] in SANITY_TEMPLATES)]
    san = [x for x in rows if x.get("sanity", x["template"] in SANITY_TEMPLATES)]
    out = {"accuracy": _NAN, "macro_accuracy": _NAN, "n_questions": len(main), "per_template": {}}
    if main:
        acc = accuracy([x["correct"] for x in main], [x["template"] for x in main])
        out.update(accuracy=acc["accuracy"], macro_accuracy=acc["macro_accuracy"], per_template=acc["per_category"])
    out["sanity"] = (accuracy([x["correct"] for x in san], [x["template"] for x in san]) if san else None)
    out["n_missing"] = sum(x["status"] == "missing" for x in rows)
    out["n_unparsed"] = sum(x["status"] == "unparsed" for x in rows)
    return out


def summarize(out_dir) -> dict:
    """Aggregate the per-sample JSONs (+ retrieval.json) into ``summary.json`` / ``summary.md``. QA headline numbers
    (accuracy per sample, pooled, macro over templates) leave out the sanity templates (SANITY_TEMPLATES), which are
    reported apart. ``pred_root`` = the prediction root of the run in ``run.json`` (None without it)."""
    out_dir = Path(out_dir)

    def load(name):
        f = out_dir / name
        return json.loads(f.read_text()) if f.exists() else None

    summary = {"task": TASK, "protocol": load("protocol.json"), "pred_root": run_pred_root(out_dir), "subsets": {},
               "retrieval": load("retrieval.json") or {}, "meta": load("meta.json")}
    all_rows = []
    for sub in SUBSETS:
        d = out_dir / sub
        if not d.is_dir():
            continue
        recs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        recs = [r for r in recs if r.get("task") == TASK]
        s = {"n_samples": len(recs)}
        qa = [r["qa"] for r in recs if r.get("qa")]
        if qa:
            rows = [x for a in qa for x in a["questions"]]
            all_rows += rows
            s["qa"] = _qa_block(rows)
            s["qa"]["accuracy_per_sample"] = mean_se(_qa_accuracy(a["questions"])["accuracy"] for a in qa)
        gr = [r["grounding"] for r in recs if r.get("grounding")]
        if gr:
            qrows = [x for g in gr for x in g["queries"]]
            s["grounding"] = {"per_sample": {k: mean_se(g[k] for g in gr) for k in
                                             ("miou",) + tuple(f"R@{t}" for t in GROUNDING_THRESHOLDS)},
                              "pooled": grounding_metrics(x["iou"] for x in qrows),
                              "n_missing": sum(x["status"] == "missing" for x in qrows),
                              "n_invalid": sum(x["status"] in ("invalid", "empty") for x in qrows)}
        summary["subsets"][sub] = s
    if all_rows:
        b = _qa_block(all_rows)
        summary["qa_overall"] = {"accuracy": b["accuracy"], "macro_accuracy": b["macro_accuracy"],
                                 "n": b["n_questions"], "sanity": b["sanity"]}
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_summary_md(summary))
    return summary


def _summary_md(summary: dict) -> str:
    lines = ["# understanding", ""]
    qa = [(sub, s["qa"]) for sub, s in summary["subsets"].items() if s.get("qa")]
    if qa:
        lines += ["## multiple-choice QA (percent; without the sanity template " + ", ".join(SANITY_TEMPLATES) + ")",
                  "", "| subset | questions | accuracy (mean ± s.e. over samples) | pooled | macro over templates |",
                  "|---|---|---|---|---|"]
        for sub, a in qa:
            lines.append(f"| {sub} | {a['n_questions']} | {fmt_ms(a['accuracy_per_sample'])} | {_fmt(a['accuracy'])} | "
                         f"{_fmt(a['macro_accuracy'])} |")
        if summary.get("qa_overall"):
            o = summary["qa_overall"]
            lines.append(f"| all | {o['n']} | | {_fmt(o['accuracy'])} | {_fmt(o['macro_accuracy'])} |")
        lines += ["", "| subset | template | n | accuracy |", "|---|---|---|---|"]
        for sub, a in qa:
            for t, v in a["per_template"].items():
                lines.append(f"| {sub} | {t} | {v['n']} | {_fmt(v['accuracy'])} |")
            for t, v in ((a.get("sanity") or {}).get("per_category") or {}).items():
                lines.append(f"| {sub} | {t} (sanity) | {v['n']} | {_fmt(v['accuracy'])} |")
    if summary.get("retrieval"):
        lines += ["", "## caption retrieval (percent; ranks 1-based)", "",
                  "| gallery | direction | n | R@1 | R@5 | R@10 | MdR | MnR |", "|---|---|---|---|---|---|---|---|"]
        for tag, r in summary["retrieval"].items():
            for dname in ("text_to_video", "video_to_text"):
                m = r[dname]
                lines.append(f"| {tag} | {dname} | {m['n_queries']} | {_fmt(m['R@1'])} | {_fmt(m['R@5'])} | "
                             f"{_fmt(m['R@10'])} | {_fmt(m['median_rank'])} | {_fmt(m['mean_rank'])} |")
    gr = [(sub, s["grounding"]) for sub, s in summary["subsets"].items() if s.get("grounding")]
    if gr:
        lines += ["", "## temporal grounding (Real-General; percent)", "",
                  "| subset | queries | mIoU (per sample) | R@0.3 | R@0.5 | R@0.7 | pooled mIoU | invalid |",
                  "|---|---|---|---|---|---|---|---|"]
        for sub, g in gr:
            ps = g["per_sample"]
            lines.append(f"| {sub} | {g['pooled']['n']} | {fmt_ms(ps['miou'])} | {fmt_ms(ps['R@0.3'])} | "
                         f"{fmt_ms(ps['R@0.5'])} | {fmt_ms(ps['R@0.7'])} | {_fmt(g['pooled']['miou'])} | "
                         f"{g.get('n_invalid', 0)} |")
    for n in (summary.get("meta") or {}).get("notes") or []:
        lines += ["", f"note: {n}"]
    lines.append("")
    return "\n".join(lines)


def _map(fn, work, jobs: int):
    if jobs and jobs > 1 and len(work) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(fn, work, chunksize=4))
    return [fn(w) for w in work]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
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
    """`vpmwb understanding {export,baseline,evaluate,summarize}` (also `python -m vpm_worldbench.tasks.understanding`).
    Shared option names: --root, --pred, --out, --protocol, --subsets, --ids, --jobs, --overwrite; window overrides
    --ref-frame / --horizon (>= 1) / --frame-step of the window templates (a changed value adds "+field=value" to the
    protocol name). Exit status: 1 when --ids match no sample of the selected subsets, or evaluate finds no prediction
    file (nothing is scored); 2 for a usage error (unknown subset or sub-task, --protocol neither the preset nor a
    file, an invalid window override, a missing --answer-key file or summarize --out folder: usage + one line) or an
    out dir of another run without --overwrite (`common.RunDirError`; it propagates, `vpmwb` reports it as one line
    `vpmwb: error: ...`)."""
    import argparse

    from ..utils import parse_list
    ap = argparse.ArgumentParser(prog="vpmwb understanding",
                                 description="understanding: multiple-choice QA, caption retrieval, temporal grounding")
    sp = ap.add_subparsers(dest="cmd", required=True, metavar="{export,baseline,evaluate,summarize}")
    helps = {"export": "write the model inputs (questions, retrieval galleries, grounding queries)",
             "baseline": "write the predictions of a reference baseline (--kind)",
             "evaluate": "score predictions", "summarize": "rebuild summary.json / summary.md"}
    parsers = {}
    for name in ("export", "baseline", "evaluate", "summarize"):
        q = parsers[name] = sp.add_parser(name, help=helps[name], description=helps[name])
        if name != "summarize":
            q.add_argument("--root", required=True, help="VP-WorldBench data root")
            q.add_argument("--protocol", default=None,
                           help=f"'{UnderstandingProtocol().name}' (default) or a JSON / YAML file of "
                                "UnderstandingProtocol fields (a file without 'name' is named after the file)")
            q.add_argument("--subsets", nargs="*", default=None, help="subset names or aliases (space / comma separated)")
            q.add_argument("--ids", nargs="*", default=None,
                           help="sample ids or uids (QA questions and grounding queries; retrieval galleries are whole "
                                "subsets)")
            q.add_argument("--jobs", type=int, default=1, help="worker processes")
            q.add_argument("--ref-frame", type=int, default=None, help="override the first window frame")
            q.add_argument("--horizon", type=int, default=None,
                           help="override the window length in frame steps (>= 1)")
            q.add_argument("--frame-step", type=int, default=None, help="override the window frame step")
        if name in ("baseline", "evaluate"):
            q.add_argument("--pred", required=True, help="prediction root")
        if name in ("export", "evaluate", "summarize"):
            q.add_argument("--out", required=True, help="output folder")
        if name in ("export", "evaluate"):
            q.add_argument("--overwrite", action="store_true", help="replace an out dir written by another run")
        if name in ("export", "baseline"):
            q.add_argument("--tasks", nargs="*", default=None,
                           help=f"sub-tasks among {', '.join(SUBTASKS)} (default {' '.join(DEFAULT_SUBTASKS)}; "
                                "retrieval_segment only on request: it lists the grounding GT segments)")
        if name == "export":
            q.add_argument("--answer-key-out", default=None,
                           help="also write the QA answer key to this file (keep it outside --out: a model may read "
                                "the export folder); for evaluate --answer-key")
        if name == "baseline":
            q.add_argument("--kind", default="prior", choices=BASELINES)
        if name == "evaluate":
            q.add_argument("--answer-key", default=None,
                           help="answer key written by export --answer-key-out (default: regenerated from the data)")
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
    if a.horizon is not None and a.horizon < 1:
        q.error(f"--horizon {a.horizon}: must be >= 1 (the window templates use a fixed window of horizon frame steps)")
    p = cli_protocol(q, a.protocol, UnderstandingProtocol, get_protocol)
    over = {k: v for k, v in (("ref_frame", a.ref_frame), ("horizon", a.horizon), ("frame_step", a.frame_step))
            if v is not None}
    if over:
        try:
            p = override_protocol(p, over, UnderstandingProtocol)
        except (KeyError, TypeError, ValueError) as e:
            q.error(str(e))
    if getattr(a, "answer_key", None) and not Path(a.answer_key).is_file():
        q.error(f"--answer-key {a.answer_key}: no such file")
    subsets, ids = cli_subsets(q, a.subsets), parse_list(a.ids)
    try:
        tasks = _check_subtasks(parse_list(getattr(a, "tasks", None)) or DEFAULT_SUBTASKS)
    except ValueError as e:
        q.error(f"--tasks: {e}")
    if getattr(a, "answer_key_out", None) and "qa" not in tasks:
        q.error("--answer-key-out needs the qa sub-task (--tasks qa ...)")
    from ..data.benchmark import Benchmark
    bench = Benchmark(a.root)
    if ids is not None and len(bench.unmatched_ids(ids, subsets)) == len(ids):
        print(f"error: {TASK}: no sample matches the selection (--subsets / --ids): {ids[:10]}", file=sys.stderr)
        return 1
    if a.cmd == "export":
        print(json.dumps(export_inputs(bench, a.out, p, subsets, ids, tasks, a.jobs, a.overwrite,
                                       answer_key_out=a.answer_key_out), indent=1))
    elif a.cmd == "baseline":
        print(json.dumps(run_baseline(bench, a.pred, kind=a.kind, protocol=p, subsets=subsets, ids=ids, jobs=a.jobs,
                                      tasks=tasks)))
    else:
        try:
            evaluate(bench, a.pred, a.out, p, subsets, ids, a.answer_key, a.jobs, a.overwrite)
        except NoPredictionFiles as e:
            print(f"error: {TASK}: {e}: nothing is scored", file=sys.stderr)
            return 1
        print((Path(a.out) / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
