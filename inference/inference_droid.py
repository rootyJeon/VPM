# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DROID inference with a VPM stage-3 checkpoint.

From the last clean frame of the DROID exterior cameras, their calibration, the robot joint state and a task
instruction, VPM jointly generates the next 32 frames of every view and a 32-step joint-position action chunk.
Optionally it then tracks points on the generated clip with its point-track head.

Inputs (one of):
  * an episode in the VPM data layout: ``--data-root <root>/droid --episode <scene> --frame <t0>``. The sample is
    built by the training dataset class, so preprocessing is identical to training. Only frames up to ``t0`` and the
    joint state at ``t0`` reach the model.
  * raw arrays: ``--images v1.png,v2.png --cameras cameras.json --state q0,..,q6,g --prompt "..."``.

Outputs in ``<out-dir>/<name>/``:
  * ``video/<view>.mp4``: clean past frames followed by the generated frames, 832x480, 15 fps.
  * ``actions.npz``: ``actions`` [32, 8] absolute joint positions (rad) and gripper (0 open .. 1 closed), one row per
    generated frame; ``state`` [8] the input state in the same convention; ``model_rows`` [33, 8] the raw model rows
    (row 0 = state, gripper as ``1 - g``); ``dt`` seconds per row.
  * with ``--tracks``: ``video/<view>_tracks.mp4`` and ``tracks.npz`` with ``tracks_px`` [V, T, N, 2] in the
    832x480 frame, ``tracks_input_px`` in input-image pixels, ``vis`` (visibility x confidence), ``conf``, ``depth``
    [V, T, N] (metres along the camera z axis), ``xyz_cam`` and ``xyz_world`` [V, T, N, 3] (metres).
  * ``meta.json``: inputs, sampling settings, timing and peak GPU memory.

Launch with ``torchrun --nproc_per_node=N`` (see ``inference/infer_droid.sh``): the model is sharded over the N GPUs and
the requested episodes run one per rank in lockstep.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import pad_action_to_max_dim
from cosmos_framework.data.vfm.joint_dataloader import custom_collate_fn
from cosmos_framework.data.vfm.local_datasets.camera_bundle import build_camera_bundle
from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils import log, misc

DEFAULT_CKPT = "vpm-project/VPM-DROID"
DEFAULT_TOML = "examples/toml/vpm_stage3.toml"
WAN_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_VAE_FILE = "Wan2.2_VAE.pth"
WAN_VAE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
DROID_DOMAIN = "oxe/droid"
STATE_DIM = 8
DROID_FPS = 15.0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("VPM_CHECKPOINT", DEFAULT_CKPT),
                    help="Hugging Face repo id or local dir: a release dir ([model/]*.safetensors) or a training "
                         "iter_* dir (model/ DCP). Default: $VPM_CHECKPOINT")
    ap.add_argument("--revision", default=None, help="Hugging Face revision of --ckpt")
    ap.add_argument("--toml", default=DEFAULT_TOML, help="stage-3 experiment TOML the model config is built from")
    ap.add_argument("--vae-path", default=os.environ.get("WAN_VAE_PATH", ""),
                    help=f"Wan2.2 VAE weights (default: $WAN_VAE_PATH, else downloaded from {WAN_VAE_REPO})")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Hydra override applied to the config (repeatable)")

    a = ap.add_argument_group("input A: episode in the VPM data layout")
    a.add_argument("--data-root", default=os.path.join(os.environ.get("VPM_DATA_ROOT", "data"), "droid"),
                   help="DROID dataset root with videos/, camera/, tracks/, actions/ (default: $VPM_DATA_ROOT/droid)")
    a.add_argument("--jsonl", default="", help="episode index (default: <data-root>/train/video_dataset_file.jsonl)")
    a.add_argument("--episode", default="", help="scene name(s) or uuid(s), comma separated")
    a.add_argument("--frame", default="", help="last clean frame t0, one for all episodes or one per episode")

    b = ap.add_argument_group("input B: raw multi-view observation")
    b.add_argument("--images", default="", help="one RGB image per view, comma separated (the last clean frame)")
    b.add_argument("--past-frames", default="",
                   help="npz with 'frames' uint8 [V, P, H, W, 3] ending at the last clean frame "
                        "(P >= 4 * (past_latents - 1) + 1); replaces --images")
    b.add_argument("--cameras", default="",
                   help="json {view: {intrinsics: 3x3 pixel K at the image size, extrinsics: 4x4 world-to-camera "
                        "in metres}} (the data-layout keys measured_intrinsics / optimized_extrinsics also work)")
    b.add_argument("--view-names", default="", help="views to use, comma separated, in image order "
                                                    "(default: the --cameras order)")
    b.add_argument("--state", default="", help="q0,...,q6,g: joint positions (rad) and gripper (0 open .. 1 closed)")
    b.add_argument("--name", default="sample", help="output sub-directory name")

    ap.add_argument("--prompt", default=None,
                    help="task instruction (input A default: the episode caption; '' = unconditional)")
    ap.add_argument("--past-latents", type=int, default=1,
                    help="clean past latents: 1 = last frame only, p = the last 4 * (p - 1) + 1 frames")
    ap.add_argument("--fps", type=float, default=DROID_FPS, help="input B: frame rate of the observation")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracks", choices=("none", "grid", "dataset"), default="none",
                    help="point tracks on the generated clip: queries on a regular grid in every view, or (input A) "
                         "the dataset's cross-view track points at the first frame")
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--track-sigma", type=float, default=0.15, help="noise level of the tracking forward pass")
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("OUTPUT_ROOT", "outputs"), "droid_inference"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build the input batches and print their contents; no GPU, no checkpoint")
    argv = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(argv) - 1:
        if argv[i] == "--state" and argv[i + 1].startswith("-"):
            argv[i : i + 2] = [f"--state={argv[i + 1]}"]
        i += 1
    args = ap.parse_args(argv)
    raw = bool(args.images or args.past_frames)
    if bool(args.episode) == raw:
        ap.error("give either --episode (input A) or --images/--past-frames (input B)")
    if raw and (not args.cameras or not args.state or args.prompt is None):
        ap.error("input B needs --cameras, --state and --prompt")
    if raw and args.tracks == "dataset":
        ap.error("--tracks dataset needs input A")
    if args.past_latents < 1:
        ap.error("--past-latents must be >= 1")
    return args


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p)) if p else p


def _hydra_str(value: str) -> str:
    if "'" in value or "\\" in value:
        raise ValueError(f"unsupported character in {value!r}")
    return f"'{value}'"


def load_config(toml_path: str, vae_path: str, extra: list[str]):
    """The stage-3 experiment config with the inference overrides (no EMA copy, no warm start)."""
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    overrides = [
        "model.config.ema.enabled=false",
        "model.config.rectified_flow_training_config.tracking_corr_weight=0.0",
        f"model.config.tokenizer.vae_path={_hydra_str(vae_path)}",
        "checkpoint.load_path=''",
        *extra,
    ]
    return load_experiment_from_toml(toml_path, extra_overrides=overrides)


def build_dataset(config, slot: str = "droid"):
    """The training dataset of ``slot`` without metadata and without caption dropout."""
    from omegaconf import open_dict

    from cosmos_framework.utils.lazy_config import instantiate

    node = copy.deepcopy(config.dataloader_train.dataloader.datasets[slot].dataset)
    with open_dict(node):
        node.jsonl_paths = []
        node.cfg_dropout_rate = 0.0
        node.min_window_frames = 1
    ds = instantiate(node)
    cc = getattr(ds, "conditioning_config", None)
    if isinstance(cc, dict) and any(isinstance(k, str) for k in cc):
        ds.conditioning_config = {int(k): v for k, v in cc.items()}
    ds.s3_client = None
    return ds


def window_geometry(ds, p_lat: int) -> tuple[int, int]:
    """(p_px, T): pixel frames of the clean past and of the whole window."""
    tcf = int(ds.temporal_compression_factor)
    chunk = int(ds._mvt_get("_mvt_action_chunk", 32))
    p_px = tcf * (p_lat - 1) + 1
    return p_px, p_px + chunk


def resize_crop(width: int, height: int, out_w: int, out_h: int) -> dict:
    """Resize to cover the output size, then center crop (the data loader's geometry)."""
    r = max(out_w / width, out_h / height)
    rh, rw = round(height * r), round(width * r)
    return dict(ratio=r, input_hw=(height, width), resize_hw=(rh, rw),
                crop_yx=(round((rh - out_h) / 2), round((rw - out_w) / 2)), output_hw=(out_h, out_w))


def to_batch(sample: dict, device: "str | None") -> dict:
    """Collate one sample like the training loader, with ``image_size`` flat per vision item."""
    batch = custom_collate_fn([sample])
    batch["image_size"] = [
        torch.tensor([v.shape[-2], v.shape[-1], v.shape[-2], v.shape[-1]], dtype=torch.float32)
        for views in batch["video"] for v in views
    ]
    return misc.to(batch, device=device) if device else batch


def _hide_future(sample: dict, p_px: int) -> None:
    """Zero every frame and action row after the clean past, so only the observation reaches the model."""
    sample["video"] = [v.clone() for v in sample["video"]]
    for v in sample["video"]:
        v[:, p_px:] = 0.0
    act = sample["action"].clone()
    act[1:] = 0.0
    sample["action"] = act


def find_episode(jsonl: Path, data_root: str, episode: str) -> dict:
    """Loader metadata of ``episode`` (scene name or uuid); videos are read from ``<data_root>/videos``."""
    with open(jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            vp = rec["vision_path"]
            scene = Path(vp).stem.rsplit("__", 1)[0]
            if episode not in (scene, rec.get("uuid")):
                continue
            local = Path(data_root) / "videos" / Path(vp).name
            if local.is_file():
                vp = str(local)
            elif "://" not in vp and not os.path.isabs(vp):
                vp = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(jsonl)), vp))
            return {
                "uuid": rec["uuid"],
                "vision_path": vp,
                "width": rec["width"],
                "height": rec["height"],
                "aspect_ratio": get_aspect_ratio(rec["width"], rec["height"]),
                "t2w_windows": rec.get("t2w_windows") or [],
            }
    raise FileNotFoundError(f"episode {episode!r} not found in {jsonl}")


def episode_sample(ds, md: dict, t0: int, p_lat: int, prompt: "str | None", seed: int = 0) -> tuple[dict, dict]:
    """The training sample of the window whose clean past ends at episode frame ``t0``, future hidden."""
    anchor = Path(os.path.abspath(md["vision_path"]))
    scene, anchor_view = anchor.stem.rsplit("__", 1)
    with np.load(anchor.parents[1] / "actions" / f"{scene}.npz") as z:
        joint_states = z["joint_states"].astype(np.float32)
    n = int(joint_states.shape[0])
    p_px, T = window_geometry(ds, p_lat)
    start = t0 - p_px + 1
    if start < 0 or start + T > n:
        raise ValueError(f"{scene}: --frame {t0} out of range; with --past-latents {p_lat} it must lie in "
                         f"[{p_px - 1}, {n - T + p_px - 1}] ({n} frames)")
    win0 = dict((md.get("t2w_windows") or [{}])[0])
    caption = win0.get("qwen3_32b_rewrite-dense", win0.get("caption")) if prompt is None else (prompt or ".")
    if caption is None:
        raise ValueError(f"{scene}: no caption in the index; pass --prompt")
    md = dict(md, window_start=start, t2w_windows=[{"start_frame": start, "end_frame": start + T - 1,
                                                    "temporal_interval": 1, "caption": caption}])

    saved_past = ds._mvt_get("_mvt_variable_past_latents", None)
    saved_state = random.getstate()
    ds._mvt_variable_past_latents = [int(p_lat)]
    ds._mvt_rng_obj = random.Random(seed)
    random.seed(seed)
    try:
        sample = ds.process_one_sample(md)
    finally:
        ds._mvt_variable_past_latents = saved_past
        random.setstate(saved_state)
    if sample is None:
        raise RuntimeError(f"{scene}: the dataset rejected the window starting at frame {start} (see the log)")
    if int(sample["frame_start"]) != start or int(sample["num_frames"]) != T:
        raise RuntimeError(f"{scene}: window not pinned (start {sample['frame_start']} != {start} or "
                           f"T {sample['num_frames']} != {T})")
    if list(sample["sequence_plan"].condition_frame_indexes_vision) != list(range(p_lat)):
        raise RuntimeError(f"{scene}: clean past not pinned to {p_lat} latents")
    if prompt == "":
        sample["ai_caption"] = ""
        sample["text_token_ids"] = torch.tensor(ds._tokenize_caption("")[0])
    _hide_future(sample, p_px)

    out_w, out_h = ds.output_sizes[md["aspect_ratio"]]
    info = {
        "input": "episode",
        "episode": scene,
        "t0": int(t0),
        "window_start": int(start),
        "views": _episode_view_names(sample, anchor.parents[1] / "camera" / f"{scene}_cameras.json", anchor_view),
        "state": joint_states[t0],
        "fps": float(sample["fps"]),
        "geometry": resize_crop(int(md["width"]), int(md["height"]), out_w, out_h),
    }
    return sample, info


def _episode_view_names(sample: dict, cameras_json: Path, anchor_view: str) -> list[str]:
    """View names of the sample, in sample order (anchor first), parsed from the dataset's ``__key__``."""
    V = len(sample["video"])
    views = sorted((k for k in json.loads(cameras_json.read_text()) if k != "optimization_summary"),
                   key=len, reverse=True)
    parts = str(sample.get("__key__", "")).split("__")
    for i, part in enumerate(parts[:-1]):
        if not re.fullmatch(r"\d+views_fut\d+", part):
            continue
        rest, names = parts[i + 1], []
        while rest:
            name = next((k for k in views if rest == k or rest.startswith(k + "_")), None)
            if name is None:
                break
            names.append(name)
            rest = rest[len(name) + 1:]
        if len(names) == V and not rest:
            return names
    return [anchor_view] + [f"sibling{i}" for i in range(1, V)]


def load_cameras(path: str, names: "list[str] | None") -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Per-view pixel K (3x3) and world-to-camera (4x4) from a cameras json."""
    data = json.loads(Path(path).read_text())
    views = {k: v for k, v in data.items()
             if isinstance(v, dict) and ("intrinsics" in v or "measured_intrinsics" in v)}
    names = names or list(views)
    missing = [k for k in names if k not in views]
    if missing:
        raise KeyError(f"views {missing} not in {path} (has {list(views)})")
    Ks = [np.asarray(views[k].get("intrinsics", views[k].get("measured_intrinsics")), np.float64) for k in names]
    Es = [np.asarray(views[k].get("extrinsics", views[k].get("optimized_extrinsics")), np.float64) for k in names]
    for k, K, E in zip(names, Ks, Es):
        if K.shape != (3, 3) or E.shape != (4, 4):
            raise ValueError(f"view {k}: expected a 3x3 intrinsics and a 4x4 extrinsics, got {K.shape}, {E.shape}")
    return Ks, Es, names


def load_frames(args, p_px: int) -> list[np.ndarray]:
    """Per view uint8 [p_px, H, W, 3] ending at the last clean frame."""
    from PIL import Image

    if args.past_frames:
        with np.load(args.past_frames) as z:
            arr = z["frames"] if "frames" in z.files else z[z.files[0]]
        frames = [np.asarray(v, np.uint8) for v in arr]
    else:
        frames = [np.asarray(Image.open(p).convert("RGB"), np.uint8)[None] for p in args.images.split(",")]
    for f in frames:
        if f.ndim != 4 or f.shape[-1] != 3:
            raise ValueError(f"expected frames [P,H,W,3], got {f.shape}")
        if f.shape[0] < p_px:
            raise ValueError(f"--past-latents {args.past_latents} needs {p_px} past frames per view, got {f.shape[0]}")
    return [f[-p_px:] for f in frames]


def _resize_frames(frames: np.ndarray, geo: dict) -> np.ndarray:
    from PIL import Image

    (rh, rw), (cy, cx), (oh, ow) = geo["resize_hw"], geo["crop_yx"], geo["output_hw"]
    return np.stack([np.asarray(Image.fromarray(f).resize((rw, rh), Image.BICUBIC))[cy:cy + oh, cx:cx + ow]
                     for f in frames])


def raw_sample(ds, frames: list[np.ndarray], Ks: list[np.ndarray], w2cs: list[np.ndarray], state: np.ndarray,
               prompt: str, view_names: list[str], p_lat: int, fps: float = DROID_FPS,
               name: str = "sample") -> tuple[dict, dict]:
    """The DROID training sample contract built from raw arrays.

    ``frames``: per view uint8 ``[p_px, H, W, 3]`` ending at the last clean frame. ``Ks``: 3x3 pixel intrinsics at
    ``(W, H)``. ``w2cs``: 4x4 world-to-camera (metres). ``state``: ``[q0..q6, gripper]``, gripper 0 open .. 1 closed.
    """
    V = len(frames)
    if not (len(Ks) == len(w2cs) == len(view_names) == V):
        raise ValueError(f"{V} views of frames but {len(Ks)} cameras / {len(view_names)} names")
    H, W = frames[0].shape[1:3]
    if any(f.shape[1:3] != (H, W) for f in frames):
        raise ValueError("all views must share one image size")
    p_px, T = window_geometry(ds, p_lat)
    out_w, out_h = ds.output_sizes[get_aspect_ratio(W, H)]
    geo = resize_crop(W, H, out_w, out_h)
    r, (rh, rw), (cy, cx) = geo["ratio"], geo["resize_hw"], geo["crop_yx"]

    video = []
    for f in frames:
        clip = torch.zeros(3, T, out_h, out_w, dtype=torch.float32)
        past = torch.from_numpy(np.ascontiguousarray(_resize_frames(f[-p_px:], geo))).permute(3, 0, 1, 2)
        clip[:, :p_px] = past.float().div_(127.5).sub_(1.0)
        video.append(clip)

    E = [np.repeat(np.asarray(e, np.float64)[None], T, axis=0) for e in w2cs]
    camera = build_camera_bundle(
        c2w_views=[np.linalg.inv(e) for e in E], intrinsics_views=[np.asarray(k, np.float64) for k in Ks],
        native_sizes=[(H, W)] * V, unit_to_meters=1.0, input_height=H, input_width=W, resize_height=rh,
        resize_width=rw, crop_y=cy, crop_x=cx, output_height=out_h, output_width=out_w,
    )
    Kn = np.zeros((V, 3, 3), np.float32)
    for i, k in enumerate(Ks):
        Kn[i, 0, 0] = k[0, 0] * r / out_w
        Kn[i, 1, 1] = k[1, 1] * r / out_h
        Kn[i, 0, 2] = (k[0, 2] * r - cx) / out_w
        Kn[i, 1, 2] = (k[1, 2] * r - cy) / out_h
        Kn[i, 2, 2] = 1.0

    caption = ""
    if prompt:
        caption = prompt.strip().rstrip(".") + "."
        if ds.caption_suffix:
            caption = f"{caption} {ds.caption_suffix}"
    cond_fps = float(fps) if float(ds.conditioning_fps) < 0 else float(ds.conditioning_fps)
    if caption and (ds.append_duration_fps_timestamps or ds.append_resolution_info):
        from cosmos_framework.data.vfm.local_datasets.sft_dataset import _DURATION_TEMPLATE, _RESOLUTION_TEMPLATE

        if ds.append_duration_fps_timestamps:
            caption += " " + _DURATION_TEMPLATE.format(duration=T / cond_fps, fps=cond_fps)
        if ds.append_resolution_info:
            caption += " " + _RESOLUTION_TEMPLATE.format(height=out_h, width=out_w)
        caption = caption.strip()
    text_ids, caption = ds._tokenize_caption(caption)

    rows = np.zeros((1 + T - p_px, STATE_DIM), np.float32)
    rows[0] = np.asarray(state, np.float32)
    if ds._mvt_get("_mvt_gripper_flip", True):
        rows[0, -1] = 1.0 - rows[0, -1]
    image_size = torch.tensor([out_h, out_w, out_h, out_w], dtype=torch.float32)

    sample = {
        "__key__": name,
        "__url__": "",
        "fps": float(fps),
        "n_orig_video_frames": T,
        "chunk_index": 0,
        "frame_start": 0,
        "frame_end": T - 1,
        "num_frames": T,
        "video": video,
        "num_multiplier": 1,
        "conditioning_fps": cond_fps,
        "padding_mask": torch.zeros((1, out_h, out_w), dtype=torch.float32),
        "image_size": [image_size] * V,
        "ai_caption": caption,
        "sampled_caption_style": "caption",
        "text_token_ids": torch.tensor(text_ids),
        "track_intrinsics": torch.from_numpy(Kn),
        "track_extrinsics": torch.from_numpy(np.stack([e.astype(np.float32) for e in E])),
        "action": pad_action_to_max_dim(torch.from_numpy(rows), int(ds._mvt_get("_mvt_max_action_dim", 64))),
        "domain_id": torch.tensor(get_domain_id(DROID_DOMAIN), dtype=torch.long),
        "raw_action_dim": torch.tensor(STATE_DIM, dtype=torch.long),
        "mode": "policy",
        "sequence_plan": SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=list(range(p_lat)),
            num_source_views=0,
            has_action=True,
            condition_frame_indexes_action=[0],
            action_start_frame_offset=p_px - 1,
            action_single_stream=True,
            action_view_index=0,
        ),
        **camera,
    }
    info = {"input": "raw", "views": list(view_names), "state": np.asarray(state, np.float32), "fps": float(fps),
            "geometry": geo}
    return sample, info


def raw_inputs(args, ds) -> tuple[dict, dict]:
    p_px, _ = window_geometry(ds, args.past_latents)
    names = [s for s in args.view_names.split(",") if s] or None
    Ks, w2cs, names = load_cameras(args.cameras, names)
    frames = load_frames(args, p_px)
    if len(frames) != len(names):
        raise ValueError(f"{len(frames)} views of frames but {len(names)} cameras {names}; use --view-names")
    state = np.asarray([float(x) for x in args.state.split(",")], np.float64)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"--state needs {STATE_DIM} values (7 joints + gripper), got {state.size}")
    return raw_sample(ds, frames, Ks, w2cs, state, args.prompt, names, args.past_latents, fps=args.fps,
                      name=args.name)


def resolve_checkpoint(ckpt: str, revision: "str | None", rank: int) -> Path:
    path = Path(ckpt).expanduser()
    if path.exists():
        return Path(os.path.abspath(path))
    if ckpt.startswith("<"):
        raise SystemExit(f"--ckpt {ckpt!r} is a placeholder: pass the released Hugging Face repo id or a local dir")
    if ckpt.count("/") != 1:
        raise FileNotFoundError(f"checkpoint {ckpt!r} is neither a local path nor a Hugging Face repo id")
    from huggingface_hub import snapshot_download

    from cosmos_framework.utils import distributed

    if rank == 0:
        snapshot_download(ckpt, revision=revision)
    distributed.barrier()
    return Path(snapshot_download(ckpt, revision=revision, local_files_only=rank != 0))


def resolve_vae(vae_path: str, rank: int) -> str:
    if vae_path:
        return _abs(vae_path)
    from huggingface_hub import hf_hub_download

    from cosmos_framework.utils import distributed

    if rank == 0:
        hf_hub_download(WAN_VAE_REPO, WAN_VAE_FILE, revision=WAN_VAE_REVISION)
    distributed.barrier()
    return hf_hub_download(WAN_VAE_REPO, WAN_VAE_FILE, revision=WAN_VAE_REVISION, local_files_only=rank != 0)


def storage_reader(ckpt_dir: Path):
    """Safetensors release dir (``[model/]*.safetensors``) or a training checkpoint (``[iter_*/]model`` DCP)."""
    from torch.distributed.checkpoint import FileSystemReader, HuggingFaceStorageReader

    for d in (ckpt_dir / "model", ckpt_dir):
        if d.is_dir() and any(d.glob("*.safetensors")):
            return HuggingFaceStorageReader(str(d)), "safetensors"
    for d in (ckpt_dir / "model", ckpt_dir):
        if (d / ".metadata").is_file():
            return FileSystemReader(str(d)), "dcp"
    raise FileNotFoundError(f"no *.safetensors or DCP .metadata under {ckpt_dir} (or its model/)")


def build_model(config, ckpt_dir: Path):
    """Instantiate the model from the config and load the ``net.*`` weights strictly."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

    from cosmos_framework.utils.lazy_config import instantiate

    model = instantiate(config.model).to("cuda")
    model.eval()
    state = get_model_state_dict(model)
    stray = [k for k in state if not k.startswith("net.")]
    if stray:
        raise RuntimeError(f"unexpected model state keys (EMA must be disabled): {stray[:5]}")
    reader, kind = storage_reader(ckpt_dir)
    dcp.load(state, storage_reader=reader)
    set_model_state_dict(model, state, options=StateDictOptions(strict=True))
    return model, kind


@torch.no_grad()
def run_clip(model, batch: dict, p_lat: int, args) -> dict:
    """Generate the future frames of every view and the action chunk; optionally track points on the result."""
    tcf = int(model.tokenizer_vision_gen.temporal_compression_factor)
    clips = model.nested_clips(batch)
    T = int(clips[0][0].shape[1])
    p_px = tcf * (p_lat - 1) + 1
    cond = list(range(p_lat))
    win = model.build_window_batch(batch, clips, 0, T, cond)
    roll = model.generate_window(win, clips, p_px, T - p_px, T, args.num_steps, args.guidance, args.seed,
                                 shift=float(args.shift))
    action = (roll.get("action") or [None])[0]
    if action is not None:
        while action.dim() > 2:
            action = action[0]
    tracks = None
    if args.tracks != "none":
        db = model.build_window_batch(win, roll["pred"], 0, T, cond)
        if action is not None:
            rows = db["action"][0].clone()
            k = min(rows.shape[0], action.shape[0])
            rows[1:k] = action[1:k].to(device=rows.device, dtype=rows.dtype)
            db["action"] = [rows]
        torch.manual_seed(args.seed)
        res = model.predict_tracks(db, grid_size=args.grid_size, seed=args.seed,
                                   use_gt_queries=args.tracks == "dataset", sigma=float(args.track_sigma))
        tracks = res[0] if res else None
    return {
        "pred": [v.detach().float().cpu() for v in roll["pred"][0]],
        "action": None if action is None else action.detach().float().cpu(),
        "tracks": tracks,
        "p_px": p_px,
    }


def _u8(clip: torch.Tensor) -> np.ndarray:
    """[3,T,H,W] in [-1,1] -> [T,H,W,3] uint8."""
    return ((clip.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    from cosmos_framework.utils.easy_io import easy_io

    easy_io.dump(frames, str(path), file_format="mp4", fps=float(fps), quality=8)


def _np(x):
    return None if x is None else (x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x))


def _first(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return x


def write_outputs(d: Path, res: dict, batch: dict, info: dict, meta: dict, fps: float, gripper_flip: bool) -> None:
    (d / "video").mkdir(parents=True, exist_ok=True)
    views = info["views"]
    frames = [_u8(v) for v in res["pred"]]
    for name, f in zip(views, frames):
        _write_mp4(d / "video" / f"{name}.mp4", f, fps)

    if res["action"] is not None:
        rows = res["action"][:, :STATE_DIM].numpy().astype(np.float32)
        actions = rows[1:].copy()
        if gripper_flip:
            actions[:, -1] = 1.0 - actions[:, -1]
        actions[:, -1] = np.clip(actions[:, -1], 0.0, 1.0)
        np.savez(d / "actions.npz", actions=actions, state=np.asarray(info["state"], np.float32), model_rows=rows,
                 dt=np.float32(1.0 / fps))

    tr = res.get("tracks")
    if tr is not None:
        uv = _np(tr["tracks"]).astype(np.float32)
        geo = info["geometry"]
        (oh, ow), (cy, cx), r = geo["output_hw"], geo["crop_yx"], geo["ratio"]
        out = {
            "tracks_px": uv,
            "tracks_input_px": ((uv + np.array([cx, cy], np.float32)) / r).astype(np.float32),
            "sigma": np.float32(meta["track_sigma"]),
            "views": np.array(views),
        }
        for key in ("vis", "conf", "depth"):
            if tr.get(key) is not None:
                out[key] = _np(tr[key]).astype(np.float32)
        if "depth" in out:
            Kn = _np(_first(batch["track_intrinsics"])).astype(np.float64)
            w2c = _np(_first(batch["track_extrinsics"])).astype(np.float64)[:, : uv.shape[1]]
            uvn = uv / np.array([ow, oh], np.float64)
            pix = np.concatenate([uvn, np.ones_like(uvn[..., :1])], axis=-1)
            xc = np.einsum("vij,vtnj->vtni", np.linalg.inv(Kn), pix) * out["depth"][..., None]
            c2w = np.linalg.inv(w2c)
            xw = np.einsum("vtij,vtnj->vtni", c2w[..., :3, :3], xc) + c2w[:, :, None, :3, 3]
            out["xyz_cam"], out["xyz_world"] = xc.astype(np.float32), xw.astype(np.float32)
        np.savez(d / "tracks.npz", **out)
        from cosmos_framework.model.vfm.tracking.viz import draw_tracks

        for v, name in enumerate(views):
            _write_mp4(d / "video" / f"{name}_tracks.mp4", draw_tracks(frames[v], uv[v]), fps)

    meta = dict(meta, views=views, fps=fps, caption=_first(batch.get("ai_caption")), past_frames=res["p_px"],
                generated_frames=int(frames[0].shape[0] - res["p_px"]),
                geometry={k: (list(v) if isinstance(v, tuple) else v) for k, v in info["geometry"].items()})
    for key in ("episode", "t0", "window_start"):
        if key in info:
            meta[key] = info[key]
    meta["state"] = np.asarray(info["state"], np.float32).tolist()
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def summarize(batch: dict) -> dict:
    """Key -> shape/dtype (or value) of a collated batch, for --dry-run."""
    def desc(x):
        if torch.is_tensor(x):
            return f"{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"
        if isinstance(x, (list, tuple)):
            return [desc(e) for e in x]
        if isinstance(x, SequencePlan):
            return {k: getattr(x, k) for k in ("condition_frame_indexes_vision", "condition_frame_indexes_action",
                                                 "action_start_frame_offset", "action_single_stream")}
        return x
    return {k: desc(v) for k, v in sorted(batch.items())}


def main(argv=None) -> None:
    args = parse_args(argv)
    for key in ("vae_path", "data_root", "jsonl", "past_frames", "cameras", "out_dir"):
        setattr(args, key, _abs(getattr(args, key)))
    if args.images:
        args.images = ",".join(_abs(p) for p in args.images.split(","))
    if not Path(args.toml).is_absolute() and not Path(args.toml).exists():
        args.toml = str(REPO_ROOT / args.toml)
    args.toml = _abs(args.toml)
    if Path(args.ckpt).expanduser().exists():
        args.ckpt = _abs(args.ckpt)
    os.chdir(REPO_ROOT)

    if args.episode:
        episodes = [e for e in args.episode.split(",") if e]
        frames = [int(f) for f in args.frame.split(",") if f != ""] or [4 * (args.past_latents - 1)]
        if len(frames) == 1:
            frames = frames * len(episodes)
        if len(frames) != len(episodes):
            raise SystemExit("--frame needs one value or one per --episode")
        clips = list(zip(episodes, frames))
        jsonl = Path(args.jsonl) if args.jsonl else Path(args.data_root) / "train" / "video_dataset_file.jsonl"
    else:
        clips, jsonl = [(None, None)], None

    if args.dry_run:
        rank, world, vae = 0, 1, args.vae_path or "unused"
    else:
        from cosmos_framework.utils import distributed

        distributed.init()
        rank, world = distributed.get_rank(), distributed.get_world_size()
        vae = resolve_vae(args.vae_path, rank)
        ckpt_dir = resolve_checkpoint(args.ckpt, args.revision, rank)

    config = load_config(args.toml, vae, list(args.override))
    ds = build_dataset(config)
    gripper_flip = bool(ds._mvt_get("_mvt_gripper_flip", True))
    device = None if args.dry_run else "cuda"

    def make(clip):
        episode, t0 = clip
        if episode is None:
            sample, info = raw_inputs(args, ds)
            name = args.name
        else:
            md = find_episode(jsonl, args.data_root, episode)
            sample, info = episode_sample(ds, md, t0, args.past_latents, args.prompt, seed=args.seed)
            name = f"{info['episode']}_t{t0}"
        return to_batch(sample, device), info, name

    if args.dry_run:
        for clip in clips:
            batch, info, name = make(clip)
            print(json.dumps({"name": name, "views": info["views"], "geometry": info["geometry"],
                              "batch": summarize(batch)}, indent=1, default=str))
        return

    t_load = time.time()
    model, kind = build_model(config, ckpt_dir)
    log.info(f"[droid] model ready in {time.time() - t_load:.0f}s ({kind} weights from {ckpt_dir})")
    out_root = Path(args.out_dir)
    base_meta = {
        "checkpoint": args.ckpt, "checkpoint_format": kind, "revision": args.revision, "seed": args.seed,
        "num_steps": args.num_steps, "guidance": args.guidance, "shift": args.shift,
        "past_latents": args.past_latents, "gripper_flip": gripper_flip,
        "tracks": args.tracks, "grid_size": args.grid_size, "track_sigma": args.track_sigma,
        "num_gpus": world,
    }
    stand_in = None
    for rnd in range((len(clips) + world - 1) // world):
        idx = rnd * world + rank
        real = idx < len(clips)
        built = None
        if real:
            try:
                built = make(clips[idx])
            except Exception:
                traceback.print_exc()
                real = False
        if built is None:
            if stand_in is None:
                stand_in = make(clips[0])
            built = stand_in
        batch, info, name = built
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        res = run_clip(model, batch, args.past_latents, args)
        torch.cuda.synchronize()
        seconds = time.time() - t_start
        if not real:
            continue
        d = out_root / name
        meta = dict(base_meta, seconds=round(seconds, 1),
                    max_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    max_memory_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
        write_outputs(d, res, batch, info, meta, info["fps"], gripper_flip)
        log.info(f"[droid] rank {rank}: {name} -> {d} ({seconds:.0f}s)", rank0_only=False)

    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
