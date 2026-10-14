# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DROID inference with VPM-DROID-v2 (three cameras, joint-position actions, no camera calibration).

From the clean past frames of the three DROID cameras (``exterior_1_left``, ``exterior_2_left``, ``wrist_left``),
the robot joint state, the joint actions that led to the current frame and a task instruction, VPM-DROID-v2 jointly
generates the next 16 frames of every camera and a 16-step joint-position action chunk. Optionally it then tracks
points on the generated clip with its point-track head: 2D tracks with visibility and depth, lifted to 3D when the
camera calibration is given (``--cameras``).

Inputs (one of):
  * an episode of DROID-Raw (``rooty2020/DROID-Raw``): ``--data-root <DROID-Raw dir> --episode <row> --frame <t0>``.
    The sample is built exactly as in training, without the image augmentation and the caption dropout. Only the
    frames up to ``t0``, the state at ``t0`` and the actions before ``t0`` reach the model; the ground-truth frames
    and actions after ``t0`` are saved next to the prediction.
  * raw arrays: ``--past-frames frames.npz --past-actions actions.npy --state q0,..,q6,g --prompt "..."`` (the last
    9 frames of every camera and the 8 actions between them), or ``--images ext1.png,ext2.png,wrist.png`` for the
    current frame only (``--past-latents 1``; less accurate actions).

Outputs in ``<out-dir>/<name>/``:
  * ``video/<view>.mp4``: clean past frames followed by the 16 generated frames, 320x192, 15 fps. With an episode
    input also ``video/<view>_gt.mp4``, the same frames from the dataset.
  * ``actions.npz``: ``actions`` [16, 8] absolute joint positions (rad) and gripper (0 open .. 1 closed), the DROID
    actions of frames ``t0 .. t0+15``; ``state`` [8] the input state in the same convention; ``model_rows`` [R, 8]
    the raw model rows (state, past actions, generated actions; gripper as ``1 - g``); ``dt`` seconds per row.
    With an episode input also ``gt_actions`` [16, 8].
  * with ``--tracks grid``: ``video/<view>_tracks.mp4`` and ``tracks.npz`` with ``tracks_px`` [3, T, N, 2] in the
    320x192 frame, ``tracks_input_px`` in input-image pixels, ``vis`` (visibility x confidence), ``conf`` and ``depth``
    [3, T, N] (metres along the camera z axis); with ``--cameras`` also ``xyz_cam`` and ``xyz_world`` [3, T, N, 3].
  * ``meta.json``: inputs, sampling settings, timing, peak GPU memory and, for episodes, the action error.

Launch with ``torchrun --nproc_per_node=N`` (see ``inference/infer_droid_v2.sh``): the model is sharded over the N
GPUs and the requested episodes run one per rank in lockstep.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
for _p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_droid import (  # noqa: E402
    WAN_VAE_REPO,
    _abs,
    _first,
    _np,
    _u8,
    _write_mp4,
    build_model,
    load_cameras,
    load_config,
    resolve_checkpoint,
    resolve_vae,
    run_clip,
    summarize,
    to_batch,
)

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id  # noqa: E402
from cosmos_framework.data.vfm.action.transforms import pad_action_to_max_dim  # noqa: E402
from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio  # noqa: E402
from cosmos_framework.data.vfm.sequence_packing import SequencePlan  # noqa: E402
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO  # noqa: E402
from cosmos_framework.utils import log  # noqa: E402

DEFAULT_CKPT = "rooty2020/VPM-DROID-v2"
DEFAULT_TOML = "examples/toml/vpm_stage3.toml"
MODEL_OVERRIDES = [
    "model.config.camera_conditioning_enabled=false",
    "model.config.camera_roce_mode=none",
    "model.config.diffusion_expert_config.max_vae_latent_side_after_patchify=20",
]
VIEWS = ["exterior_1_left", "exterior_2_left", "wrist_left"]
VIEWPOINT_NOTE = "3 views: third-person camera, third-person camera, wrist camera."
DROID_DOMAIN = "oxe/droid"
RESOLUTION = "256"
CHUNK = 16
STATE_DIM = 8
MAX_ACTION_DIM = 64
MAX_TEXT_TOKENS = 1024
DROID_FPS = 15.0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("VPM_CHECKPOINT", DEFAULT_CKPT),
                    help="Hugging Face repo id or local dir: a release dir ([model/]*.safetensors) or a training "
                         f"iter_* dir (model/ DCP). Default: $VPM_CHECKPOINT, else {DEFAULT_CKPT}")
    ap.add_argument("--revision", default=None, help="Hugging Face revision of --ckpt")
    ap.add_argument("--toml", default=DEFAULT_TOML, help="stage-3 experiment TOML the model config is built from")
    ap.add_argument("--vae-path", default=os.environ.get("WAN_VAE_PATH", ""),
                    help=f"Wan2.2 VAE weights (default: $WAN_VAE_PATH, else downloaded from {WAN_VAE_REPO})")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="extra Hydra override applied to the config (repeatable)")

    a = ap.add_argument_group("input A: episode of DROID-Raw")
    a.add_argument("--data-root", default=os.path.join(os.environ.get("VPM_DATA_ROOT", "data"), "DROID-Raw"),
                   help="DROID-Raw root with oxe_policy_cache/ and lerobot/ (default: $VPM_DATA_ROOT/DROID-Raw)")
    a.add_argument("--episode", default="", help="episode row(s) of oxe_policy_cache/droid/episodes.npz, comma separated")
    a.add_argument("--frame", default="",
                   help="current frame t0, one for all episodes or one per episode (default: the start of the "
                        "episode's first non-idle range of at least 17 frames)")
    a.add_argument("--list", type=int, default=0, metavar="N", help="print the first N usable episode rows and exit")

    b = ap.add_argument_group("input B: raw three-camera observation")
    b.add_argument("--images", default="",
                   help="the current frame of exterior_1_left, exterior_2_left and wrist_left, comma separated")
    b.add_argument("--past-frames", default="",
                   help="npz with 'frames' uint8 [3, P, H, W, 3] ending at the current frame "
                        "(P >= 4 * (past_latents - 1) + 1); replaces --images")
    b.add_argument("--past-actions", default="",
                   help="npy/npz [4 * (past_latents - 1), 8]: the DROID joint actions of the past frames, oldest first "
                        "(gripper 0 open .. 1 closed); needed when --past-latents > 1")
    b.add_argument("--state", default="", help="q0,...,q6,g: joint positions (rad) and gripper (0 open .. 1 closed)")
    ap.add_argument("--cameras", default="",
                    help="optional json {view: {intrinsics: 3x3 pixel K at the input image size, extrinsics: 4x4 "
                         "world-to-camera in metres at the current frame}} for exterior_1_left, exterior_2_left and "
                         "wrist_left; with --tracks the head is conditioned on them and the tracks are lifted to 3D. "
                         "The extrinsics are held for the whole clip, so xyz_world of the moving wrist camera is only "
                         "exact at the current frame")
    b.add_argument("--name", default="sample", help="output sub-directory name")

    ap.add_argument("--prompt", default=None,
                    help="task instruction (input A default: the episode instruction; '' = unconditional)")
    ap.add_argument("--past-latents", type=int, default=None, choices=(1, 2, 3),
                    help="clean past latents: 1 = current frame only, p = the last 4 * (p - 1) + 1 frames and the "
                         "actions between them (default: 3 for episodes and --past-frames, 1 for --images); the 9-frame "
                         "context gives much more accurate actions")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracks", choices=("none", "grid"), default="none",
                    help="point tracks on the generated clip (experimental), queried on a regular grid in every camera")
    ap.add_argument("--grid-size", type=int, default=12)
    ap.add_argument("--track-sigma", type=float, default=0.15, help="noise level of the tracking forward pass")
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("OUTPUT_ROOT", "outputs"), "droid_v2_inference"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build the input batches and print their contents; no GPU, no checkpoint")
    argv = list(sys.argv[1:] if argv is None else argv)
    i = 0
    while i < len(argv) - 1:
        if argv[i] == "--state" and argv[i + 1].startswith("-"):
            argv[i : i + 2] = [f"--state={argv[i + 1]}"]
        i += 1
    args = ap.parse_args(argv)
    if args.list:
        return args
    raw = bool(args.images or args.past_frames)
    if bool(args.episode) == raw:
        ap.error("give either --episode (input A) or --images/--past-frames (input B)")
    if raw and (not args.state or args.prompt is None):
        ap.error("input B needs --state and --prompt")
    if args.past_latents is None:
        args.past_latents = 1 if args.images and not args.past_frames else 3
    if raw and args.past_latents > 1 and not (args.past_frames and args.past_actions):
        ap.error("--past-latents > 1 needs --past-frames and --past-actions (or pass --past-latents 1)")
    return args


class DroidRaw:
    """Read-only access to DROID-Raw: episode index, raw 8D joint actions / states and the JPEG frame caches."""

    def __init__(self, root: str):
        self.root = Path(root)
        c = self.root / "oxe_policy_cache" / "droid"
        if not (c / "episodes.npz").is_file():
            raise FileNotFoundError(f"{c}/episodes.npz not found: --data-root must be a DROID-Raw download")
        self.meta = json.loads((c / "meta.json").read_text())
        self.tasks: list[str] = json.loads((c / "tasks.json").read_text())
        self.ep = dict(np.load(c / "episodes.npz"))
        self.actions = np.load(c / "joint_actions.npy", mmap_mode="r")
        self.states = np.load(c / "joint_states.npy", mmap_mode="r")
        self.cams: list[str] = list(self.meta["camera_keys"])
        self.fps = float(self.meta["fps"])
        self._files: dict[tuple[int, int, int], tuple[np.ndarray, io.BufferedReader]] = {}

    def usable_rows(self) -> np.ndarray:
        """Episodes the training run sampled from (kept, instructed, all views cached, a non-idle range >= 17 frames)."""
        e = self.ep
        ok = e["keep"] & np.array([bool(t and t.strip()) for t in self.tasks]) & e["cam_ok"].all(1)
        good = (e["range_end"] - e["range_start"]) >= CHUNK + 1
        long_enough = np.zeros_like(ok)
        long_enough[e["range_ep"][good]] = True
        return np.flatnonzero(ok & long_enough)

    def nonidle_ranges(self, row: int) -> list[tuple[int, int]]:
        m = self.ep["range_ep"] == row
        return list(zip(self.ep["range_start"][m].tolist(), self.ep["range_end"][m].tolist()))

    def length(self, row: int) -> int:
        return int(self.ep["length"][row])

    def rows(self, table: np.ndarray, row: int, frames) -> np.ndarray:
        """Rows of ``table`` at episode-relative ``frames`` (clamped into the episode)."""
        ids = np.clip(np.asarray(frames, np.int64), 0, self.length(row) - 1) + int(self.ep["row_start"][row])
        return np.asarray(table[ids], np.float32)

    def frames(self, row: int, cam: int, frames) -> np.ndarray:
        """uint8 [n, H, W, 3] RGB frames of camera ``cam`` at episode-relative ``frames`` (clamped)."""
        from PIL import Image

        e = self.ep
        if not e["cam_ok"][row, cam]:
            raise ValueError(f"episode row {row}: camera {self.cams[cam]} is not in DROID-Raw")
        key = (cam, int(e["cam_chunk"][row, cam]), int(e["cam_file"][row, cam]))
        if key not in self._files:
            d = (self.root / "lerobot" / "droid_1.0.1" / "videos" / self.cams[cam] / f"chunk-{key[1]:03d}"
                 / f"file-{key[2]:03d}.framecache")
            self._files[key] = (np.load(d / "index.npy", mmap_mode="r"), open(d / "frames.bin", "rb"))
        index, fh = self._files[key]
        base, L = int(e["cam_base"][row, cam]), self.length(row)
        out = []
        for f in frames:
            off, n = (int(x) for x in index[base + max(0, min(L - 1, int(f)))])
            fh.seek(off)
            out.append(np.asarray(Image.open(io.BytesIO(fh.read(n))).convert("RGB")))
        return np.stack(out)


def build_tokenizer(config):
    """The Qwen3-VL text tokenizer of the training data pipeline (with the VPM special tokens)."""
    from cosmos_framework.data.vfm.sequence_packing import add_special_tokens
    from cosmos_framework.utils.lazy_config import instantiate

    node = config.dataloader_train.dataloader.datasets["droid"].dataset.tokenizer_config
    tokenizer = instantiate(node).tokenizer
    return add_special_tokens(tokenizer)[0]


def make_caption(prompt: str) -> str:
    """The training caption: the instruction as a sentence plus the viewpoint note ('' stays unconditional)."""
    caption = prompt.strip()
    if not caption:
        return ""
    caption = caption[0].upper() + caption[1:]
    return f"{caption.rstrip('.')}. {VIEWPOINT_NOTE}"


def geometry(H: int, W: int) -> dict:
    """Cover resize of an ``H x W`` image to the model size, then center crop (the training data pipeline)."""
    tw, th = VIDEO_RES_SIZE_INFO[RESOLUTION][get_aspect_ratio(W, H)]
    ratio = max(tw / W, th / H)
    rh, rw = max(th, int(round(H * ratio))), max(tw, int(round(W * ratio)))
    return dict(input_hw=(H, W), resize_hw=(rh, rw), crop_yx=((rh - th) // 2, (rw - tw) // 2), output_hw=(th, tw))


def process_frames(frames: np.ndarray) -> torch.Tensor:
    """uint8 [n, H, W, 3] -> float [3, n, 192, 320] in [-1, 1]: cover resize (bilinear, antialiased) + center crop."""
    geo = geometry(*frames.shape[1:3])
    (rh, rw), (y0, x0), (th, tw) = geo["resize_hw"], geo["crop_yx"], geo["output_hw"]
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False, antialias=True)
    x = x[:, :, y0 : y0 + th, x0 : x0 + tw].permute(1, 0, 2, 3)
    return x.mul_(2.0).sub_(1.0).contiguous()


def normalized_intrinsics(K: np.ndarray, geo: dict) -> np.ndarray:
    """Pixel K at the input size -> K of the model frame, normalized by its width / height."""
    (H, W), (rh, rw), (y0, x0), (th, tw) = geo["input_hw"], geo["resize_hw"], geo["crop_yx"], geo["output_hw"]
    sx, sy = rw / W, rh / H
    Kn = np.zeros((3, 3), np.float32)
    Kn[0, 0], Kn[0, 2] = K[0, 0] * sx / tw, (K[0, 2] * sx - x0) / tw
    Kn[1, 1], Kn[1, 2] = K[1, 1] * sy / th, (K[1, 2] * sy - y0) / th
    Kn[2, 2] = 1.0
    return Kn


def build_sample(tokenizer, past: list[np.ndarray], state: np.ndarray, past_actions: np.ndarray, prompt: str,
                 p_lat: int, fps: float = DROID_FPS, name: str = "sample", cameras: "tuple | None" = None) -> dict:
    """The training sample of one window with the future hidden.

    ``past``: per view uint8 ``[4 * (p_lat - 1) + 1, H, W, 3]`` ending at the current frame. ``state``: ``[8]`` at the
    current frame. ``past_actions``: ``[4 * (p_lat - 1), 8]`` actions of the past frames. Joints in rad, gripper
    0 open .. 1 closed (the model sees ``1 - g``). ``cameras``: optional (pixel Ks, world-to-camera 4x4s) per view at
    the input image size, for the point-track head.
    """
    hist = 4 * (p_lat - 1)
    p_px, T = hist + 1, hist + 1 + CHUNK
    if len(past) != len(VIEWS):
        raise ValueError(f"need {len(VIEWS)} views ({', '.join(VIEWS)}), got {len(past)}")
    if any(f.ndim != 4 or f.shape[0] != p_px or f.shape[-1] != 3 for f in past):
        raise ValueError(f"expected {p_px} past frames [H, W, 3] per view, got {[f.shape for f in past]}")
    if any(f.shape[1:3] != past[0].shape[1:3] for f in past):
        raise ValueError("all views must share one image size")
    video = []
    for f in past:
        clip = process_frames(f)
        video.append(torch.cat([clip, clip.new_zeros(3, T - p_px, *clip.shape[2:])], dim=1))
    th, tw = video[0].shape[-2:]

    caption = make_caption(prompt)
    from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption

    text_ids = tokenize_caption(caption, tokenizer, is_video=True, use_system_prompt=False)[:MAX_TEXT_TOKENS]

    rows = np.zeros((1 + hist + CHUNK, STATE_DIM), np.float32)
    rows[0] = np.asarray(state, np.float32)
    rows[1 : 1 + hist] = np.asarray(past_actions, np.float32).reshape(hist, STATE_DIM)
    rows[: 1 + hist, -1] = 1.0 - rows[: 1 + hist, -1]
    image_size = torch.tensor([th, tw, th, tw], dtype=torch.float32)
    extra = {}
    if cameras is not None:
        geo = geometry(*past[0].shape[1:3])
        Ks, w2cs = cameras
        extra = dict(
            track_intrinsics=torch.from_numpy(np.stack([normalized_intrinsics(K, geo) for K in Ks])),
            track_extrinsics=torch.from_numpy(np.stack([np.repeat(np.asarray(E, np.float32)[None], T, axis=0)
                                                        for E in w2cs])),
        )
    return dict(
        **extra,
        __key__=name,
        __url__="",
        dataset_name="droid",
        fps=float(fps),
        conditioning_fps=float(fps),
        num_frames=T,
        video=video,
        image_size=[image_size] * len(video),
        padding_mask=torch.zeros((1, th, tw), dtype=torch.float32),
        ai_caption=caption,
        text_token_ids=torch.tensor(text_ids, dtype=torch.long),
        action=pad_action_to_max_dim(torch.from_numpy(rows), MAX_ACTION_DIM),
        domain_id=torch.tensor(get_domain_id(DROID_DOMAIN), dtype=torch.long),
        raw_action_dim=torch.tensor(STATE_DIM, dtype=torch.long),
        mode="policy",
        sequence_plan=SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=list(range(p_lat)),
            num_source_views=0,
            has_action=True,
            condition_frame_indexes_action=list(range(1 + hist)),
            action_start_frame_offset=0,
            action_single_stream=True,
            action_view_index=0,
        ),
    )


def episode_input(data: DroidRaw, row: int, t0: "int | None", p_lat: int, prompt: "str | None", tokenizer,
                  cameras: "tuple | None" = None) -> tuple:
    """(sample, info) of episode ``row`` with the current frame ``t0``; ``info`` carries the ground truth."""
    if not data.ep["cam_ok"][row].all():
        raise ValueError(f"episode row {row} does not have all three cameras in DROID-Raw")
    L = data.length(row)
    if t0 is None:
        ranges = [r for r in data.nonidle_ranges(row) if r[1] - r[0] >= CHUNK + 1]
        t0 = ranges[0][0] if ranges else 0
    if not 0 <= t0 <= L - CHUNK - 1:
        raise ValueError(f"episode row {row}: --frame {t0} out of range [0, {L - CHUNK - 1}] ({L} frames)")
    hist = 4 * (p_lat - 1)
    past_ids = list(range(t0 - hist, t0 + 1))
    future_ids = list(range(t0 + 1, t0 + CHUNK + 1))
    past = [data.frames(row, j, past_ids) for j in range(len(VIEWS))]
    state = data.rows(data.states, row, [t0])[0]
    past_actions = data.rows(data.actions, row, range(t0 - hist, t0))
    instruction = data.tasks[row] if prompt is None else prompt
    name = f"row{row}_t{t0}"
    sample = build_sample(tokenizer, past, state, past_actions, instruction, p_lat, fps=data.fps, name=name,
                          cameras=cameras)
    info = {
        "geometry": geometry(*past[0].shape[1:3]),
        "input": "episode",
        "episode_row": int(row),
        "episode_index": int(data.ep["episode_index"][row]),
        "t0": int(t0),
        "instruction": data.tasks[row],
        "state": state,
        "fps": data.fps,
        "gt_actions": data.rows(data.actions, row, range(t0, t0 + CHUNK)),
        "gt_future": [data.frames(row, j, future_ids) for j in range(len(VIEWS))],
        "name": name,
    }
    return sample, info


def _load_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as z:
            return np.asarray(z[z.files[0]] if len(z.files) == 1 else z["frames" if "frames" in z.files else "actions"])
    return np.load(path)


def raw_input(args, tokenizer, cameras: "tuple | None" = None) -> tuple:
    from PIL import Image

    hist = 4 * (args.past_latents - 1)
    if args.past_frames:
        frames = [np.asarray(v, np.uint8) for v in _load_array(args.past_frames)]
        if any(f.shape[0] < hist + 1 for f in frames):
            raise ValueError(f"--past-latents {args.past_latents} needs {hist + 1} past frames per view")
        frames = [f[-(hist + 1):] for f in frames]
    else:
        frames = [np.asarray(Image.open(p).convert("RGB"), np.uint8)[None] for p in args.images.split(",")]
    state = np.asarray([float(x) for x in args.state.split(",")], np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"--state needs {STATE_DIM} values (7 joints + gripper), got {state.size}")
    past_actions = np.zeros((0, STATE_DIM), np.float32)
    if hist:
        past_actions = np.asarray(_load_array(args.past_actions), np.float32)
        if past_actions.ndim != 2 or past_actions.shape[1] != STATE_DIM or past_actions.shape[0] < hist:
            raise ValueError(f"--past-actions needs [{hist}, {STATE_DIM}], got {past_actions.shape}")
        past_actions = past_actions[-hist:]
    sample = build_sample(tokenizer, frames, state, past_actions, args.prompt, args.past_latents, name=args.name,
                          cameras=cameras)
    return sample, {"input": "raw", "state": state, "fps": DROID_FPS, "name": args.name,
                    "geometry": geometry(*frames[0].shape[1:3])}


def check_checkpoint(ckpt_dir: Path) -> None:
    """Refuse a VPM-DROID (v1) release: it carries the camera-conditioning weights and needs infer_droid.sh."""
    for idx in (ckpt_dir / "model" / "model.safetensors.index.json", ckpt_dir / "model.safetensors.index.json"):
        if idx.is_file():
            keys = json.loads(idx.read_text())["weight_map"]
            if any(k.startswith("net.camera_conditioner.") for k in keys):
                raise SystemExit(f"{ckpt_dir} is a camera-conditioned VPM checkpoint (VPM-DROID); "
                                 "run it with inference/infer_droid.sh")
            return


def write_outputs(d: Path, res: dict, batch: dict, info: dict, meta: dict, p_lat: int) -> None:
    (d / "video").mkdir(parents=True, exist_ok=True)
    fps = info["fps"]
    frames = [_u8(v) for v in res["pred"]]
    for name, f in zip(VIEWS, frames):
        _write_mp4(d / "video" / f"{name}.mp4", f, fps)
    if "gt_future" in info:
        for name, f, gt in zip(VIEWS, frames, info["gt_future"]):
            p_px = res["p_px"]
            gt_clip = _u8(process_frames(gt))
            _write_mp4(d / "video" / f"{name}_gt.mp4", np.concatenate([f[:p_px], gt_clip]), fps)

    hist = 4 * (p_lat - 1)
    extra = {}
    if res["action"] is not None:
        rows = res["action"][:, :STATE_DIM].numpy().astype(np.float32)
        actions = rows[1 + hist :].copy()
        actions[:, -1] = np.clip(1.0 - actions[:, -1], 0.0, 1.0)
        arrays = dict(actions=actions, state=np.asarray(info["state"], np.float32), model_rows=rows,
                      dt=np.float32(1.0 / fps))
        if "gt_actions" in info:
            gt = np.asarray(info["gt_actions"], np.float32)
            arrays["gt_actions"] = gt
            extra = {"joint_mae_rad": float(np.abs(actions[:, :7] - gt[:, :7]).mean()),
                     "gripper_mae": float(np.abs(actions[:, 7] - gt[:, 7]).mean())}
        np.savez(d / "actions.npz", **arrays)

    tr = res.get("tracks")
    if tr is not None:
        uv = _np(tr["tracks"]).astype(np.float32)
        geo = info["geometry"]
        (H, W), (rh, rw), (y0, x0), (th, tw) = geo["input_hw"], geo["resize_hw"], geo["crop_yx"], geo["output_hw"]
        out = {
            "tracks_px": uv,
            "tracks_input_px": ((uv + np.array([x0, y0], np.float32)) * np.array([W / rw, H / rh], np.float32)),
            "sigma": np.float32(meta["track_sigma"]),
            "views": np.array(VIEWS),
        }
        for key in ("vis", "conf", "depth"):
            if tr.get(key) is not None:
                out[key] = _np(tr[key]).astype(np.float32)
        if "depth" in out and batch.get("track_intrinsics") is not None:
            Kn = _np(_first(batch["track_intrinsics"])).astype(np.float64)
            w2c = _np(_first(batch["track_extrinsics"])).astype(np.float64)[:, : uv.shape[1]]
            uvn = uv / np.array([tw, th], np.float64)
            pix = np.concatenate([uvn, np.ones_like(uvn[..., :1])], axis=-1)
            xc = np.einsum("vij,vtnj->vtni", np.linalg.inv(Kn), pix) * out["depth"][..., None]
            c2w = np.linalg.inv(w2c)
            xw = np.einsum("vtij,vtnj->vtni", c2w[..., :3, :3], xc) + c2w[:, :, None, :3, 3]
            out["xyz_cam"], out["xyz_world"] = xc.astype(np.float32), xw.astype(np.float32)
        np.savez(d / "tracks.npz", **out)
        from cosmos_framework.model.vfm.tracking.viz import draw_tracks

        for v, name in enumerate(VIEWS):
            vis = out["vis"][v] > 0.5 if "vis" in out else None
            _write_mp4(d / "video" / f"{name}_tracks.mp4", draw_tracks(frames[v], uv[v], visible=vis, trail=8, radius=2),
                       fps)

    meta = dict(meta, **extra, views=VIEWS, fps=fps, past_frames=res["p_px"],
                generated_frames=int(frames[0].shape[0] - res["p_px"]), state=np.asarray(info["state"]).tolist())
    for key in ("input", "episode_row", "episode_index", "t0", "instruction"):
        if key in info:
            meta[key] = info[key]
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def main(argv=None) -> None:
    args = parse_args(argv)
    for key in ("vae_path", "data_root", "past_frames", "past_actions", "cameras", "out_dir"):
        setattr(args, key, _abs(getattr(args, key)))
    if args.images:
        args.images = ",".join(_abs(p) for p in args.images.split(","))
    if not Path(args.toml).is_absolute() and not Path(args.toml).exists():
        args.toml = str(REPO_ROOT / args.toml)
    args.toml = _abs(args.toml)
    if Path(args.ckpt).expanduser().exists():
        args.ckpt = _abs(args.ckpt)
    os.chdir(REPO_ROOT)

    if args.list:
        data = DroidRaw(args.data_root)
        for r in data.usable_rows()[: args.list]:
            print(int(r), data.length(int(r)), data.nonidle_ranges(int(r))[0], data.tasks[r])
        return

    data = None
    if args.episode:
        rows = [int(e) for e in args.episode.split(",") if e]
        t0s = [int(f) for f in args.frame.split(",") if f != ""] or [None]
        if len(t0s) == 1:
            t0s = t0s * len(rows)
        if len(t0s) != len(rows):
            raise SystemExit("--frame needs one value or one per --episode")
        clips = list(zip(rows, t0s))
        data = DroidRaw(args.data_root)
    else:
        clips = [(None, None)]

    if args.dry_run:
        rank, world, vae = 0, 1, args.vae_path or "unused"
    else:
        from cosmos_framework.utils import distributed

        distributed.init()
        rank, world = distributed.get_rank(), distributed.get_world_size()
        vae = resolve_vae(args.vae_path, rank)
        ckpt_dir = resolve_checkpoint(args.ckpt, args.revision, rank)
        check_checkpoint(ckpt_dir)

    config = load_config(args.toml, vae, MODEL_OVERRIDES + list(args.override))
    tokenizer = build_tokenizer(config)
    device = None if args.dry_run else "cuda"
    cameras = None
    if args.cameras:
        Ks, w2cs, _ = load_cameras(args.cameras, VIEWS)
        cameras = (Ks, w2cs)

    def make(clip):
        row, t0 = clip
        if row is None:
            sample, info = raw_input(args, tokenizer, cameras)
        else:
            sample, info = episode_input(data, row, t0, args.past_latents, args.prompt, tokenizer, cameras)
        return to_batch(sample, device), info

    if args.dry_run:
        for clip in clips:
            batch, info = make(clip)
            print(json.dumps({"name": info["name"], "caption": batch["ai_caption"], "batch": summarize(batch)},
                             indent=1, default=str))
        return

    t_load = time.time()
    model, kind = build_model(config, ckpt_dir)
    log.info(f"[droid-v2] model ready in {time.time() - t_load:.0f}s ({kind} weights from {ckpt_dir})")
    out_root = Path(args.out_dir)
    base_meta = {
        "checkpoint": args.ckpt, "checkpoint_format": kind, "revision": args.revision, "seed": args.seed,
        "num_steps": args.num_steps, "guidance": args.guidance, "shift": args.shift,
        "past_latents": args.past_latents, "num_gpus": world,
        "tracks": args.tracks, "grid_size": args.grid_size, "track_sigma": args.track_sigma, "cameras": bool(cameras),
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
        batch, info = built
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        res = run_clip(model, batch, args.past_latents, args)
        torch.cuda.synchronize()
        seconds = time.time() - t_start
        if not real:
            continue
        d = out_root / info["name"]
        meta = dict(base_meta, caption=batch["ai_caption"][0], seconds=round(seconds, 1),
                    max_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    max_memory_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
        write_outputs(d, res, batch, info, meta, args.past_latents)
        log.info(f"[droid-v2] rank {rank}: {info['name']} -> {d} ({seconds:.0f}s)", rank0_only=False)

    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
