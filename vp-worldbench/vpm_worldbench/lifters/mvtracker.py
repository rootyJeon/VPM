"""MVTracker + MoGe-2: the reference lifter of the δ metric.

    from vpm_worldbench.lifters.base import Anchor
    from vpm_worldbench.lifters.mvtracker import MVTrackerLifter

    lifter = MVTrackerLifter(device="cuda")                       # public weights via torch.hub / the HF hub
    lifter = MVTrackerLifter(hub_repo="/path/to/mvtracker",       # local clone of github.com/ethz-vlg/mvtracker
                             moge="/path/to/moge-2-vitl-normal/model.pt")
    anchor = Anchor("gt_depth", depth=gt_depth_ref, far_clip_m=80.0)              # Simul-*, Real-General
    anchor = lifter.prepare_anchor(Anchor("gt_points", points=X, points_vis=vis, far_clip_m=20.0),
                                   real_frames, K, w2c)                           # Real-Robot (no depth)
    xyz, vis = lifter.track(frames, K, w2c, queries, anchor)

Pipeline (`DepthAnchoredLifter.track`, shapes in lifters/base.py):
1. MoGe-2 ViT-L metric depth of every frame of every view: `MoGeModel.infer(rgb / 255, fov_x)` with
   fov_x = 2 atan(W / 2 fx) from K, resolution level 9, fp16 on CUDA, batches of 16 frames per view; pixels outside
   MoGe's mask are invalid.
2. Per view anchoring on window position 0 (`Anchor`): scale+shift fit against GT depth, or the median ratio at the
   visible GT points; anchored depth outside (0, far_clip_m] -> 0.
3. MVTracker-style scene normalisation (radius 6.3, floor quantile 0.12, true camera centres, +z up) from
   `anchor.norm_depth`, applied to depth, extrinsics and queries.
4. One call of the MVTracker hub predictor with all queries (predictor defaults of the MVTracker release: inputs
   resized to 384x512 with nearest neighbour, a 4x4 support grid per view from the depth of window position 0,
   6 iterations, visibility = sigmoid > 0.5; fp32, no autocast), predictions mapped back to the world.

RGB scale: `MVTracker.forward` normalises its input as 2 * (rgb / 255) - 1 itself, so it expects RGB in 0..255 (as in
MVTracker's own evaluation). The internal pre-release numbers - like MVTracker's README demo - passed rgb / 255, so the
feature encoder saw nearly constant images (values in [-1, -0.992]). The default passes 0..255;
`rgb_scale_quirk=True` reproduces the pre-release behaviour (legacy protocol).

Weights: MVTracker through `torch.hub.load(hub_repo, hub_model, pretrained=True)`; `hub_repo` is a GitHub repo
'owner/repo[:ref]' or a local clone (loaded with source='local'). The default, DEFAULT_HUB_REPO =
'ethz-vlg/mvtracker:ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072', is the commit of the reference numbers, fetched without
torch.hub's branch / tag check (which only finds the head commits of branches and tags), so later commits to MVTracker's
main branch do not change the default; 'ethz-vlg/mvtracker' alone follows the main branch. Its hubconf downloads the
checkpoint hf://ethz-vlg/mvtracker::mvtracker_200000_june2025.pth (HF hub cache, honours HF_HUB_OFFLINE). δ records
store hub_repo and the commit it names (models.delta.paths: hub_repo, hub_commit). MoGe-2 comes from the `moge`
package (`MoGeModel.from_pretrained`: HF repo id 'Ruicheng/moge-2-vitl-normal' or a local model.pt). Every δ record
stores weight fingerprints (models.delta.tracker_weights / moge_weights, `tasks.generation.weights_fingerprint`); the
reference weights give 01943de6871bce32 / 2e283b2674fab4d4.

Installation (Python >= 3.10, which torch 2.10, numpy 2.2 and moge 3.0 need; the stack of the reference numbers:
MVTracker commit ceea8ad, moge 3.0.0 at commit 74fbce0, torch 2.10.0 with CUDA 13.0, Python 3.13; the same recipe with
CPU torch 2.10.0 on Python 3.12 loads the same weights):

    # torch + torchvision built for your CUDA (reference: 2.10.0 / 0.25.0; cu130 wheels for x86_64 and aarch64)
    pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
    pip install -e ".[track]"
    # MVTracker: only what its hub entry imports (its requirements.txt pins numpy 1.24, which cannot be installed
    # next to this package's dependencies); moviepy must stay 1.x (moviepy.editor); mediapy 1.2.6 does not import
    # with numpy 2.5
    pip install numpy==2.2.6 huggingface_hub==0.36.2 easydict==1.13 einops==0.8.2 pandas==2.2.3 kornia==0.8.2 \\
        matplotlib==3.10.8 mediapy==1.2.6 rerun-sdk==0.38.1 pypng==0.20220715.0 flow-vis==0.1 moviepy==1.0.3
    git clone https://github.com/ethz-vlg/mvtracker
    git -C mvtracker checkout ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072
    # MoGe-2 without its app / training dependencies (MoGeModel v2 does not import them; `pip check` lists them)
    pip install --no-deps "moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
    pip install \\
        "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"

then pass the clone: `vpmwb generation evaluate ... --mvtracker-repo ./mvtracker`, `vpmwb track3d baseline --kind
lifter ... --hub-repo ./mvtracker`, or `MVTrackerLifter(hub_repo="./mvtracker")`. Without a clone, the default
hub_repo fetches the same commit through torch.hub (into its cache, ~/.cache/torch/hub or $TORCH_HOME/hub). Another
pinned commit ('owner/repo:<sha>') needs hub_kwargs={"skip_validation": True} once it is no longer the head of a
branch or tag. The same packages unpinned (2026-10: torch 2.14, numpy 2.5, pandas 3.0,
huggingface_hub 2.2, mediapy 1.2.7, moviepy 1.0.3) also loaded the reference weights and tracked. Offline nodes:
download both checkpoints once (HF hub cache) or pass a local MoGe model.pt, and set HF_HUB_OFFLINE=1.

Determinism: no sampling anywhere; repeated runs on the same GPU / software stack gave bit-identical tracks. But the
tracker amplifies tiny input differences: on Simul-Robot, re-decoding the same video with a different YUV->RGB
implementation (+-1-2 levels), a 1-ulp change of the anchor depth or of the queries moved the per-clip delta by
~0.3-2 points on average (up to ~5 on hard moving-camera clips), and the handedness of the world frame by ~5 (up to
16; see base.source_world_transform). Compare models on means over many clips, decode real and generated videos the
same way (vpm_worldbench.io.video), and treat per-clip differences of a few points as noise.
"""
from __future__ import annotations

import os

import numpy as np

from .base import DepthAnchoredLifter

REFERENCE_COMMIT = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
DEFAULT_HUB_REPO = f"ethz-vlg/mvtracker:{REFERENCE_COMMIT}"
DEFAULT_MOGE = "Ruicheng/moge-2-vitl-normal"


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("MVTrackerLifter needs PyTorch: pip install 'vpm-worldbench[track]' (pinned δ recipe: "
                          "docstring of vpm_worldbench/lifters/mvtracker.py)") from e
    return torch


class MVTrackerLifter(DepthAnchoredLifter):
    """MoGe-2 depth + MVTracker (see the module docstring).

    device                 torch device for both models ("cuda", "cuda:1", "cpu"; CPU works but is very slow)
    hub_repo               torch.hub repo of MVTracker: 'owner/repo[:ref]' on GitHub or a local directory (default:
                           DEFAULT_HUB_REPO, the commit of the reference numbers)
    moge                   MoGe-2 weights: HF repo id or a local model.pt
    rgb_scale_quirk        True = pass RGB / 255 to MVTracker (pre-release bug, legacy protocol); False = 0..255
    hub_source             force torch.hub source ('github' | 'local'); None = 'local' iff hub_repo is a directory
    hub_model              hub entry point ('mvtracker' or 'mvtracker_cleandepth')
    hub_kwargs             extra torch.hub.load kwargs (GitHub repos default to trust_repo=True: the repo you name
                           is trusted; DEFAULT_HUB_REPO also to skip_validation=True: a pinned commit is no branch or
                           tag head, which is all torch.hub's check accepts)
    moge_batch             frames per MoGe forward pass (16 as in the pre-release runs; fp16 results can depend on it)
    moge_resolution_level  MoGe `infer` resolution level (0-9; 9 = most tokens, the MoGe default)
    moge_fp16              MoGe mixed precision (CUDA only)
    track_size, plain_K_scaling: see `DepthAnchoredLifter`.
    Models are loaded on first use (or with `load()`).
    """

    name = "mvtracker"

    def __init__(self, device: str = "cuda", hub_repo: str = DEFAULT_HUB_REPO, moge: str = DEFAULT_MOGE,
                 rgb_scale_quirk: bool = False, *, hub_source: str | None = None, hub_model: str = "mvtracker",
                 hub_kwargs: dict | None = None, moge_batch: int = 16, moge_resolution_level: int = 9,
                 moge_fp16: bool = True, track_size=None, plain_K_scaling: bool = False):
        super().__init__(track_size=track_size, plain_K_scaling=plain_K_scaling)
        if hub_source not in (None, "github", "local"):
            raise ValueError("hub_source must be None, 'github' or 'local'")
        self.device = str(device)
        self.hub_repo = str(hub_repo)
        self.moge_weights = str(moge)
        self.rgb_scale_quirk = bool(rgb_scale_quirk)
        self.hub_source = hub_source
        self.hub_model = hub_model
        self.hub_kwargs = dict(hub_kwargs or {})
        self.moge_batch = int(moge_batch)
        self.moge_resolution_level = int(moge_resolution_level)
        self.moge_fp16 = bool(moge_fp16)
        self._tracker = None
        self._moge = None

    def __repr__(self):
        return (f"MVTrackerLifter(device={self.device!r}, hub_repo={self.hub_repo!r}, moge={self.moge_weights!r}, "
                f"rgb_scale_quirk={self.rgb_scale_quirk}, track_size={self.track_size}, "
                f"plain_K_scaling={self.plain_K_scaling})")

    def load(self) -> "MVTrackerLifter":
        """Load both models now (otherwise they are loaded on first use)."""
        self._get_moge()
        self._get_tracker()
        return self

    def _get_tracker(self):
        if self._tracker is None:
            torch = _torch()
            local = os.path.isdir(os.path.expanduser(self.hub_repo))
            source = self.hub_source or ("local" if local else "github")
            repo = os.path.expanduser(self.hub_repo) if source == "local" else self.hub_repo
            kw = dict(self.hub_kwargs)
            if source == "github":
                kw.setdefault("trust_repo", True)
                if self.hub_repo == DEFAULT_HUB_REPO:
                    kw.setdefault("skip_validation", True)
            try:
                self._tracker = torch.hub.load(repo, self.hub_model, source=source, pretrained=True,
                                               device=self.device, **kw)
            except ImportError as e:
                raise ImportError(f"loading MVTracker from {repo!r} failed ({e}); install the packages its hub entry "
                                  "imports at the pinned versions of the installation recipe in the docstring of "
                                  "vpm_worldbench/lifters/mvtracker.py") from e
            self._tracker.eval()
        return self._tracker

    def _get_moge(self):
        if self._moge is None:
            _torch()
            try:
                from moge.model.v2 import MoGeModel
            except ImportError as e:
                raise ImportError("MVTrackerLifter needs MoGe-2 (the 'moge' package, commit 74fbce0): pip install "
                                  "--no-deps 'moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0"
                                  "ad0e83495065719a' plus utils3d_moge, see the installation recipe in the docstring "
                                  "of vpm_worldbench/lifters/mvtracker.py") from e
            path = os.path.expanduser(self.moge_weights)
            src = path if os.path.exists(path) else self.moge_weights
            self._moge = MoGeModel.from_pretrained(src).to(self.device).eval()
        return self._moge

    def estimate_depth(self, frames: np.ndarray, fov_x: np.ndarray) -> np.ndarray:
        """MoGe-2 metric depth. frames (n, H, W, 3) uint8 RGB of one view, fov_x (n,) or scalar horizontal FoV in
        degrees -> (n, H, W) float32 planar depth in metres, NaN outside MoGe's mask."""
        torch = _torch()
        model = self._get_moge()
        n = len(frames)
        fov = np.array(np.broadcast_to(np.asarray(fov_x, np.float64).reshape(-1), (n,)))
        constant = bool(np.all(fov == fov[0]))
        fp16 = self.moge_fp16 and self.device.startswith("cuda")
        out = []
        with torch.no_grad():
            for i in range(0, n, self.moge_batch):
                im = torch.from_numpy(np.ascontiguousarray(frames[i:i + self.moge_batch])).to(self.device)
                im = im.permute(0, 3, 1, 2).float() / 255.0
                fx = float(fov[0]) if constant else torch.as_tensor(fov[i:i + self.moge_batch], dtype=torch.float32,
                                                                    device=self.device)
                r = model.infer(im, fov_x=fx, resolution_level=self.moge_resolution_level, use_fp16=fp16)
                d = r["depth"].float()
                if "mask" in r:
                    d = torch.where(r["mask"], d, torch.nan)
                out.append(d.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def run_tracker(self, frames: np.ndarray, depths: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                    queries: np.ndarray) -> tuple:
        """One MVTracker call in the normalised scene. frames (V, n, H, W, 3) uint8, depths (V, n, H, W) float32
        (0 = invalid), K (V, n, 3, 3), w2c (V, n, 4, 4) or (V, n, 3, 4), queries (N, 4) [t, x, y, z] ->
        (xyz (n, N, 3) float32 normalised coordinates, vis (n, N) bool)."""
        torch = _torch()
        tracker = self._get_tracker()
        dev = self.device

        def tensor(x):
            return torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(dev)

        with torch.no_grad():
            rgbs = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 1, 4, 2, 3)[None].float().to(dev)
            if self.rgb_scale_quirk:
                rgbs = rgbs / 255.0
            res = tracker(rgbs=rgbs, depths=tensor(np.asarray(depths)[:, :, None]), query_points_3d=tensor(queries),
                          intrs=tensor(K), extrs=tensor(np.asarray(w2c)[..., :3, :]))
            tr, vs = res["traj_e"], res["vis_e"]
            tr = (tr[0] if tr.dim() == 4 else tr).float().cpu().numpy()
            vs = (vs[0] if vs.dim() == 3 else vs).cpu().numpy()
        del rgbs, res
        return tr, vs.astype(bool)
