"""Cross-view MEt3R: the distance of a clip's cross-view MEt3R to the real video's.

MEt3R (Asim et al., "MEt3R: Measuring Multi-View Consistency in Generated Images", CVPR 2025) scores an ordered
image pair (I1, I2): MASt3R predicts the point maps of both images in I1's camera, FeatUp-upsampled DINO ViT-S/16
features of both images are attached to the points and rendered into I1's camera (PyTorch3D point rasterizer,
radius 0.01, 10 points per pixel, alpha compositing, focal from the median of per-pixel votes, principal point at the
image centre), and the score is the masked mean of the cosine dissimilarity ``1 - cos(f1, f2)`` over the pixels that
both renderings cover. Range [0, 2]; lower = more consistent. Reference values at 256x256: an image with itself
≈ 0.003, two consecutive real video frames ≈ 0.01, real multi-view pairs 0.4-0.6, independent noise ≈ 0.43. A pair
whose renderings do not overlap at all scores exactly 0 (the official eps in the denominator): read the score
together with its overlap.

VP-WorldBench uses the OFFICIAL implementation with its README defaults (MASt3R backbone, ``dino16`` + FeatUp JBU
upsampler, cosine distance, 256x256 inputs, naive point rasterizer ``bin_size=0``) and symmetrises every pair:
``MEt3R(I1, I2) = (S(I1, I2) + S(I2, I1)) / 2``. This module only adds bookkeeping around it:

* `MEt3R` (the scorer) computes the FeatUp features of every image ONCE and reuses them in all pairs the image is
  part of (a view at one timestep appears in 2(V-1) cross-view pairs). The pair score is the unmodified official
  ``met3r.MEt3R.forward`` with its feature computation swapped for a lookup of those per-image features, which are
  computed with the same ops (``_interpolate(_get_features((x + 1) / 2), (x + 1) / 2)``).
* `prepare` maps RGB uint8 frames to the official input format: square images (centre crop by default, or a squash),
  ``cv2.INTER_AREA`` to 256x256, float ``x / 127.5 - 1`` in [-1, 1].
* `cross_view` scores every view pair (a < b) at the same timestep for a list of window positions.
* `delta_table` turns per-clip cross-view MEt3R of generated and real videos into the benchmark table.

Benchmark protocol (generation task): timesteps ``t_ref + 4k`` (k >= 1, `Protocol.met3r_positions()`), all view
pairs, per clip ``m(c)`` = mean over pairs and timesteps, the score ``mean_c |m_gen(c) - m_real(c)|`` (lower is better),
plus the signed Δ, the share of clips below the real video and the raw MEt3R. Raw MEt3R is not a quality ordering:
views of a generator that drift apart overlap less, and low overlap is cheap, so a generator can score below the real
video; hence the paired comparison with the real video of the same clip.

    scorer = MEt3R("cuda")
    pos = protocol.met3r_positions()                      # window positions of t_ref + 4, t_ref + 8, ...
    real = cross_view(scorer, ref_frames, pos)            # (V, n, H, W, 3) uint8 on the window grid
    gen = cross_view(scorer, pred_frames, pos)
    per_clip[sample.uid] = {"real": real["cross"], "my_model": gen["cross"]}
    ...
    table = delta_table(per_clip, ["my_model"])           # table["my_model"]["abs_delta"], table["_floor"]["mean"]

Numerics and process-wide torch state
    Loading the official stack changes process-wide torch state: CroCo (inside MASt3R) sets
    ``torch.backends.cuda.matmul.allow_tf32 = True`` when it is imported, and FeatUp calls ``torch.manual_seed(42)``
    when its adaptive-conv extension is imported. `MEt3R` keeps both out of the caller's process: the constructor
    restores the caller's TF32 settings and RNG states (CPU, and every CUDA device when CUDA is in use). Every scoring
    call (`MEt3R.features`, `MEt3R.forward_pairs`, `MEt3R.score_pairs` and so `cross_view`) runs with fixed numerics
    whatever the caller has set - TF32 for float32 matmuls and cuDNN convolutions (``tf32=True``, what the official
    stack runs with and what the reference numbers used; ``tf32=False`` = full float32, which moves single pair
    values by up to ~2e-3), autocast off, cuDNN autotuning off, no autograd - and restores the caller's settings on
    exit (both the legacy flags and, on torch >= 2.9, the per-backend ``fp32_precision`` settings). These are
    process-wide flags, so CUDA work in other threads during a call sees them too. Calling ``scorer.model`` directly
    bypasses all of this. Left as the official package does it: importing it puts its MASt3R / DUSt3R checkouts on
    ``sys.path``, and CroCo's modules become the top-level package ``models``.

Reproducibility
    MEt3R magnifies tiny numeric differences (MASt3R's point maps feed a hard-edged rasterizer and a median focal), so
    single pair values are only defined up to ~1e-3:
    * batch composition: MASt3R runs on ``batch`` ordered pairs at a time and the TF32 GEMM rounding depends on the
      batch; this alone moves single pair values by up to ~1e-3 and clip means by up to ~1e-4;
    * feature cache: the FeatUp features carry TF32 rounding that depends on the batch they are computed in, so
      ``feature_chunk`` moves single pair values by up to ~5e-5; with the same MASt3R batches the cached features and
      the official in-forward ones give scores within ~5e-5 (bit-identical when computed in the same batch);
    * rasterizer: naive and binned agree within 2e-6 on the same batches (but the binned one can overflow its
      per-bin point budget on dense point maps and drop points, hence the naive default);
    * input pixels: MEt3R is far more sensitive to its INPUT than to the numerics above. `vpm_worldbench.io.video`
      converts YUV to RGB with its own fixed formula, which differs from OpenCV's by 1-2 levels on ~30% of the
      values; that alone moved the clip mean of a Simul-Robot clip by up to 6e-4, but by up to ~0.04 on a Real-Robot
      generated clip (single low-overlap pairs by up to ~0.45), and a near-lossless re-encode (H.264 CRF 4) moved
      clip means by up to ~0.04 as well. The floors of the published vpm-v1 tables (0.0038-0.0296; noise floors
      0.0008-0.0111) measure the real video's temporal fluctuation on fixed frames, not input noise: such a shift is
      ~4-10x the floor of the main Simul-Robot and Real-* tables, ~3x on the Simul-Robot hard split and only ~1.4x
      on Simul-General. Decode real and generated videos the same way (this module's callers do) and score lossless
      or near-lossless predictions.
    Given the same RGB frames as the pre-release run (decoded with OpenCV), the defaults reproduce its stored values
    within ~1.2e-3 per pair and ~1e-4 per clip mean. The pre-release batching brings this to ~2.5e-4 per pair:
    ``MEt3R(..., feature_chunk=32)`` and ``cross_view(scorer, window, range(0, 65, 4), temporal=True, batch=64)``
    on the 65-frame window (cross and temporal pairs of t_ref, t_ref + 4, ..., t_ref + 64 in one call; then drop
    row 0, the conditioning frame).

Installation (``pip install "vpm-worldbench[met3r]"`` installs only torch; the official stack is installed by hand).
The reference numbers were computed with these commits, torch 2.10 / CUDA 13, timm 0.4.12, on NVIDIA GH200:

    # 1. PyTorch3D 0.7.9 @ 88e182f, compiled against your torch / CUDA (needs nvcc)
    pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@88e182f989c80836f4bd744e0d9cb1852762ce01"
    # 2. FeatUp @ 6b5a6c0 (JBU upsampler + its adaptive-conv CUDA extension), from a checkout so that its
    #    hubconf.py is used locally (see `featup_repo`)
    git clone https://github.com/mhamilton723/FeatUp && git -C FeatUp checkout 6b5a6c0
    pip install --no-build-isolation -e ./FeatUp
    # 3. MEt3R @ ee0e175 with MASt3R @ f5209af (met3r itself pins e06b009; MASt3R pins DUSt3R 3cc8c88 / CroCo d7de070)
    git clone https://github.com/mohammadasim98/met3r && git -C met3r checkout ee0e175
    git -C met3r submodule update --init --recursive
    git -C met3r/mast3r checkout f5209af && git -C met3r/mast3r submodule update --init --recursive
    (cd met3r/mast3r/dust3r/croco/models/curope && python setup.py build_ext --inplace)    # CroCo's RoPE kernel
    pip install "timm==0.4.12" roma einops jaxtyping torchmetrics lpips iopath "huggingface-hub>=0.22"
    export PYTHONPATH="$PWD/met3r:$PYTHONPATH"     # the checkout itself (met3r finds MASt3R next to its own files)
    # check: git -C met3r rev-parse --short HEAD -> ee0e175; git -C met3r/mast3r rev-parse --short HEAD -> f5209af

Weights are downloaded on first use and cached:
    MASt3R  ``naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric``  (Hugging Face hub; ``backbone_weights=`` for
            a local copy)
    FeatUp  ``dino16_jbu_stack_cocostuff.ckpt`` through torch.hub ``mhamilton723/FeatUp`` (``featup_repo=`` for a
            local FeatUp checkout; by default the checkout the installed ``featup`` package lives in, if any)
    DINO    ``dino_deitsmall16_pretrain.pth`` through torch.hub ``facebookresearch/dino:main``
Offline: populate the Hugging Face and torch.hub caches once, then set ``HF_HUB_OFFLINE=1``.

GPU memory (float32): model 2.7 GiB; feature cache 96 MiB per scored image (V x positions); FeatUp ~0.85 GiB per image
of a ``feature_chunk``; the MASt3R + rendering forward ~1.3 GiB per ordered pair of a ``batch``. Peak ≈ model + cache
+ max(0.85 x feature_chunk, 1.3 x batch) GiB. Measured for 64 images (4 views x 16 positions, cache 6 GiB, GH200):
29.6 GiB with the defaults (feature_chunk=8, batch=16); 19.2 GiB with batch=8 (24 GB GPUs, +5% time); 12.1 GiB with
batch=2, feature_chunk=4 (16 GB GPUs, +50% time).
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..geometry import image_map, warp_image

__all__ = ["MEt3R", "prepare", "cross_view", "delta_table", "clip_mean", "ASPECTS", "RASTERS"]

ASPECTS = ("center_crop", "resize")
RASTERS = {"naive": 0, "binned": None}
FEATUP_HUB = "mhamilton723/FeatUp"
RANGE_TOL = 1e-3
_INSTALL = ("install the official MEt3R stack: pip install 'vpm-worldbench[met3r]', then pytorch3d, FeatUp and met3r "
            "(with its MASt3R submodule) as described in the docstring of vpm_worldbench.metrics.met3r")


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"MEt3R needs PyTorch: {_INSTALL}") from e
    return torch


def _official():
    """The official `met3r.met3r` module (imported lazily; it pulls in pytorch3d, FeatUp, MASt3R/DUSt3R)."""
    _torch()
    try:
        import met3r.met3r as mod
    except ImportError as e:
        raise ImportError(f"the official MEt3R package could not be imported ({type(e).__name__}: {e}); "
                          f"{_INSTALL}") from e
    return mod


def _featup_source(featup_repo) -> tuple:
    """-> (repo_or_dir, torch.hub source) for the FeatUp hub entry point.

    'auto' / None: the FeatUp checkout the installed `featup` package lives in, if it has a hubconf.py (editable
    install), else torch.hub GitHub 'mhamilton723/FeatUp' (the official behaviour). A directory: that local checkout.
    Anything else: a torch.hub GitHub spec 'owner/repo[:ref]'."""
    if featup_repo in (None, "auto"):
        try:
            import featup
            root = Path(featup.__file__).resolve().parent.parent
            if (root / "hubconf.py").is_file():
                return str(root), "local"
        except (ImportError, TypeError):
            pass
        return FEATUP_HUB, "github"
    p = Path(str(featup_repo)).expanduser()
    if p.is_dir():
        if not (p / "hubconf.py").is_file():
            raise FileNotFoundError(f"featup_repo={p}: no hubconf.py, not a FeatUp checkout")
        return str(p.resolve()), "local"
    return str(featup_repo), "github"


@contextlib.contextmanager
def _official_loaders(mod, featup_src: tuple, backbone: str, backbone_weights):
    """While the official MEt3R is constructed: route torch.hub 'mhamilton723/FeatUp' to `featup_src` and the
    MASt3R/DUSt3R weights to `backbone_weights` (both restored afterwards)."""
    torch = _torch()
    hub_load = torch.hub.load
    repo, source = featup_src

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == FEATUP_HUB and (repo, source) != (FEATUP_HUB, "github"):
            kwargs["source"] = source
            return hub_load(repo, model, *args, **kwargs)
        return hub_load(repo_or_dir, model, *args, **kwargs)

    table = getattr(mod, "backbone_to_weights", None)
    if backbone_weights is not None and not isinstance(table, dict):
        raise RuntimeError("this met3r version has no `backbone_to_weights` table; backbone_weights cannot be set")
    saved = dict(table) if isinstance(table, dict) else None
    torch.hub.load = load
    if backbone_weights is not None:
        table[backbone] = str(backbone_weights)
    try:
        yield
    finally:
        torch.hub.load = hub_load
        if saved is not None:
            table.clear()
            table.update(saved)


_PRECISION_KNOBS = (("cuda", "matmul"), ("mkldnn", "matmul"), ("cudnn", "conv"), ("cudnn", "rnn"))


def _backend_state() -> dict:
    """Snapshot of the process-wide settings that change float32 results: TF32 use of cuBLAS matmuls and cuDNN
    convolutions (legacy flags + torch >= 2.9 per-backend strings; None = not readable / not present) and cuDNN
    autotuning."""
    torch = _torch()
    b = torch.backends
    state = {"benchmark": bool(b.cudnn.benchmark)}
    try:
        state["matmul_precision"] = torch.get_float32_matmul_precision()
    except RuntimeError:
        state["matmul_precision"] = None
    try:
        state["cudnn_allow_tf32"] = bool(b.cudnn.allow_tf32)
    except RuntimeError:
        state["cudnn_allow_tf32"] = None
    try:
        state["fp32_precision"] = tuple(getattr(getattr(b, mod), op).fp32_precision for mod, op in _PRECISION_KNOBS)
    except AttributeError:
        state["fp32_precision"] = None
    return state


def _restore_backend_state(state: dict) -> None:
    torch = _torch()
    b = torch.backends
    if state["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(state["matmul_precision"])
    if state["cudnn_allow_tf32"] is not None:
        b.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    if state["fp32_precision"] is not None:
        for (mod, op), value in zip(_PRECISION_KNOBS, state["fp32_precision"]):
            getattr(getattr(b, mod), op).fp32_precision = value
    b.cudnn.benchmark = state["benchmark"]


def _set_scoring_backend(tf32: bool) -> None:
    """TF32 on/off for float32 matmuls and cuDNN convolutions (legacy API, so both APIs agree), autotuning off."""
    torch = _torch()
    b = torch.backends
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    b.cudnn.allow_tf32 = bool(tf32)
    if not tf32 and _backend_state()["fp32_precision"] is not None:
        b.cudnn.conv.fp32_precision = "ieee"
        b.cudnn.rnn.fp32_precision = "ieee"
    b.cudnn.benchmark = False


def _models_conflict() -> str:
    """A hint when a top-level package `models` other than CroCo's is already imported (it shadows CroCo's)."""
    import sys
    other = sys.modules.get("models")
    if other is None:
        return ""
    where = str(getattr(other, "__file__", None) or list(getattr(other, "__path__", []) or []))
    if "croco" in where:
        return ""
    return (f" CroCo (inside MASt3R / DUSt3R) imports its own top-level package `models`, but another `models` "
            f"({where}) was imported first.")


def _rng_devices(torch, device) -> list:
    """CUDA devices whose RNG state the constructor preserves (FeatUp's import re-seeds every device)."""
    if not torch.cuda.is_available() or (device.type != "cuda" and not torch.cuda.is_initialized()):
        return []
    return list(range(torch.cuda.device_count()))


def _float_tensor(images, what: str = "images"):
    """images (tensor or array) -> torch tensor; it must be floating point (values in [-1, 1])."""
    torch = _torch()
    x = images if torch.is_tensor(images) else torch.from_numpy(np.asarray(images))
    if not torch.is_floating_point(x):
        raise TypeError(f"{what} must be floating point with values in [-1, 1] (x / 127.5 - 1; use prepare() for "
                        f"uint8 frames), got {x.dtype}")
    return x


def _check_range(x, what: str = "images") -> None:
    """Raise unless every value of the float tensor x is finite and within [-1, 1] (+- RANGE_TOL)."""
    torch = _torch()
    if x.numel() == 0:
        return
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{what} contain NaN or inf")
    lo, hi = float(x.min()), float(x.max())
    if lo < -1 - RANGE_TOL or hi > 1 + RANGE_TOL:
        raise ValueError(f"{what} must be in [-1, 1] (x / 127.5 - 1, see prepare), got values in [{lo:.4g}, {hi:.4g}]")


class MEt3R:
    """The official MEt3R (MASt3R + FeatUp dino16, cosine) with per-image cached features and symmetric pairs.

    Args:
        device: torch device for all models.
        img_size: input / rasterization size S (official default 256; images must be S x S).
        raster: 'naive' (official, ``bin_size=0``) or 'binned' (``bin_size=None``, coarse-to-fine; faster, but
            pytorch3d may overflow its per-bin point budget on dense MASt3R point maps and drop points).
        tf32: run float32 CUDA matmuls / convolutions in TF32 while scoring (default; the official stack's effective
            setting and the reference numbers) or in full float32 (False). See "Numerics" in the module docstring.
        backbone: 'mast3r' (benchmark) or 'dust3r'.
        featup_repo: 'auto' (default), a local FeatUp checkout, or a torch.hub GitHub spec (see module docstring).
        backbone_weights: None (official Hugging Face id) or a local directory / id for ``from_pretrained``.
        feature_chunk: images per FeatUp forward when filling the feature cache (~0.85 GiB of GPU memory per image;
            moves single pair values by up to ~5e-5 through TF32 rounding; the pre-release run used 32).

    The constructor leaves the caller's TF32 settings and RNG states as they were (see module docstring).
    Attributes: ``model`` (the official ``met3r.MEt3R`` module), ``img_size``, ``device``, ``raster``, ``tf32``,
    ``feature_chunk`` (both may be changed between calls).
    """

    def __init__(self, device="cuda", img_size: int | None = 256, raster: str = "naive", *, tf32: bool = True,
                 backbone: str = "mast3r", featup_repo="auto", backbone_weights=None, feature_chunk: int = 8):
        if raster not in RASTERS:
            raise ValueError(f"raster must be one of {tuple(RASTERS)}, got {raster!r}")
        if backbone not in ("mast3r", "dust3r"):
            raise ValueError("backbone must be 'mast3r' or 'dust3r' (point-map backbones)")
        if img_size is None and raster != "naive":
            raise ValueError("img_size=None rebuilds the (naive) rasterizer per call; raster must be 'naive'")
        if int(feature_chunk) < 1:
            raise ValueError(f"feature_chunk must be >= 1, got {feature_chunk}")
        torch = _torch()
        self.device = torch.device(device)
        self.img_size = None if img_size is None else int(img_size)
        self.raster = raster
        self.tf32 = bool(tf32)
        self.backbone = backbone
        self.feature_chunk = int(feature_chunk)
        saved = _backend_state()
        try:
            with torch.random.fork_rng(devices=_rng_devices(torch, self.device)):
                self.featup_source = _featup_source(featup_repo)
                mod = _official()
                try:
                    with _official_loaders(mod, self.featup_source, backbone, backbone_weights):
                        model = mod.MEt3R(img_size=self.img_size, use_norm=True, backbone=backbone,
                                          feature_backbone="dino16", feature_backbone_weights=FEATUP_HUB,
                                          upsampler="featup", distance="cosine", freeze=True)
                except ImportError as e:
                    raise ImportError(f"building the official MEt3R failed ({type(e).__name__}: {e});"
                                      f"{_models_conflict()} {_INSTALL}") from e
                self.model = model.to(self.device).eval()
        finally:
            _restore_backend_state(saved)
        if self.img_size is not None:
            rs = self.model.rasterizer.raster_settings
            if rs.bin_size != RASTERS[raster]:
                self.model.set_rasterizer(image_size=rs.image_size, points_per_pixel=rs.points_per_pixel,
                                          bin_size=RASTERS[raster])

    def __repr__(self):
        return (f"MEt3R(device={self.device}, img_size={self.img_size}, raster={self.raster!r}, tf32={self.tf32}, "
                f"backbone={self.backbone!r})")

    @contextlib.contextmanager
    def _scoring(self):
        """Fixed numerics for a scoring call (TF32 per ``self.tf32``, cuDNN autotuning off, autocast off, no
        autograd); the caller's process-wide settings are restored on exit."""
        torch = _torch()
        saved = _backend_state()
        no_autocast = (torch.autocast(self.device.type, enabled=False) if self.device.type in ("cuda", "cpu")
                       else contextlib.nullcontext())
        try:
            _set_scoring_backend(self.tf32)
            with torch.no_grad(), no_autocast:
                yield
        finally:
            _restore_backend_state(saved)

    def features(self, images):
        """FeatUp-upsampled DINO features, the same ops as ``met3r.MEt3R.forward``.

        images: float tensor (N, 3, S, S) in [-1, 1] (any device), N >= 1.
        Returns float32 (N, 384, S, S) on ``device`` (96 MiB per image at S = 256)."""
        x = _float_tensor(images)
        if x.ndim != 4 or x.shape[1] != 3 or len(x) == 0:
            raise ValueError(f"images must be (N >= 1, 3, S, S), got {tuple(x.shape)}")
        _check_range(x)
        with self._scoring():
            return self._features(x)

    def _features(self, x):
        """`features` without checks; inside `_scoring`. Fills a preallocated cache chunk by chunk."""
        torch = _torch()
        m, n, out = self.model, len(x), None
        for i in range(0, n, self.feature_chunk):
            im = (x[i:i + self.feature_chunk].to(self.device, torch.float32) + 1) / 2
            f = m._interpolate(m._get_features(im), im)
            if out is None:
                out = f.new_empty((n, *f.shape[1:]))
            out[i:i + len(im)] = f
            del f, im
        return out

    def forward_pairs(self, pairs, feats=None):
        """Ordered pair scores S(I1, I2) with the official forward.

        pairs: float tensor (b, 2, 3, S, S) in [-1, 1]; feats: (b, 2, C, S, S) features of the same images from
        `features` (None = the unmodified official forward, which computes them itself).
        Returns (score (b,) float32, overlap mask (b, S, S) float32 in {0, 1}) on ``device``."""
        torch = _torch()
        p = _float_tensor(pairs, "pairs")
        if p.ndim != 5 or p.shape[1] != 2 or p.shape[2] != 3:
            raise ValueError(f"pairs must be (b, 2, 3, S, S), got {tuple(p.shape)}")
        if feats is not None and (feats.ndim != 5 or tuple(feats.shape[:2]) != tuple(p.shape[:2])
                                  or tuple(feats.shape[-2:]) != tuple(p.shape[-2:])):
            raise ValueError(f"feats {tuple(feats.shape)} do not match pairs {tuple(p.shape)}")
        _check_range(p, "pairs")
        with self._scoring():
            return self._forward_pairs(p.to(self.device, torch.float32), feats)

    def _forward_pairs(self, pairs, feats):
        """`forward_pairs` without checks; inside `_scoring`, pairs already float32 on ``device``."""
        m = self.model
        if feats is None:
            score, mask = m(images=pairs, return_overlap_mask=True)
            return score, mask
        hr = feats.to(self.device).flatten(0, 1)
        calls = []

        def get_features(images):
            calls.append("features")
            return None

        def interpolate(lr_feat, images):
            calls.append("interpolate")
            if tuple(images.shape[:1]) + tuple(images.shape[-2:]) != tuple(hr.shape[:1]) + tuple(hr.shape[-2:]):
                raise RuntimeError(f"cached features {tuple(hr.shape)} do not match the images "
                                   f"{tuple(images.shape)} seen by met3r.MEt3R.forward")
            return hr

        m._get_features, m._interpolate = get_features, interpolate
        try:
            score, mask = m(images=pairs, return_overlap_mask=True)
        finally:
            del m._get_features, m._interpolate
        if calls != ["features", "interpolate"]:
            raise RuntimeError(f"unexpected feature calls {calls} in met3r.MEt3R.forward; this met3r version is "
                               "not supported by the feature cache")
        return score, mask

    def score_pairs(self, images, pairs, batch: int = 16):
        """Symmetric MEt3R of image pairs with per-image cached features.

        images: float tensor or array (V, T, 3, S, S) with values in [-1, 1] (e.g. from `prepare`), CPU or GPU.
        pairs: sequence of ((v, t), (v2, t2)) index pairs into images (non-negative).
        batch: ordered pairs per official forward (both orders of a pair are scored, so 2 * len(pairs) in total;
            ~1.3 GiB of GPU memory per ordered pair; see "Reproducibility" in the module docstring).
        Returns (scores (P,), overlap (P,)) float64 numpy: MEt3R = (S(I1,I2) + S(I2,I1)) / 2 and the mean fraction of
        pixels covered by both renderings, averaged over the two orders the same way (overlap 0 -> score 0)."""
        torch = _torch()
        x = _float_tensor(images)
        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError(f"images must be (V, T, 3, S, S), got {tuple(x.shape)}")
        V, T, _, H, W = x.shape
        if H != W or (self.img_size is not None and H != self.img_size):
            raise ValueError(f"images must be {self.img_size} x {self.img_size} squares, got {H} x {W}")
        if int(batch) < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        batch = int(batch)
        pairs = [((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))) for a, b in pairs]
        n = len(pairs)
        if n == 0:
            return np.zeros(0), np.zeros(0)
        for a, b in pairs:
            for v, t in (a, b):
                if not (0 <= v < V and 0 <= t < T):
                    raise IndexError(f"pair index {(v, t)} outside images (V={V}, T={T})")
        flat = [a[0] * T + a[1] for a, _ in pairs], [b[0] * T + b[1] for _, b in pairs]
        used = sorted(set(flat[0]) | set(flat[1]))
        slot = {k: i for i, k in enumerate(used)}
        seq = [(slot[i], slot[j]) for i, j in zip(*flat)] + [(slot[j], slot[i]) for i, j in zip(*flat)]
        sim, ovl = [], []
        with self._scoring():
            xs = x.reshape(V * T, 3, H, W)[used].to(self.device, torch.float32)
            _check_range(xs)
            feat = self._features(xs)
            for i in range(0, len(seq), batch):
                ii = torch.tensor(seq[i:i + batch], device=self.device)
                s, mask = self._forward_pairs(xs[ii], feat[ii])
                sim.append(s.float().cpu())
                ovl.append(mask.float().mean((-1, -2)).cpu())
                del s, mask
            del feat, xs
        sim, ovl = torch.cat(sim).numpy(), torch.cat(ovl).numpy()
        return ((sim[:n] + sim[n:]) / 2).astype(np.float64), ((ovl[:n] + ovl[n:]) / 2).astype(np.float64)

    __call__ = score_pairs


def prepare(frames, size: int = 256, aspect: str = "center_crop"):
    """RGB uint8 frames -> MEt3R input tensor.

    frames: uint8 array (..., H, W, 3), e.g. (V, T, H, W, 3).
    aspect: 'center_crop' (default; the largest centred square, side min(H, W)) or 'resize' (squash to a square).
    Returns a float32 torch tensor (..., 3, size, size) with values x / 127.5 - 1 in [-1, 1]. Resampling is
    cv2.INTER_AREA (also when enlarging), exactly as the reference implementation."""
    torch = _torch()
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {ASPECTS}, got {aspect!r}")
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        raise TypeError(f"prepare expects RGB uint8 frames, got {a.dtype}")
    if a.ndim < 3 or a.shape[-1] != 3:
        raise ValueError(f"frames must be (..., H, W, 3), got {a.shape}")
    size = int(size)
    lead, (H, W) = a.shape[:-3], a.shape[-3:-1]
    flat = a.reshape((-1, H, W, 3))
    if (H, W) != (size, size) and len(flat):
        m = image_map((W, H), (size, size), aspect)
        flat = np.stack([warp_image(f, m, interp="area") for f in flat])
    elif (H, W) != (size, size):
        flat = np.zeros((0, size, size, 3), np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(flat)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return t.contiguous().reshape(*lead, 3, size, size)


def cross_view(scorer, frames, positions: Sequence[int], *, aspect: str = "center_crop", size: int | None = None,
               batch: int = 16, temporal: bool = False) -> dict:
    """Cross-view MEt3R of one clip: every view pair (a < b) at the same timestep, for each window position.

    scorer: a `MEt3R` (or any object with ``score_pairs(images, pairs, batch)`` and ``img_size``).
    frames: RGB uint8 (V, n, H, W, 3) on the window grid (position 0 = t_ref), or a list of V arrays (n, H_v, W_v, 3),
        or an already prepared float tensor (V, n, 3, S, S) in [-1, 1].
    positions: non-negative indices into the window (axis 1), e.g. `Protocol.met3r_positions()` = [4, 8, ..., 64].
    aspect / size: see `prepare` (size defaults to ``scorer.img_size``, else 256). For a prepared tensor nothing is
        resampled: size must be None or S, and the result records ``aspect: None`` (the preprocessing is unknown).
    batch: ordered pairs per MASt3R forward (see `MEt3R.score_pairs`).
    temporal: also score consecutive positions inside each view (temporal MEt3R), in the same call.
    Returns {"pairs": [[a, b], ...] (P), "positions": [...] (T'), "cross": [[P floats] per position],
    "overlap": [[P floats] per position], "mean": m(c) = mean of cross (None if V < 2 or no positions),
    "img_size": S, "aspect": aspect or None} plus, with temporal, "temporal" / "temporal_overlap": [[T'-1 floats] per
    view]. Overlap = fraction of pixels covered by both renderings; a pair with overlap 0 scores 0."""
    torch = _torch()
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {ASPECTS}, got {aspect!r}")
    pos = [int(p) for p in positions]
    if any(p < 0 for p in pos):
        raise ValueError(f"positions must be non-negative window indices, got {pos}")
    if torch.is_tensor(frames):
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"a prepared tensor must be (V, n, 3, S, S), got {tuple(frames.shape)}")
        S = int(frames.shape[-1])
        if size is not None and int(size) != S:
            raise ValueError(f"size={size} but the prepared tensor is {S} x {S} (it is not resampled)")
        x, size, used_aspect = frames[:, pos], S, None
    else:
        size = int(size) if size is not None else int(getattr(scorer, "img_size", None) or 256)
        if isinstance(frames, (list, tuple)):
            x = torch.stack([prepare(np.asarray(f)[pos], size, aspect) for f in frames])
        else:
            x = prepare(np.asarray(frames)[:, pos], size, aspect)
        used_aspect = aspect
    V, n = int(x.shape[0]), len(pos)
    vpairs = [(a, b) for a in range(V) for b in range(a + 1, V)]
    cross_pairs = [((a, k), (b, k)) for k in range(n) for a, b in vpairs]
    temp_pairs = [((v, k), (v, k + 1)) for v in range(V) for k in range(n - 1)] if temporal else []
    sc, oc = scorer.score_pairs(x, cross_pairs + temp_pairs, batch=batch)
    sc, oc = np.asarray(sc, np.float64), np.asarray(oc, np.float64)
    nc, P = len(cross_pairs), len(vpairs)
    out = {"pairs": [[a, b] for a, b in vpairs], "positions": pos,
           "cross": sc[:nc].reshape(n, P).tolist(), "overlap": oc[:nc].reshape(n, P).tolist(),
           "mean": float(sc[:nc].mean()) if nc else None, "img_size": int(size), "aspect": used_aspect}
    if temporal:
        out["temporal"] = sc[nc:].reshape(V, max(n - 1, 0)).tolist()
        out["temporal_overlap"] = oc[nc:].reshape(V, max(n - 1, 0)).tolist()
    return out


def _se(x) -> float:
    """Standard error of the mean, std(ddof=1) / sqrt(n); 0.0 when n < 2."""
    x = np.asarray(x, np.float64)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def clip_mean(cross) -> float:
    """Clip-level MEt3R m(c): mean of a cross array (T', P) over all view pairs and timesteps (non-empty)."""
    a = np.asarray(cross, np.float64)
    if a.size == 0:
        raise ValueError("clip_mean of an empty cross array (no view pairs or no timesteps)")
    return float(a.mean())


def delta_table(per_clip: Mapping[str, Mapping], sources: Sequence[str], reference: str = "real") -> dict:
    """The cross-view MEt3R table relative to the real video.

    per_clip: {clip_id: {"real": cross (T', P), "<source>": cross (T', P) or None}}; cross[k][p] = MEt3R of view pair
        p at scored timestep k (the conditioning frame excluded), e.g. `cross_view(...)["cross"]`.
    sources: generated sources to tabulate (a single name may be given as a str). Only clips where the reference
        and EVERY source exist (not None, all values finite) are used (paired rows); all arrays of a clip must have
        the same shape and all clips the same T' >= 2.
    Per clip the arrays are averaged over pairs and timesteps FIRST (m(c)), then compared: Δ(c) = m_s(c) - m_real(c).

    Returns {reference: {"raw", "raw_se", "n"},
             source: {"abs_delta": mean_c |m_s(c) - m_real(c)| (main number, lower is better), "abs_delta_se", "delta": mean Δ
                      (> 0: less consistent than the real video), "delta_se", "below": share of clips with Δ < 0,
                      "raw": mean m_s, "raw_se", "n"},
             "_floor": {"mean", "se", "n"}, "_noise_floor": {"mean", "se", "n"},
             "_clips": [clip ids used], "_skipped": [clip ids of per_clip not used]}
    ``*_se`` = std(ddof=1) / sqrt(n) over clips (0.0 for n < 2).
    Floor of the score (what a generator exactly as consistent as the real video would still score): x(c, k) = real MEt3R
    averaged over pairs; remove the subset-average time profile x'(c, k) = x(c, k) - mean_c x(c, k); per clip
    |mean of the first half - mean of the second half| / sqrt(2) (the first T'//2 timesteps vs the rest), mean over
    clips. Noise floor (measurement noise alone): the same with interleaved timesteps (even vs odd k) on x without
    profile removal. Both depend on the clip set (recompute them for any subset of clips)."""
    if isinstance(sources, str):
        sources = [sources]
    srcs = [s for s in dict.fromkeys(sources) if s != reference]
    for s in [reference, *srcs]:
        if str(s).startswith("_"):
            raise ValueError(f"source names must not start with '_': {s!r}")
    names = [reference, *srcs]
    clips, arrays = [], {}
    for clip in sorted(per_clip):
        d = per_clip[clip] or {}
        got = {}
        for s in names:
            v = d.get(s)
            if v is None:
                break
            a = np.asarray(v, np.float64)
            if a.ndim != 2 or a.size == 0:
                raise ValueError(f"clip {clip!r}, source {s!r}: cross must be a non-empty (T', P) array, got {a.shape}")
            if not np.isfinite(a).all():
                break
            got[s] = a
        else:
            shapes = sorted({a.shape for a in got.values()})
            if len(shapes) != 1:
                raise ValueError(f"clip {clip!r}: sources disagree on (timesteps, pairs): {shapes}")
            clips.append(clip)
            arrays[clip] = got
    if not clips:
        raise ValueError(f"no clip has {reference!r} and all of {srcs}")
    steps = sorted({arrays[c][reference].shape[0] for c in clips})
    if len(steps) != 1:
        raise ValueError(f"clips differ in the number of scored timesteps {steps}; the floor needs one time grid")
    if steps[0] < 2:
        raise ValueError("the floor needs at least 2 scored timesteps per clip")

    real = np.array([arrays[c][reference].mean() for c in clips])
    x = np.array([arrays[c][reference].mean(1) for c in clips])
    xp = x - x.mean(0, keepdims=True)
    h = x.shape[1] // 2
    floor = np.abs(xp[:, :h].mean(1) - xp[:, h:].mean(1)) / np.sqrt(2)
    noise = np.abs(x[:, 0::2].mean(1) - x[:, 1::2].mean(1)) / np.sqrt(2)
    n = len(clips)
    out = {reference: {"raw": float(real.mean()), "raw_se": _se(real), "n": n}}
    for s in srcs:
        g = np.array([arrays[c][s].mean() for c in clips])
        dl = g - real
        out[s] = {"abs_delta": float(np.abs(dl).mean()), "abs_delta_se": _se(np.abs(dl)),
                  "delta": float(dl.mean()), "delta_se": _se(dl), "below": float((dl < 0).mean()),
                  "raw": float(g.mean()), "raw_se": _se(g), "n": n}
    out["_floor"] = {"mean": float(floor.mean()), "se": _se(floor), "n": n}
    out["_noise_floor"] = {"mean": float(noise.mean()), "se": _se(noise), "n": n}
    out["_clips"] = clips
    out["_skipped"] = sorted(set(per_clip) - set(clips))
    return out
