"""Frame-level appearance metrics of a generated video against the real one: PSNR, SSIM and LPIPS.

All functions compare two videos of the SAME view on the SAME pixel grid, frame by frame:

    real, fake : RGB uint8 arrays (n, H, W, 3) (a single frame (H, W, 3) is accepted and treated as n = 1),
                 e.g. the protocol eval frames t_ref+1 .. t_ref+horizon of one view on the eval grid
                 (`predictions.load_reference` / `predictions.load_prediction`, position 0 = t_ref excluded).

and return one value per frame (float64 arrays of shape (n,)). The benchmark aggregation (docs/METRICS.md) is the
mean over the scored frames of a view (`appearance`), then the mean over views, then the mean over clips.

    p = psnr_frames(real, fake)                  # (n,) dB, higher is better
    s = ssim_frames(real, fake)                  # (n,) in [-1, 1], higher is better      (needs scikit-image)
    lp = LPIPS(device="cuda")                    # AlexNet LPIPS v0.1, built once and reused (needs torch + lpips)
    d = lp(real, fake)                           # (n,) lower is better
    appearance(real, fake, lpips=lp)             # {"psnr", "ssim", "lpips", "psnr_per_frame", ...}

Definitions (those of the pre-release code that produced the published numbers; on identical inputs the per-frame
values agree to < 4e-6 dB PSNR, bit-exactly for SSIM and to < 3e-8 for LPIPS on the same GPU):
  PSNR_i  = 10 * log10(255^2 / max(MSE_i, 1e-8)), MSE_i over all H*W*3 values of frame i (exact, float64; the
            pre-release code accumulated it in float32); the clip value is the MEAN OF THE PER-FRAME PSNRs, not the
            PSNR of the pooled MSE.
  SSIM_i  = skimage.metrics.structural_similarity(real_i, fake_i, channel_axis=2, data_range=255) with all other
            arguments at their defaults (7x7 uniform window, sample covariance, K1=0.01, K2=0.03).
  LPIPS_i = lpips.LPIPS(net="alex", version="0.1") on x / 127.5 - 1, frames processed in chunks of `batch` (16).

Heavy dependencies are imported lazily: SSIM needs scikit-image, LPIPS needs torch + lpips
(`pip install vpm-worldbench[metrics]`).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["psnr_frames", "ssim_frames", "LPIPS", "appearance", "PSNR_MSE_FLOOR"]

PSNR_MSE_FLOOR = 1e-8
_METRICS_EXTRA = "pip install 'vpm-worldbench[metrics]'"


def _as_frames(x, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        raise TypeError(f"{name}: expected RGB uint8 frames, got dtype {x.dtype} (convert with np.uint8 on the 0..255 "
                        "scale; float images in [0, 1] would be scored on the wrong range)")
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"{name}: expected shape (n, H, W, 3), got {x.shape}")
    return x


def _check_pair(real, fake):
    real = _as_frames(real, "real")
    fake = _as_frames(fake, "fake")
    if real.shape != fake.shape:
        raise ValueError(f"real {real.shape} and fake {fake.shape} must have the same shape (same frames, same grid)")
    if len(real) == 0:
        raise ValueError("no frames to compare")
    return real, fake


def psnr_frames(real, fake) -> np.ndarray:
    """Per-frame PSNR in dB.

    real, fake: uint8 (n, H, W, 3) on the same grid. Returns float64 (n,):
    10 * log10(255^2 / max(MSE, 1e-8)) with MSE the mean squared difference over the H*W*3 values of the frame,
    computed exactly in float64 (squared differences of uint8 values are integers, so the sum is exact).
    """
    real, fake = _check_pair(real, fake)
    out = np.empty(len(real), np.float64)
    for i in range(len(real)):
        d = real[i].astype(np.float64) - fake[i]
        mse = float(np.mean(d * d))
        out[i] = 10.0 * np.log10(255.0 ** 2 / max(mse, PSNR_MSE_FLOOR))
    return out


def ssim_frames(real, fake) -> np.ndarray:
    """Per-frame SSIM (scikit-image defaults on RGB, data_range 255).

    real, fake: uint8 (n, H, W, 3), H and W >= 7. Returns float64 (n,):
    `skimage.metrics.structural_similarity(real_i, fake_i, channel_axis=2, data_range=255)` (7x7 uniform window,
    sample covariance, K1 = 0.01, K2 = 0.03; mean of the per-channel SSIM maps).
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as e:
        raise ImportError(f"SSIM needs scikit-image: {_METRICS_EXTRA}") from e
    real, fake = _check_pair(real, fake)
    return np.array([structural_similarity(a, b, channel_axis=2, data_range=255) for a, b in zip(real, fake)],
                    dtype=np.float64)


class LPIPS:
    """LPIPS perceptual distance (Zhang et al. 2018), AlexNet backbone, linear calibration v0.1.

    Wraps `lpips.LPIPS(net=net, version="0.1")`. The linear layers ship with the `lpips` pip package (override with
    `model_path`); the backbone is torchvision's ImageNet AlexNet (downloaded once into the torch hub cache, see
    `TORCH_HOME`). Build it once and reuse it:

        lp = LPIPS(device="cuda")
        d = lp(real, fake)               # (n,) float64, lower = more similar

    Args:
        device: torch device string ("cuda", "cuda:1", "cpu").
        net: backbone, "alex" (benchmark default), "vgg" or "squeeze".
        model_path: optional local path of the linear-layer weights (default: the package's v0.1 weights).
    """

    def __init__(self, device: str = "cuda", net: str = "alex", model_path: Optional[str] = None):
        try:
            import torch
            import lpips as _lpips
        except ImportError as e:
            raise ImportError(f"LPIPS needs torch and lpips: {_METRICS_EXTRA}") from e
        self._torch = torch
        self.device = torch.device(device)
        self.net = net
        self.version = "0.1"
        self.model = _lpips.LPIPS(net=net, version=self.version, model_path=model_path, verbose=False)
        self.model = self.model.to(self.device).eval()

    def __repr__(self):
        return f"LPIPS(net={self.net!r}, version={self.version!r}, device={str(self.device)!r})"

    def __call__(self, real, fake, batch: int = 16) -> np.ndarray:
        """Per-frame LPIPS distance.

        real, fake: uint8 (n, H, W, 3) on the same grid (H, W >= 32 for AlexNet). Frames are scaled to [-1, 1]
        (x / 127.5 - 1) and processed in chunks of `batch` frames. Returns float64 (n,).
        """
        torch = self._torch
        real, fake = _check_pair(real, fake)
        if batch < 1:
            raise ValueError("batch must be >= 1")
        out = []
        with torch.no_grad():
            for i in range(0, len(real), batch):
                a = self._to_tensor(real[i:i + batch])
                b = self._to_tensor(fake[i:i + batch])
                out.append(self.model(a, b).flatten().double().cpu().numpy())
        return np.concatenate(out)

    def _to_tensor(self, frames: np.ndarray):
        t = self._torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        return t.permute(0, 3, 1, 2).float() / 127.5 - 1


def appearance(real, fake, lpips: Optional[Callable] = None, ssim: bool = True) -> dict:
    """PSNR (+ SSIM, + LPIPS) of one view, averaged over its frames.

    Args:
        real, fake: uint8 (n, H, W, 3), the scored frames of one view on the same grid (the protocol excludes the
            conditioning frame t_ref: pass window positions 1..horizon).
        lpips: an `LPIPS` instance (or any callable (real, fake) -> (n,) distances); None skips LPIPS.
        ssim: compute SSIM (requires scikit-image).

    Returns:
        {"psnr": float dB, "ssim": float | None, "lpips": float | None, "n_frames": int,
         "psnr_per_frame": [n floats], "ssim_per_frame": [n] | None, "lpips_per_frame": [n] | None}
        where "psnr"/"ssim"/"lpips" are means over the n frames.
    """
    if lpips is not None and not callable(lpips):
        raise TypeError("lpips must be an LPIPS instance (or a callable (real, fake) -> per-frame distances) or None")
    real, fake = _check_pair(real, fake)
    p = psnr_frames(real, fake)
    out = {"psnr": float(np.mean(p)), "ssim": None, "lpips": None, "n_frames": int(len(p)),
           "psnr_per_frame": p.tolist(), "ssim_per_frame": None, "lpips_per_frame": None}
    if ssim:
        s = ssim_frames(real, fake)
        out["ssim"] = float(np.mean(s))
        out["ssim_per_frame"] = s.tolist()
    if lpips is not None:
        d = np.asarray(lpips(real, fake), dtype=np.float64).reshape(-1)
        if len(d) != len(p):
            raise ValueError(f"lpips returned {len(d)} values for {len(p)} frames")
        out["lpips"] = float(np.mean(d))
        out["lpips_per_frame"] = d.tolist()
    return out
