"""Fréchet Video Distance (FVD) with the Kinetics-400 I3D used by TATS / cd-fvd.

Protocol (docs/METRICS.md). For every view of every sample, the video on the protocol window grid (position 0 = the
conditioning frame t_ref, positions 1..horizon = the scored future frames) is cut into non-overlapping 16-frame
windows starting at position 1 (horizon 64 -> 4 windows), every frame is resized to 224x224 with cv2.INTER_AREA
(non-square grids are squashed), and every window is mapped to the 400-d I3D logits averaged over time. Real and
generated windows are taken at the same samples, views and positions. The Fréchet distance is computed once per
subset over the POOLED window features of all samples and views (it is not a per-sample average):

    FVD = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))

with S the covariance of the features: np.cov ("unbiased", 1/(N-1), the default) or the 1/N moment estimate used by
cd-fvd ("biased"). Always report the number of windows next to the value: with 400-d features FVD is strongly biased
upwards for small N.

    wins = fvd_windows(frames)                  # (4, 16, 224, 224, 3) uint8 from a 65-frame window grid
    i3d = I3D(device="cuda")                    # cd-fvd's InceptionI3d + TATS weights i3d_pretrained_400.pt
    f_real = i3d.features(wins)                 # (4, 400) float64
    ...                                         # collect per sample and view, real and generated
    frechet(F_real, F_gen)                      # float; pooled_fvd(...) also returns the window counts

The I3D is cd-fvd's PyTorch port (`cdfvd.third_party.i3d.pytorch_i3d.InceptionI3d`, 400 classes) with the TATS
checkpoint `i3d_pretrained_400.pt` (md5 c275f5caff95bea0b712515feedad130), input preprocessing identical to
`cdfvd.fvd.get_i3d_logits` (bilinear resize to 224 with align_corners=False, identity for 224 inputs; x*2/255 - 1;
float32; batches of 16 windows). Install with `pip install vpm-worldbench[fvd]`. GPU feature extraction runs with
PyTorch's default precision settings (cuDNN may use TF32 convolutions on Ampere or newer GPUs), as for the published
numbers; the Fréchet distance itself is plain numpy/scipy on the CPU.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["WINDOW", "SIZE", "I3D_URL", "I3D_MD5", "COV_MODES", "window_starts", "fvd_windows", "I3D", "frechet",
           "fvd_from_features", "pooled_fvd"]

WINDOW = 16
SIZE = 224
I3D_URL = "https://github.com/songweige/TATS/raw/main/tats/fvd/i3d_pretrained_400.pt"
I3D_MD5 = "c275f5caff95bea0b712515feedad130"
COV_MODES = ("unbiased", "biased")
_FVD_EXTRA = "pip install 'vpm-worldbench[fvd]'"

FeatureInput = Union[np.ndarray, Sequence[np.ndarray]]


def window_starts(n_frames: int, window: int = WINDOW, n_windows: Optional[int] = None, start: int = 1) -> list:
    """First positions of the FVD windows on a grid of `n_frames` positions (position 0 = conditioning frame).

    The candidate windows are the non-overlapping blocks [start + k*window, start + (k+1)*window) that fit in the
    grid, k = 0 .. n_fit-1 with n_fit = (n_frames - start) // window. n_windows=None takes all of them. A smaller
    n_windows takes blocks spread evenly over the horizon, k_i = floor(i * n_fit / n_windows): on a 64-frame horizon
    n_windows=2 gives positions 1-16 and 33-48 (the pre-release Real-General setting), n_windows=4 gives all four.
    Raises ValueError if n_windows > n_fit or if no window fits.
    """
    n_frames, window, start = int(n_frames), int(window), int(start)
    if window < 1:
        raise ValueError("window must be >= 1")
    if start < 0:
        raise ValueError("start must be >= 0")
    n_fit = max(0, (n_frames - start) // window)
    if n_windows is None:
        blocks = list(range(n_fit))
    else:
        n_windows = int(n_windows)
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1 (or None for all windows)")
        if n_windows > n_fit:
            raise ValueError(f"{n_windows} windows of {window} frames requested from position {start}, "
                             f"but only {n_fit} fit in {n_frames} frames")
        blocks = [i * n_fit // n_windows for i in range(n_windows)]
    if not blocks:
        raise ValueError(f"no complete {window}-frame window from position {start} in {n_frames} frames")
    return [start + k * window for k in blocks]


def fvd_windows(frames, window: int = WINDOW, n_windows: Optional[int] = None, start: int = 1,
                size: int = SIZE) -> np.ndarray:
    """Cut a video on the protocol window grid into FVD windows for the I3D.

    Args:
        frames: RGB uint8 (n, H, W, 3); position 0 is the conditioning frame t_ref, position j is window_indices[j].
        window: frames per window (16).
        n_windows: number of windows (None = every complete non-overlapping window); see `window_starts` for the
            placement when fewer windows than fit are requested.
        start: grid position of the first window (1 = the first generated frame).
        size: output side; every frame is resized to (size, size) with cv2.INTER_AREA (aspect ratio not kept).

    Returns:
        uint8 (k, window, size, size, 3).
    """
    import cv2

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be RGB uint8 (n, H, W, 3), got {frames.dtype} {frames.shape}")
    starts = window_starts(len(frames), window, n_windows, start)
    out = np.empty((len(starts), window, size, size, 3), np.uint8)
    for i, s in enumerate(starts):
        for j in range(window):
            out[i, j] = cv2.resize(np.ascontiguousarray(frames[s + j]), (size, size), interpolation=cv2.INTER_AREA)
    return out


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class I3D:
    """Kinetics-400 I3D feature extractor for FVD (cd-fvd's PyTorch port with the TATS weights).

        i3d = I3D(device="cuda")                    # or I3D(weights="/path/to/i3d_pretrained_400.pt")
        feats = i3d.features(windows)               # windows uint8 (B, 16, 224, 224, 3) -> (B, 400) float64

    Args:
        device: torch device string.
        weights: local path of `i3d_pretrained_400.pt`. None = cd-fvd's packaged location
            (`cdfvd/third_party/i3d/i3d_pretrained_400.pt`). A missing file is downloaded to that path by cd-fvd from
            the TATS release (I3D_URL). A checksum different from I3D_MD5 triggers a warning.
        batch: windows per forward pass (16, as cd-fvd).

    Attributes: `weights` (path used), `weights_md5`, `device`, `model` (the torch module, eval mode), `info`.

    With these settings the features are bit-identical to the pre-release cache (checked on a GH200 with torch 2.10:
    Simul-Robot, Simul-General and Real-Robot windows, real and generated).
    """

    def __init__(self, device: str = "cuda", weights: Optional[str] = None, batch: int = 16):
        try:
            import torch
            from cdfvd.third_party.i3d import utils as cd_i3d
        except ImportError as e:
            raise ImportError(f"FVD needs torch and cd-fvd (I3D model + weights): {_FVD_EXTRA}") from e
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self._torch = torch
        self.device = torch.device(device)
        self.batch = int(batch)
        default = os.path.join(os.path.dirname(os.path.abspath(cd_i3d.__file__)), "i3d_pretrained_400.pt")
        self.weights = os.path.abspath(os.path.expanduser(str(weights))) if weights is not None else default
        if weights is not None:
            os.makedirs(os.path.dirname(self.weights), exist_ok=True)
        self.model = cd_i3d.load_i3d_model(self.device, ckpt_path=None if weights is None else self.weights).eval()
        self.weights_md5 = _file_md5(self.weights)
        if self.weights_md5 != I3D_MD5:
            warnings.warn(f"I3D weights {self.weights} have md5 {self.weights_md5}, expected {I3D_MD5} "
                          "(TATS i3d_pretrained_400.pt); FVD values will not be comparable", RuntimeWarning)

    def __repr__(self):
        return f"I3D(weights={self.weights!r}, device={str(self.device)!r}, batch={self.batch})"

    @property
    def info(self) -> dict:
        """Provenance to store next to FVD values: weights path + md5, device, batch, torch version and whether cuDNN
        may use TF32 convolutions (it changes the logits by ~1e-2; the published features used the default, True)."""
        torch = self._torch
        return {"model": "i3d_kinetics400_tats", "weights": self.weights, "weights_md5": self.weights_md5,
                "device": str(self.device), "batch": self.batch, "torch": torch.__version__,
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}

    def _preprocess(self, videos: np.ndarray):
        """uint8 (b, t, h, w, 3) -> float32 (b, 3, t, 224, 224) in [-1, 1], exactly as cdfvd's preprocess_i3d."""
        torch = self._torch
        import torch.nn.functional as F

        b, t, h, w, c = videos.shape
        x = torch.from_numpy(np.ascontiguousarray(videos)).float().flatten(end_dim=1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        x = x.view(b, t, c, SIZE, SIZE).transpose(1, 2).contiguous()
        return 2. * x / 255. - 1

    def features(self, windows) -> np.ndarray:
        """I3D logits of video windows.

        windows: uint8 (B, T, H, W, 3) (a single window (T, H, W, 3) is accepted); the protocol uses T = 16 and
        224x224 windows from `fvd_windows` (other sizes are resized bilinearly to 224 as in cd-fvd). Returns float64
        (B, 400): the 400 Kinetics logits averaged over time.
        """
        torch = self._torch
        x = np.asarray(windows)
        if x.dtype != np.uint8:
            raise TypeError(f"windows must be uint8 (0..255), got {x.dtype}")
        if x.ndim == 4:
            x = x[None]
        if x.ndim != 5 or x.shape[-1] != 3:
            raise ValueError(f"windows must be (B, T, H, W, 3), got {x.shape}")
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch):
                inp = self._preprocess(x[i:i + self.batch]).to(device=self.device, dtype=torch.float32)
                out.append(self.model(inp).float().cpu().numpy().astype(np.float64))
        return np.concatenate(out) if out else np.zeros((0, 400), np.float64)

    __call__ = features


def _as_features(x: FeatureInput, name: str) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x
    else:
        parts = [np.asarray(p) for p in x]
        if not parts:
            raise ValueError(f"{name}: no features")
        a = np.concatenate([p.reshape(-1, p.shape[-1]) for p in parts])
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name}: features must be (N, d), got {a.shape}")
    if len(a) < 2:
        raise ValueError(f"{name}: need at least 2 feature vectors, got {len(a)}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite features")
    return a


def _mean_cov(a: np.ndarray, cov: str):
    mu = a.mean(0)
    if cov == "unbiased":
        return mu, np.atleast_2d(np.cov(a, rowvar=False))
    if cov == "biased":
        return mu, a.T @ a / len(a) - np.outer(mu, mu)
    raise ValueError(f"cov must be one of {COV_MODES}, got {cov!r}")


def frechet(real: FeatureInput, fake: FeatureInput, cov: str = "unbiased") -> float:
    """Fréchet distance between two Gaussian fits of feature sets (FVD when the features are I3D logits).

    Args:
        real, fake: (N, d) and (M, d) features (any float dtype, computed in float64), or sequences of per-sample
            (k_i, d) arrays that are pooled (concatenated).
        cov: "unbiased" (np.cov, 1/(N-1); VPM default and Simul-* pre-release numbers) or "biased" (E[xx^T] - mu mu^T,
            1/N; cd-fvd's estimate, used for the Real-* pre-release numbers).

    Returns:
        ||mu_r - mu_f||^2 + Tr(S_r + S_f - 2 Re(sqrtm(S_r S_f))) as a Python float.

    Means and covariances are computed in float64. (The pre-release Simul-* tables took the feature means in float32;
    on their cached features this changes the published values by at most 1.2e-4, i.e. < 1e-6 relative.)
    """
    from scipy import linalg

    a = _as_features(real, "real")
    b = _as_features(fake, "fake")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dimensions differ: {a.shape[1]} vs {b.shape[1]}")
    mu_a, s_a = _mean_cov(a, cov)
    mu_b, s_b = _mean_cov(b, cov)
    root = linalg.sqrtm(s_a.dot(s_b))
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * np.real(root)))


def fvd_from_features(real: FeatureInput, fake: FeatureInput, cov: str = "unbiased") -> float:
    """Alias of `frechet` (FVD from precomputed I3D features)."""
    return frechet(real, fake, cov=cov)


def pooled_fvd(real: FeatureInput, fake: FeatureInput, cov: str = "unbiased") -> dict:
    """FVD over pooled windows with the counts that must be reported with it.

    real, fake: (N, d) arrays or iterables of per-sample (k_i, d) arrays (concatenated in order).
    Returns {"fvd": float, "n_real": N, "n_fake": M, "cov": cov}.
    """
    a = _as_features(real if isinstance(real, np.ndarray) else list(real), "real")
    b = _as_features(fake if isinstance(fake, np.ndarray) else list(fake), "fake")
    return {"fvd": frechet(a, b, cov=cov), "n_real": int(len(a)), "n_fake": int(len(b)), "cov": cov}
