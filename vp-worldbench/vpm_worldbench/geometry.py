"""Camera geometry shared by every task.

Conventions (docs/DATA_FORMAT.md): world is right-handed, metres, +z up; extrinsics are world-to-camera 4x4 (OpenCV:
x right, y down, z forward); intrinsics are in pixels with integer coordinates at pixel centres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESIZE_MODES = ("resize", "center_crop", "letterbox")


@dataclass(frozen=True)
class ImageMap:
    """Affine pixel map from a source image (W, H) to a target grid (w, h): u' = sx*(u + 0.5) - 0.5 + ox, same for v.
    `crop` is the source crop box (x0, y0, x1, y1) actually used (whole image unless mode == center_crop)."""
    src_size: tuple
    dst_size: tuple
    sx: float
    sy: float
    ox: float
    oy: float
    crop: tuple
    mode: str

    def apply_uv(self, uv: np.ndarray) -> np.ndarray:
        out = np.array(uv, dtype=np.float64, copy=True)
        out[..., 0] = self.sx * (out[..., 0] + 0.5) - 0.5 + self.ox
        out[..., 1] = self.sy * (out[..., 1] + 0.5) - 0.5 + self.oy
        return out

    def apply_K(self, K: np.ndarray) -> np.ndarray:
        K = np.array(K, dtype=np.float64, copy=True)
        K[..., 0, 0] *= self.sx
        K[..., 1, 1] *= self.sy
        K[..., 0, 1] *= self.sx
        K[..., 0, 2] = self.sx * (K[..., 0, 2] + 0.5) - 0.5 + self.ox
        K[..., 1, 2] = self.sy * (K[..., 1, 2] + 0.5) - 0.5 + self.oy
        return K


def image_map(src_size, dst_size, mode: str = "resize") -> ImageMap:
    """src_size / dst_size = (W, H). Modes:
    resize       stretch to (w, h) (may change the aspect ratio; sx != sy)
    center_crop  crop the largest centred region with the target aspect, then resize (sx == sy)
    letterbox    resize to fit inside (w, h) keeping the aspect, pad symmetrically (sx == sy)
    """
    W, H = (int(v) for v in src_size)
    w, h = (int(v) for v in dst_size)
    if mode == "resize":
        return ImageMap((W, H), (w, h), w / W, h / H, 0.0, 0.0, (0, 0, W, H), mode)
    if mode == "center_crop":
        if W * h >= H * w:
            ch, cw = H, int(round(H * w / h))
        else:
            cw, ch = W, int(round(W * h / w))
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        sx, sy = w / cw, h / ch
        return ImageMap((W, H), (w, h), sx, sy, -sx * x0, -sy * y0, (x0, y0, x0 + cw, y0 + ch), mode)
    if mode == "letterbox":
        s = min(w / W, h / H)
        nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        return ImageMap((W, H), (w, h), nw / W, nh / H, float(x0), float(y0), (0, 0, W, H), mode)
    raise ValueError(f"resize mode {mode!r} not in {RESIZE_MODES}")


def warp_image(img: np.ndarray, m: ImageMap, interp: str = "auto", border_value=0) -> np.ndarray:
    """Apply `m` to an image (H, W[, C]). interp: 'auto' (INTER_AREA when shrinking, INTER_CUBIC when enlarging),
    'nearest' (labels, depth: target pixel i takes source pixel floor((i + 0.5) * W / w), the pixel-centre rule of
    `ImageMap`, in exact integer arithmetic; OpenCV's INTER_NEAREST picks floor(i * W / w), up to a source pixel off,
    and INTER_NEAREST_EXACT rounds in fixed point), 'area', 'linear', 'cubic'."""
    import cv2
    H, W = img.shape[:2]
    assert (W, H) == tuple(m.src_size), ((W, H), m.src_size)
    w, h = m.dst_size
    if interp == "nearest":
        return _warp_nearest(img, m, border_value)
    if interp == "auto":
        flag = cv2.INTER_AREA if (m.sx < 1.0 or m.sy < 1.0) else cv2.INTER_CUBIC
    else:
        flag = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[interp]
    if m.mode == "resize":
        return cv2.resize(img, (w, h), interpolation=flag)
    if m.mode == "center_crop":
        x0, y0, x1, y1 = (int(v) for v in m.crop)
        return cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=flag)
    nw, nh = int(round(W * m.sx)), int(round(H * m.sy))
    small = cv2.resize(img, (nw, nh), interpolation=flag)
    out = np.full((h, w) + img.shape[2:], border_value, dtype=img.dtype)
    x0, y0 = int(m.ox), int(m.oy)
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def _centre_index(n_dst: int, n_src: int) -> np.ndarray:
    """Source index of every target pixel when n_src pixels are stretched onto n_dst: floor((i + 0.5) * n_src / n_dst)."""
    return ((2 * np.arange(n_dst, dtype=np.int64) + 1) * n_src) // (2 * n_dst)


def _warp_nearest(img: np.ndarray, m: ImageMap, border_value=0) -> np.ndarray:
    H, W = img.shape[:2]
    w, h = m.dst_size
    if m.mode == "resize":
        return img[_centre_index(h, H)][:, _centre_index(w, W)]
    if m.mode == "center_crop":
        x0, y0, x1, y1 = (int(v) for v in m.crop)
        return img[y0 + _centre_index(h, y1 - y0)][:, x0 + _centre_index(w, x1 - x0)]
    nw, nh = int(round(W * m.sx)), int(round(H * m.sy))
    out = np.full((h, w) + img.shape[2:], border_value, dtype=img.dtype)
    x0, y0 = int(m.ox), int(m.oy)
    out[y0:y0 + nh, x0:x0 + nw] = img[_centre_index(nh, H)][:, _centre_index(nw, W)]
    return out


def to_homogeneous(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones(x.shape[:-1] + (1,), x.dtype)], -1)


def transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to points (..., 3)."""
    return X @ T[:3, :3].T + T[:3, 3]


def project(X: np.ndarray, K: np.ndarray, w2c: np.ndarray):
    """World points (..., 3) -> (uv (..., 2), z (...)) for one camera. NaN in -> NaN out."""
    Xc = transform(np.asarray(w2c, np.float64), np.asarray(X, np.float64))
    z = Xc[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = K[0, 0] * Xc[..., 0] / z + K[0, 1] * Xc[..., 1] / z + K[0, 2]
        v = K[1, 1] * Xc[..., 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


def unproject_depth(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Depth map (H, W) planar z in metres -> world points (M, 3) for pixels with depth > 0 (and mask)."""
    H, W = depth.shape
    m = depth > 0 if mask is None else (depth > 0) & mask
    v, u = np.nonzero(m)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    Xc = np.stack([x, y, z], -1)
    c2w = np.linalg.inv(w2c)
    return Xc @ c2w[:3, :3].T + c2w[:3, 3]


def camera_center(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[..., :3, :3], w2c[..., :3, 3]
    return -np.einsum("...ji,...j->...i", R, t)


def inside(uv: np.ndarray, size, margin: float = 0.0) -> np.ndarray:
    """Pixel-centre convention: valid u in [-0.5, W-0.5)."""
    W, H = size
    return (uv[..., 0] >= -0.5 + margin) & (uv[..., 0] < W - 0.5 - margin) & \
        (uv[..., 1] >= -0.5 + margin) & (uv[..., 1] < H - 0.5 - margin)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity dst ≈ s R src + t (Umeyama 1991). src, dst (N, 3). Returns (s, R, t)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max(xs.var(0).sum(), 1e-12)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotation matrices (..., 3, 3)."""
    Rr = np.einsum("...ji,...jk->...ik", R1, R2)
    c = (np.trace(Rr, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))
