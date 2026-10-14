"""Video reading / writing with PyAV (FFmpeg).

Deterministic colour conversion: H.264 decoding is bit-exact by specification, but the YUV -> RGB step of FFmpeg's
swscale (used by PyAV's to_ndarray("rgb24") and by OpenCV) differs between library versions by up to 3-4 levels on
most pixels. So the benchmark converts the decoded planes itself with fixed, documented formulas:
  - yuv420p / yuv422p / yuv444p (limited range, untagged = BT.601, the convention of every source video):
    each chroma sample is replicated over its 2x2 (4:2:0) or 2x1 (4:2:2) luma block - what FFmpeg's unscaled
    yuv2rgb path (used by PyAV and OpenCV) does - then the BT.601 limited-range matrix in float64, round half up,
    clip to [0, 255]. Agrees with FFmpeg/OpenCV to within a few levels (their fixed-point tables differ by version);
  - yuvj* (full range): BT.601 full-range matrix;
  - rgb / gbrp (e.g. lossless libx264rgb): exact channel reordering.
Other pixel formats fall back to PyAV's rgb24 conversion (and are reported by `decode_backend_info`).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_BT601_LIMITED = np.array([[1.164383562, 0.0, 1.596026786],
                           [1.164383562, -0.391762290, -0.812967647],
                           [1.164383562, 2.017232143, 0.0]])
_BT601_FULL = np.array([[1.0, 0.0, 1.402],
                        [1.0, -0.344136286, -0.714136286],
                        [1.0, 1.772, 0.0]])
_FALLBACK_WARNED = set()


def _plane(frame, i: int) -> np.ndarray:
    p = frame.planes[i]
    a = np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)
    return a[:, : p.width]


def frame_to_rgb(frame) -> np.ndarray:
    """PyAV VideoFrame -> RGB uint8 (H, W, 3) with the deterministic conversion described in the module docstring."""
    fmt = frame.format.name
    H, W = frame.height, frame.width
    if fmt in ("rgb24",):
        return frame.to_ndarray()
    if fmt in ("gbrp", "bgr24", "rgba", "bgra", "gbrap"):
        return frame.to_ndarray(format="rgb24")
    if fmt in ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p"):
        y = _plane(frame, 0).astype(np.float64)
        u, v = _plane(frame, 1), _plane(frame, 2)
        if fmt.endswith("420p"):
            u = np.repeat(np.repeat(u, 2, 0), 2, 1)[:H, :W].astype(np.float64)
            v = np.repeat(np.repeat(v, 2, 0), 2, 1)[:H, :W].astype(np.float64)
        elif fmt.endswith("422p"):
            u = np.repeat(u, 2, 1)[:, :W].astype(np.float64)
            v = np.repeat(v, 2, 1)[:, :W].astype(np.float64)
        else:
            u, v = u.astype(np.float64), v.astype(np.float64)
        if fmt.startswith("yuvj"):
            M, yy = _BT601_FULL, y
        else:
            M, yy = _BT601_LIMITED, y - 16.0
        cb, cr = u - 128.0, v - 128.0
        rgb = np.stack([M[0, 0] * yy + M[0, 2] * cr,
                        M[1, 0] * yy + M[1, 1] * cb + M[1, 2] * cr,
                        M[2, 0] * yy + M[2, 1] * cb], -1)
        return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)
    if fmt not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(fmt)
        warnings.warn(f"pixel format {fmt}: falling back to FFmpeg's rgb24 conversion (not bit-reproducible "
                      f"across FFmpeg versions)")
    return frame.to_ndarray(format="rgb24")


def _av():
    try:
        import av
    except ImportError as e:
        raise ImportError("PyAV is required to read videos: pip install av") from e
    import av
    return av


def video_info(path) -> dict:
    """{'width', 'height', 'fps', 'n_frames'} without decoding (n_frames from the container, else counted)."""
    av = _av()
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        width, height = s.codec_context.width, s.codec_context.height
        fps = float(s.average_rate) if s.average_rate else float(s.guessed_rate)
        n = s.frames
        if not n:
            n = sum(1 for p in c.demux(s) if p.size > 0)
        return {"width": width, "height": height, "fps": fps, "n_frames": int(n)}


def read_frames(path, indices: Sequence[int] | None = None) -> np.ndarray:
    """Decode RGB uint8 frames (n, H, W, 3) in presentation order. `indices` (any order, repeats allowed) selects
    frames by 0-based index; None = all frames. Raises IndexError if an index is past the end of the video."""
    av = _av()
    want = None if indices is None else sorted(set(int(i) for i in indices))
    if want is not None and want and want[0] < 0:
        raise IndexError(f"negative frame index in {path}")
    got = {}
    frames = []
    last = want[-1] if want else None
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            if want is None:
                frames.append(frame_to_rgb(frame))
                continue
            if i in want:
                got[i] = frame_to_rgb(frame)
            if i >= last:
                break
    if want is None:
        return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)
    missing = [i for i in want if i not in got]
    if missing:
        raise IndexError(f"{path}: frames {missing[:5]}{'...' if len(missing) > 5 else ''} not in the video")
    return np.stack([got[int(i)] for i in indices])


def iter_frames(path, rgb: bool = True):
    """Stream the frames of a video one at a time (constant memory). Yields RGB uint8 (H, W, 3) when rgb=True, else the
    raw PyAV VideoFrame (decoding only - enough to check integrity and count frames cheaply)."""
    av = _av()
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            yield frame_to_rgb(frame) if rgb else frame


def write_video(path, frames: np.ndarray, fps: float, crf: int = 10, lossless: bool = False,
                pix_fmt: str = "yuv420p") -> Path:
    """Write RGB uint8 frames (n, H, W, 3) as H.264 mp4. lossless=True uses libx264rgb with qp 0 (bit-exact RGB)."""
    av = _av()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.mp4")
    n, H, W, _ = frames.shape
    from fractions import Fraction
    with av.open(str(tmp), mode="w") as c:
        if lossless:
            s = c.add_stream("libx264rgb", rate=Fraction(fps).limit_denominator(1000))
            s.pix_fmt = "rgb24"
            s.options = {"qp": "0", "preset": "medium"}
        else:
            s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
            s.pix_fmt = pix_fmt
            s.options = {"crf": str(crf), "preset": "medium"}
        s.width, s.height = W - W % 2 if not lossless else W, H - H % 2 if not lossless else H
        for f in frames:
            if not lossless and (W % 2 or H % 2):
                f = f[: s.height, : s.width]
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
            for p in s.encode(vf):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    tmp.replace(path)
    return path


def iter_chunks(indices: Iterable[int], size: int):
    buf = []
    for i in indices:
        buf.append(i)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
