"""Pre-release ("legacy") recipes of the generation metrics: what `Protocol(name="legacy-prerelease")` changes.

The legacy preset approximately reproduces the internal pre-release numbers of the baseline generators. The per-subset
PROTOCOL switches live in `presets.LEGACY_PRERELEASE` and are read with `protocol.legacy_option`:

    FVD windows / covariance   Simul-* 4 windows, unbiased (np.cov); Real-Robot 4 windows, biased (1/N);
                               Real-General 2 windows (positions 1-16 and 33-48), biased
    MEt3R aspect               Real-General: squash ('resize'); the others: centre crop
    queries                    Simul-*: every point with default_gt at t_ref (no visibility requirement)
    GT visibility              Simul-General: in_frustum (frustum only); the others: visible
    RGB scale                  MVTracker received RGB / 255 (rgb_scale_quirk)
    K scaling                  K rows scaled by (w/W, h/H) instead of the pixel-centre mapping (plain_K_scaling)
    Real-General frames        the lossless 480x480 squashed frames of GT 16..80 from <root>/_reproduce

This module holds the remaining recipes of the pre-release evaluation code, which are not Protocol fields
(`LEGACY_GENERATION`, applied by `vpm_worldbench.tasks.generation` when `is_legacy(protocol)`):

    intrinsics   Simul-*: the vendor-delivered K (cameras.npz `K_delivered`, cx = cy = 640, the UE/CARLA pixel-corner
                 convention) instead of the released K (cx = cy = 639.5).
    track grid   Real-General: the squashed 480x480 frames are un-squashed to 640x480 (INTER_CUBIC) for depth
                 estimation and tracking, with K = float32(K_undistort_raw) * 640/2048 (the pre-release scaled the
                 float32 copy of K_undistort stored with the orhtrack tracks; the released K = K_undistort / 2 in
                 float64 differs by one float32 ulp in some entries).
    MoGe FoV     Simul-*: sample.json views[k].hfov_deg (vendor metadata); Real-Robot: 2 atan(W / 2 fx) of the native K;
                 Real-General: from K on the 640x480 track grid.
    world frame  Simul-*: the lifter works in the source world (sample.json world.to_source): the left-handed Unreal
                 world in metres for Simul-Robot, the un-centred vendor world for Simul-General. MVTracker is neither
                 mirror- nor translation-invariant (see lifters.base.source_world_transform); Real-*: released world.
    GT depth     Simul-*: decoded as raw / 100 in float32 (the pre-release arithmetic; the package multiplies by 0.01,
                 which differs by one float32 ulp on some pixels), cv2.INTER_NEAREST to the eval grid, <= 1 cm or
                 > 80 m invalid. Real-General: the reconstruction depth of frame 16 on the 480x480 grid from
                 <root>/_reproduce/Real-General/<id>/depth_cond.npz, cv2.INTER_NEAREST to the 640x480 track grid.
                 (cv2.INTER_NEAREST takes source pixel floor(i * W / w); the package's own nearest resampling,
                 geometry.warp_image(interp="nearest"), uses pixel centres, floor((i + 0.5) * W / w).)
    decoder      'opencv' (legacy default): videos decoded with cv2.VideoCapture (the FFmpeg/swscale YUV->RGB of the
                 installed OpenCV), then INTER_AREA to the eval grid, as the pre-release loaders did; this depends on
                 the OpenCV / FFmpeg build. 'pyav' = the package's deterministic decoder (vpm_worldbench.io.video).
    MEt3R        feature_chunk 32, 64 ordered pairs per MASt3R forward, cross-view AND temporal pairs of the window
                 positions 0, 4, ..., 64 scored in one call (position 0 = the conditioning frame, not scored). Peak GPU
                 memory ~90-94 GiB (a 96 GB GH200 fits it); on smaller GPUs pass met3r_batch / met3r_feature_chunk
                 (e.g. 16 / 8: ~31 GiB), which moves single pairs by ~1e-3 and clip means by ~1e-4.

Remaining, documented differences (exact reproduction is not expected):
  * generated videos: the pre-release Simul-* and Real-Robot PSNR / SSIM / LPIPS / delta were computed on the in-memory
    generated frames, before they were written to the eval-grid mp4s that are scored now (imageio libx264 quality 9:
    ~0.03 dB PSNR); MVTracker amplifies such pixel differences (per-clip delta moves by up to several points).
    FVD and MEt3R of the pre-release read those mp4s (OpenCV), so they are reproduced closely.
  * Real-Robot GT: the released tracks are time-aligned to the videos (-2.3 frames) and their visibility is a 12 px
    z-buffer at the native 1280x720 resolution; the pre-release used the un-aligned forward-kinematics tracks and an
    8 px z-buffer on the 480x848 grid (queries and scored entries differ).
  * Simul-* GT: the released visible / in_frustum flags were recomputed with the corrected K (cx 639.5) and the
    renderer near plane (Simul-Robot visible = vendor depth test AND that frustum); positions went through the world
    re-centring (float32 rounding: some query coordinates differ by one ulp in the source frame).
  * Simul-* frozen row: the pre-release reference row was "camera_only" (points frozen in camera 0 and moved with it),
    equal to frozen points only when camera 0 is static (most Simul-Robot clips; not the moving Simul-General cameras,
    where it is a different baseline altogether).
  * MEt3R: the pre-release production run used pytorch3d's binned rasterizer (<= 5e-5 per pair vs the naive one) and
    stored values rounded to 5 decimals.
  * FVD: the pre-release Simul-* tables took feature means in float32 (<= 1.2e-4).

Observed differences (2026-10-08, GH200, legacy preset vs the stored pre-release per-clip values; 7 samples per subset
x HunyuanVideo-1.5 / Wan2.2-5B / Cosmos3-Nano = 21 clip-model pairs per subset: the 2 samples first compared, 1 from
an independent check and 4 more spread over the index order). These are OBSERVED MAXIMA of |difference| on these
samples, not bounds: the first 8 samples gave delta gen. <= 10.9 / 7.7 / 4.0 and delta real <= 0.40 / 0.58 / 0.73,
and the out-of-sample clips went beyond that. Mean |difference| in parentheses.
                   PSNR (dB)  SSIM     LPIPS    delta gen.    delta real    delta frozen  MEt3R clip mean  I3D features
    Simul-Robot    0.040      0.008    0.002    12.6 (4.1)    0.87 (0.43)   0.01          1.0e-4           identical
    Simul-General  0.008      0.001    0.004    7.7 (1.1)     1.13 (0.43)   82 (*)        2.4e-5           identical
    Real-Robot     0.014      0.004    0.005    7.4 (1.5)     2.82 (1.21)   1.12          3.6e-5           identical
    Real-General   1e-6       0.0      -        0.0           0.0           0.0           1.6e-5           -
(*) the pre-release "camera_only" row with moving cameras (see above). Single MEt3R pairs differ by up to 5e-3
(Simul-Robot; <= 7e-4 elsewhere). Real-General (lossless _reproduce frames, the same K and depth recipe) is exact.
The generated-video delta residuals are consistent with the in-memory-vs-mp4 explanation above: an independent
re-encode of the baseline videos at CRF 18 (PSNR vs GT changed by <= 0.025 dB) moved per-clip delta_avg by up to ~7
points, and a decoder swap or a CRF-4 re-encode by up to ~1.3 points. The real-video residuals on Simul-* come from the
recomputed GT / query positions (one-ulp coordinate differences that MVTracker amplifies), on Real-Robot from the
re-aligned GT tracks (see above).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import canonical_subset

LEGACY_NAME = "legacy-prerelease"

LEGACY_GENERATION = {
    "Simul-Robot": {"K": "delivered", "fov": "hfov_deg", "world": "source", "depth": "raw_div100", "track_size": None,
                    "depth_min_m": 0.01},
    "Simul-General": {"K": "delivered", "fov": "hfov_deg", "world": "source", "depth": "raw_div100", "track_size": None,
                      "depth_min_m": 0.01},
    "Real-Robot": {"K": "released", "fov": "native_K", "world": None, "depth": None, "track_size": None,
                   "depth_min_m": 0.0},
    "Real-General": {"K": "undistort_raw_f32", "fov": None, "world": None, "depth": "_reproduce",
                     "track_size": (640, 480), "depth_min_m": 0.0},
}

DEFAULT_GENERATION = {"K": "released", "fov": None, "world": None, "depth": None, "track_size": None,
                      "depth_min_m": 0.0}

MET3R_LEGACY = {"batch": 64, "feature_chunk": 32, "temporal": True}
DECODERS = ("pyav", "opencv")


def is_legacy(protocol) -> bool:
    """True for the 'legacy-prerelease' preset (the same test as protocol.legacy_option)."""
    return getattr(protocol, "name", None) == LEGACY_NAME


def generation_recipe(protocol, subset: str) -> dict:
    """The non-Protocol recipe of `subset`: LEGACY_GENERATION in legacy mode, else DEFAULT_GENERATION (a copy)."""
    if is_legacy(protocol):
        return dict(LEGACY_GENERATION[canonical_subset(subset)])
    return dict(DEFAULT_GENERATION)


def read_frames_opencv(path, indices=None, limit: int | None = None) -> np.ndarray:
    """Decode RGB uint8 frames (n, H, W, 3) with cv2.VideoCapture (the pre-release loaders: BGR -> RGB).

    indices: frame indices (any order, repeats allowed) or None for every frame (at most `limit` frames from the
    start when limit is given). Raises IndexError when a requested frame is past the end, IOError when nothing can be
    decoded. Results depend on the OpenCV / FFmpeg build."""
    import cv2
    if indices is None and limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be >= 1")
    want = None if indices is None else sorted(set(int(i) for i in indices))
    if want is not None and want and want[0] < 0:
        raise IndexError(f"negative frame index in {path}")
    if want == []:
        raise ValueError("no frame index given")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    got, frames, i = {}, [], 0
    wanted = set(want) if want is not None else None
    last = want[-1] if want else None
    try:
        while (last is None or i <= last) and (want is not None or limit is None or i < int(limit)):
            ok, fr = cap.read()
            if not ok:
                break
            if wanted is None:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            elif i in wanted:
                got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
    if want is None:
        if not frames:
            raise IOError(f"OpenCV decoded no frame from {path}")
        return np.stack(frames)
    missing = [k for k in want if k not in got]
    if missing:
        raise IndexError(f"{path}: frames {missing[:5]}{'...' if len(missing) > 5 else ''} not in the video")
    return np.stack([got[int(k)] for k in indices])


def simul_depth_raw_div100(sample, view: int, t: int, size, far_clip_m: float = 80.0,
                           min_m: float = 0.01) -> np.ndarray:
    """Pre-release Simul-* GT depth of (view, t) on a (W, H) grid: raw uint16 / 100 in float32, cv2.INTER_NEAREST,
    values <= min_m, > far_clip_m or non-finite set to 0 (the pre-release depth loader). Only for 'resize' grids."""
    import cv2
    spec = sample.meta.get("depth") or {}
    if abs(float(spec.get("unit_scale", 0.0)) - 0.01) > 1e-12:
        raise ValueError(f"{sample.uid}: the raw/100 depth recipe needs centimetre depth (unit_scale 0.01)")
    arch = sample._depth
    raw = arch._z().read(arch.member(view, t))
    d = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"cannot decode {arch.member(view, t)} in {arch.path}")
    d = d.astype(np.float32) / 100.0
    W, H = (int(v) for v in size)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    d[(d <= min_m) | (d > far_clip_m) | ~np.isfinite(d)] = 0.0
    return d


def reproduce_dir(sample) -> Path:
    return sample.root / "_reproduce" / sample.subset / sample.id


def resize_nearest_opencv(img: np.ndarray, size) -> np.ndarray:
    """cv2.resize(..., INTER_NEAREST) to (W, H): target pixel i takes source pixel floor(i * W_src / W), the pre-release
    rule (up to one source pixel off the pixel-centre rule of `geometry.warp_image(interp="nearest")`)."""
    import cv2
    W, H = (int(v) for v in size)
    if img.shape[:2] == (H, W):
        return img
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)


def reproduce_depth(sample, views, frame: int = 16, size=None) -> np.ndarray:
    """(len(views), 480, 480) float32 metres: the pre-release Real-General reconstruction depth of frame 16 on the
    squashed 480x480 grid (<root>/_reproduce/Real-General/<id>/depth_cond.npz), in the order of `views`; with `size`
    (W, H) resized like the pre-release evaluator (cv2.INTER_NEAREST, `resize_nearest_opencv`).
    Raises FileNotFoundError when the pack is absent and ValueError when it is not the depth of `frame`."""
    p = reproduce_dir(sample) / "depth_cond.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} (legacy Real-General depth; part of the _reproduce pack)")
    with np.load(p, allow_pickle=False) as z:
        d = z["depth"].astype(np.float32)
        names = [str(s) for s in z["views"]] if "views" in z.files else None
        stored = int(z["frame"]) if "frame" in z.files else 16
    if stored != int(frame):
        raise ValueError(f"{p} holds the depth of frame {stored}, not of frame {frame}")
    src = [str(s) for s in sample.cameras.source_view_names]
    if names is not None and names != src:
        raise ValueError(f"{p}: views {names} do not match the sample's views {src}")
    d = d[list(views)]
    if size is not None:
        d = np.stack([resize_nearest_opencv(x, size) for x in d])
    return d


def native_fov_deg(sample, view: int) -> float:
    """2 atan(W / 2 fx) of the released K at the stored resolution (the pre-release Real-Robot MoGe FoV)."""
    W = sample.image_size(view)[0]
    return float(np.degrees(2 * np.arctan(W / 2 / sample.cameras.K[view][0, 0])))
