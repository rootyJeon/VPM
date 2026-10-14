"""One VP-WorldBench sample: lazy, read-only access to everything in <root>/<Subset>/<id>/.

    s = Sample(root, "Real-General", "C001")
    s.n_views, s.n_frames, s.fps, s.view_names
    s.frames(0, [16, 17], size=(640, 480))          # RGB uint8 (n, 480, 640, 3)
    s.K(0, size=(640, 480)), s.w2c(0, [16])        # intrinsics on that grid, extrinsics per frame
    tr = s.tracks                                   # GT 3D tracks (world metres)
    uv, vis, in_frame = s.tracks2d(0, size=(640, 480))   # projected 2D tracks, visibility, inside the grid
    d = s.depth(0, [16])                            # metres, 0 = invalid (None if the subset has no depth)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ..constants import SCHEMA_SAMPLE
from ..geometry import ImageMap, image_map, inside, project, warp_image
from ..io.depth import DepthArchive
from ..io.video import read_frames


@dataclass
class Cameras:
    view_names: np.ndarray
    source_view_names: np.ndarray
    image_size: np.ndarray
    K: np.ndarray
    w2c: np.ndarray
    static: np.ndarray
    extra: dict = field(default_factory=dict)


@dataclass
class Tracks:
    xyz: np.ndarray
    valid: np.ndarray
    visible: np.ndarray
    in_frustum: np.ndarray
    default_gt: np.ndarray
    group: np.ndarray
    group_names: np.ndarray
    part: np.ndarray
    part_names: np.ndarray
    kind: np.ndarray
    kind_names: np.ndarray
    side: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.xyz.shape[1]

    def group_mask(self, name: str | None) -> np.ndarray:
        if name in (None, "all"):
            return np.ones(self.n_points, bool)
        names = [str(g) for g in self.group_names]
        if name not in names:
            raise KeyError(f"group {name!r} not in {names}")
        return self.group == names.index(name)


_CORE_TRACK_KEYS = {"xyz", "valid", "visible", "in_frustum", "default_gt", "group", "group_names", "part", "part_names",
                    "kind", "kind_names", "side"}
_CORE_CAM_KEYS = {"view_names", "source_view_names", "image_size", "K", "w2c", "static"}


class Sample:
    def __init__(self, root, subset: str, sid: str):
        self.root = Path(root)
        self.subset = subset
        self.id = sid
        self.dir = self.root / subset / sid
        p = self.dir / "sample.json"
        if not p.exists():
            raise FileNotFoundError(p)
        self.meta = json.loads(p.read_text())
        if self.meta.get("schema") != SCHEMA_SAMPLE:
            raise ValueError(f"{p}: schema {self.meta.get('schema')!r}, expected {SCHEMA_SAMPLE!r}")

    def __repr__(self):
        return f"Sample({self.uid}, V={self.n_views}, T={self.n_frames}, fps={self.fps})"

    @property
    def uid(self) -> str:
        return self.meta["uid"]

    @property
    def n_views(self) -> int:
        return int(self.meta["n_views"])

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def view_names(self) -> list:
        return [v["name"] for v in self.meta["views"]]

    @property
    def caption(self) -> str:
        return self.meta["caption"]

    @property
    def attributes(self) -> dict:
        return self.meta.get("attributes", {})

    @property
    def flags(self) -> list:
        return self.meta.get("flags", [])

    def image_size(self, view: int):
        v = self.meta["views"][view]
        return int(v["width"]), int(v["height"])

    @cached_property
    def cameras(self) -> Cameras:
        with np.load(self.dir / self.meta["files"]["cameras"], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        return Cameras(**{k: d[k] for k in _CORE_CAM_KEYS},
                       extra={k: v for k, v in d.items() if k not in _CORE_CAM_KEYS})

    @cached_property
    def tracks(self) -> Tracks:
        with np.load(self.dir / self.meta["files"]["tracks3d"], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        return Tracks(**{k: d[k] for k in _CORE_TRACK_KEYS},
                      extra={k: v for k, v in d.items() if k not in _CORE_TRACK_KEYS})

    @cached_property
    def captions(self) -> dict:
        return json.loads((self.dir / self.meta["files"]["captions"]).read_text())

    @property
    def has_depth(self) -> bool:
        return bool(self.meta.get("depth"))

    @cached_property
    def _depth(self) -> DepthArchive | None:
        if not self.has_depth:
            return None
        return DepthArchive(self.dir, self.meta["depth"], self.view_names)

    def label_path(self, name: str) -> Path:
        rel = self.meta["files"].get("labels", {}).get(name)
        if rel is None:
            raise KeyError(f"{self.uid} has no label {name!r}; available: {sorted(self.meta['files'].get('labels', {}))}")
        return self.dir / rel

    def image_map(self, view: int, size=None, mode: str = "resize") -> ImageMap:
        src = self.image_size(view)
        return image_map(src, src if size is None else size, mode)

    def K(self, view: int, size=None, mode: str = "resize") -> np.ndarray:
        """Intrinsics of `view` on a (W, H) grid obtained with resize mode `mode` (default: stored resolution)."""
        return self.image_map(view, size, mode).apply_K(self.cameras.K[view])

    def w2c(self, view: int, indices=None) -> np.ndarray:
        w = self.cameras.w2c[view]
        return w if indices is None else w[np.asarray(indices)]

    def video_path(self, view: int) -> Path:
        return self.dir / self.meta["views"][view]["video"]

    def frames(self, view: int, indices=None, size=None, mode: str = "resize") -> np.ndarray:
        """RGB uint8 (n, H, W, 3) on the requested grid (size = (W, H))."""
        fr = read_frames(self.video_path(view), indices)
        if size is None or tuple(size) == self.image_size(view):
            return fr
        m = self.image_map(view, size, mode)
        return np.stack([warp_image(f, m) for f in fr])

    def depth(self, view: int, indices, size=None, mode: str = "resize") -> np.ndarray | None:
        """Planar depth in metres (n, H, W), 0 = invalid, on the requested grid (nearest-neighbour resampling).
        The archive resolution may differ from the video; `size` is in the VIDEO's frame of reference (K(view, size))."""
        if self._depth is None:
            return None
        d = self._depth.read_many(view, indices)
        target = self.image_size(view) if size is None else tuple(size)
        dm = image_map(self._depth.size, self.image_size(view), "resize")
        vm = image_map(self.image_size(view), target, mode)
        out = []
        for x in d:
            if self._depth.size != self.image_size(view):
                x = warp_image(x, dm, interp="nearest")
            if target != self.image_size(view):
                x = warp_image(x, vm, interp="nearest")
            out.append(x)
        return np.stack(out)

    def tracks2d(self, view: int, size=None, mode: str = "resize", indices=None):
        """GT tracks projected into `view` on a (W, H) grid: uv (n, N, 2) float64 (NaN where invalid), visible (n, N),
        in_frame (n, N). Visibility is the stored occlusion-aware flag restricted to points inside the target grid."""
        tr = self.tracks
        idx = np.arange(self.n_frames) if indices is None else np.asarray(indices)
        K = self.K(view, size, mode)
        W, H = self.image_size(view) if size is None else size
        uv = np.full((len(idx), tr.n_points, 2), np.nan)
        in_frame = np.zeros((len(idx), tr.n_points), bool)
        for i, t in enumerate(idx):
            p, z = project(tr.xyz[t], K, self.cameras.w2c[view, t])
            ok = tr.valid[t] & (z > 0)
            uv[i, ok] = p[ok]
            in_frame[i] = ok & inside(p, (W, H))
        vis = tr.visible[idx, :, view] & in_frame
        return uv, vis, in_frame
