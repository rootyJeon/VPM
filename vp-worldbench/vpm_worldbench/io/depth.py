"""Depth archives: depth/depth.zip with one 16-bit PNG per (view, frame), described by sample.json["depth"].

    {"file": "depth/depth.zip", "member": "{source_view}/frame_{t:04d}.png", "view_member_names": [...],
     "dtype": "uint16", "unit_scale": 0.01, "invalid_values": [65504], "invalid_ge": 65000, "planar_z": true,
     "width": 1280, "height": 1280, "kind": "..."}

Decoded depth is planar camera z in metres; invalid pixels are 0.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np


class DepthArchive:
    def __init__(self, sample_dir: Path, spec: dict, view_names: list):
        self.path = Path(sample_dir) / spec["file"]
        self.spec = spec
        self.view_names = list(view_names)
        self.member_names = spec.get("view_member_names") or self.view_names
        self._zip = None

    @property
    def size(self):
        return int(self.spec["width"]), int(self.spec["height"])

    def _z(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def member(self, view: int, t: int) -> str:
        return self.spec["member"].format(source_view=self.member_names[view], view=self.view_names[view], t=int(t))

    def read(self, view: int, t: int) -> np.ndarray:
        """(H, W) float32 metres, 0 = invalid."""
        import cv2
        raw = self._z().read(self.member(view, t))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"cannot decode {self.member(view, t)} in {self.path}")
        bad = np.zeros(img.shape, bool)
        for v in self.spec.get("invalid_values", []):
            bad |= img == v
        if self.spec.get("invalid_ge") is not None:
            bad |= img >= self.spec["invalid_ge"]
        d = img.astype(np.float32) * float(self.spec["unit_scale"])
        d[bad] = 0.0
        return d

    def read_many(self, view: int, indices) -> np.ndarray:
        return np.stack([self.read(view, t) for t in indices])

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
