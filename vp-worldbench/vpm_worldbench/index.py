"""Build <root>/index.json: one record per sample (fast filtering without opening every sample).

    vpmwb index --root /data/VP-WorldBench
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import SUBSETS

INDEX_SCHEMA = "vpm-worldbench/index/1.0"
_ATTR_KEYS = ("difficulty", "category", "subcategory", "scene", "map", "task", "hand", "task_family", "n_objects",
              "hands_used", "session", "recording", "wholebody_qa_level")


def sample_record(sample_dir: Path, ref_frame: int = 16) -> dict:
    m = json.loads((sample_dir / "sample.json").read_text())
    rec = {
        "uid": m["uid"], "subset": m["subset"], "id": m["id"], "n_views": m["n_views"], "n_frames": m["n_frames"],
        "fps": m["fps"], "duration_s": m["duration_s"],
        "image_size": [[v["width"], v["height"]] for v in m["views"]],
        "has_depth": bool(m.get("depth")), "labels": sorted(m["files"].get("labels", {})),
        "n_points": m["tracks"].get("n_points"), "flags": m.get("flags", []),
        "caption": m["caption"],
    }
    attrs = m.get("attributes", {})
    rec["attributes"] = {k: attrs[k] for k in _ATTR_KEYS if k in attrs}
    for k, v in rec["attributes"].items():
        rec.setdefault(k, v)
    tr = np.load(sample_dir / m["files"]["tracks3d"])
    q = tr["default_gt"] & tr["visible"].any(-1)
    rec["queries_per_frame"] = q.sum(1).astype(int).tolist()
    rec["queries_at_ref_frame"] = int(q[ref_frame].sum()) if ref_frame < m["n_frames"] else 0
    return rec


def build_index(root, version: str = "1.0") -> dict:
    """Write <root>/index.json and return it. A root that is not a directory, or that holds no sample folder
    (<root>/<Subset>/<id>/sample.json), raises DataRootError (a UsageError and FileNotFoundError; `vpmwb index` reports
    it in one line) and writes nothing, so a wrong --root never leaves an empty index.json behind."""
    from .data.benchmark import DataRootError
    root = Path(root)
    if not root.is_dir():
        raise DataRootError(f"data root {str(root)!r} does not exist or is not a directory")
    samples, subsets = [], {}
    for sub in SUBSETS:
        d = root / sub
        if not d.is_dir():
            continue
        ids = sorted(p.name for p in d.iterdir() if (p / "sample.json").exists())
        recs = [sample_record(d / i) for i in ids]
        samples += recs
        subsets[sub] = {"n_samples": len(recs), "n_views": sorted({r["n_views"] for r in recs}),
                        "fps": sorted({r["fps"] for r in recs}),
                        "n_frames_min": min((r["n_frames"] for r in recs), default=0),
                        "n_frames_max": max((r["n_frames"] for r in recs), default=0)}
    if not samples:
        raise DataRootError(f"data root {str(root)!r} holds no sample (expected <root>/<Subset>/<id>/sample.json for "
                            f"the subsets {', '.join(SUBSETS)}); index.json not written")
    idx = {"schema": INDEX_SCHEMA, "version": version, "subsets": subsets, "samples": samples}
    (root / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False))
    return idx


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="vpmwb index")
    ap.add_argument("--root", required=True)
    ap.add_argument("--version", default="1.0")
    a = ap.parse_args(argv)
    idx = build_index(a.root, a.version)
    for sub, s in idx["subsets"].items():
        print(f"{sub:14s} {s['n_samples']} samples, views {s['n_views']}, fps {s['fps']}, "
              f"frames {s['n_frames_min']}-{s['n_frames_max']}")
    return 0


if __name__ == "__main__":
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb(["index"] + sys.argv[1:]))
