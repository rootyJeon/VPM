"""Export the inputs a video generator needs for the generation protocol.

    vpmwb export-inputs --root /data/VP-WorldBench --out inputs/ [--protocol vpm-v1|<yaml>] [--size 1280x720]

Per evaluable sample:
    <out>/<Subset>/<id>/context/view{k}/frame_{t:04d}.png     context frames (t_ref-c+1 .. t_ref), native size unless --size
    <out>/<Subset>/<id>/inputs.json                            prompts, frame indices, fps, cameras (K at the exported size,
                                                               the resize mode that reached it (null: native size), w2c
                                                               of the context frames, and of the window frames when
                                                               protocol.camera_conditioning is set)
    <out>/manifest.jsonl                                       one line per sample
--size reaches the requested grid with the protocol's resize mode of each subset (protocol.resize_mode_for: the mode
that maps predictions and reference frames to the eval grid), so context frames are framed like the scored reference.
An explicit --resize-mode that differs from it, or a --size whose framing cannot match the reference (e.g. a
centre-cropping protocol with a size of neither the native nor the eval-grid aspect), gives a warning.
Generate `len(window_indices)` frames per view starting with the conditioning frame (or only the future frames and set
`first_frame: future` in the prediction meta.json), at `fps_window` = clip fps / frame_step (then frame j is window
position j), or at any other rate declared as "fps" in meta.json; see predictions.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from .geometry import RESIZE_MODES, camera_center, image_map
from .protocol import Protocol


def camera_phrase(w2c_a: np.ndarray, w2c_b: np.ndarray) -> str:
    """Camera-motion sentence between two poses (z-up world). Same wording rules as the pre-release prompts, with the
    heading measured as the yaw of the optical axis around +z (the pre-release Simul-General prompts used a wrong axis)."""
    ca, cb = camera_center(w2c_a), camera_center(w2c_b)
    travel = float(np.linalg.norm(cb - ca))
    fa, fb = w2c_a[2, :3], w2c_b[2, :3]
    ya, yb = np.arctan2(fa[1], fa[0]), np.arctan2(fb[1], fb[0])
    dyaw = float(np.degrees((yb - ya + np.pi) % (2 * np.pi) - np.pi))
    if travel < 0.05 and abs(dyaw) < 2.0:
        return "The camera is static."
    s = f"The camera travels about {travel:.1f} m"
    if abs(dyaw) > 8.0:
        s += ", panning left" if dyaw > 0 else ", panning right"
    return s + "."


def window_caption(sample, protocol: Protocol) -> str:
    """Caption of the evaluated window if the sample has window captions (Real-General), else the global caption."""
    caps = sample.captions
    lo, hi = protocol.ref_frame, protocol.last_frame
    best, best_ov = None, -1
    for w in caps.get("windows", []):
        ov = min(hi, w["end_frame"]) - max(lo, w["start_frame"])
        if ov > best_ov or (ov == best_ov and best is not None and abs(w["start_frame"] - lo) < abs(best["start_frame"] - lo)):
            best, best_ov = w, ov
    return best["caption"] if best is not None and best_ov > 0 else caps["caption"]


def prompts(sample, protocol: Protocol) -> dict:
    views = protocol.views_for(sample)
    out = {}
    for k in views:
        name = f"view{k}"
        if protocol.text == "none":
            out[name] = ""
        elif protocol.text == "legacy":
            out[name] = sample.captions.get("legacy_prompts", {}).get(name, sample.caption)
        elif protocol.text == "window":
            out[name] = window_caption(sample, protocol)
        elif protocol.text == "caption+camera":
            w = sample.cameras.w2c[k]
            out[name] = f"{sample.caption} {camera_phrase(w[protocol.ref_frame], w[protocol.last_frame])}"
        else:
            out[name] = sample.caption
    return out


def _framing_matches(src, size, mode: str, eval_size, eval_mode: str, tol_px: float = 1.0) -> bool:
    """True when native frames (src) taken to `size` with `mode` and then to the eval grid with `eval_mode` (how a
    prediction of that size is scored, predictions.load_prediction) land where the native frames taken directly to the
    eval grid do (the reference frames), within `tol_px` eval-grid pixels."""
    a = image_map(src, size, mode)
    b = image_map(size, eval_size, eval_mode)
    r = image_map(src, eval_size, eval_mode)
    corners = np.array([[-0.5, -0.5], [src[0] - 0.5, src[1] - 0.5]], float)
    return bool(np.abs(b.apply_uv(a.apply_uv(corners)) - r.apply_uv(corners)).max() <= tol_px)


def export_notes(sample, protocol: Protocol, size, mode: str | None = None) -> list:
    """Warnings for exporting `sample` at `size` (W, H) with resize mode `mode` (None = the protocol's mode of the
    subset): an explicit mode other than the protocol's, and views whose exported framing cannot match the reference
    frames on the eval grid. [] for a native-size export (size None)."""
    if size is None:
        return []
    pmode = protocol.resize_mode_for(sample.subset)
    mode = pmode if mode is None else mode
    ev = protocol.eval_size_for(sample.subset)
    notes = []
    if mode != pmode:
        notes.append(f"{sample.subset}: --resize-mode {mode} differs from the protocol's resize_mode {pmode} (protocol "
                     f"{protocol.name!r})")
    bad = sorted({sample.image_size(k) for k in protocol.views_for(sample)
                  if not _framing_matches(sample.image_size(k), tuple(size), mode, ev, pmode)})
    if bad:
        notes.append(f"{sample.subset}: frames of {', '.join(f'{w}x{h}' for w, h in bad)} exported at "
                     f"{size[0]}x{size[1]} with {mode} are framed differently from the reference frames on the "
                     f"{ev[0]}x{ev[1]} eval grid ({pmode}): predictions of this size would be scored against other "
                     f"image content; use a size with the native or the eval-grid aspect, or the protocol's resize "
                     f"mode")
    return notes


def export_sample(sample, protocol: Protocol, out_dir, size=None, mode: str | None = None) -> dict | None:
    """Export one sample (module docstring); None when the protocol window does not fit the sample. `mode` (how
    `size` is reached) defaults to the protocol's resize mode of the sample's subset."""
    import cv2
    ok, why = protocol.evaluable(sample)
    if not ok:
        return None
    mode = protocol.resize_mode_for(sample.subset) if mode is None else mode
    d = Path(out_dir) / sample.subset / sample.id
    views = protocol.views_for(sample)
    cams = {}
    for k in views:
        fr = sample.frames(k, protocol.context_indices, size=size, mode=mode)
        vd = d / "context" / f"view{k}"
        vd.mkdir(parents=True, exist_ok=True)
        for t, f in zip(protocol.context_indices, fr):
            cv2.imwrite(str(vd / f"frame_{t:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        W, H = (fr.shape[2], fr.shape[1])
        c = {"K": sample.K(k, (W, H), mode).tolist(), "size": [W, H],
             "resize_mode": None if (W, H) == sample.image_size(k) else mode,
             "w2c_context": sample.w2c(k, protocol.context_indices).tolist()}
        if protocol.camera_conditioning:
            c["w2c_window"] = sample.w2c(k, protocol.window_indices).tolist()
        cams[f"view{k}"] = c
    rec = {"uid": sample.uid, "subset": sample.subset, "id": sample.id, "views": [f"view{k}" for k in views],
           "fps": sample.fps, "fps_window": sample.fps / protocol.frame_step,
           "ref_frame": protocol.ref_frame, "context_indices": protocol.context_indices,
           "window_indices": protocol.window_indices, "n_frames_to_generate": len(protocol.window_indices) - 1,
           "prompts": prompts(sample, protocol), "cameras": cams, "protocol": protocol.name,
           "context_dir": f"{sample.subset}/{sample.id}/context"}
    (d / "inputs.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def export_inputs(bench, out_dir, protocol: Protocol, subsets=None, size=None, mode: str | None = None,
                  ids=None) -> dict:
    """Export the model inputs of every evaluable sample (module docstring). `size` (W, H) resizes the context frames
    (and K) with `mode` (None = the protocol's resize mode per subset, see `export_notes`); None keeps the native size.
    Returns {"exported": n, "skipped": n} (+ "unmatched_ids": requested ids that match no selected sample)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, skipped = 0, []
    subs = list(subsets or bench.subsets())
    warned = set()
    with open(out_dir / "manifest.jsonl", "w") as man:
        for sample in bench.samples(subs, ids=ids, **protocol.clip_filter):
            for note in export_notes(sample, protocol, size, mode):
                if note not in warned:
                    warned.add(note)
                    warnings.warn(note, stacklevel=2)
            rec = export_sample(sample, protocol, out_dir, size=size, mode=mode)
            if rec is None:
                skipped.append({"uid": sample.uid, "reason": protocol.evaluable(sample)[1]})
                continue
            man.write(json.dumps({k: rec[k] for k in ("uid", "subset", "id", "views", "fps", "fps_window", "ref_frame",
                                                    "n_frames_to_generate", "prompts", "context_dir")}, ensure_ascii=False)
                      + "\n")
            n_ok += 1
    (out_dir / "protocol.json").write_text(json.dumps(protocol.to_dict(), indent=1))
    (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))
    rep = {"exported": n_ok, "skipped": len(skipped)}
    if ids is not None:
        unmatched = bench.unmatched_ids(ids, subs, **protocol.clip_filter)
        if unmatched:
            rep["unmatched_ids"] = unmatched
    return rep


def main(argv=None) -> int:
    import argparse

    from .data.benchmark import Benchmark
    from .protocol import get_protocol
    from .utils import parse_list, parse_size, parse_subsets
    ap = argparse.ArgumentParser(prog="vpmwb generation export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="VP-WorldBench data root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="vpm-v1", help="vpm-v1 | legacy-prerelease | <yaml/json>")
    ap.add_argument("--subsets", nargs="*", default=None, help="subset names or aliases (space or comma separated)")
    ap.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (space or comma separated)")
    ap.add_argument("--size", default=None, help="WxH of the exported frames (default: native)")
    ap.add_argument("--resize-mode", default=None, choices=RESIZE_MODES,
                    help="how --size is reached from the native frames (default: the protocol's resize_mode of each "
                         "subset, the framing of the scored reference frames)")
    a = ap.parse_args(argv)
    try:
        subsets = parse_subsets(a.subsets)
    except KeyError as e:
        ap.error(str(e))
    try:
        size = parse_size(a.size)
    except ValueError:
        ap.error(f"--size must be WxH (e.g. 1280x720), got {a.size!r}")
    rep = export_inputs(Benchmark(a.root), a.out, get_protocol(a.protocol), subsets, size, a.resize_mode,
                        parse_list(a.ids))
    print(f"exported {rep['exported']} samples, skipped {rep['skipped']} (see {Path(a.out) / 'skipped.json'})")
    if rep["exported"] + rep["skipped"] == 0:
        print("error: generation: no sample matches the selection (--subsets / --ids / clip_filter)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    from .cli import main as _cli_main
    raise SystemExit(_cli_main(["generation", "export"] + sys.argv[1:]))
