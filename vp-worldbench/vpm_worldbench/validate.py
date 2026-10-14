"""Dataset validator: checks that every sample of a VP-WorldBench folder is complete and self-consistent.

    vpmwb validate --root /data/VP-WorldBench [--deep] [--hash] [--jobs 32] [--out validation.json]
                   [--subsets sr rr] [--ids Real-Robot/episode_000 ...]

Exit status: 0 = every selected sample is valid, 1 = a sample failed, 2 = the selection is empty, names a subset folder
that does not exist or an id that matches no sample, or --root is not a directory (an unknown subset name is a usage
error).
Fast checks (default): schema and files, video container metadata (frames, size, fps), camera validity, track invariants
(NaN iff invalid, visible ⊆ in_frustum ⊆ valid, projections of in-frustum points inside the image), depth archive member set and
depth-vs-track agreement at the reference frame, captions, declared labels.
--deep additionally decodes every video frame and every depth member; --hash verifies files_sha256.
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .constants import KIND_NAMES, SCHEMA_SAMPLE, SUBSETS, canonical_subset
from .geometry import inside, project
from .utils import canonical_uid

REQUIRED_META = ("schema", "uid", "subset", "id", "source", "n_views", "n_frames", "fps", "views", "world", "files", "tracks",
                 "caption", "attributes", "flags")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def validate_sample(root, subset: str, sid: str, deep: bool = False, check_hash: bool = False, ref_frame: int = 16) -> dict:
    from .data.sample import Sample
    from .io.video import frame_to_rgb, iter_frames, video_info

    errs, warns, stats = [], [], {}
    uid = f"{subset}/{sid}"

    def err(msg):
        errs.append(msg)

    try:
        s = Sample(root, subset, sid)
    except Exception as e:
        return {"uid": uid, "ok": False, "errors": [f"load: {type(e).__name__}: {e}"], "warnings": [], "stats": {}}
    m = s.meta
    for k in REQUIRED_META:
        if k not in m:
            err(f"sample.json missing '{k}'")
    if m.get("schema") != SCHEMA_SAMPLE:
        err(f"schema {m.get('schema')}")
    if m.get("uid") != uid or m.get("subset") != subset or m.get("id") != sid:
        err(f"uid/subset/id mismatch: {m.get('uid')}")
    V, T = s.n_views, s.n_frames
    stats.update(V=V, T=T, fps=s.fps)
    if len(m.get("views", [])) != V:
        err(f"{len(m.get('views', []))} view entries for n_views={V}")

    files = [m["files"]["cameras"], m["files"]["tracks3d"], m["files"]["captions"]] + [v["video"] for v in m["views"]]
    if m["files"].get("depth"):
        files.append(m["files"]["depth"])
    files += list(m["files"].get("labels", {}).values())
    for f in files:
        if not (s.dir / f).exists():
            err(f"missing file {f}")
    if errs:
        return {"uid": uid, "ok": False, "errors": errs, "warnings": warns, "stats": stats}
    if check_hash:
        for f, h in m.get("files_sha256", {}).items():
            p = s.dir / f
            if not p.exists():
                err(f"hashed file missing: {f}")
            elif _sha256(p) != h:
                err(f"sha256 mismatch: {f}")

    for k in range(V):
        info = video_info(s.video_path(k))
        W, H = s.image_size(k)
        if (info["width"], info["height"]) != (W, H):
            err(f"view{k}: video {info['width']}x{info['height']} != declared {W}x{H}")
        if info["n_frames"] != T:
            err(f"view{k}: {info['n_frames']} frames != T={T}")
        if abs(info["fps"] - s.fps) > 1e-3:
            err(f"view{k}: fps {info['fps']} != {s.fps}")
        if deep:
            n_dec, means = 0, []
            for i, fr in enumerate(iter_frames(s.video_path(k), rgb=False)):
                if (fr.width, fr.height) != (W, H):
                    err(f"view{k}: decoded frame {i} is {fr.width}x{fr.height}")
                    break
                if i % 16 == 0:
                    means.append(float(frame_to_rgb(fr).mean()))
                n_dec += 1
            if n_dec != T:
                err(f"view{k}: decoded {n_dec} frames != T={T}")
            stats[f"view{k}_mean_intensity"] = float(np.mean(means)) if means else None
            if means and (max(means) < 2 or min(means) > 253):
                warns.append(f"view{k}: video looks blank (mean intensity {np.mean(means):.1f})")

    cams = s.cameras
    if cams.K.shape != (V, 3, 3) or cams.w2c.shape != (V, T, 4, 4):
        err(f"camera shapes K{cams.K.shape} w2c{cams.w2c.shape}")
        return {"uid": uid, "ok": False, "errors": errs, "warnings": warns, "stats": stats}
    if not (np.isfinite(cams.K).all() and np.isfinite(cams.w2c).all()):
        err("non-finite camera values")
    R = cams.w2c[..., :3, :3]
    orth = np.abs(np.einsum("vtij,vtkj->vtik", R, R) - np.eye(3)).max()
    det = np.linalg.det(R)
    if orth > 1e-5 or np.abs(det - 1).max() > 1e-5:
        err(f"w2c rotations not proper (max |RR^T-I| {orth:.2e}, det range {det.min():.6f}..{det.max():.6f})")
    if np.abs(cams.w2c[..., 3, :] - np.array([0, 0, 0, 1])).max() > 1e-9:
        err("w2c bottom row is not [0,0,0,1]")
    for k in range(V):
        W, H = s.image_size(k)
        fx, fy, cx, cy = cams.K[k, 0, 0], cams.K[k, 1, 1], cams.K[k, 0, 2], cams.K[k, 1, 2]
        if not (fx > 0 and fy > 0 and -1 <= cx <= W and -1 <= cy <= H):
            err(f"view{k}: implausible K fx={fx} fy={fy} cx={cx} cy={cy} for {W}x{H}")
        if tuple(cams.image_size[k]) != (W, H):
            err(f"view{k}: cameras.image_size {cams.image_size[k]} != {W}x{H}")
        motion = np.abs(cams.w2c[k] - cams.w2c[k, :1]).max()
        if bool(cams.static[k]) and motion > 1e-6:
            err(f"view{k}: marked static but w2c changes by {motion:.2e}")

    tr = s.tracks
    N = tr.n_points
    stats["N"] = N
    if tr.xyz.shape != (T, N, 3) or tr.valid.shape != (T, N) or tr.visible.shape != (T, N, V) \
            or tr.in_frustum.shape != (T, N, V) or tr.default_gt.shape != (T, N):
        err("track array shapes inconsistent")
        return {"uid": uid, "ok": False, "errors": errs, "warnings": warns, "stats": stats}
    fin = np.isfinite(tr.xyz).all(-1)
    if (fin != tr.valid).any():
        err(f"xyz finite != valid at {(fin != tr.valid).sum()} entries")
    if (tr.visible & ~tr.in_frustum).any():
        err("visible not subset of in_frustum")
    if (tr.in_frustum & ~tr.valid[..., None]).any():
        err("in_frustum not subset of valid")
    if (tr.default_gt & ~tr.valid).any():
        err("default_gt not subset of valid")
    if [str(x) for x in tr.kind_names] != list(KIND_NAMES):
        err("kind_names differ from the unified taxonomy")
    if tr.kind.min() < 0 or tr.kind.max() >= len(KIND_NAMES):
        err("kind codes out of range")
    if not set(np.unique(tr.side).tolist()) <= {-1, 0, 1}:
        err("side codes out of range")
    if tr.group.min() < 0 or tr.group.max() >= len(tr.group_names) or tr.part.min() < 0 or tr.part.max() >= len(tr.part_names):
        err("group/part codes out of range")
    stats["valid_frac"] = float(tr.valid.mean())
    stats["visible_any_frac"] = float(tr.visible.any(-1).mean())
    if tr.visible.any(-1).sum() == 0:
        err("no visible GT point in any frame")
    bad = 0
    tot = 0
    ts = sorted(set(np.linspace(0, T - 1, 9).astype(int).tolist()))
    for k in range(V):
        W, H = s.image_size(k)
        for t in ts:
            uv, z = project(tr.xyz[t], cams.K[k], cams.w2c[k, t])
            fr_flag = tr.in_frustum[t, :, k]
            ok = inside(uv, (W, H)) & (z > 0)
            bad += int((fr_flag & ~ok).sum())
            tot += int(fr_flag.sum())
    stats["in_frustum_projection_violations"] = bad
    if tot and bad / tot > 1e-3:
        err(f"{bad}/{tot} in_frustum points project outside the image")
    d = np.linalg.norm(np.diff(tr.xyz, axis=0), axis=-1)
    both = tr.valid[1:] & tr.valid[:-1]
    if both.any():
        vmax = float(np.nanmax(np.where(both, d, np.nan))) * s.fps
        stats["max_speed_mps"] = vmax
        if vmax > 60:
            warns.append(f"max point speed {vmax:.1f} m/s")
    if ref_frame < T:
        q = tr.default_gt[ref_frame] & tr.visible[ref_frame].any(-1)
        stats["queries_at_ref_frame"] = int(q.sum())
        if q.sum() == 0:
            warns.append(f"no default-GT point visible at frame {ref_frame}")

    if s.has_depth:
        import zipfile
        spec = m["depth"]
        with zipfile.ZipFile(s.dir / spec["file"]) as z:
            names = set(z.namelist())
        member_names = spec.get("view_member_names") or s.view_names
        expect = {spec["member"].format(source_view=member_names[k], view=s.view_names[k], t=t) for k in range(V)
                  for t in range(T)}
        miss = expect - names
        if miss:
            err(f"depth archive misses {len(miss)} members (e.g. {sorted(miss)[:2]})")
        else:
            t = min(ref_frame, T - 1)
            rel = []
            for k in range(V):
                dep = s.depth(k, [t])[0]
                Kd = s.K(k)
                uv, zc = project(tr.xyz[t], Kd, cams.w2c[k, t])
                vis = tr.visible[t, :, k]
                if vis.sum() == 0:
                    continue
                ui = np.clip(np.round(uv[vis, 0]).astype(int), 0, dep.shape[1] - 1)
                vi = np.clip(np.round(uv[vis, 1]).astype(int), 0, dep.shape[0] - 1)
                dv = dep[vi, ui]
                good = dv > 0
                if good.any():
                    rel.append(np.abs(dv[good] - zc[vis][good]) / zc[vis][good])
            if rel:
                r = np.concatenate(rel)
                stats["depth_vs_track_median_rel_err"] = float(np.median(r))
                if np.median(r) > 0.05:
                    warns.append(f"depth vs visible track z: median relative error {np.median(r):.3f}")
            if deep:
                for k in range(V):
                    for t in range(T):
                        s.depth(k, [t])

    cap = s.captions
    if not isinstance(cap.get("caption"), str) or not cap["caption"].strip():
        err("empty caption")
    elif cap["caption"].strip() != str(m.get("caption", "")).strip():
        warns.append("captions.json caption differs from sample.json caption")
    stats["n_labels"] = len(m["files"].get("labels", {}))
    stats["n_flags"] = len(m.get("flags", []))
    return {"uid": uid, "ok": not errs, "errors": errs, "warnings": warns, "stats": stats}


def _one(args):
    root, subset, sid, deep, check_hash = args
    try:
        return validate_sample(root, subset, sid, deep=deep, check_hash=check_hash)
    except Exception as e:
        return {"uid": f"{subset}/{sid}", "ok": False, "errors": [f"crash: {type(e).__name__}: {e}",
                                                                  traceback.format_exc(limit=3)], "warnings": [], "stats": {}}


def _run_pool(work: list, jobs: int) -> list:
    """Run _one over `work`; a worker that dies (e.g. OOM-killed) breaks the pool -> the unfinished items are retried in
    a fresh pool with half the workers, and finally one by one, so the report is always complete."""
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool
    done, todo = {}, list(work)
    while todo:
        if jobs <= 1:
            for w in todo:
                done[w] = _one(w)
            break
        try:
            with ProcessPoolExecutor(jobs) as ex:
                futs = {ex.submit(_one, w): w for w in todo}
                for f in as_completed(futs):
                    done[futs[f]] = f.result()
        except BrokenProcessPool:
            pass
        todo = [w for w in todo if w not in done]
        jobs = max(1, jobs // 2)
    return [done[w] for w in work]


def validate(root, subsets=None, deep: bool = False, check_hash: bool = False, jobs: int = 8, ids=None) -> dict:
    """Validate the samples of `subsets` (names or aliases; default all four), optionally only `ids` (sample ids or
    uids, aliases accepted). The report also lists "missing_subsets" (requested subset folders that do not exist) and
    "unmatched_ids" (requested ids that match no sample folder). A root that is not a directory raises DataRootError
    (a UsageError and FileNotFoundError: `vpmwb validate` reports it in one line, exit status 2)."""
    from .data.benchmark import DataRootError
    root = Path(root)
    if not root.is_dir():
        raise DataRootError(f"data root {str(root)!r} does not exist or is not a directory")
    subs = list(dict.fromkeys(canonical_subset(s) for s in subsets)) if subsets else list(SUBSETS)
    keep = None if not ids else {canonical_uid(i) for i in ([ids] if isinstance(ids, str) else ids)}
    work, missing, matched = [], [], set()
    for sub in subs:
        d = root / sub
        if not d.is_dir():
            if subsets:
                missing.append(sub)
            continue
        for p in sorted(d.iterdir()):
            if p.name.startswith("_") or not p.is_dir():
                continue
            if keep is not None:
                hit = [x for x in (p.name, f"{sub}/{p.name}") if x in keep]
                if not hit:
                    continue
                matched.update(hit)
            work.append((str(root), sub, p.name, deep, check_hash))
    unmatched = [] if keep is None else [i for i in ([ids] if isinstance(ids, str) else ids)
                                         if canonical_uid(i) not in matched]
    res = _run_pool(work, jobs)
    by_subset = {}
    for r in res:
        sub = r["uid"].split("/")[0]
        b = by_subset.setdefault(sub, {"n": 0, "ok": 0, "with_warnings": 0})
        b["n"] += 1
        b["ok"] += int(r["ok"])
        b["with_warnings"] += int(bool(r["warnings"]))
    return {"root": str(root), "deep": deep, "hash": check_hash, "summary": by_subset,
            "n_samples": len(res), "n_ok": sum(r["ok"] for r in res), "missing_subsets": missing,
            "unmatched_ids": unmatched, "samples": res}


def main(argv=None):
    import argparse

    from .utils import error_line, parse_list, parse_subsets
    ap = argparse.ArgumentParser(prog="vpmwb validate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--subsets", nargs="*", default=None, help="subset names or aliases (space or comma separated; "
                                                                 "default all)")
    ap.add_argument("--ids", nargs="*", default=None, help="sample ids or uids (space or comma separated)")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    try:
        subsets = parse_subsets(a.subsets)
    except KeyError as e:
        ap.error(str(e))
    rep = validate(a.root, subsets, a.deep, a.hash, a.jobs, parse_list(a.ids))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    for sub, b in rep["summary"].items():
        print(f"{sub:14s} {b['ok']}/{b['n']} ok ({b['with_warnings']} with warnings)")
    bad = [r for r in rep["samples"] if not r["ok"]]
    for r in bad[:20]:
        print("FAIL", r["uid"], "; ".join(r["errors"][:3]))
    print(f"{rep['n_ok']}/{rep['n_samples']} samples valid")
    selection = []
    if rep["missing_subsets"]:
        selection.append(f"no subset folder {', '.join(rep['missing_subsets'])} under {a.root}")
    if rep["unmatched_ids"]:
        u = rep["unmatched_ids"]
        selection.append(f"{len(u)} requested id(s) match no sample: {u[:10]}")
    if rep["n_samples"] == 0:
        selection.append("no sample selected")
    for msg in selection:
        print(error_line(msg), file=sys.stderr)
    return 2 if selection else (1 if bad else 0)


if __name__ == "__main__":
    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb(["validate"] + sys.argv[1:]))
