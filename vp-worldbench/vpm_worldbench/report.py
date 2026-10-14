"""Result tables of the generation task: per-sample JSONs of several models + shared reference rows -> summary.

    from vpm_worldbench.report import summarize_generation
    summarize_generation({"hunyuan": "res/hunyuan", "wan": "res/wan"}, out="res/summary",
                         reference_dir="res/_reference", splits=["difficulty=hard"])
    vpmwb generation summarize --run hunyuan=res/hunyuan --run wan=res/wan \\
        --reference-dir res/_reference --out res/summary [--split difficulty=hard] [--protocol vpm-v1]

Inputs (written by `vpm_worldbench.tasks.generation.evaluate`):
    <run_dir>/protocol.json                          protocol, its hash, the resolved per-subset settings and the
                                                     prediction root of the run ("pred_root"; absent in older runs)
    <run_dir>/<Subset>/<id>.json                     per-sample record of one model; status ok | skipped (not evaluable) |
                                                     missing (prediction files) | invalid (unusable prediction) | error
    <run_dir>/_fvd/<Subset>/<id>.npz                 I3D features of the model's FVD windows ("feats" (V*k, 400))
    <reference_dir>/<hash>/<Subset>/<id>.json        reference rows of the sample: real-video MEt3R, real / frozen delta
    <reference_dir>/<hash>/_fvd/<Subset>/<id>.npz    I3D features of the real windows
The reference directory is shared by every model evaluated with the same protocol (same hash). The runs of one table
must share the hash; the FVD covariance estimator (fvd_cov, not part of the hash) comes from `protocol` when given,
else from the runs' protocol.json (which must agree).

Clip sets. Per subset (and per attribute split), the PAIRED set = clips with status "ok" for every model of the table
and an "ok" reference record. The reference record is required only when the tables read it: a subset whose model
records hold only appearance metrics (evaluate(metrics=["appearance"]): PSNR / SSIM / LPIPS) pairs without reference
rows, and such runs need no reference directory. Every number of the table is computed on that set (a column whose
metric is missing for some paired clip, e.g. LPIPS of a run evaluated with lpips=False, uses the clips that have it for
every source, and reports its own n). Not evaluable clips (protocol window outside the clip) and excluded clips (with
the reason per model) are listed in summary.json. The expected clips of a run are those its evaluate() calls were asked
for (<run_dir>/_selection/<hash>_*.json, written before sharding): an expected clip without a record (a shard that
died) is excluded as "not evaluated" and counted (summary.md says how many). Records of another protocol hash, and
records scored from another prediction root than the one in the run's protocol.json (left from before
evaluate(overwrite=True) with a new root), are never used (status "stale"). Provenance: every record stores the
identity of the weights of each metric ("models") and the data version; differences between the records of a table
are listed as warnings.

Columns (docs/METRICS.md):
    FVD             Fréchet distance of the I3D features of all windows of the paired clips and views, real vs generated,
                    pooled over the subset (metrics.fvd.pooled_fvd; covariance per subset from the protocol), with the
                    number of windows. Not a clip mean, so no s.e.
    PSNR/SSIM/LPIPS mean ± s.e. over clips of the clip value (per view: mean over the scored frames t_ref+1..; then the
                    mean over views).
    MEt3R           metrics.met3r.delta_table on the scored timesteps (conditioning frame excluded): cross-view MEt3R ± s.e. (main
                    number), signed difference ± s.e., share of clips below the real video, raw MEt3R ± s.e.; the floor and
                    noise floor of the same clips (Real video row: raw MEt3R only).
    δ               mean ± s.e. over clips of the sample's main point group (protocol delta_group, else sample.json
                    tracks.default_report_group): δ_avg, δ_avg of dynamic tracks (path > 10 cm), δ on the moving
                    entries, AJ, OA, survival (all %), ADE (cm, scored entries); reference rows Real video (real frames
                    through the same lifter) and Frozen points (query positions held fixed, always visible). Other point
                    groups are in summary.json. δ_avg dyn. / δ mov. are undefined on clips without dynamic tracks /
                    moving entries and use the other clips: "columns_n" (delta_dyn, delta_mov) and the footnote give
                    their n.
s.e. = std(ddof=1) / sqrt(n). Every mean ± s.e. in summary.json carries its n and n_inf: infinite values (e.g. the
ADE of a clip with non-finite predicted tracks) are kept, so the mean is ∞ and n_inf counts those clips.
"""
from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path

import numpy as np

from .constants import SUBSETS

TASK = "generation"
REFERENCE_TASK = "generation-reference"
REAL_ROW, FROZEN_ROW = "Real video", "Frozen points"
REFERENCE_SOURCE = "real"

DELTA_COLUMNS = (("any/delta_avg", "δ_avg", True, 1.0, 1), ("dynamic/delta_avg", "δ_avg dyn.", True, 1.0, 1),
                 ("moving/delta_avg", "δ mov.", True, 1.0, 1),
                 ("any/aj", "AJ", True, 1.0, 1), ("any/oa", "OA", True, 1.0, 1),
                 ("any/survival", "Surv.", True, 1.0, 1), ("any/ade", "ADE (cm)", False, 100.0, 1))
DELTA_KEYS = tuple(c[0] for c in DELTA_COLUMNS) + ("any/mte", "ade_all", "mte_all", "fde", "allframes/ade",
                                                     "any/n_tracks", "dynamic/n_tracks", "moving/n_entries",
                                                     "n_queries")
DELTA_SUB_COLUMNS = {"delta_dyn": "dynamic/delta_avg", "delta_mov": "moving/delta_avg"}
APPEARANCE_COLUMNS = (("psnr", "PSNR", True, 2), ("ssim", "SSIM", True, 3), ("lpips", "LPIPS", False, 3))


def jsonable(x):
    """JSON-safe copy: numpy scalars / arrays -> Python, NaN -> None, +-inf -> "inf" / "-inf", tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj, indent=1) -> Path:
    """Atomic JSON write (tmp file + rename) of jsonable(obj)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=indent, ensure_ascii=False))
    os.replace(tmp, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def num(x) -> float:
    """JSON value -> float (None -> NaN, "inf" / "-inf" strings -> infinities)."""
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return float(x)
    return float(x)


def _defined(x) -> bool:
    """A value that mean_se counts (not None / NaN; infinities count)."""
    return not math.isnan(num(x))


def mean_se(values) -> dict:
    """{"mean", "se", "n", "n_inf"} over the defined values (None / NaN dropped, +-inf kept: the mean becomes infinite
    and the s.e. NaN, so a failed clip is never averaged away; n_inf counts them, n includes them).
    se = std(ddof=1) / sqrt(n), NaN for n < 2."""
    v = np.asarray([num(x) for x in values], np.float64)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0, "n_inf": 0}
    with np.errstate(invalid="ignore"):
        m = float(np.mean(v))
        se = float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 and np.isfinite(v).all() else float("nan")
    return {"mean": m, "se": se, "n": n, "n_inf": int(np.isinf(v).sum())}


def fmt(x, nd: int = 2, se: bool = True, scale: float = 1.0, signed: bool = False) -> str:
    """'m ± se' from a mean_se dict (or a bare number); '–' when undefined."""
    if isinstance(x, dict):
        m, s = x.get("mean"), x.get("se")
    else:
        m, s = x, None
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "–"
    m = num(m)
    if math.isinf(m):
        return "∞" if m > 0 else "-∞"
    sign = "+" if signed else ""
    out = f"{m * scale:{sign}.{nd}f}"
    if se and s is not None and not (isinstance(s, float) and math.isnan(s)) and not isinstance(s, str):
        out += f" ± {num(s) * scale:.{nd}f}"
    return out


def load_records(run_dir, task: str = TASK) -> dict:
    """{subset: {id: record}} from <run_dir>/<Subset>/*.json of `task`."""
    out = {}
    run_dir = Path(run_dir)
    for sub in SUBSETS:
        d = run_dir / sub
        if not d.is_dir():
            continue
        recs = {}
        for f in sorted(d.glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if r.get("task") == task:
                recs[r["id"]] = r
        if recs:
            out[sub] = recs
    return out


SELECTION_DIR = "_selection"


def load_selection(run_dir, protocol_hash: str):
    """{subset: {id: {"attributes", "flags"}}}: union of the samples evaluate() was asked for in `run_dir` under
    `protocol_hash` (<run_dir>/_selection/<hash>_*.json), or None when the run has no selection file (runs written
    before selection files existed)."""
    files = sorted((Path(run_dir) / SELECTION_DIR).glob(f"{protocol_hash}_*.json"))
    if not files:
        return None
    out = {}
    for f in files:
        try:
            sel = read_json(f)
        except (OSError, json.JSONDecodeError):
            continue
        for uid, meta in (sel.get("samples") or {}).items():
            sub, sid = uid.split("/", 1)
            out.setdefault(sub, {})[sid] = meta
    return out


def mark_stale(recs: dict, protocol_hash: str, pred_root: str | None = None) -> dict:
    """Records of another protocol hash (e.g. left in a run directory whose protocol.json was replaced) and, when
    `pred_root` (the run's prediction root) is given, evaluable records scored from another prediction root are not
    used: each becomes a placeholder with status "stale" (attributes / flags kept), so the clip is excluded with the
    reason. Records without "pred_root" (written before it was stored) are kept."""
    out = {}
    for sub, rs in recs.items():
        out[sub] = {}
        for i, r in rs.items():
            why = None
            if r.get("protocol_hash") != protocol_hash:
                why = f"protocol hash {r.get('protocol_hash')} != {protocol_hash}"
            elif (pred_root is not None and r.get("pred_root") not in (None, pred_root)
                  and r.get("status") != "skipped"):
                why = f"scored from the predictions in {r.get('pred_root')}, the run scores {pred_root}"
            if why is None:
                out[sub][i] = r
            else:
                out[sub][i] = {"task": r.get("task"), "subset": sub, "id": i, "status": "stale", "reason": why,
                               "attributes": r.get("attributes"), "flags": r.get("flags", [])}
    return out


def resolve_reference_dir(reference_dir, protocol_hash: str):
    """<reference_dir>/<hash> if it exists, else reference_dir itself when its protocol.json has that hash, else None."""
    if reference_dir is None:
        return None
    d = Path(reference_dir)
    if (d / protocol_hash).is_dir():
        return d / protocol_hash
    p = d / "protocol.json"
    if p.exists():
        try:
            if read_json(p).get("hash") == protocol_hash:
                return d
        except json.JSONDecodeError:
            pass
    return None


def parse_split(spec: str):
    """'key=value' / 'key!=value' (sample attribute; list attributes test membership), 'flag:NAME' / '!flag:NAME'
    (a sample quality flag starting with NAME) -> predicate(record) -> bool."""
    spec = spec.strip()
    if spec.startswith("!flag:") or spec.startswith("flag:"):
        neg = spec.startswith("!")
        name = spec.split(":", 1)[1]

        def has_flag(rec):
            hit = any(str(f).startswith(name) for f in rec.get("flags", []))
            return hit != neg
        return has_flag
    if "!=" in spec:
        key, val = spec.split("!=", 1)
        neg = True
    elif "=" in spec:
        key, val = spec.split("=", 1)
        neg = False
    else:
        raise ValueError(f"split {spec!r}: expected key=value, key!=value, flag:NAME or !flag:NAME")
    key, val = key.strip(), val.strip()

    def match(rec):
        a = (rec.get("attributes") or {}).get(key)
        if isinstance(a, (list, tuple)):
            hit = val in [str(x) for x in a]
        else:
            hit = a is not None and str(a) == val
        return hit != neg
    return match


def _clip_sets(sub, recs, refs, use_ref: bool, split=None, expected=None) -> dict:
    """Paired / excluded / not-evaluable clips of one subset (see the module docstring).

    expected: {id: {"attributes", "flags"}} of the samples the runs were asked for (selection files), or None (then
    the ids of the reference records count as expected, the behaviour for runs without selection files). The
    attribute split is decided on the first record (model, reference or selection) that carries attributes."""
    names = list(recs)
    ids = set(refs) if expected is None else set(expected)
    for n in names:
        ids |= set(recs[n].get(sub, {}))
    ids = sorted(ids)
    not_eval, paired, excluded = {}, [], {}
    for i in ids:
        rs = [recs[n].get(sub, {}).get(i) for n in names]
        sk = [r for r in rs if r is not None and r.get("status") == "skipped"]
        if sk:
            not_eval[i] = sk[0].get("reason", "")
            continue
        if split is not None:
            src = next((r for r in [*rs, refs.get(i), (expected or {}).get(i)]
                        if r is not None and isinstance(r.get("attributes"), dict)), None)
            if src is None:
                excluded[i] = ["attributes unknown (no record carries them)"]
                continue
            if not split(src):
                continue
        why = []
        for n, r in zip(names, rs):
            if r is None:
                why.append(f"{n}: not evaluated")
            elif r.get("status") != "ok":
                why.append(f"{n}: {r.get('status')}" + (f" ({r.get('reason')})" if r.get("reason") else ""))
        if use_ref:
            rr = refs.get(i)
            if rr is None or rr.get("status") != "ok":
                why.append("reference: " + ("missing" if rr is None else str(rr.get("status")))
                           + (f" ({rr.get('reason')})" if rr is not None and rr.get("reason") else ""))
        if why:
            excluded[i] = why
        else:
            paired.append(i)
    return {"paired": paired, "excluded": excluded, "not_evaluable": not_eval}


REFERENCE_METRICS = ("fvd", "met3r", "delta")


def needs_reference(recs: dict, sub: str) -> bool:
    """Whether the tables of `sub` read reference rows: a model record of the subset holds FVD, MEt3R or delta (or is
    "ok" without a "metrics_done" list, the safe guess). Runs with only appearance metrics need none."""
    for rs in recs.values():
        for r in rs.get(sub, {}).values():
            done = r.get("metrics_done")
            if done is None and r.get("status") == "ok":
                return True
            if set(done or ()) & set(REFERENCE_METRICS):
                return True
    return False


def _scored_cross(met3r: dict, positions=None):
    """(T', P) cross-view MEt3R of the scored timesteps (window position 0, the conditioning frame, dropped)."""
    if not met3r or met3r.get("cross") is None:
        return None
    pos = met3r.get("positions")
    rows = np.asarray(met3r["cross"], np.float64)
    if rows.size == 0:
        return None
    keep = [k for k, q in enumerate(pos) if q != 0 and (positions is None or q in positions)]
    return rows[keep] if keep else None


def delta_main(rec: dict, source: str | None = None):
    """(metrics of the main point group, {group: metrics}) of a record's delta, or None when it has none.
    Model records: rec["delta"]["groups"]; reference records: rec["delta"][source]["groups"] (source "real" / "frozen")."""
    d = rec.get("delta") if rec else None
    if not d:
        return None
    part = d if source is None else d.get(source)
    if not part or not part.get("groups"):
        return None
    m = part["groups"].get(d.get("main_group"))
    if not m or m.get("any/delta_avg", "absent") == "absent":
        return None
    return m, part["groups"]


def _load_feats(base: Path, entry: dict):
    if not entry or not entry.get("file"):
        return None
    p = Path(base) / entry["file"]
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return np.asarray(z["feats"], np.float64)


def _subset_table(sub, clips, recs, refs, run_dirs, ref_dir, settings) -> dict:
    """All rows of one subset on the clip list `clips` (already paired)."""
    from .metrics.fvd import pooled_fvd
    from .metrics.met3r import delta_table
    names = list(recs)
    R = {n: recs[n].get(sub, {}) for n in names}
    rows = {n: {} for n in names}
    cols = {}

    for key, *_ in APPEARANCE_COLUMNS:
        hk = [c for c in clips if all((R[n][c].get("appearance") or {}).get(key) is not None for n in names)]
        cols[key] = len(hk)
        for n in names:
            rows[n][key] = mean_se([R[n][c]["appearance"][key] for c in hk])
    have = [c for c in clips if all((R[n][c].get("appearance") or {}).get("psnr") is not None for n in names)]
    for n in names:
        cp = [R[n][c].get("cond_psnr") for c in have if R[n][c].get("cond_psnr")]
        rows[n]["cond_psnr_min"] = mean_se([min(num(v) for v in x) for x in cp])
        rows[n]["cond_psnr_low"] = int(sum(1 for c in have if R[n][c].get("cond_psnr_ok") is False))

    cov = settings.get("fvd_cov", "unbiased")
    if ref_dir is not None:
        have = [c for c in clips if refs[c].get("fvd") and all(R[n][c].get("fvd") for n in names)]
        real = {c: _load_feats(ref_dir, refs[c]["fvd"]) for c in have}
        fake = {n: {c: _load_feats(run_dirs[n], R[n][c]["fvd"]) for c in have} for n in names}
        have = [c for c in have if real[c] is not None and all(fake[n][c] is not None and fake[n][c].shape == real[c].shape
                                                                  for n in names)]
        cols["fvd"] = len(have)
        for n in names:
            if len(have) and sum(len(real[c]) for c in have) >= 2:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = pooled_fvd([real[c] for c in have], [fake[n][c] for c in have], cov=cov)
                dim = int(real[have[0]].shape[1])
                rows[n]["fvd"] = {**f, "n_clips": len(have), "feature_dim": dim,
                                  "reliable": bool(f["n_real"] > dim and math.isfinite(f["fvd"]))}
            else:
                rows[n]["fvd"] = None

    met = {}
    if ref_dir is not None:
        per_clip = {}
        for c in clips:
            d = {REFERENCE_SOURCE: _scored_cross(refs[c].get("met3r"))}
            for n in names:
                d[n] = _scored_cross(R[n][c].get("met3r"))
            if all(v is not None for v in d.values()):
                per_clip[c] = d
        cols["met3r"] = len(per_clip)
        if per_clip:
            try:
                met = delta_table(per_clip, names, reference=REFERENCE_SOURCE)
                cols["met3r"] = len(met["_clips"])
            except ValueError as e:
                met = {"_error": str(e)}
                cols["met3r"] = 0
    for n in names:
        rows[n]["met3r"] = met.get(n)
    from .tasks.generation import DEFAULT_MET3R_BATCH, DEFAULT_MET3R_CHUNK
    from .legacy import MET3R_LEGACY
    want = (MET3R_LEGACY["batch"], MET3R_LEGACY["feature_chunk"]) if settings.get("legacy") else \
        (DEFAULT_MET3R_BATCH, DEFAULT_MET3R_CHUNK)
    other = {}
    used = list(met.get("_clips", [])) if met and "_clips" in met else []
    for src, get in [(REFERENCE_SOURCE, lambda c: refs[c])] + [(n, lambda c, n=n: R[n][c]) for n in names]:
        for c in used:
            m = get(c).get("met3r") or {}
            bc = (m.get("batch"), m.get("feature_chunk"))
            if None not in bc and tuple(bc) != want:
                other.setdefault(src, {}).setdefault(f"{bc[0]}/{bc[1]}", []).append(c)
    met3r_batching = {"default": list(want), "other": other}

    use_ref_delta = ref_dir is not None
    have = []
    for c in clips:
        ok = all(delta_main(R[n][c]) is not None for n in names)
        if use_ref_delta:
            ok = ok and all(delta_main(refs[c], s) is not None for s in ("real", "frozen"))
        if ok:
            have.append(c)
    cols["delta"] = len(have)

    def delta_rows(get):
        main = {k: mean_se([get(c)[0].get(k) for c in have]) for k in DELTA_KEYS}
        groups = {}
        for c in have:
            for g, m in get(c)[1].items():
                groups.setdefault(g, []).append(m)
        return main, {g: {k: mean_se([m.get(k) for m in v]) for k in DELTA_KEYS} for g, v in groups.items()}

    for n in names:
        rows[n]["delta"], rows[n]["delta_groups"] = delta_rows(lambda c, n=n: delta_main(R[n][c]))
    ref_rows = {}
    if use_ref_delta or met:
        ref_rows[REAL_ROW] = {}
        ref_rows[FROZEN_ROW] = {}
        if met and REFERENCE_SOURCE in met:
            ref_rows[REAL_ROW]["met3r"] = met[REFERENCE_SOURCE]
        if use_ref_delta and have:
            for name, src in ((REAL_ROW, "real"), (FROZEN_ROW, "frozen")):
                ref_rows[name]["delta"], ref_rows[name]["delta_groups"] = delta_rows(
                    lambda c, s=src: delta_main(refs[c], s))
    gets = [lambda c, n=n: delta_main(R[n][c]) for n in names]
    if use_ref_delta:
        gets += [lambda c, s=s: delta_main(refs[c], s) for s in ("real", "frozen")]
    for col, key in DELTA_SUB_COLUMNS.items():
        cols[col] = sum(1 for c in have if all(_defined(get(c)[0].get(key)) for get in gets))
    main_groups = sorted({str((R[names[0]][c].get("delta") or {}).get("main_group")) for c in have}) if names else []
    out = {"n": len(clips), "columns_n": cols, "rows": {**ref_rows, **rows}, "main_groups": main_groups,
           "met3r_floor": met.get("_floor"), "met3r_noise_floor": met.get("_noise_floor"),
           "met3r_batching": met3r_batching}
    if "_error" in met:
        out["met3r_error"] = met["_error"]
    return out


def summarize_generation(run_dirs: dict, out=None, reference_dir=None, protocol=None, splits=(),
                         list_ids: bool = True) -> dict:
    """Combine the per-sample results of several models (and the shared reference rows) into tables.

    run_dirs: {model name: evaluate() out_dir}; names must not start with "_" and must not be "real".
    out: directory receiving summary.json and summary.md (None: nothing written).
    reference_dir: the reference directory given to evaluate() (the <hash> subdirectory is found automatically; an
        explicit directory that holds no reference rows of the runs' protocol raises FileNotFoundError, unless the runs
        hold only appearance metrics); None: <run_dir>/_reference or <run_dir>/../_reference, the first that holds
        this protocol (when none does, the reference-dependent columns are omitted and a warning is written, unless
        the runs hold only appearance metrics, which need no reference rows).
    protocol: optional Protocol / preset name / file to check the runs against (the runs' protocol.json is used).
    splits: attribute filters, e.g. ["difficulty=hard", "category=vehicle", "!flag:no_target_motion"]; each adds a
        table per subset restricted to the matching paired clips (floors / FVD recomputed on them).

    Only records of the runs' protocol hash and prediction root are used (others: status "stale", excluded). The
    expected samples of a run are the union of its selection files (written by evaluate() before sharding); expected
    samples without a record are excluded as "not evaluated" and counted in "n_not_evaluated". "warnings" lists
    provenance mismatches between the records of the table (weights / options of a metric, data version), a missing
    reference directory, and subsets whose reference directory holds no row. Returns the summary dict (written as
    summary.json)."""
    runs = {str(k): Path(v) for k, v in dict(run_dirs).items()}
    if not runs:
        raise ValueError("no run directory given")
    for n in runs:
        if n.startswith("_") or n == REFERENCE_SOURCE:
            raise ValueError(f"model name {n!r} is reserved (names must not start with '_' or be '{REFERENCE_SOURCE}')")
    infos = {}
    for n, d in runs.items():
        p = d / "protocol.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: not a generation run directory (run evaluate() first)")
        infos[n] = read_json(p)
    hashes = {i["hash"] for i in infos.values()}
    if len(hashes) != 1:
        raise ValueError(f"the runs were evaluated with different protocols: "
                         f"{ {n: i['hash'] for n, i in infos.items()} }")
    h = hashes.pop()
    if protocol is not None:
        from .tasks.generation import protocol_hash, resolve_protocol
        info0 = next(iter(infos.values()))
        want = protocol_hash(resolve_protocol(protocol), info0.get("track_size"), info0.get("decoder"))
        if want != h:
            raise ValueError(f"the runs used protocol hash {h}, the given protocol has {want}")
    info = next(iter(infos.values()))
    settings_all = _table_settings(infos, protocol)
    warnings, notes = [], []
    recs = {n: mark_stale(load_records(d), h, infos[n].get("pred_root")) for n, d in runs.items()}
    need_ref = {sub: needs_reference(recs, sub) for sub in SUBSETS}
    ref_dir = resolve_reference_dir(reference_dir, h)
    if reference_dir is not None and ref_dir is None:
        if any(need_ref.values()):
            raise FileNotFoundError(f"reference directory {reference_dir} holds no reference rows of protocol hash {h} "
                                    f"(neither {Path(reference_dir) / h} nor a protocol.json with that hash)")
        notes.append(f"reference directory {reference_dir} holds no reference rows; the runs hold only appearance "
                     f"metrics, which need none")
    elif ref_dir is None:
        cands = [d / "_reference" for d in runs.values()] + [d.parent / "_reference" for d in runs.values()]
        for cand in cands:
            ref_dir = resolve_reference_dir(cand, h)
            if ref_dir is not None:
                break
        if ref_dir is None and any(need_ref.values()):
            warnings.append(f"no reference directory of protocol hash {h} found (searched "
                            f"{', '.join(sorted({str(c) for c in cands}))}): FVD, cross-view MEt3R and the reference rows are "
                            f"omitted; pass --reference-dir")
    refs_all = mark_stale(load_records(ref_dir, REFERENCE_TASK), h) if ref_dir is not None else {}
    sels = {n: load_selection(d, h) for n, d in runs.items()}
    have_sel = any(v is not None for v in sels.values())
    run_protocols = {n: {k: (i.get("protocol") or {}).get(k) for k in RUN_INPUT_FIELDS} for n, i in infos.items()}
    summary = {"task": TASK, "protocol": info.get("protocol"), "protocol_hash": h, "models": list(runs),
               "run_dirs": {n: str(d) for n, d in runs.items()}, "run_protocols": run_protocols,
               "pred_roots": {n: i.get("pred_root") for n, i in infos.items()},
               "reference_dir": None if ref_dir is None else str(ref_dir), "splits": list(splits),
               "selection_files": {n: v is not None for n, v in sels.items()}, "n_not_evaluated": 0,
               "warnings": warnings, "notes": notes, "subsets": {}}
    differ = [k for k in RUN_INPUT_FIELDS if k != "name" and len({json.dumps(v.get(k), sort_keys=True)
                                                                   for v in run_protocols.values()}) > 1]
    if differ:
        summary["notes"].append("the runs differ in model inputs (" + ", ".join(differ) + "): " + "; ".join(
            f"{n}: " + ", ".join(f"{k} {json.dumps(v.get(k))}" for k in differ) for n, v in run_protocols.items()))
    for sub in SUBSETS:
        expected = None
        if have_sel:
            expected = {}
            for v in sels.values():
                expected.update((v or {}).get(sub, {}))
        if not any(sub in r for r in recs.values()) and not expected:
            continue
        settings = settings_all.get(sub, {})
        use_ref = ref_dir is not None and need_ref[sub]
        refs = refs_all.get(sub, {}) if use_ref else {}
        sub_ref_dir = ref_dir if use_ref else None
        if use_ref and not refs and any(r.get("status") == "ok" for rs in recs.values()
                                        for r in rs.get(sub, {}).values()):
            warnings.append(f"{sub}: the reference directory {ref_dir} holds no reference row of this subset, so no "
                            f"clip is paired; compute them (`vpmwb generation reference`, or evaluate without "
                            f"--no-references)")
        cs = _clip_sets(sub, recs, refs, use_ref, expected=expected)
        not_evaluated = {}
        if expected:
            for n in runs:
                miss = sorted(i for i in expected if i not in recs[n].get(sub, {}) and i not in cs["not_evaluable"])
                if miss:
                    not_evaluated[n] = miss
        n_ne = sum(len(v) for v in not_evaluated.values())
        summary["n_not_evaluated"] += n_ne
        S = {"n_records": len(set().union(*[set(r.get(sub, {})) for r in recs.values()])),
             "n_expected": None if expected is None else len(set(expected) | set().union(
                 *[set(r.get(sub, {})) for r in recs.values()])),
             "n_not_evaluated": n_ne, "n_not_evaluable": len(cs["not_evaluable"]), "n_paired": len(cs["paired"]),
             "n_excluded": len(cs["excluded"]), "settings": settings,
             "status_counts": {n: _status_counts(recs[n].get(sub, {})) for n in runs},
             "table": _subset_table(sub, cs["paired"], recs, refs, runs, sub_ref_dir, settings), "splits": {}}
        if list_ids:
            S["paired_ids"] = cs["paired"]
            S["excluded"] = cs["excluded"]
            S["not_evaluable"] = cs["not_evaluable"]
            S["not_evaluated"] = not_evaluated
        for spec in splits:
            cs2 = _clip_sets(sub, recs, refs, use_ref, parse_split(spec), expected=expected)
            S["splits"][spec] = {"n_paired": len(cs2["paired"]), "n_excluded": len(cs2["excluded"]),
                                 "table": _subset_table(sub, cs2["paired"], recs, refs, runs, sub_ref_dir, settings)}
            if list_ids:
                S["splits"][spec]["excluded"] = cs2["excluded"]
        summary["subsets"][sub] = S
        warnings += [f"{sub}: {w}" for w in _provenance_warnings(
            {n: recs[n].get(sub, {}) for n in runs}, refs, cs["paired"])]
    if summary["n_not_evaluated"]:
        warnings.insert(0, f"{summary['n_not_evaluated']} expected sample record(s) are missing (never evaluated; "
                           f"per run and subset in subsets.<subset>.not_evaluated): re-run evaluate on the same "
                           f"selection to complete them")
    if out is not None:
        out = Path(out)
        write_json(out / "summary.json", summary)
        (out / "summary.md").write_text(format_markdown(summary))
    return summary


RUN_INPUT_FIELDS = ("name", "context_frames", "text", "camera_conditioning", "clip_filter")
PROVENANCE_METRICS = ("appearance", "fvd", "met3r", "delta")


def _provenance_warnings(recs: dict, refs: dict, clips) -> list:
    """Mismatches of the per-metric model identities (record["models"][metric] without "paths": weight fingerprints,
    numeric options) and of the data version between the records (models and reference) of the paired clips.
    Records written before provenance existed carry none and are not compared."""
    out = []
    srcs = [(n, rs) for n, rs in recs.items()] + [(REFERENCE_SOURCE, refs)]
    for metric in PROVENANCE_METRICS:
        seen = {}
        for name, rs in srcs:
            for c in clips:
                ident = ((rs.get(c) or {}).get("models") or {}).get(metric)
                if not ident:
                    continue
                key = json.dumps({k: v for k, v in ident.items() if k != "paths"}, sort_keys=True)
                seen.setdefault(key, {}).setdefault(name, 0)
                seen[key][name] += 1
        if len(seen) > 1:
            idents = [json.loads(k) for k in seen]
            keys = sorted({k for d in idents for k in d if len({json.dumps(x.get(k)) for x in idents}) > 1})
            out.append(f"{metric} computed with different weights / options ({', '.join(keys)}): " + "; ".join(
                ", ".join(f"{k}={d.get(k)}" for k in keys) + " in " + ", ".join(f"{n} x{c}" for n, c in who.items())
                for d, who in zip(idents, seen.values())))
    dv = {}
    for name, rs in srcs:
        for c in clips:
            v = (rs.get(c) or {}).get("data_version")
            if v is not None:
                dv.setdefault(v, {}).setdefault(name, 0)
                dv[v][name] += 1
    if len(dv) > 1:
        out.append("records of different data versions: " + "; ".join(
            f"{v} in " + ", ".join(f"{n} x{c}" for n, c in who.items()) for v, who in dv.items()))
    return out


def _table_settings(infos: dict, protocol=None) -> dict:
    """{subset: resolved settings} of the table: those of `protocol` (on the runs' track_size / decoder) when given,
    else the runs' protocol.json, whose summary-time settings (fvd_cov) must then agree."""
    info0 = next(iter(infos.values()))
    if protocol is not None:
        from .tasks.generation import resolve_protocol, subset_settings
        p = resolve_protocol(protocol)
        return {s: subset_settings(p, s, info0.get("track_size"), info0.get("decoder")) for s in SUBSETS}
    out = {}
    for sub in SUBSETS:
        per = {n: (i.get("settings") or {}).get(sub, {}) for n, i in infos.items()}
        covs = {n: s.get("fvd_cov", "unbiased") for n, s in per.items()}
        if len(set(covs.values())) > 1:
            raise ValueError(f"{sub}: the runs were evaluated with different fvd_cov {covs}; pass protocol= "
                             f"(--protocol) to choose one")
        out[sub] = per[next(iter(per))]
    return out


def _status_counts(recs: dict) -> dict:
    c = {}
    for r in recs.values():
        c[r.get("status", "?")] = c.get(r.get("status", "?"), 0) + 1
    return c


def _best(values, higher: bool, nd: int = 4):
    """Indices of the best values (ties at the displayed precision `nd` all count); None with fewer than 2 values."""
    v = [(i, num(x["mean"] if isinstance(x, dict) else x)) for i, x in enumerate(values) if x is not None]
    v = [(i, round(x, nd)) for i, x in v if not math.isnan(x)]
    if len(v) < 2:
        return None
    best = (max if higher else min)(x for _, x in v)
    return {i for i, x in v if x == best}


def _row_cells(table: dict, names: list) -> dict:
    """Formatted cells per row for the main subset table."""
    cells = {}
    model_rows = [r for r in names if r in table["rows"]]
    bold = {}
    for key, _, higher, nd in APPEARANCE_COLUMNS:
        bold[key] = _best([table["rows"][n].get(key) for n in model_rows], higher, nd)
    bold["fvd"] = _best([(table["rows"][n].get("fvd") or {}).get("fvd") for n in model_rows], False, 1)
    if any(not (table["rows"][n].get("fvd") or {}).get("reliable", True) for n in model_rows):
        bold["fvd"] = None
    bold["met3r"] = _best([(table["rows"][n].get("met3r") or {}).get("abs_delta") for n in model_rows], False, 4)
    for key, _, higher, scale, nd in DELTA_COLUMNS:
        vals = [(table["rows"][n].get("delta") or {}).get(key) for n in model_rows]
        bold[key] = _best([None if v is None else num(v.get("mean")) * scale for v in vals], higher, nd)

    def b(text, key, i):
        return f"**{text}**" if bold.get(key) and i in bold[key] and text != "–" else text

    for name, row in table["rows"].items():
        i = model_rows.index(name) if name in model_rows else None
        c = []
        c.append(b(_fvd_cell(row.get("fvd")), "fvd", i))
        for key, _, _, nd in APPEARANCE_COLUMNS:
            c.append(b(fmt(row.get(key), nd), key, i) if row.get(key) else "–")
        m = row.get("met3r")
        c.append(b(fmt({"mean": m["abs_delta"], "se": _se_or_none(m, "abs_delta_se")}, 4), "met3r", i)
                 if m and "abs_delta" in m else "–")
        d = row.get("delta") or {}
        for key, _, _, scale, nd in DELTA_COLUMNS:
            c.append(b(fmt(d.get(key), nd, scale=scale), key, i) if d.get(key) else "–")
        cells[name] = c
    return cells


def _fvd_cell(f) -> str:
    """FVD value; '–' when undefined; a dagger when there are no more windows than feature dimensions (singular
    covariance: the value is not a reliable estimate). sqrtm noise around 0 (identical sets) prints as 0.0."""
    if not f or f.get("fvd") is None or not math.isfinite(num(f["fvd"])):
        return "–"
    v = num(f["fvd"])
    return f"{0.0 if abs(v) < 0.05 else v:.1f}" + ("" if f.get("reliable", True) else "†")


def _se_or_none(m: dict, key: str):
    """The s.e. of a delta_table entry, or None when it rests on fewer than 2 clips (delta_table reports 0.0)."""
    return m.get(key) if m.get("n", 0) >= 2 else None


def _table_md(table: dict, names: list) -> list:
    head = (["Method", "FVD ↓"] + [f"{t} {'↑' if hi else '↓'}" for _, t, hi, _ in APPEARANCE_COLUMNS]
            + ["Cross-view MEt3R ↓"] + [f"{t} {'↑' if hi else '↓'}" for _, t, hi, _, _ in DELTA_COLUMNS])
    lines = ["| " + " | ".join(head) + " |", "|:--|" + "--:|" * (len(head) - 1)]
    for name, c in _row_cells(table, names).items():
        lines.append(f"| {name} | " + " | ".join(c) + " |")
    return lines


def _met3r_md(table: dict, names: list) -> list:
    rows = table["rows"]
    if not any((rows.get(n) or {}).get("met3r") for n in names):
        return []
    lines = ["| Method | MEt3R (raw) ↓ | gen − real | Cross-view MEt3R ↓ | clips below real |", "|:--|--:|--:|--:|--:|"]
    real = (rows.get(REAL_ROW) or {}).get("met3r")
    if real:
        lines.append(f"| {REAL_ROW} | {fmt({'mean': real['raw'], 'se': _se_or_none(real, 'raw_se')}, 4)} | – | – | – |")
    for n in names:
        m = rows[n].get("met3r")
        if not m:
            continue
        lines.append(f"| {n} | {fmt({'mean': m['raw'], 'se': _se_or_none(m, 'raw_se')}, 4)} | "
                     f"{fmt({'mean': m['delta'], 'se': _se_or_none(m, 'delta_se')}, 4, signed=True)} | "
                     f"{fmt({'mean': m['abs_delta'], 'se': _se_or_none(m, 'abs_delta_se')}, 4)} | "
                     f"{100 * m['below']:.0f}% |")
    fl, nf = table.get("met3r_floor"), table.get("met3r_noise_floor")
    if fl:
        lines.append(f"| floor (noise floor) | | | {fmt({'mean': fl['mean'], 'se': _se_or_none(fl, 'se')}, 4)} "
                     f"({fmt({'mean': nf['mean'], 'se': _se_or_none(nf, 'se')}, 4)}) | |")
    return lines


def _footnotes(sub: str, S: dict, table: dict) -> list:
    st = S.get("settings") or {}
    notes = []
    fvds = [r.get("fvd") for r in table["rows"].values() if r.get("fvd")]
    if fvds:
        f = fvds[0]
        per = st.get("fvd_windows") or "all"
        notes.append(f"FVD: I3D on {per} non-overlapping {st.get('fvd_window', 16)}-frame windows per view from the first "
                     f"generated frame, pooled over {f['n_real']} windows of {f.get('n_clips')} clips, "
                     f"{f['cov']} covariance."
                     + (f" † {f['n_real']} windows ≤ {f.get('feature_dim')} feature dimensions: singular covariance, "
                        f"not a reliable FVD." if not f.get("reliable", True) else ""))
    if st:
        lad = ", ".join(f"{100 * x:g}" for x in st.get("ladder_m", []))
        groups = ", ".join(table.get("main_groups", [])) or "–"
        notes.append(f"δ: point group {groups}; thresholds {lad} cm, survival {100 * st.get('survival_m', 0):g} cm; "
                     f"queries at t_ref = default GT points{' visible in ≥ 1 view' if st.get('query_requires_visible') else ''}; "
                     f"GT visibility = {st.get('gt_visibility')}; lifter {st.get('lifter')} on a "
                     f"{'x'.join(str(x) for x in st.get('track_size', []))} grid.")
        notes.append(f"MEt3R: cross-view pairs at window positions {st.get('met3r_positions', [])[1:]} "
                     f"({st.get('met3r_aspect')} to {st.get('met3r_size')}²); eval grid "
                     f"{'x'.join(str(x) for x in st.get('eval_size', []))} ({st.get('resize_mode')}).")
    low = {name: r.get("cond_psnr_low") for name, r in table["rows"].items() if r.get("cond_psnr_low")}
    if low:
        notes.append("I2V check: clips whose generated frame 0 is below 25 dB PSNR from the conditioning frame "
                     "(the model may not start from the given frame): "
                     + ", ".join(f"{k} {v}" for k, v in low.items()) + ".")
    mb = table.get("met3r_batching") or {}
    if mb.get("other"):
        d = mb.get("default") or []
        notes.append(f"MEt3R batching other than the protocol default {d[0]}/{d[1]} (pairs per forward / FeatUp chunk; "
                     f"CUDA OOM fallback or --met3r-batch; moves clip means slightly): "
                     + "; ".join(f"{src}: " + ", ".join(f"{len(v)} clip(s) at {k}" for k, v in o.items())
                                 for src, o in mb["other"].items()) + ".")
    fewer = _fewer_clips_note(table)
    if fewer:
        notes.append(fewer)
    infs = []
    for name, r in table["rows"].items():
        for key, title, *_ in DELTA_COLUMNS:
            k = ((r.get("delta") or {}).get(key) or {}).get("n_inf")
            if k:
                infs.append(f"{name} {title} {k}")
    if infs:
        notes.append("∞: clips with non-finite predicted tracks (failures) are kept in the mean; such clips per row and "
                     "column: " + ", ".join(infs) + ".")
    return notes


_SUB_COLUMN_NOTES = {"delta_dyn": "δ_avg dyn. n = {} (clips with dynamic tracks)",
                     "delta_mov": "δ mov. n = {} (clips with moving entries)"}


def _fewer_clips_note(table: dict) -> str:
    """'Columns on fewer clips: ...' for the columns of a table whose n differs from its paired clips (δ_avg dyn. /
    δ mov.: from the clips of the δ columns), or ''."""
    cols, n = table.get("columns_n", {}), table.get("n")
    odd = [_SUB_COLUMN_NOTES[k].format(v) if k in _SUB_COLUMN_NOTES else f"{k} n = {v}" for k, v in cols.items()
           if v != (cols.get("delta", n) if k in _SUB_COLUMN_NOTES else n)]
    return "Columns on fewer clips: " + ", ".join(odd) + "." if odd else ""


def format_markdown(summary: dict) -> str:
    """summary.md: an overview table (rows = methods, columns = subsets x {FVD, PSNR, cross-view MEt3R, δ_avg, δ mov.}) and per subset
    the full table, the MEt3R detail table and footnotes (clip counts, settings)."""
    names = summary["models"]
    p = summary.get("protocol") or {}
    rp = summary.get("run_protocols") or {}
    ctx = {n: v.get("context_frames") for n, v in rp.items()}
    ctx_txt = (f"{p.get('context_frames')} context frame(s)" if len(set(ctx.values())) <= 1 else
               "context frames per method: " + ", ".join(f"{n} {c}" for n, c in ctx.items()))
    refs_txt = (f"Reference rows: *{REAL_ROW}* (the real frames through the same metric) and *{FROZEN_ROW}* (query "
                f"points held still)." if summary.get("reference_dir") else
                "No reference directory: FVD, cross-view MEt3R and the reference rows are omitted.")
    L = ["# VP-WorldBench generation results", "",
         f"Protocol `{p.get('name', '?')}` (hash `{summary.get('protocol_hash')}`): conditioning frame t_ref = "
         f"{p.get('ref_frame')}, {ctx_txt}, {p.get('horizon')} scored frames "
         f"(step {p.get('frame_step')}). Mean ± s.e. over the paired clips (scored for every method); FVD pooled over "
         f"windows. ↑ / ↓: higher / lower is better; best generated method in bold. {refs_txt}", ""]
    for w in summary.get("warnings") or []:
        L += [f"> **Warning:** {w}", ""]
    for w in summary.get("notes") or []:
        L += [f"> Note: {w}", ""]
    subs = list(summary["subsets"])
    if subs:
        head = ["Method"]
        for s in subs:
            head += [f"{s} FVD ↓", "PSNR ↑", "Cross-view MEt3R ↓", "δ_avg ↑", "δ mov. ↑"]
        L += ["## Overview", "", "| " + " | ".join(head) + " |", "|:--|" + "--:|" * (len(head) - 1)]
        rows = [REAL_ROW, FROZEN_ROW] + names
        for r in rows:
            c = [r]
            present = False
            for s in subs:
                row = summary["subsets"][s]["table"]["rows"].get(r) or {}
                present |= bool(row)
                c.append(_fvd_cell(row.get("fvd")))
                c.append(fmt(row.get("psnr"), 2, se=False) if row.get("psnr") else "–")
                m = row.get("met3r")
                c.append(f"{m['abs_delta']:.4f}" if m and "abs_delta" in m else
                         (f"({m['raw']:.3f})" if m and "raw" in m else "–"))
                for key in ("any/delta_avg", "moving/delta_avg"):
                    c.append(fmt((row.get("delta") or {}).get(key), 1, se=False))
            if present:
                L.append("| " + " | ".join(c) + " |")
        L += ["", "Real-video MEt3R in the overview is the raw value (in parentheses); n paired clips: "
              + ", ".join(f"{s} {summary['subsets'][s]['n_paired']}" for s in subs) + ".", ""]
    for s in subs:
        S = summary["subsets"][s]
        T = S["table"]
        exp = (f"; {S['n_not_evaluated']} expected sample record(s) of {S['n_expected']} expected samples missing "
               f"(never evaluated, see `not_evaluated`)" if S.get("n_not_evaluated") else
               (f"; {S['n_expected']} expected" if S.get("n_expected") is not None else ""))
        L += [f"## {s}", "",
              f"{S['n_paired']} paired clips ({S['n_records']} with records{exp}; {S['n_not_evaluable']} not evaluable "
              f"under the protocol; {S['n_excluded']} excluded because a method or the reference has no result).", ""]
        if S["n_paired"]:
            L += _table_md(T, names) + [""]
            met = _met3r_md(T, names)
            if met:
                L += met + [""]
            L += [f"- {x}" for x in _footnotes(s, S, T)] + [""]
        else:
            L += ["No clip is scored for every method (see `excluded` in summary.json).", ""]
        for spec, sp in S.get("splits", {}).items():
            L += [f"### {s}, {spec} ({sp['n_paired']} paired clips)", ""]
            if not sp["n_paired"]:
                L += ["No paired clip in this split.", ""]
                continue
            L += _table_md(sp["table"], names) + [""]
            met = _met3r_md(sp["table"], names)
            if met:
                L += met + [""]
            fewer = _fewer_clips_note(sp["table"])
            if fewer:
                L += [f"- {fewer}", ""]
        counts = "; ".join(f"{n}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                           for n, c in S["status_counts"].items())
        L += [f"Per-model record status: {counts}.", ""]
    return "\n".join(L) + "\n"
