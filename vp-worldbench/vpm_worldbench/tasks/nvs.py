"""Task `nvs`: held-out-view synthesis (render a camera of the rig that the model never sees).

Task
    One view of every sample is held out as the target (default the last view: Simul-Robot view2, Simul-General view3,
    Real-Robot view1 = the ego camera, Real-General view3; `target_view`). Given the videos of the other views on the
    window t = ref_frame + j * frame_step, j = 0..horizon (default frames 16..80, 65 frames), their cameras, and the
    target camera (intrinsics and per-frame world-to-camera pose), render the target view at every window frame. For a
    leave-one-out study run the task once per target view with separate prediction / result folders (the exported inputs
    of one run never contain the target view's frames).

Inputs (`export_inputs`), per evaluable sample, in <out>/<Subset>/<id>/
    view{k}.mp4 | view{k}/frame_{t:04d}.png | nothing   window frames of the SOURCE views only (as `tasks.depth`;
                                                         frames="mp4" default, "png", "none")
    inputs.json        frame indices, fps; per source view: file, size, K at that size, w2c (n, 4, 4); target: view,
                       render size = the evaluation grid (W, H), K on that grid, w2c (n, 4, 4), native size and K,
                       and "fov_mask" (file name or null)
    target_fov_mask.png   (Real-General only) valid field of view of the target on the evaluation grid (255 = scored)
    <out>/manifest.jsonl, protocol.json, skipped.json

Prediction format
    <pred_root>/<Subset>/<id>/view{target}.mp4 on the window grid: frame j = GT frame window_indices[j] (frame 0 = the
    first window frame; at least horizon + 1 frames), any resolution (mapped to the evaluation grid with the protocol
    resize mode, default 'resize' = stretch). Optional <pred_root>/<Subset>/<id>/meta.json {"fps": f}: frames are taken
    by time ((t - ref_frame) / clip fps * f; nearest frame for f >= 12, linear blend below), see
    `vpm_worldbench.predictions`; f must be a finite number > 0. "first_frame" may only be "ref" ("future" is not
    accepted: no target frame is given). A missing video makes the sample "missing"; an unreadable video, a too-short
    one or a malformed meta.json "invalid" (common.STATUSES).

Ground truth: the held-out view's real video (every subset), decoded on the evaluation grid
(`predictions.load_reference`).

Metrics (`vpm_worldbench.metrics.appearance`): PSNR [dB], SSIM and LPIPS (AlexNet v0.1) per window frame on the
evaluation grid, averaged over the window frames of the sample (every window frame is scored: no frame of the target is
given), then mean +- s.e. over samples per subset. Pixels outside the target's valid field of view (Real-General's black
undistortion border, `tasks.common.fov_mask`) are set to black in both videos (they never count as errors). SSIM needs
scikit-image and LPIPS torch + lpips (pip install 'vpm-worldbench[metrics]'); `evaluate` checks them before the first
sample and stops with a one-line message (`missing_dependencies`). Turn them off with NVSProtocol(ssim=False /
lpips=False), on the command line `evaluate --no-ssim / --no-lpips`.

Default protocol (`NVSProtocol`): ref_frame 16, horizon 64, frame_step 1, target_view -1 (last view), the generation
evaluation grid (Simul-* 480x480, Real-Robot 848x480, Real-General 640x480) with resize mode 'resize', LPIPS on.
Per-subset fields (target_view, eval_size, resize_mode) accept {subset or alias: value}.

Baselines (`BASELINES`, sanity checks)
    nearest_view   per window frame, the source view whose camera centre is nearest to the target's (ties: lowest
                   index), resized to the evaluation grid
    oracle         the GT target video written losslessly: PSNR = 128.13 dB (the MSE floor), SSIM 1, LPIPS 0

Usage
    from vpm_worldbench.tasks import nvs
    nvs.export_inputs(bench, "inputs/nvs")
    nvs.run_baseline(bench, "preds/nearest", kind="nearest_view")
    nvs.evaluate(bench, "preds/my_model", "results/nvs/my_model")
    vpmwb nvs baseline --kind nearest_view --root /data/VP-WorldBench --pred preds/nearest
    vpmwb nvs evaluate --root /data/VP-WorldBench --pred preds/nearest --out results/nvs/nearest [--no-lpips]
    (--target-view k holds out view k instead of the last view: give the same value to export, baseline and evaluate,
    e.g. `vpmwb nvs baseline ... --target-view 0` then `vpmwb nvs evaluate ... --target-view 0`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..geometry import camera_center
from ..metrics.appearance import appearance
from ..predictions import has_prediction, load_prediction, load_reference, prediction_meta
from ..presets import SUBSET_DEFAULTS
from ..protocol import Protocol
from .common import (InvalidPrediction, MissingPrediction, TaskProtocol, as_size, baseline_args, evaluate_task,
                     export_view_frames, fov_mask, run_baseline_task, run_jobs, select_samples, summarize_records,
                     task_cli, write_json, write_manifest)

__all__ = ["NVSProtocol", "export_inputs", "evaluate", "evaluate_sample", "summarize", "missing_dependencies",
           "check_meta", "nearest_source_views", "write_nearest_view", "write_oracle", "run_baseline", "BASELINES",
           "main"]

TASK = "nvs"
NVS_EVAL_SIZE = {s: tuple(SUBSET_DEFAULTS[s]["eval_size"]) for s in SUBSET_DEFAULTS}
NVS_RESIZE_MODE = {s: SUBSET_DEFAULTS[s]["resize_mode"] for s in SUBSET_DEFAULTS}


@dataclass
class NVSProtocol(TaskProtocol):
    """Protocol of the nvs task (module docstring). target_view: an int (negative = from the end, -1 = last view) or
    {subset: int}; eval_size {subset: (W, H)}; resize_mode {subset: 'resize' | 'center_crop' | 'letterbox'} (subset
    aliases accepted; missing subsets use the defaults)."""
    SUBSET_FIELDS = ("target_view", "eval_size", "resize_mode")

    name: str = "nvs-v1"
    target_view: Union[int, dict] = -1
    eval_size: dict = field(default_factory=dict)
    resize_mode: dict = field(default_factory=dict)
    lpips: bool = True
    ssim: bool = True
    use_fov_mask: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.horizon < 1:
            raise ValueError("nvs needs horizon >= 1 (the window has horizon + 1 frames)")

    def eval_size_for(self, subset):
        return as_size(self.per_subset("eval_size", subset, NVS_EVAL_SIZE))

    def resize_mode_for(self, subset):
        return self.per_subset("resize_mode", subset, NVS_RESIZE_MODE)

    def target_for(self, sample) -> int:
        t = self.target_view.get(sample.subset, -1) if isinstance(self.target_view, dict) else self.target_view
        t = int(t)
        return t + sample.n_views if t < 0 else t

    def sources_for(self, sample) -> list:
        t = self.target_for(sample)
        return [v for v in self.views_for(sample) if v != t]

    def evaluable(self, sample) -> tuple:
        ok, why = super().evaluable(sample)
        if not ok:
            return ok, why
        t = self.target_for(sample)
        if not 0 <= t < sample.n_views:
            return False, f"target view {t} not in 0..{sample.n_views - 1}"
        if not self.sources_for(sample):
            return False, "no source view"
        return True, ""

    def generation_protocol(self) -> Protocol:
        """The equivalent generation `Protocol` (window grid, eval grid, resize mode) used by the prediction loaders."""
        return Protocol(name=self.name, ref_frame=self.ref_frame, horizon=self.horizon, frame_step=self.frame_step,
                        eval_size={k: tuple(v) for k, v in self.eval_size.items()}, resize_mode=dict(self.resize_mode))


_LPIPS_CACHE = {}


def _has_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_dependencies(protocol: NVSProtocol) -> Optional[str]:
    """A one-line message naming the optional packages the protocol's metrics need but the environment lacks (SSIM:
    scikit-image; LPIPS: torch + lpips) and the switches that turn them off; None when nothing is missing."""
    need, off = [], []
    if protocol.ssim and not _has_module("skimage"):
        need.append("SSIM needs scikit-image")
        off.append(("--no-ssim", "ssim=False"))
    if protocol.lpips and not (_has_module("torch") and _has_module("lpips")):
        need.append("LPIPS needs torch and lpips")
        off.append(("--no-lpips", "lpips=False"))
    if not need:
        return None
    return (f"{'; '.join(need)}: pip install 'vpm-worldbench[metrics]', or skip them with "
            f"{' '.join(f for f, _ in off)} (NVSProtocol({', '.join(k for _, k in off)}))")


def check_meta(pred_root, sample) -> dict:
    """The prediction's meta.json ({} without one) with the checks of every prediction loader
    (`predictions.prediction_meta`: a JSON object, "fps" a finite number > 0) and the nvs rule "first_frame" = "ref"
    ("future" is not accepted: no target frame is given). Raises InvalidPrediction otherwise."""
    f = Path(pred_root) / sample.subset / sample.id / "meta.json"
    try:
        meta = prediction_meta(pred_root, sample)
    except ValueError as e:
        raise InvalidPrediction(f"{f}: {e}") from e
    ff = meta.get("first_frame", "ref")
    if ff != "ref":
        raise InvalidPrediction(f"{f}: first_frame={ff!r}; nvs predictions start at the first window frame (\"ref\"; "
                                "\"future\" is not accepted)")
    return meta


def _load_errors() -> tuple:
    errors = (ValueError, OSError, EOFError, IndexError)
    try:
        import av
        errors += (av.error.FFmpegError,)
    except (ImportError, AttributeError):
        pass
    return errors


def _lpips(device: str):
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError as e:
            raise ImportError("LPIPS needs torch and lpips: pip install 'vpm-worldbench[metrics]' "
                              "(or NVSProtocol(lpips=False))") from e
    if device not in _LPIPS_CACHE:
        from ..metrics.appearance import LPIPS
        _LPIPS_CACHE[device] = LPIPS(device=device)
    return _LPIPS_CACHE[device]


def evaluate_sample(sample, pred_root, protocol: NVSProtocol, lpips_device: str = "auto") -> dict:
    """PSNR / SSIM / LPIPS of the held-out view of one sample. Returns {"status": "ok", "target_view", "source_views",
    "eval_size", "resize_mode", "fov_masked_px", "metrics": {"psnr", "ssim", "lpips"}, "per_frame": {...}}."""
    t = protocol.target_for(sample)
    gp = protocol.generation_protocol()
    path = Path(pred_root) / sample.subset / sample.id / f"view{t}.mp4"
    if not has_prediction(pred_root, sample, [t]):
        raise MissingPrediction(f"missing prediction {path}")
    check_meta(pred_root, sample)
    real = load_reference(sample, t, gp)
    try:
        fake = load_prediction(pred_root, sample, t, gp, ref_frame=None)
    except _load_errors() as e:
        raise InvalidPrediction(f"{path}: {type(e).__name__}: {e}") from e
    W, H = protocol.eval_size_for(sample.subset)
    mode = protocol.resize_mode_for(sample.subset)
    n_masked = 0
    if protocol.use_fov_mask:
        m = fov_mask(sample, t, (W, H), mode)
        if m is not None:
            real = real.copy()
            fake = fake.copy()
            real[:, ~m] = 0
            fake[:, ~m] = 0
            n_masked = int((~m).sum())
    lp = _lpips(lpips_device) if protocol.lpips else None
    a = appearance(real, fake, lpips=lp, ssim=protocol.ssim)
    metrics = {"psnr": a["psnr"]}
    if a["ssim"] is not None:
        metrics["ssim"] = a["ssim"]
    if a["lpips"] is not None:
        metrics["lpips"] = a["lpips"]
    return {"status": "ok", "target_view": t, "source_views": protocol.sources_for(sample), "eval_size": [W, H],
            "resize_mode": mode, "fov_masked_px": n_masked, "n_frames": a["n_frames"], "metrics": metrics,
            "per_frame": {"psnr": a["psnr_per_frame"], "ssim": a["ssim_per_frame"], "lpips": a["lpips_per_frame"]}}


def evaluate(bench, pred_root, out_dir, protocol: Optional[NVSProtocol] = None, subsets=None, ids=None,
             jobs: int = 1, overwrite: bool = False, strict: bool = False, progress: bool = True,
             lpips_device: str = "auto") -> dict:
    """Score <pred_root>/<Subset>/<id>/view{target}.mp4; writes <out_dir>/<Subset>/<id>.json per sample, run.json,
    protocol.json and summary.json|md (mean +- s.e. over samples per subset). Finished samples of the same run are
    reused while their prediction files are unchanged; another run in the same out_dir raises `common.RunDirError`
    unless overwrite=True (--overwrite). With jobs > 1 every process builds its own LPIPS network (on `lpips_device`:
    "auto" = cuda if available). Raises ImportError before the first sample when SSIM / LPIPS are on but their
    packages are missing (`missing_dependencies`; `vpmwb`: one line, exit status 2). Statuses: ok | skipped | missing |
    invalid | error. Returns the summary dict."""
    protocol = protocol or NVSProtocol()
    msg = missing_dependencies(protocol)
    if msg:
        raise ImportError(msg)
    return evaluate_task(TASK, evaluate_sample, protocol, NVSProtocol, bench, pred_root, out_dir, subsets, ids, jobs,
                         overwrite, strict, progress, summarize_fn=summarize, lpips_device=lpips_device)


def summarize(out_dir) -> dict:
    """(Re)build <out_dir>/summary.json and summary.md from the per-sample JSONs."""
    head = [("psnr", "PSNR [dB]", 3), ("ssim", "SSIM", 4), ("lpips", "LPIPS", 4)]
    notes = ["target = the held-out view (default the last view); every window frame is scored",
             "Real-General: pixels outside the valid undistorted field of view are not scored"]
    return summarize_records(out_dir, TASK, "VP-WorldBench - nvs (held-out view synthesis)", head, notes)


def export_sample(sample, protocol: NVSProtocol, out_dir, frames: str = "mp4", size=None, crf: int = 10,
                  lossless: bool = False) -> Optional[dict]:
    ok, _ = protocol.evaluable(sample)
    if not ok:
        return None
    d = Path(out_dir) / sample.subset / sample.id
    d.mkdir(parents=True, exist_ok=True)
    win = protocol.window_indices
    t = protocol.target_for(sample)
    sources = protocol.sources_for(sample)
    srec = {}
    for k in sources:
        r = export_view_frames(sample, k, win, d, frames, size, "resize", crf, lossless)
        r["w2c"] = sample.w2c(k, win).tolist()
        srec[f"view{k}"] = r
    W, H = protocol.eval_size_for(sample.subset)
    mode = protocol.resize_mode_for(sample.subset)
    fov_file = None
    m = fov_mask(sample, t, (W, H), mode) if protocol.use_fov_mask else None
    if m is not None:
        import cv2
        fov_file = "target_fov_mask.png"
        cv2.imwrite(str(d / fov_file), m.astype(np.uint8) * 255)
    target = {"view": f"view{t}", "size": [W, H], "resize_mode": mode, "K": sample.K(t, (W, H), mode).tolist(),
              "w2c": sample.w2c(t, win).tolist(), "native_size": list(sample.image_size(t)),
              "K_native": sample.K(t).tolist(), "fov_mask": fov_file}
    rec = {"task": TASK, "uid": sample.uid, "subset": sample.subset, "id": sample.id, "fps": sample.fps,
           "window_fps": sample.fps / protocol.frame_step, "frame_indices": win, "n_frames": len(win),
           "target": target, "sources": srec, "dir": f"{sample.subset}/{sample.id}",
           "prediction": {"file": f"{sample.subset}/{sample.id}/view{t}.mp4", "frames": len(win),
                          "fps": sample.fps / protocol.frame_step, "size": [W, H]}}
    write_json(d / "inputs.json", rec)
    return rec


def _export_one(root, subset, sid, protocol_dict, out_dir, frames, size, crf, lossless):
    from ..data.sample import Sample
    s = Sample(root, subset, sid)
    p = NVSProtocol.from_dict(protocol_dict)
    rec = export_sample(s, p, out_dir, frames, size, crf, lossless)
    return rec if rec is not None else {"uid": s.uid, "skipped": p.evaluable(s)[1]}


def export_inputs(bench, out_dir, protocol: Optional[NVSProtocol] = None, subsets=None, ids=None, frames: str = "mp4",
                  size=None, crf: int = 10, lossless: bool = False, jobs: int = 1, progress: bool = True) -> dict:
    """Write the model inputs of every evaluable sample (module docstring) plus manifest.jsonl, protocol.json and
    skipped.json. Returns {"exported": n, "skipped": m}."""
    protocol = protocol or NVSProtocol()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [(str(bench.root), s.subset, s.id, protocol.to_dict(), str(out_dir), frames, as_size(size), crf, lossless)
            for s in select_samples(bench, subsets, ids, protocol.clip_filter)]
    res = run_jobs(_export_one, args, jobs, desc="export nvs", progress=progress)
    recs = [r for r in res if "skipped" not in r]
    skipped = [r for r in res if "skipped" in r]
    for r in recs:
        r["target_view"] = r["target"]["view"]
        r["source_views"] = sorted(r["sources"])
    write_manifest(out_dir, recs, skipped, protocol, ("uid", "subset", "id", "target_view", "source_views", "fps",
                                                      "window_fps", "frame_indices", "dir"))
    return {"exported": len(recs), "skipped": len(skipped)}


def nearest_source_views(sample, protocol: NVSProtocol) -> list:
    """Per window frame, the source view whose camera centre is nearest to the target camera centre (ties: lowest
    view index)."""
    t = protocol.target_for(sample)
    sources = protocol.sources_for(sample)
    win = protocol.window_indices
    ct = camera_center(sample.w2c(t, win))
    cs = np.stack([camera_center(sample.w2c(k, win)) for k in sources])
    d = np.linalg.norm(cs - ct[None], axis=-1)
    return [sources[i] for i in np.argmin(d, axis=0)]


def _write(sample, pred_root, protocol: NVSProtocol, frames: np.ndarray, lossless: bool, crf: int) -> Path:
    from ..io.video import write_video
    t = protocol.target_for(sample)
    p = Path(pred_root) / sample.subset / sample.id / f"view{t}.mp4"
    return write_video(p, frames, fps=sample.fps / protocol.frame_step, crf=crf, lossless=lossless)


def write_nearest_view(sample, pred_root, protocol: NVSProtocol, lossless: bool = False, crf: int = 10) -> None:
    """Copy of the nearest source view (per frame) on the evaluation grid."""
    win = protocol.window_indices
    choice = nearest_source_views(sample, protocol)
    size = protocol.eval_size_for(sample.subset)
    mode = protocol.resize_mode_for(sample.subset)
    out = None
    for k in sorted(set(choice)):
        js = [j for j, c in enumerate(choice) if c == k]
        fr = sample.frames(k, [win[j] for j in js], size=size, mode=mode)
        if out is None:
            out = np.empty((len(win),) + fr.shape[1:], np.uint8)
        out[js] = fr
    _write(sample, pred_root, protocol, out, lossless, crf)


def write_oracle(sample, pred_root, protocol: NVSProtocol, lossless: bool = True, crf: int = 10) -> None:
    """The GT target view on the evaluation grid (lossless by default)."""
    gp = protocol.generation_protocol()
    _write(sample, pred_root, protocol, load_reference(sample, protocol.target_for(sample), gp), lossless, crf)


BASELINES = {"nearest_view": write_nearest_view, "oracle": write_oracle}


def run_baseline(bench, pred_root=None, kind: str = "nearest_view", protocol: Optional[NVSProtocol] = None,
                 subsets=None, ids=None, jobs: int = 1, progress: bool = True, **opts) -> dict:
    """Write the predictions of baseline `kind` ("nearest_view" | "oracle") for every evaluable sample. The call form
    of every task, run_baseline(bench, pred_root, kind="oracle", ...); the earlier run_baseline("oracle", bench,
    pred_root, ...) is still accepted."""
    bench, pred_root, kind = baseline_args(bench, pred_root, kind, opts)
    return run_baseline_task(kind, BASELINES, protocol or NVSProtocol(), NVSProtocol, bench, pred_root, subsets, ids,
                             jobs, progress, **opts)


def main(argv=None) -> int:
    """`vpmwb nvs {export,baseline,evaluate,summarize}` (`common.task_cli`; --target-view overrides the target;
    evaluate --no-lpips / --no-ssim turn the optional metrics off)."""
    return task_cli(argv, TASK, NVSProtocol, export_fn=export_inputs, evaluate_fn=evaluate, summarize_fn=summarize,
                    baseline_fn=run_baseline, baselines=BASELINES, default_baseline="nearest_view",
                    description="held-out-view synthesis (docs/TASKS.md)",
                    protocol_args=[("--target-view", "target_view", int,
                                    "held-out view index (negative = from the end; default -1 = last view)")],
                    evaluate_args=[("--lpips-device", "lpips_device", str, "auto",
                                    "LPIPS device: auto (cuda if available) | cpu | cuda")],
                    evaluate_flags=[("--no-lpips", "lpips", False, "skip LPIPS (needs torch + lpips)"),
                                    ("--no-ssim", "ssim", False, "skip SSIM (needs scikit-image)")])


if __name__ == "__main__":
    import sys

    from vpm_worldbench.cli import main as _vpmwb
    raise SystemExit(_vpmwb([TASK] + sys.argv[1:]))
