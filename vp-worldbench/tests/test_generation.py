"""Tests for vpm_worldbench.tasks.generation and vpm_worldbench.report.

CPU tests: a synthetic mini benchmark (tmp_path) and stand-in models with the real interfaces (LPIPS / I3D / MEt3R /
a DepthAnchoredLifter whose depth and tracker are fake), so the whole evaluate -> summarize pipeline, resumability,
statuses, legacy settings and the table arithmetic are checked without weights.

Opt-in GPU end-to-end check with the real models (CUDA, MVTracker, MoGe-2, MEt3R stack, cd-fvd, LPIPS):
    VPMWB_GEN_ROOT=<VP-WorldBench root>  VPMWB_GEN_PRED=<prediction root of one model>
    [VPMWB_GEN_IDS=uid,uid]  [VPMWB_GEN_LIFTER_KW='{"hub_repo": "...", "moge": "..."}']  [VPMWB_GEN_OUT=<dir>]
It scores the samples, checks that every record is complete, and that a lossless copy of the real frames ("oracle"
prediction) reproduces the reference rows exactly (cross-view MEt3R = 0, δ identical to the real-video row).
"""
import json
import math
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.constants import KIND_NAMES, SCHEMA_SAMPLE, SUBSETS
from vpm_worldbench.data.benchmark import Benchmark
from vpm_worldbench.geometry import image_map, inside, project, warp_image
from vpm_worldbench.lifters.base import DepthAnchoredLifter
from vpm_worldbench.predictions import load_reference
from vpm_worldbench.protocol import Protocol, legacy_prerelease
from vpm_worldbench import report as R
from vpm_worldbench.tasks import generation as G

try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed (MEt3R bookkeeping)")

TQ, HZ = 4, 16
W0, H0 = 64, 48


def _look_at(C, target=(0.0, 0.0, 0.0)):
    C = np.asarray(C, float)
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    E = np.eye(4)
    E[:3, :3] = np.stack([r, d, f])
    E[:3, 3] = -E[:3, :3] @ C
    return E


def make_sample(root: Path, subset: str, sid: str, V=2, T=24, N=12, groups=("object", "robot"), main="object",
                depth=True, seed=0, frames_offset=0):
    """Look-at cameras around the origin, linearly moving points, random occlusions, smooth frames, depth.zip (cm)."""
    import cv2
    from vpm_worldbench.io.video import write_video
    rng = np.random.default_rng(seed)
    d = root / subset / sid
    (d / "videos").mkdir(parents=True)
    K = np.array([[40.0, 0, (W0 - 1) / 2], [0, 40.0, (H0 - 1) / 2], [0, 0, 1]])
    Kd = K.copy()
    Kd[:2, 2] += 0.5
    centres = np.array([(3.0, 0.2, 1.0), (0.3, 3.0, 1.2), (-3.0, -0.4, 0.9)])[:V]
    w2c = np.stack([np.repeat(_look_at(c)[None], T, 0) for c in centres])
    xyz0 = rng.uniform(-0.4, 0.4, (N, 3))
    vel = rng.normal(0, 0.012, (N, 3))
    vel[: N // 4] = 0.0
    xyz = (xyz0[None] + vel[None] * np.arange(T)[:, None, None]).astype(np.float32)
    valid = np.ones((T, N), bool)
    infr = np.zeros((T, N, V), bool)
    for v in range(V):
        for t in range(T):
            uv, z = project(xyz[t], K, w2c[v, t])
            infr[t, :, v] = valid[t] & (z > 0.1) & inside(uv, (W0, H0))
    vis = infr & (rng.random((T, N, V)) > 0.2)
    vis[TQ, 0, :] = True
    vis[TQ, 1, :] = False
    gi = np.array([0] * (N // 2) + [1] * (N - N // 2), np.int16) if len(groups) > 1 else np.zeros(N, np.int16)
    np.savez(d / "tracks3d.npz", xyz=xyz, valid=valid, visible=vis, in_frustum=infr, default_gt=valid.copy(), group=gi,
             group_names=np.array(groups), part=np.zeros(N, np.int16), part_names=np.array(["p"]),
             kind=np.zeros(N, np.int8), kind_names=np.array(KIND_NAMES), side=np.full(N, -1, np.int8))
    np.savez(d / "cameras.npz", view_names=np.array([f"view{v}" for v in range(V)]),
             source_view_names=np.array([f"cam{v}" for v in range(V)]), image_size=np.array([[W0, H0]] * V, np.int32),
             K=np.repeat(K[None], V, 0), K_delivered=np.repeat(Kd[None], V, 0), w2c=w2c, static=np.ones(V, bool))
    (d / "captions.json").write_text(json.dumps({"caption": "points move."}))
    hf = float(np.degrees(2 * np.arctan(W0 / 2 / K[0, 0])))
    views = [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W0, "height": H0, "source_name": f"cam{v}",
              "hfov_deg": round(hf, 3)} for v in range(V)]
    to_src = {"A": [[100.0, 0, 0], [0, -100.0, 0], [0, 0, 100.0]], "b": [1200.0, -340.0, 15.0], "source_units": "cm"}
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid, "n_views": V,
            "n_frames": T, "fps": 10.0, "views": views, "world": {"to_source": to_src},
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": None, "labels": {}},
            "tracks": {"n_points": N, "groups": list(groups), "default_report_group": main},
            "caption": "points move.", "attributes": {"difficulty": "hard" if seed % 2 else "normal", "n": seed},
            "flags": ["few_default_queries_at_ref_frame:3"] if seed == 2 else [], "depth": None}
    if depth:
        (d / "depth").mkdir()
        with zipfile.ZipFile(d / "depth" / "depth.zip", "w") as z:
            for v in range(V):
                for t in range(T):
                    img = (np.full((H0, W0), 300 + 3 * v + t, np.uint16) + np.arange(W0, dtype=np.uint16)[None])
                    z.writestr(f"cam{v}/frame_{t:04d}.png", cv2.imencode(".png", img)[1].tobytes())
        meta["depth"] = {"file": "depth/depth.zip", "member": "{source_view}/frame_{t:04d}.png",
                         "view_member_names": [f"cam{v}" for v in range(V)], "dtype": "uint16", "unit_scale": 0.01,
                         "invalid_values": [0], "width": W0, "height": H0, "kind": "rendered (synthetic GT)"}
        meta["files"]["depth"] = "depth/depth.zip"
    (d / "sample.json").write_text(json.dumps(meta))
    yy, xx = np.mgrid[0:H0, 0:W0]
    for v in range(V):
        fr = np.zeros((T, H0, W0, 3), np.uint8)
        for t in range(T):
            base = (xx * 3 + yy * 2 + 7 * t + 40 * v + frames_offset) % 256
            fr[t] = np.stack([base, (base * 2) % 256, 255 - base], -1).astype(np.uint8)
        write_video(d / "videos" / f"view{v}.mp4", fr, fps=10.0, lossless=True)
    return d


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("vpmwb_gen")
    make_sample(root, "Simul-Robot", "a000", seed=1)
    make_sample(root, "Simul-Robot", "a001", seed=2)
    make_sample(root, "Simul-Robot", "a002", seed=3, T=12)
    make_sample(root, "Real-Robot", "r000", groups=("arm", "hand"), main="all", depth=False, seed=4)
    make_sample(root, "Real-General", "g000", V=3, groups=("dynamic_surface", "human_body"), main="all", seed=5)
    return Benchmark(root)


def proto(**kw):
    sizes = {s: (W0, H0) for s in SUBSETS}
    base = dict(name="synthetic", ref_frame=TQ, horizon=HZ, fvd_window=8, eval_size=sizes)
    base.update(kw)
    return Protocol(**base)


class FakeLPIPS:
    def __call__(self, real, fake, batch=16):
        return np.abs(real.astype(np.float64) - fake).mean((1, 2, 3)) / 255.0


class FakeI3D:
    info = {"model": "fake-i3d"}

    def features(self, windows):
        w = np.asarray(windows, np.float64)
        B = len(w)
        f = np.concatenate([w.mean((1, 2, 3)), w.std((1, 2, 3)),
                            w[:, :, :24:2, :24:2].mean(1).reshape(B, -1)[:, :394]], 1)
        assert f.shape == (B, 400)
        return f


class FakeMEt3R:
    """score_pairs on the prepared tensors: 0.4 + |mean(I1) - mean(I2)|; records its calls."""
    img_size = 256

    def __init__(self):
        self.calls = []
        self.feature_chunk = 8

    def score_pairs(self, images, pairs, batch=16):
        self.calls.append((tuple(images.shape), len(pairs), batch, self.feature_chunk))
        x = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
        m = x.mean((2, 3, 4))
        s = np.array([0.4 + abs(m[a] - m[b]) for a, b in pairs], np.float64)
        return s, np.full(len(pairs), 0.5)


class FakeLifter(DepthAnchoredLifter):
    """Constant 3 m depth; the 'tracker' keeps every query at its position plus a drift proportional to the frame
    brightness change since position 0 (so identical frames give identical tracks); optional NaN tracks."""
    name = "fake"
    rgb_scale_quirk = False

    def __init__(self, nan_tracks=0, **kw):
        super().__init__(**kw)
        self.nan_tracks = nan_tracks
        self.calls = []

    def estimate_depth(self, frames, fov_x):
        return np.full(frames.shape[:3], 3.0, np.float32)

    def run_tracker(self, frames, depths, K, w2c, queries):
        self.calls.append({"frames": frames.shape, "K": K.copy(), "w2c": w2c.copy(), "queries": queries.copy(),
                           "depth_valid": float((depths > 0).mean())})
        n = frames.shape[1]
        b = frames.astype(np.float64).mean((0, 2, 3, 4))
        drift = (b - b[0]) / 255.0
        xyz = np.repeat(queries[None, :, 1:], n, 0).astype(np.float32)
        xyz[..., 0] += drift[:, None].astype(np.float32)
        if self.nan_tracks:
            xyz[1:, : self.nan_tracks] = np.nan
        return xyz, np.ones((n, len(queries)), bool)


def fake_models(nan_tracks=0):
    m = G.Models(device="cpu")
    m.lpips, m.i3d, m.met3r, m.lifter = FakeLPIPS(), FakeI3D(), FakeMEt3R(), FakeLifter(nan_tracks)
    return m


def write_pred(pred_root, sample, frames, meta=None):
    from vpm_worldbench.io.video import write_video
    d = Path(pred_root) / sample.subset / sample.id
    d.mkdir(parents=True, exist_ok=True)
    for v, fr in enumerate(frames):
        write_video(d / f"view{v}.mp4", np.asarray(fr), fps=10.0, lossless=True)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return d


def real_window(sample, p):
    return np.stack([load_reference(sample, v, p) for v in range(sample.n_views)])


def write_models(bench, root, p):
    """'copy' = lossless real window; 'dim' = darker and drifting; both for every evaluable sample."""
    for s in bench.samples():
        if not p.evaluable(s)[0]:
            continue
        ref = real_window(s, p)
        write_pred(root / "copy", s, ref)
        dim = (ref.astype(np.int16) * 0.8 - np.arange(ref.shape[1])[None, :, None, None, None] * 2).clip(0, 255)
        write_pred(root / "dim", s, dim.astype(np.uint8))


def test_settings_default_and_legacy():
    p = Protocol()
    s = G.subset_settings(p, "Real-General")
    assert s["eval_size"] == [640, 480] and s["track_size"] == [640, 480] and s["decoder"] == "pyav"
    assert s["met3r_positions"] == [0] + list(range(4, 65, 4)) and not s["met3r_temporal"]
    assert s["fvd_windows"] is None and s["fvd_cov"] == "unbiased" and s["query_requires_visible"]
    assert G.subset_settings(p, "Real-Robot")["depth_anchor"] == "gt_points"
    lp = legacy_prerelease()
    rg = G.subset_settings(lp, "Real-General")
    assert rg["eval_size"] == [480, 480] and rg["track_size"] == [640, 480] and rg["decoder"] == "opencv"
    assert rg["fvd_windows"] == 2 and rg["fvd_cov"] == "biased" and rg["met3r_aspect"] == "resize"
    assert rg["depth_recipe"] == "_reproduce" and rg["met3r_temporal"] and rg["rgb_scale_quirk"]
    sg = G.subset_settings(lp, "Simul-General")
    assert sg["gt_visibility"] == "in_frustum" and not sg["query_requires_visible"] and sg["K_source"] == "delivered"
    assert sg["world"] == "source" and sg["fov"] == "hfov_deg" and sg["depth_min_m"] == 0.01 and sg["plain_K_scaling"]
    rr = G.subset_settings(lp, "Real-Robot")
    assert rr["fvd_cov"] == "biased" and rr["fov"] == "native_K" and rr["query_requires_visible"]
    assert G.subset_settings(p, "Simul-Robot", track_size={"Simul-Robot": (320, 320)})["track_size"] == [320, 320]
    with pytest.raises(ValueError):
        G.subset_settings(p, "Simul-Robot", decoder="ffmpeg")


def test_protocol_hash_tracks_effective_settings(tmp_path):
    h = G.protocol_hash(Protocol())
    assert h == G.protocol_hash("vpm-v1") == G.protocol_hash(Protocol(name="x", text="none", camera_conditioning=True,
                                                                      clip_filter={"difficulty": "hard"},
                                                                      context_frames=9, fvd_cov="biased"))
    explicit = Protocol(eval_size={"Simul-Robot": [480, 480], "Simul-General": (480, 480), "Real-Robot": (848, 480),
                                   "Real-General": (640, 480)}, resize_mode={"Real-Robot": "resize"}, fvd_windows=4,
                        delta_ladder_m={"Real-Robot": [0.01, 0.02, 0.05, 0.1, 0.2]}, depth_anchor={"Simul-Robot": "gt_depth"},
                        delta_group={"Simul-Robot": "object"})
    assert G.protocol_hash(explicit) == h
    assert G.protocol_hash(Protocol(), track_size={"Real-General": (640, 480)}, decoder="pyav") == h
    for other in (Protocol(horizon=32), Protocol(met3r_step=8), Protocol(eval_size={"Real-Robot": (640, 360)}),
                  Protocol(gt_visibility="in_frustum"), Protocol(fvd_windows=2), Protocol(fvd_window=8),
                  Protocol(delta_ladder_m={"Real-Robot": [0.01, 0.02]}), Protocol(views=[0, 1]), legacy_prerelease(),
                  Protocol(rgb_scale_quirk=True), Protocol(query_requires_visible=False)):
        assert G.protocol_hash(other) != h
    assert G.protocol_hash(Protocol(), decoder="opencv") != h
    assert G.protocol_hash(Protocol(), track_size={"Real-General": (320, 240)}) != h
    assert G.protocol_hash(legacy_prerelease()) == G.protocol_hash(replace(legacy_prerelease(), text="none"))
    cfg = Path(__file__).resolve().parents[1] / "configs"
    for name, same in (("vpm-v1", True), ("camera_control", True), ("hard_only", True), ("v2v_9", True),
                       ("half_rate", False), ("square_center_crop", False)):
        if (cfg / f"{name}.yaml").exists():
            assert (G.protocol_hash(str(cfg / f"{name}.yaml")) == h) is same, name
    assert G.fvd_starts(Protocol(), G.subset_settings(Protocol(), "Simul-Robot")) == [1, 17, 33, 49]
    assert G.fvd_starts(Protocol(horizon=8), G.subset_settings(Protocol(horizon=8), "Simul-Robot")) is None


def test_track_intrinsics_maps(bench):
    s = bench.sample("Simul-Robot/a000")
    st = G.subset_settings(proto(eval_size={"Simul-Robot": (32, 24)}), "Simul-Robot")
    K = G.track_intrinsics(s, 0, st)
    np.testing.assert_allclose(K, image_map((W0, H0), (32, 24)).apply_K(s.cameras.K[0]))
    st2 = dict(st, plain_K_scaling=True, K_source="delivered")
    K2 = G.track_intrinsics(s, 0, st2)
    np.testing.assert_allclose(K2[:2, :3], s.cameras.extra["K_delivered"][0][:2, :3] * 0.5)
    st3 = dict(st, track_size=[64, 48])
    np.testing.assert_allclose(G.track_intrinsics(s, 0, st3), s.cameras.K[0])
    st4 = dict(st, resize_mode="center_crop", eval_size=[24, 24], track_size=[48, 48])
    m1 = image_map((W0, H0), (24, 24), "center_crop")
    np.testing.assert_allclose(G.track_intrinsics(s, 0, st4), image_map((24, 24), (48, 48)).apply_K(m1.apply_K(s.cameras.K[0])))


def test_sample_case_queries_and_visibility(bench):
    s = bench.sample("Simul-Robot/a000")
    p = proto()
    c = G.SampleCase(s, p, G.subset_settings(p, s.subset))
    tr = s.tracks
    seen = tr.visible[TQ].any(-1)
    assert np.array_equal(c.query_index, np.nonzero(seen)[0]) and 1 not in c.query_index and 0 in c.query_index
    assert c.gt.shape == (HZ + 1, len(c.query_index), 3) and c.main_group == "object"
    assert list(c.groups) == ["object", "robot", "all"]
    np.testing.assert_array_equal(c.gt_vis, tr.visible[TQ:TQ + HZ + 1][:, c.query_index].any(-1))
    lp = proto(name="legacy-prerelease")
    sg = G.subset_settings(lp, "Simul-Robot")
    c2 = G.SampleCase(s, lp, sg)
    assert len(c2.query_index) == s.tracks.n_points
    assert c2.world_transform is not None and c2.fov_x == [v["hfov_deg"] for v in s.meta["views"]]
    s2 = bench.sample("Simul-Robot/a002")
    c3 = G.SampleCase(s2, p, G.subset_settings(p, s2.subset))
    assert not c3.ok and "window ends" in c3.reason


def test_anchor_depth_recipes(bench):
    s = bench.sample("Simul-Robot/a000")
    p = proto()
    c = G.SampleCase(s, p, G.subset_settings(p, s.subset))
    a = c.base_anchor()
    np.testing.assert_allclose(a.depth[0], s.depth(0, [TQ])[0])
    assert a.norm_depth is a.depth
    lp = proto(name="legacy-prerelease")
    c2 = G.SampleCase(s, lp, G.subset_settings(lp, s.subset))
    d2 = c2.anchor_depth()
    raw = np.full((H0, W0), 300 + TQ, np.float32) + np.arange(W0, dtype=np.float32)[None]
    np.testing.assert_array_equal(d2[0], raw / 100.0)
    r = bench.sample("Real-Robot/r000")
    c3 = G.SampleCase(r, p, G.subset_settings(p, r.subset))
    a3 = c3.base_anchor()
    assert a3.mode == "gt_points" and a3.points_vis.shape == (2, r.tracks.n_points) and a3.norm_depth is None


@needs_torch
def test_evaluate_end_to_end(bench, tmp_path):
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    ref_dir = tmp_path / "ref"
    runs = {}
    for name in ("copy", "dim"):
        m = fake_models()
        res = G.evaluate(bench, tmp_path / "pred" / name, tmp_path / "res" / name, p, reference_dir=ref_dir, models=m,
                         device="cpu", verbose=False)
        assert res["counts"]["Simul-Robot"] == {"ok": 2, "skipped": 1}
        assert res["counts"]["Real-Robot"] == {"ok": 1} and res["counts"]["Real-General"] == {"ok": 1}
        runs[name] = tmp_path / "res" / name
    h = G.protocol_hash(p)
    rec = json.loads((runs["copy"] / "Simul-Robot/a000.json").read_text())
    assert rec["status"] == "ok" and rec["metrics_done"] == list(G.METRICS) and rec["protocol_hash"] == h
    assert rec["appearance"]["psnr"] > 100 and rec["appearance"]["lpips"] == 0 and rec["cond_psnr_ok"]
    assert rec["appearance"]["n_frames"] == HZ and len(rec["appearance"]["per_frame"]["psnr"][0]) == HZ
    assert rec["fvd"]["n_windows"] == 2 * 2 and rec["fvd"]["starts"] == [1, 9]
    feats = np.load(runs["copy"] / rec["fvd"]["file"])["feats"]
    assert feats.shape == (4, 400) and feats.dtype == np.float32
    assert rec["met3r"]["positions"] == [0, 4, 8, 12, 16] and rec["met3r"]["scored_rows"] == [1, 2, 3, 4]
    assert rec["delta"]["main_group"] == "object" and set(rec["delta"]["groups"]) == {"object", "robot", "all"}
    ref = json.loads((ref_dir / h / "Simul-Robot/a000.json").read_text())
    assert ref["task"] == G.REFERENCE_TASK and ref["status"] == "ok"
    assert ref["met3r"]["cross"] == rec["met3r"]["cross"]
    assert ref["delta"]["real"]["groups"] == rec["delta"]["groups"]
    np.testing.assert_array_equal(np.load(ref_dir / h / ref["fvd"]["file"])["feats"], feats)
    dim = json.loads((runs["dim"] / "Simul-Robot/a000.json").read_text())
    assert dim["appearance"]["psnr"] < 30 and dim["delta"]["groups"]["object"]["any/ade"] > 0
    assert (ref_dir / h / "_norm/Real-Robot/r000.npz").exists()
    rr = json.loads((runs["dim"] / "Real-Robot/r000.json").read_text())
    rr_ref = json.loads((ref_dir / h / "Real-Robot/r000.json").read_text())
    assert rr["delta"]["lifter"]["scene_norm"] == rr_ref["delta"]["real"]["lifter"]["scene_norm"]
    assert rr["delta"]["lifter"]["scene_norm"]["from"] == "norm_depth"
    skipped = json.loads((runs["copy"] / "Simul-Robot/a002.json").read_text())
    assert skipped["status"] == "skipped" and "window" in skipped["reason"]

    res = G.evaluate(bench, tmp_path / "pred" / "copy", runs["copy"], p, reference_dir=ref_dir, models=fake_models(),
                     device="cpu", verbose=False)
    assert res["counts"]["Simul-Robot"] == {"cached": 2, "skipped": 1} and res["reference"]["computed"] == 0
    res = G.evaluate(bench, tmp_path / "pred" / "copy", runs["copy"], p, reference_dir=ref_dir, models=fake_models(),
                     device="cpu", verbose=False, ids=["a000"], metrics=("appearance",), overwrite=True)
    assert res["counts"] == {"Simul-Robot": {"ok": 1}}
    again = json.loads((runs["copy"] / "Simul-Robot/a000.json").read_text())
    assert again["met3r"] == rec["met3r"] and again["appearance"]["psnr"] == rec["appearance"]["psnr"]

    rng = np.random.default_rng(7)
    for i in ("a000", "a001"):
        f_real = rng.normal(size=(4, 3))
        np.savez(ref_dir / h / f"_fvd/Simul-Robot/{i}.npz", feats=f_real)
        np.savez(runs["copy"] / f"_fvd/Simul-Robot/{i}.npz", feats=f_real)
        np.savez(runs["dim"] / f"_fvd/Simul-Robot/{i}.npz", feats=rng.normal(1.0, 2.0, size=(4, 3)))
    out = tmp_path / "summary"
    summ = G.summarize(runs, reference_dir=ref_dir, out=out, splits=["difficulty=hard"])
    S = summ["subsets"]["Simul-Robot"]
    assert S["n_paired"] == 2 and S["n_not_evaluable"] == 1 and S["paired_ids"] == ["a000", "a001"]
    T = S["table"]
    real = np.concatenate([np.load(ref_dir / h / f"_fvd/Simul-Robot/{i}.npz")["feats"] for i in ("a000", "a001")])
    fake = np.concatenate([np.load(runs["dim"] / f"_fvd/Simul-Robot/{i}.npz")["feats"] for i in ("a000", "a001")])
    from vpm_worldbench.metrics.fvd import frechet
    assert T["rows"]["dim"]["fvd"]["fvd"] == pytest.approx(frechet(real, fake, cov="unbiased"))
    assert T["rows"]["dim"]["fvd"]["n_real"] == 8 and abs(T["rows"]["copy"]["fvd"]["fvd"]) < 1e-9
    ps = [json.loads((runs["dim"] / f"Simul-Robot/{i}.json").read_text())["appearance"]["psnr"] for i in ("a000", "a001")]
    assert T["rows"]["dim"]["psnr"]["mean"] == pytest.approx(np.mean(ps))
    assert T["rows"]["dim"]["psnr"]["se"] == pytest.approx(np.std(ps, ddof=1) / math.sqrt(2))
    assert T["rows"]["copy"]["met3r"]["abs_delta"] == 0.0
    md = []
    for i in ("a000", "a001"):
        g = np.asarray(json.loads((runs["dim"] / f"Simul-Robot/{i}.json").read_text())["met3r"]["cross"])[1:].mean()
        r = np.asarray(json.loads((ref_dir / h / f"Simul-Robot/{i}.json").read_text())["met3r"]["cross"])[1:].mean()
        md.append(abs(g - r))
    assert T["rows"]["dim"]["met3r"]["abs_delta"] == pytest.approx(np.mean(md))
    assert T["met3r_floor"]["n"] == 2
    assert T["rows"][R.REAL_ROW]["delta"]["any/delta_avg"] == T["rows"]["copy"]["delta"]["any/delta_avg"]
    assert T["rows"][R.FROZEN_ROW]["delta"]["any/delta_avg"]["n"] == 2
    assert set(T["rows"]["dim"]["delta_groups"]) == {"object", "robot", "all"}
    assert S["splits"]["difficulty=hard"]["n_paired"] == 1
    text = (out / "summary.md").read_text()
    assert "## Simul-Robot" in text and "| Real video |" in text and "| dim |" in text and "Cross-view MEt3R ↓" in text
    assert json.loads((out / "summary.json").read_text())["protocol_hash"] == h


@needs_torch
def test_several_models_in_one_pass(bench, tmp_path):
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    preds = {"copy": tmp_path / "pred/copy", "dim": tmp_path / "pred/dim"}
    m = fake_models()
    res = G.evaluate(bench, preds, tmp_path / "res", p, models=m, device="cpu", verbose=False,
                     subsets=["Simul-Robot", "Real-General"])
    assert res["counts"] == {n: {"Simul-Robot": {"ok": 2, "skipped": 1}, "Real-General": {"ok": 1}} for n in preds}
    assert res["reference"]["computed"] == 3
    h = G.protocol_hash(p)
    assert (tmp_path / "res/_reference" / h / "Simul-Robot/a000.json").exists()
    for n in preds:
        assert json.loads((tmp_path / "res" / n / "Simul-Robot/a000.json").read_text())["status"] == "ok"
        assert json.loads((tmp_path / "res" / n / "protocol.json").read_text())["hash"] == h
    assert len(m.met3r.calls) == 3 * 3
    summ = G.summarize({n: tmp_path / "res" / n for n in preds})
    assert summ["reference_dir"] == str(tmp_path / "res/_reference" / h)
    assert summ["subsets"]["Simul-Robot"]["n_paired"] == 2
    with pytest.raises(ValueError):
        G.evaluate(bench, {"_x": tmp_path / "pred/copy"}, tmp_path / "res2", p, models=m, device="cpu", verbose=False)


@needs_torch
def test_statuses_missing_invalid_and_pairing(bench, tmp_path):
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    s = bench.sample("Simul-Robot/a001")
    (tmp_path / "pred/dim/Simul-Robot/a001/view1.mp4").unlink()
    short = real_window(bench.sample("Real-Robot/r000"), p)[:, :10]
    write_pred(tmp_path / "pred/dim", bench.sample("Real-Robot/r000"), short)
    ref_dir = tmp_path / "ref"
    for name in ("copy", "dim"):
        G.evaluate(bench, tmp_path / "pred" / name, tmp_path / "res" / name, p, reference_dir=ref_dir,
                   models=fake_models(), device="cpu", verbose=False)
    r1 = json.loads((tmp_path / "res/dim/Simul-Robot/a001.json").read_text())
    assert r1["status"] == "missing" and "view1.mp4" in r1["reason"]
    r2 = json.loads((tmp_path / "res/dim/Real-Robot/r000.json").read_text())
    assert r2["status"] == "invalid" and "frames" in r2["reason"]
    summ = G.summarize({"copy": tmp_path / "res/copy", "dim": tmp_path / "res/dim"}, reference_dir=ref_dir)
    S = summ["subsets"]["Simul-Robot"]
    assert S["paired_ids"] == ["a000"] and any("dim: missing" in w for w in S["excluded"]["a001"])
    assert summ["subsets"]["Real-Robot"]["n_paired"] == 0
    write_pred(tmp_path / "pred/dim", s, (real_window(s, p) // 2))
    res = G.evaluate(bench, tmp_path / "pred/dim", tmp_path / "res/dim", p, reference_dir=ref_dir, models=fake_models(),
                     device="cpu", verbose=False, subsets=["Simul-Robot"])
    assert res["counts"]["Simul-Robot"] == {"cached": 1, "ok": 1, "skipped": 1}
    with pytest.raises(ValueError):
        R.summarize_generation({"real": tmp_path / "res/copy"})
    with pytest.raises(ValueError):
        G.evaluate(bench, tmp_path / "pred/dim", tmp_path / "res/dim", proto(horizon=8), reference_dir=ref_dir,
                   models=fake_models(), device="cpu", verbose=False)


@needs_torch
def test_prediction_meta_fps_and_future(bench, tmp_path):
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    ref = real_window(s, p)
    st = G.subset_settings(p, s.subset)
    import cv2
    fut = []
    for v in range(s.n_views):
        fr = np.repeat(ref[v, 1:], 2, 0)
        fr = np.concatenate([fr[:1], fr], 0)
        fut.append(np.stack([cv2.resize(f, (2 * W0, 2 * H0), interpolation=cv2.INTER_NEAREST) for f in fr]))
    write_pred(tmp_path / "p", s, fut, meta={"fps": 20.0, "first_frame": "future"})
    got, info = G.load_prediction_frames(tmp_path / "p", s, 0, p, st, ref[0, 0])
    assert got.shape == ref[0].shape and info["size"] == [2 * W0, 2 * H0]
    np.testing.assert_array_equal(got[0], ref[0, 0])
    assert np.abs(got[1:].astype(int) - ref[0, 1:]).mean() < 3
    (tmp_path / "p/Simul-Robot/a000/meta.json").write_text("{bad json")
    with pytest.raises(ValueError):
        G.load_prediction_frames(tmp_path / "p", s, 0, p, st, ref[0, 0])


@needs_torch
def test_nonfinite_tracks_are_failures(bench, tmp_path):
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, p))
    G.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, ids=["a000"], reference_dir=tmp_path / "ref",
               models=fake_models(nan_tracks=2), device="cpu", verbose=False, metrics=("delta",))
    rec = json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())
    d = rec["delta"]
    assert d["nonfinite_frac"] > 0 and d["groups"]["all"]["any/ade"] == "inf"
    ref = json.loads((tmp_path / "ref" / G.protocol_hash(p) / "Simul-Robot/a000.json").read_text())
    assert ref["delta"]["real"]["nonfinite_frac"] > 0
    assert ref["delta"]["frozen"]["nonfinite_frac"] == 0


@needs_torch
def test_reference_only_and_shards(bench, tmp_path):
    p = proto()
    ref_dir = tmp_path / "ref"
    r0 = G.evaluate(bench, None, None, p, reference_dir=ref_dir, models=fake_models(), device="cpu", verbose=False,
                    shard="0/2")
    r1 = G.evaluate(bench, None, None, p, reference_dir=ref_dir, models=fake_models(), device="cpu", verbose=False,
                    shard="1/2")
    h = G.protocol_hash(p)
    files = sorted(str(x.relative_to(ref_dir / h)) for x in (ref_dir / h).glob("*/*.json"))
    assert files == ["Real-General/g000.json", "Real-Robot/r000.json", "Simul-Robot/a000.json", "Simul-Robot/a001.json"]
    assert r0["reference"]["computed"] + r1["reference"]["computed"] == 4
    assert not list((ref_dir / h).glob("*/.*.lock"))
    with pytest.raises(ValueError):
        G.parse_shard("2/2")
    store = G.ReferenceStore(tmp_path / "ref2", h)
    s = bench.sample("Simul-Robot/a000")
    assert store.acquire(s) and not store.acquire(s)
    store.lock(s).write_text(f"{G.socket.gethostname()} 999999999 {0}")
    assert store.acquire(s)
    store.release(s)


@needs_torch
def test_legacy_mode_runs_and_switches(bench, tmp_path):
    lp = proto(name="legacy-prerelease", rgb_scale_quirk=True, plain_K_scaling=True, reference_frames="_reproduce",
               fvd_window=4)
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, lp))
    m = fake_models()
    G.evaluate(bench, tmp_path / "pred", tmp_path / "res", lp, ids=["a000"], reference_dir=tmp_path / "ref", models=m,
               device="cpu", verbose=False)
    rec = json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())
    assert rec["status"] == "ok", rec.get("reason")
    assert rec["decoder"] == "opencv" and rec["fvd"]["n_windows"] == 2 * 4
    assert rec["met3r"]["batch"] == 64 and rec["met3r"]["feature_chunk"] == 32 and "temporal" in rec["met3r"]
    assert m.met3r.calls[-1][1:] == (5 + 8, 64, 32)
    info = rec["delta"]["lifter"]
    assert info["world_transform"] is not None and rec["delta"]["depth_source"].startswith("depth.zip raw/100")
    assert m.lifter.rgb_scale_quirk is True
    call = m.lifter.calls[-1]
    Kd = s.cameras.extra["K_delivered"][0]
    np.testing.assert_allclose(call["K"][0, 0], Kd.astype(np.float32), rtol=0, atol=1e-6)
    assert rec["delta"]["groups"]["all"]["n_queries"] == s.tracks.n_points


@needs_torch
def test_cli_main(bench, tmp_path, monkeypatch, capsys):
    p = proto()
    p.save(tmp_path / "p.json")
    for uid in ("Real-General/g000", "Simul-Robot/a000", "Simul-Robot/a001"):
        s = bench.sample(uid)
        write_pred(tmp_path / "pred", s, real_window(s, p))
    monkeypatch.setattr(G, "get_models", lambda *a, **k: fake_models())
    common = ["--root", str(bench.root), "--protocol", str(tmp_path / "p.json"), "--device", "cpu", "--quiet"]
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"),
                   "--subsets", "Real-General", "Simul-Robot", "--ids", "g000,a000", "--metrics", "appearance", "fvd",
                   "--reference-dir", str(tmp_path / "ref")]) == 0
    assert json.loads((tmp_path / "res/Real-General/g000.json").read_text())["metrics_done"] == ["appearance", "fvd"]
    assert not (tmp_path / "res/Simul-Robot/a001.json").exists()
    assert G.main(["eval", *common, "--pred", f"a={tmp_path / 'pred'}", f"b={tmp_path / 'pred'}", "--out",
                   str(tmp_path / "multi"), "--ids", "g000", "a000"]) == 0
    for n in ("a", "b"):
        assert json.loads((tmp_path / "multi" / n / "Simul-Robot/a000.json").read_text())["status"] == "ok"
    assert G.main(["reference", *common, "--reference-dir", str(tmp_path / "ref2"), "--subsets", "rg"]) == 0
    h = G.protocol_hash(p)
    assert sorted(x.name for x in (tmp_path / "ref2" / h / "Real-General").glob("*.json")) == ["g000.json"]
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"),
                   "--ids", "g000", "--reference-dir", str(tmp_path / "ref")]) == 0
    capsys.readouterr()
    assert G.main(["summarize", "--run", f"m={tmp_path / 'res'}", "--reference-dir", str(tmp_path / "ref"),
                   "--out", str(tmp_path / "sum")]) == 0
    assert "## Real-General" in (tmp_path / "sum/summary.md").read_text()
    assert G.main(["summarize", "--runs", f"a={tmp_path / 'multi/a'}", f"b={tmp_path / 'multi/b'}",
                   "--out", str(tmp_path / "sum2"), "--split", "difficulty=hard"]) == 0
    summ = json.loads((tmp_path / "sum2/summary.json").read_text())
    assert summ["models"] == ["a", "b"] and summ["subsets"]["Simul-Robot"]["n_paired"] == 1
    for args, msg in ((["summarize", "--run", "nodir", "--out", str(tmp_path / "x")], "--run: expected NAME=DIR"),
                      (["summarize", "--run", "a=x", "a=y", "--out", str(tmp_path / "x")], "empty or repeated name"),
                      (["summarize", "--run", "a=", "--out", str(tmp_path / "x")], "--run: expected NAME=DIR"),
                      (["evaluate", *common, "--pred", "a", "b", "--out", str(tmp_path / "x")], "several roots need"),
                      (["evaluate", *common, "--pred", "a=x", "b", "--out", str(tmp_path / "x")], "either one prediction"),
                      (["evaluate", *common, "--pred", "a=x", "a=y", "--out", str(tmp_path / "x")], "repeated name")):
        capsys.readouterr()
        with pytest.raises(SystemExit) as e:
            G.main(args)
        err = capsys.readouterr().err
        assert e.value.code == 2 and f"vpmwb generation {args[0]}: error: " in err and msg in err, args
    assert not (tmp_path / "x").exists()
    assert G.main(["export", "--root", str(bench.root), "--out", str(tmp_path / "inputs"), "--subsets", "Real-Robot",
                   "--protocol", str(tmp_path / "p.json")]) == 0
    assert (tmp_path / "inputs/manifest.jsonl").exists()
    capsys.readouterr()
    with pytest.raises(SystemExit):
        G.main(["evaluate", "--help"])
    assert "vpmwb generation evaluate" in capsys.readouterr().out


@needs_torch
def test_records_are_deterministic_and_resumable(bench, tmp_path):
    """Two evaluations of the same predictions give identical records apart from the "runtime" section."""
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    ref = real_window(s, p)
    write_pred(tmp_path / "pred", s, (ref.astype(np.int16) * 9 // 10).astype(np.uint8))
    for k in (1, 2):
        G.evaluate(bench, tmp_path / "pred", tmp_path / f"res{k}", p, ids=["a000"], reference_dir=tmp_path / f"ref{k}",
                   models=fake_models(), device="cpu", verbose=False)
    a, b = (json.loads((tmp_path / f"res{k}/Simul-Robot/a000.json").read_text()) for k in (1, 2))
    assert set(a["runtime"]) >= {"seconds", "gpu_peak_gib", "torch_state", "host"} and "timing" not in a
    a.pop("runtime"), b.pop("runtime")
    assert a == b
    h = G.protocol_hash(p)
    ra, rb = (json.loads((tmp_path / f"ref{k}" / h / "Simul-Robot/a000.json").read_text()) for k in (1, 2))
    ra.pop("runtime"), rb.pop("runtime")
    assert ra == rb


@needs_torch
def test_sibling_runs_share_the_default_reference(bench, tmp_path):
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    r1 = G.evaluate(bench, tmp_path / "pred/copy", tmp_path / "runs/copy", p, subsets=["rg"], models=fake_models(),
                    device="cpu", verbose=False)
    r2 = G.evaluate(bench, tmp_path / "pred/dim", tmp_path / "runs/dim", p, subsets=["rg"], models=fake_models(),
                    device="cpu", verbose=False)
    assert r1["reference"]["computed"] == 1 and r2["reference"]["computed"] == 0
    assert (tmp_path / "runs/_reference" / G.protocol_hash(p) / "Real-General/g000.json").exists()
    summ = G.summarize({"copy": tmp_path / "runs/copy", "dim": tmp_path / "runs/dim"})
    assert summ["reference_dir"] == str(tmp_path / "runs/_reference" / G.protocol_hash(p))
    assert summ["subsets"]["Real-General"]["n_paired"] == 1
    r3 = G.evaluate(bench, None, None, p, reference_dir=tmp_path / "runs/_reference", models=fake_models(),
                    device="cpu", verbose=False, ids=["g000", "nope"])
    assert r3["reference"]["cached"] == 1 and r3["unmatched_ids"] == ["nope"]


@needs_torch
def test_prediction_decoding_stops_at_the_window(bench, tmp_path):
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    ref = real_window(s, p)
    st = G.subset_settings(p, s.subset)
    longer = np.concatenate([ref, ref[:, ::-1]], 1)
    write_pred(tmp_path / "p", s, longer)
    got, info = G.load_prediction_frames(tmp_path / "p", s, 0, p, st, ref[0, 0])
    assert info["n_decoded"] == HZ + 1 and np.array_equal(got, ref[0])
    for decoder in ("pyav", "opencv"):
        write_pred(tmp_path / "q", s, ref[:, :HZ])
        with pytest.raises(ValueError, match=f"has {HZ} frames, the window needs {HZ + 1}"):
            G.load_prediction_frames(tmp_path / "q", s, 0, p, dict(st, decoder=decoder), ref[0, 0])
    for bad in ({"fps": 0}, {"fps": "x"}, {"first_frame": "middle"}, [1, 2]):
        write_pred(tmp_path / "r", s, ref, meta=bad)
        with pytest.raises(ValueError):
            G.load_prediction_frames(tmp_path / "r", s, 0, p, st, ref[0, 0])
    assert G._frames_needed(p, 10.0, {}) == HZ + 1 and G._frames_needed(p, 10.0, {"first_frame": "future"}) == HZ
    assert G._frames_needed(p, 10.0, {"fps": 20.0}) == 2 * HZ + 2


@needs_torch
def test_failures_are_recorded_and_the_run_continues(bench, tmp_path):
    """Missing view, too short video, corrupt file, garbage (noise / black) frames, unreadable reference video."""
    import shutil
    p = proto()
    root = tmp_path / "bench"
    for uid in ("Simul-Robot/a000", "Simul-Robot/a001", "Real-Robot/r000", "Real-General/g000"):
        shutil.copytree(bench.root / uid, root / uid)
    b2 = Benchmark(root)
    rng = np.random.default_rng(3)
    preds = tmp_path / "pred"
    for s in b2.samples():
        ref = real_window(s, p)
        if s.uid == "Simul-Robot/a000":
            write_pred(preds, s, rng.integers(0, 256, ref.shape, dtype=np.uint8))
        elif s.uid == "Simul-Robot/a001":
            write_pred(preds, s, np.zeros((ref.shape[0], ref.shape[1], 24, 32, 3), np.uint8))
        elif s.uid == "Real-Robot/r000":
            write_pred(preds, s, ref)
            (preds / s.uid / "view1.mp4").write_bytes(b"not a video")
        else:
            write_pred(preds, s, ref)
    (root / "Real-General/g000/videos/view0.mp4").write_bytes(b"\x00" * 64)
    res = G.evaluate(b2, preds, tmp_path / "res", p, reference_dir=tmp_path / "ref", models=fake_models(),
                     device="cpu", verbose=False)
    assert res["counts"] == {"Simul-Robot": {"ok": 2}, "Real-Robot": {"invalid": 1}, "Real-General": {"error": 1}}
    a0 = json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())
    a1 = json.loads((tmp_path / "res/Simul-Robot/a001.json").read_text())
    assert not a0["cond_psnr_ok"] and a0["cond_psnr"][0] < 25 and a1["prediction"]["views"][0]["size"] == [32, 24]
    assert all(np.isfinite(x["appearance"]["psnr"]) for x in (a0, a1))
    r0 = json.loads((tmp_path / "res/Real-Robot/r000.json").read_text())
    assert r0["status"] == "invalid" and "cannot decode view1.mp4" in r0["reason"]
    g0 = json.loads((tmp_path / "res/Real-General/g000.json").read_text())
    assert g0["status"] == "error" and g0["reason"] and "traceback" in g0
    assert g0["attributes"] == {"difficulty": "hard", "n": 5} and g0["flags"] == []
    summ = G.summarize({"m": tmp_path / "res"}, reference_dir=tmp_path / "ref")
    assert summ["subsets"]["Simul-Robot"]["n_paired"] == 2 and summ["subsets"]["Real-General"]["n_paired"] == 0
    assert any("error" in w for w in summ["subsets"]["Real-General"]["excluded"]["g000"])
    m = fake_models()
    m.load = lambda *a, **k: (_ for _ in ()).throw(ImportError("no MEt3R"))
    with pytest.raises(G.ModelLoadError):
        G.evaluate(b2, preds, tmp_path / "res3", p, reference_dir=tmp_path / "ref3", models=m, device="cpu",
                   verbose=False, ids=["a000"])


@needs_torch
def test_short_horizon_skips_fvd(bench, tmp_path):
    p = proto(horizon=6)
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, p))
    G.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, ids=["a000"], reference_dir=tmp_path / "ref",
               models=fake_models(), device="cpu", verbose=False)
    rec = json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())
    assert rec["status"] == "ok" and "skipped" in rec["fvd"] and "fvd" in rec["metrics_done"]
    T = G.summarize({"m": tmp_path / "res"}, reference_dir=tmp_path / "ref")["subsets"]["Simul-Robot"]["table"]
    assert T["columns_n"]["fvd"] == 0 and T["rows"]["m"]["psnr"]["n"] == 1


@needs_torch
def test_summarize_fvd_covariance(bench, tmp_path):
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, p))
    for name, cov in (("u", "unbiased"), ("b", "biased")):
        G.evaluate(bench, tmp_path / "pred", tmp_path / name, replace(p, fvd_cov=cov), ids=["a000"],
                   reference_dir=tmp_path / "ref", models=fake_models(), device="cpu", verbose=False)
    h = G.protocol_hash(p)
    assert json.loads((tmp_path / "b/protocol.json").read_text())["hash"] == h
    runs = {"u": tmp_path / "u", "b": tmp_path / "b"}
    with pytest.raises(ValueError, match="fvd_cov"):
        G.summarize(runs, reference_dir=tmp_path / "ref")
    summ = G.summarize(runs, reference_dir=tmp_path / "ref", protocol=replace(p, fvd_cov="biased"))
    assert summ["subsets"]["Simul-Robot"]["settings"]["fvd_cov"] == "biased"
    with pytest.raises(ValueError, match="protocol hash"):
        G.summarize(runs, reference_dir=tmp_path / "ref", protocol=replace(p, horizon=8))


def test_legacy_reproduce_depth_uses_opencv_nearest(tmp_path):
    import cv2
    from types import SimpleNamespace
    from vpm_worldbench import legacy as L
    d = np.random.default_rng(0).uniform(1, 5, (2, 6, 6)).astype(np.float32)
    (tmp_path / "_reproduce/Real-General/C1").mkdir(parents=True)
    np.savez(tmp_path / "_reproduce/Real-General/C1/depth_cond.npz", depth=d, views=np.array(["a", "b"]), frame=16)
    s = SimpleNamespace(root=tmp_path, subset="Real-General", id="C1",
                        cameras=SimpleNamespace(source_view_names=np.array(["a", "b"])))
    out = L.reproduce_depth(s, [1], 16, size=(8, 6))
    np.testing.assert_array_equal(out[0], cv2.resize(d[1], (8, 6), interpolation=cv2.INTER_NEAREST))
    assert not np.array_equal(out[0], warp_image(d[1], image_map((6, 6), (8, 6)), interp="nearest"))
    with pytest.raises(ValueError):
        L.reproduce_depth(s, [0], 17)


def test_center_crop_restricts_queries_and_visibility_to_the_grid(bench):
    """resize_mode center_crop: a point visible in the native frame but outside the crop is neither a query nor a
    scored GT entry (points must project inside the track grid of a view, as in Sample.tracks2d)."""
    s = bench.sample("Simul-Robot/a000")
    p_resize = proto(eval_size={"Simul-Robot": (8, 48)})
    p_crop = proto(eval_size={"Simul-Robot": (8, 48)}, resize_mode={"Simul-Robot": "center_crop"})
    cr = G.SampleCase(s, p_resize, G.subset_settings(p_resize, s.subset))
    cc = G.SampleCase(s, p_crop, G.subset_settings(p_crop, s.subset))
    assert cr.in_grid is None and cc.in_grid is not None
    assert set(cc.query_index) < set(cr.query_index)
    tr = s.tracks
    W, H = cc.settings["track_size"]
    ins = np.zeros((len(cc.window), cc.n_queries), bool)
    for iv, v in enumerate(cc.views):
        for j, t in enumerate(cc.window):
            uv, z = project(tr.xyz[t][cc.query_index], cc.K[iv], s.w2c(v, [t])[0])
            ins[j] |= (z > 0) & inside(uv, (W, H)) & tr.visible[t][cc.query_index][:, v]
    assert ins[0].all() and not (cc.gt_vis & ~ins).any()
    assert G.protocol_hash(p_crop) != G.protocol_hash(replace(p_crop, resize_mode={}))
    assert "visibility_in_grid" not in G.hash_settings(Protocol(), "Real-Robot")
    assert G.hash_settings(p_crop, "Simul-Robot")["visibility_in_grid"] is True
    r = bench.sample("Real-Robot/r000")
    pr = proto(eval_size={"Real-Robot": (8, 48)}, resize_mode={"Real-Robot": "center_crop"})
    a = G.SampleCase(r, pr, G.subset_settings(pr, r.subset)).base_anchor()
    a0 = G.SampleCase(r, proto(), G.subset_settings(proto(), r.subset)).base_anchor()
    assert a.points_vis.sum() < a0.points_vis.sum() and not (a.points_vis & ~a0.points_vis).any()


@pytest.fixture()
def two_runs(bench, tmp_path):
    """copy / dim runs of the whole mini benchmark with a shared reference dir (fake models)."""
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    runs = {}
    for name in ("copy", "dim"):
        G.evaluate(bench, tmp_path / "pred" / name, tmp_path / "res" / name, p, reference_dir=tmp_path / "ref",
                   models=fake_models(), device="cpu", verbose=False)
        runs[name] = tmp_path / "res" / name
    return p, runs, tmp_path / "ref"


@needs_torch
def test_summary_reports_never_evaluated_stale_and_typos(bench, two_runs, tmp_path, capsys):
    p, runs, ref = two_runs
    h = G.protocol_hash(p)
    sel = list((runs["copy"] / "_selection").glob(f"{h}_*.json"))
    assert len(sel) == 1 and json.loads(sel[0].read_text())["n_samples"] == 5
    assert R.load_selection(runs["copy"], h)["Simul-Robot"]["a001"]["attributes"]["difficulty"] == "normal"
    for d in (runs["copy"], runs["dim"], ref / h):
        (d / "Simul-Robot/a001.json").unlink()
    rr = json.loads((runs["dim"] / "Real-Robot/r000.json").read_text())
    rr["protocol_hash"] = "000000000000"
    rr["appearance"]["psnr"] = 99.0
    R.write_json(runs["dim"] / "Real-Robot/r000.json", rr)
    summ = G.summarize(runs, reference_dir=ref, out=tmp_path / "sum", splits=["difficulty=normal"])
    S = summ["subsets"]["Simul-Robot"]
    assert S["n_paired"] == 1 and S["n_expected"] == 3 and S["n_not_evaluated"] == 2
    assert S["not_evaluated"] == {"copy": ["a001"], "dim": ["a001"]}
    assert S["excluded"]["a001"] == ["copy: not evaluated", "dim: not evaluated", "reference: missing"]
    assert S["splits"]["difficulty=normal"]["excluded"] == {"a001": S["excluded"]["a001"]}
    assert summ["n_not_evaluated"] == 2 and "expected sample record" in summ["warnings"][0]
    RR = summ["subsets"]["Real-Robot"]
    assert RR["n_paired"] == 0 and RR["status_counts"]["dim"] == {"stale": 1}
    assert any("protocol hash 000000000000" in w for w in RR["excluded"]["r000"])
    md = (tmp_path / "sum/summary.md").read_text()
    assert "**Warning:** 2 expected sample record(s)" in md and "2 expected sample record(s) of 3 expected" in md
    assert G.main(["summarize", "--run", f"copy={runs['copy']}", f"dim={runs['dim']}", "--reference-dir", str(ref),
                   "--out", str(tmp_path / "sum2"), "--require-complete"]) == 3
    assert (tmp_path / "sum2/summary.md").exists()
    capsys.readouterr()
    with pytest.raises(FileNotFoundError, match="holds no reference rows"):
        G.summarize(runs, reference_dir=tmp_path / "ref_typo")
    summ = G.summarize(runs, out=tmp_path / "sum3")
    assert summ["reference_dir"] is None and any("no reference directory" in w for w in summ["warnings"])
    assert R.REAL_ROW not in summ["subsets"]["Simul-Robot"]["table"]["rows"]
    assert "No reference directory" in (tmp_path / "sum3/summary.md").read_text()


@needs_torch
def test_summary_splits_use_any_record_with_attributes(bench, two_runs, tmp_path):
    """An error record without attributes (old layout) does not drop the clip from a split."""
    p, runs, ref = two_runs
    for n in ("copy", "dim"):
        R.write_json(runs[n] / "Simul-Robot/a001.json", {"task": "generation", "id": "a001", "subset": "Simul-Robot",
                                                          "protocol_hash": G.protocol_hash(p), "status": "error",
                                                          "reason": "OSError: x", "metrics_done": []})
    summ = G.summarize(runs, reference_dir=ref, splits=["difficulty=normal"])
    sp = summ["subsets"]["Simul-Robot"]["splits"]["difficulty=normal"]
    assert sp["n_paired"] == 0 and "a001" in sp["excluded"]


@needs_torch
def test_summary_appearance_columns_provenance_and_notes(bench, tmp_path):
    p = proto()
    write_models(bench, tmp_path / "pred", p)
    ref = tmp_path / "ref"
    kw = dict(reference_dir=ref, device="cpu", verbose=False, subsets=["Simul-Robot"])
    G.evaluate(bench, tmp_path / "pred/copy", tmp_path / "res/copy", p, models=fake_models(), **kw)
    G.evaluate(bench, tmp_path / "pred/dim", tmp_path / "res/dim", replace(p, context_frames=2),
               models=fake_models(), ids=["a000"], **kw)
    G.evaluate(bench, tmp_path / "pred/dim", tmp_path / "res/dim", replace(p, context_frames=2),
               models=fake_models(), ids=["a001"], lpips=False, **kw)
    runs = {"copy": tmp_path / "res/copy", "dim": tmp_path / "res/dim"}
    rec = json.loads((runs["dim"] / "Simul-Robot/a000.json").read_text())
    assert set(rec["models"]) == set(G.METRICS) and rec["data_version"] is None
    assert rec["models"]["fvd"]["model"] == "fake-i3d"
    summ = G.summarize(runs, reference_dir=ref, out=tmp_path / "sum")
    T = summ["subsets"]["Simul-Robot"]["table"]
    assert T["columns_n"]["psnr"] == 2 and T["columns_n"]["lpips"] == 1 and T["rows"]["dim"]["lpips"]["n"] == 1
    assert T["rows"]["copy"]["lpips"]["n"] == 1
    assert summ["warnings"] == [] and "context_frames" in summ["notes"][0]
    md = (tmp_path / "sum/summary.md").read_text()
    assert "lpips n = 1" in md and "context frames per method: copy 1, dim 2" in md
    rec = json.loads((runs["dim"] / "Simul-Robot/a001.json").read_text())
    rec["models"]["met3r"]["weights"] = "0123456789abcdef"
    rec["met3r"]["batch"] = 8
    rec["data_version"] = "vpm-worldbench/index/1.0 0.9"
    R.write_json(runs["dim"] / "Simul-Robot/a001.json", rec)
    rc = json.loads((runs["copy"] / "Simul-Robot/a001.json").read_text())
    R.write_json(runs["copy"] / "Simul-Robot/a001.json", dict(rc, data_version="vpm-worldbench/index/1.0 1.0"))
    summ = G.summarize(runs, reference_dir=ref, out=tmp_path / "sum")
    ws = summ["warnings"]
    assert any(w.startswith("Simul-Robot: met3r computed with different weights") and "weights=0123456789abcdef"
               in w for w in ws) and any("data versions" in w for w in ws)
    T = summ["subsets"]["Simul-Robot"]["table"]
    assert T["met3r_batching"] == {"default": [16, 8], "other": {"dim": {"8/8": ["a001"]}}}
    assert "dim: 1 clip(s) at 8/8" in (tmp_path / "sum/summary.md").read_text()


@needs_torch
def test_oom_like_failures_are_retried_once(bench, tmp_path):
    """A CUDA library that cannot create its handle after a memory-hungry stage (cuSOLVER in MoGe-2 after the legacy
    MEt3R) is retried once after releasing the allocator cache; a second failure is recorded."""
    err = RuntimeError("cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`")
    assert G._is_oom(err) and not G._is_oom(RuntimeError("shape mismatch"))
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, p))
    m = fake_models()
    calls = {"n": 0}
    real = m.lifter.estimate_depth

    def flaky(frames, fov_x):
        calls["n"] += 1
        if calls["n"] in (1, 4, 5):
            raise err
        return real(frames, fov_x)
    m.lifter.estimate_depth = flaky
    G.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, ids=["a000"], reference_dir=tmp_path / "ref", models=m,
               device="cpu", verbose=False, metrics=("delta",))
    ref = json.loads((tmp_path / "ref" / G.protocol_hash(p) / "Simul-Robot/a000.json").read_text())
    rec = json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())
    assert ref["status"] == "ok" and rec["status"] == "error" and "cusolverDnCreate" in rec["reason"]


@needs_torch
def test_weights_fingerprint_covers_non_persistent_buffers():
    lin = torch.nn.Linear(3, 4)
    a = G.weights_fingerprint(lin)
    w = lin.weight.detach().clone()
    del lin.weight
    lin.register_buffer("weight", w, persistent=False)
    assert "weight" not in lin.state_dict() and G.weights_fingerprint(lin) == a
    lin.weight.add_(1e-6)
    assert G.weights_fingerprint(lin) != a
    assert G.weights_fingerprint(torch.nn.ReLU()) is None and G.weights_fingerprint(object()) is None


def test_report_helpers():
    assert R.mean_se([1.0, None, 3.0]) == {"mean": 2.0, "se": 1.0, "n": 2, "n_inf": 0}
    m = R.mean_se([1.0, "inf", None, float("nan"), 2.0])
    assert math.isinf(m["mean"]) and math.isnan(m["se"]) and (m["n"], m["n_inf"]) == (3, 1)
    assert R.mean_se([None])["n_inf"] == 0
    assert R.fmt({"mean": 1.23456, "se": 0.1}, 2) == "1.23 ± 0.10" and R.fmt(None) == "–"
    hard = R.parse_split("difficulty=hard")
    assert hard({"attributes": {"difficulty": "hard"}}) and not hard({"attributes": {}})
    assert R.parse_split("hands_used=left")({"attributes": {"hands_used": ["left", "right"]}})
    assert R.parse_split("!flag:few_default")({"flags": []})
    assert not R.parse_split("!flag:few_default")({"flags": ["few_default_queries_at_ref_frame:3"]})
    with pytest.raises(ValueError):
        R.parse_split("difficulty")
    assert R.jsonable({"a": np.float32(np.nan), "b": np.inf, "c": (1, np.int64(2))}) == {"a": None, "b": "inf",
                                                                                         "c": [1, 2]}


def test_summary_counts_the_clips_of_the_motion_columns():
    """δ_avg dyn. / δ mov. are undefined on clips without dynamic tracks / moving entries: columns_n counts the clips
    they are on, the footnote prints that n (main and split tables), and infinite values are counted in n_inf."""
    def rec(dyn, mov, ade):
        g = {k: 50.0 for k in R.DELTA_KEYS}
        g.update({"dynamic/delta_avg": dyn, "moving/delta_avg": mov, "any/ade": ade})
        return {"status": "ok", "appearance": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.1},
                "delta": {"main_group": "all", "groups": {"all": g}}}

    clips = ["c0", "c1", "c2", "c3"]
    recs = {"m": {"Simul-Robot": {"c0": rec(10.0, 5.0, 0.1), "c1": rec(None, None, "inf"), "c2": rec(20.0, None, 0.2),
                                  "c3": rec(30.0, 7.0, 0.3)}},
            "k": {"Simul-Robot": {c: rec(1.0, 2.0, 0.1) for c in clips}}}
    recs["k"]["Simul-Robot"]["c3"] = rec(1.0, None, 0.1)
    T = R._subset_table("Simul-Robot", clips, recs, {}, {"m": Path("."), "k": Path(".")}, None, {})
    assert T["n"] == 4 and T["columns_n"]["psnr"] == 4 and T["columns_n"]["delta"] == 4
    assert T["columns_n"]["delta_dyn"] == 3 and T["columns_n"]["delta_mov"] == 1
    d = T["rows"]["m"]["delta"]
    assert d["dynamic/delta_avg"]["n"] == 3 and d["dynamic/delta_avg"]["mean"] == 20.0
    assert d["moving/delta_avg"]["n"] == 2 and math.isinf(d["any/ade"]["mean"]) and d["any/ade"]["n_inf"] == 1
    note = R._fewer_clips_note(T)
    assert note == ("Columns on fewer clips: δ_avg dyn. n = 3 (clips with dynamic tracks), δ mov. n = 1 (clips with "
                    "moving entries).")
    notes = R._footnotes("Simul-Robot", {"settings": {}}, T)
    assert note in notes and any(x.startswith("∞:") and "m ADE (cm) 1" in x for x in notes)
    T2 = {"n": 5, "columns_n": {"psnr": 5, "delta": 4, "delta_dyn": 4, "delta_mov": 2}}
    assert R._fewer_clips_note(T2) == "Columns on fewer clips: delta n = 4, δ mov. n = 2 (clips with moving entries)."
    assert R._fewer_clips_note({"n": 2, "columns_n": {"psnr": 2, "delta": 2, "delta_dyn": 2, "delta_mov": 2}}) == ""
    S = {"n_paired": 4, "n_records": 4, "n_expected": None, "n_not_evaluable": 0, "n_excluded": 0, "settings": {},
         "status_counts": {"m": {"ok": 4}, "k": {"ok": 4}}, "table": T,
         "splits": {"difficulty=hard": {"n_paired": 4, "n_excluded": 0, "table": T}}}
    md = R.format_markdown({"models": ["m", "k"], "protocol": {"name": "x"}, "subsets": {"Simul-Robot": S}})
    assert md.count(f"- {note}") == 2 and md.count("- ∞: clips with non-finite predicted tracks") == 1


def test_opencv_decoder_matches_pyav_closely(tmp_path):
    from vpm_worldbench.io.video import read_frames, write_video
    from vpm_worldbench.legacy import read_frames_opencv
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 255, (6, 48, 64, 3), dtype=np.uint8)
    write_video(tmp_path / "v.mp4", fr, fps=10.0, lossless=True)
    a = read_frames(tmp_path / "v.mp4", [5, 1])
    b = read_frames_opencv(tmp_path / "v.mp4", [5, 1])
    np.testing.assert_array_equal(a, b)
    with pytest.raises(IndexError):
        read_frames_opencv(tmp_path / "v.mp4", [9])


def test_prediction_stamp(bench, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    assert G.prediction_stamp(tmp_path, s) is None
    d = tmp_path / s.subset / s.id
    d.mkdir(parents=True)
    (d / "view0.mp4").write_bytes(b"x" * 10)
    a = G.prediction_stamp(tmp_path, s)
    (d / "notes.txt").write_text("other files do not count")
    assert G.prediction_stamp(tmp_path, s) == a
    (d / "meta.json").write_text("{}")
    b = G.prediction_stamp(tmp_path, s)
    assert b != a
    os.utime(d / "view0.mp4", ns=(10 ** 9, 10 ** 9))
    assert G.prediction_stamp(tmp_path, s) != b


@needs_torch
def test_changed_predictions_are_rescored(bench, tmp_path):
    """A record is reused only for the predictions it was scored from: another root needs overwrite, a prediction
    regenerated in place is scored again; records and run directories of older versions (no root / stamp) stay valid."""
    p = proto()
    s = bench.sample("Simul-Robot/a000")
    ref = real_window(s, p)
    noise = np.random.default_rng(0).integers(0, 256, ref.shape, dtype=np.uint8)
    write_pred(tmp_path / "gpA", s, ref)
    write_pred(tmp_path / "gpB", s, noise)
    out = tmp_path / "run"
    kw = dict(ids=["a000"], reference_dir=tmp_path / "ref", device="cpu", verbose=False, metrics=("appearance",))
    G.evaluate(bench, tmp_path / "gpA", out, p, models=fake_models(), **kw)
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    root_a = str((tmp_path / "gpA").resolve())
    assert rec["pred_root"] == root_a and rec["pred_stamp"] == G.prediction_stamp(tmp_path / "gpA", s)
    assert json.loads((out / "protocol.json").read_text())["pred_root"] == root_a
    psnr_a = rec["appearance"]["psnr"]
    with pytest.raises(G.RunDirError, match="predictions in"):
        G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), **kw)
    res = G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), overwrite=True, **kw)
    assert res["counts"] == {"Simul-Robot": {"ok": 1}}
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    assert rec["pred_root"] == str((tmp_path / "gpB").resolve()) and rec["appearance"]["psnr"] < psnr_a - 50
    write_pred(tmp_path / "gpB", s, ref)
    res = G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), **kw)
    assert res["counts"] == {"Simul-Robot": {"ok": 1}}
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    assert rec["appearance"]["psnr"] == psnr_a
    res = G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), **kw)
    assert res["counts"] == {"Simul-Robot": {"cached": 1}}
    rec.pop("pred_stamp")
    rec.pop("pred_root")
    rec["appearance"]["psnr"] = 1.0
    R.write_json(out / "Simul-Robot/a000.json", rec)
    info = json.loads((out / "protocol.json").read_text())
    info.pop("pred_root")
    R.write_json(out / "protocol.json", info)
    res = G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), **kw)
    assert res["counts"] == {"Simul-Robot": {"cached": 1}}
    assert json.loads((out / "Simul-Robot/a000.json").read_text()) == rec
    assert json.loads((out / "protocol.json").read_text())["pred_root"] == str((tmp_path / "gpB").resolve())
    summ = G.summarize({"m": out})
    assert summ["subsets"]["Simul-Robot"]["table"]["rows"]["m"]["psnr"]["mean"] == 1.0 and summ["warnings"] == []
    res = G.evaluate(bench, tmp_path / "gpB", out, p, models=fake_models(), **dict(kw, metrics=("appearance", "fvd")))
    assert res["counts"] == {"Simul-Robot": {"ok": 1}}
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    assert rec["appearance"]["psnr"] == 1.0 and rec["metrics_done"] == ["appearance", "fvd"] and rec["pred_stamp"]
    assert rec["pred_root"] == str((tmp_path / "gpB").resolve())


@needs_torch
def test_appearance_only_runs_and_stale_roots_in_summaries(bench, tmp_path):
    p = proto()
    h = G.protocol_hash(p)
    write_models(bench, tmp_path / "pred", p)
    kw = dict(device="cpu", verbose=False, metrics=("appearance",), subsets=["Simul-Robot"])
    for name in ("copy", "dim"):
        G.evaluate(bench, tmp_path / "pred" / name, tmp_path / "runs" / name, p, models=fake_models(), **kw)
    assert not (tmp_path / "runs/_reference").exists()
    runs = {"copy": tmp_path / "runs/copy", "dim": tmp_path / "runs/dim"}
    summ = G.summarize(runs, out=tmp_path / "sum")
    S = summ["subsets"]["Simul-Robot"]
    assert S["n_paired"] == 2 and S["table"]["rows"]["dim"]["psnr"]["n"] == 2 and summ["warnings"] == []
    assert summ["reference_dir"] is None and R.REAL_ROW not in S["table"]["rows"]
    assert "| dim |" in (tmp_path / "sum/summary.md").read_text()
    G._write_run_protocol(tmp_path / "runs/_reference" / h / "protocol.json", p, h, None, None, False)
    for rd in (None, tmp_path / "runs/_reference", tmp_path / "nowhere"):
        summ = G.summarize(runs, reference_dir=rd)
        assert summ["subsets"]["Simul-Robot"]["n_paired"] == 2 and summ["warnings"] == [], rd
    G.evaluate(bench, tmp_path / "pred/dim", runs["dim"], p, models=fake_models(), references=False,
               **dict(kw, metrics=("met3r",)))
    summ = G.summarize(runs)
    assert summ["subsets"]["Simul-Robot"]["n_paired"] == 0
    assert any("holds no reference row of this subset" in w for w in summ["warnings"])
    with pytest.raises(FileNotFoundError, match="holds no reference rows"):
        G.summarize(runs, reference_dir=tmp_path / "nowhere")
    G.evaluate(bench, tmp_path / "pred/copy", runs["dim"], p, models=fake_models(), overwrite=True,
               **dict(kw, ids=["a000"]))
    summ = G.summarize({"dim": runs["dim"]})
    S = summ["subsets"]["Simul-Robot"]
    assert S["status_counts"]["dim"] == {"ok": 1, "skipped": 1, "stale": 1} and S["paired_ids"] == ["a000"]
    assert any("scored from the predictions in" in w for w in S["excluded"]["a001"])
    assert summ["pred_roots"] == {"dim": str((tmp_path / "pred/copy").resolve())}


def test_frames_needed_covers_the_resampled_window():
    """Decoding stops after _frames_needed frames; the window resampled from that head equals the one from the whole
    video, also where the float frame time falls just below an integer (29 / 25 * 25 = 28.999999999999996)."""
    from vpm_worldbench.predictions import _resample
    p = Protocol(ref_frame=4, horizon=64)
    assert G._frames_needed(Protocol(ref_frame=4, horizon=29), 25.0, {"fps": 25.0}) == 31
    idx = np.arange(400)
    full = np.zeros((400, 1, 2, 3), np.uint8)
    full[:, 0, 0, 0], full[:, 0, 1, 0] = idx % 256, idx // 256
    for fps_gt in (10.0, 15.0, 20.0, 25.0, 29.97, 30.0):
        for fps in (8.0, 10.0, 12.0, 23.976, 25.0, 30.0, 50.0, 60.0):
            for first in ("ref", "future"):
                meta = {"fps": fps, "first_frame": first}
                need = G._frames_needed(p, fps_gt, meta)
                np.testing.assert_array_equal(_resample(full[:need], p, fps_gt, meta, full[0]),
                                              _resample(full, p, fps_gt, meta, full[0]), err_msg=str(meta))


def test_ids_failed_runs_identity_and_dependencies(monkeypatch):
    assert G.canonical_ids(["sr/a000", "a001", "xx/b", "Real-General/C001"]) == \
        ["Simul-Robot/a000", "a001", "xx/b", "Real-General/C001"]
    assert G.failed_runs({"counts": {"Simul-Robot": {"missing": 2, "skipped": 1}}}, False) == ["missing 2"]
    assert G.failed_runs({"counts": {"Simul-Robot": {"ok": 1, "error": 3}}}, False) == []
    assert G.failed_runs({"counts": {"Simul-Robot": {"cached": 2, "skipped": 1}}}, False) == []
    assert G.failed_runs({"counts": {"a": {"Real-General": {"error": 1}}, "b": {"Real-General": {"ok": 1}}}},
                         True) == ["a: error 1"]
    assert G.failed_runs({"counts": {}, "reference": {"computed": 0, "error": 2}}, False) == ["reference rows: error 2"]
    assert G.resolve_device("cpu") == "cpu" and G.resolve_device("auto") in ("cuda", "cpu")
    m = G.Models(device="cpu")
    m.met3r = type("M3", (), {"img_size": 256, "raster": "naive", "tf32": True, "backbone": "mast3r", "model": None,
                              "featup_source": ("/x/FeatUp", "local")})()
    assert m.identity("met3r")["paths"]["featup"] == {"repo": "/x/FeatUp", "source": "local"}
    monkeypatch.setattr(G, "_has_module", lambda name: False)
    msgs = G.missing_dependencies(G.METRICS)
    assert len(msgs) == 5 and "scikit-image" in msgs[0] and "[metrics]" in msgs[1] and "[fvd]" in msgs[2]
    assert "metrics/met3r.py" in msgs[3] and "lifters/mvtracker.py" in msgs[4]
    assert G.missing_dependencies(G.METRICS, ssim=False, models=fake_models()) == []
    assert G.missing_dependencies(("appearance",), lpips=False, ssim=False) == []
    have = {"torch", "skimage", "lpips", "cdfvd", "featup"}
    monkeypatch.setattr(G, "_has_module", lambda name: name in have)
    msgs = G.missing_dependencies(G.METRICS, models=G.Models(device="cpu"))
    assert len(msgs) == 2 and "(not installed: pytorch3d, met3r)" in msgs[0] and "(not installed: moge)" in msgs[1]
    assert G.missing_dependencies(("delta",), models=G.Models(device="cpu", lifter="my.mod:Lifter")) == []
    monkeypatch.setattr(G, "_has_module", lambda name: True)
    assert G.missing_dependencies(G.METRICS) == []
    sha = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
    assert G._hub_commit(f"ethz-vlg/mvtracker:{sha}") == sha
    assert G._hub_commit("ethz-vlg/mvtracker") is None and G._hub_commit("ethz-vlg/mvtracker:main") is None


def test_missing_packages_stop_before_anything_is_written(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_has_module", lambda name: name != "skimage")
    with pytest.raises(G.ModelLoadError, match="scikit-image"):
        G.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto(), models=fake_models(), device="cpu",
                   verbose=False, metrics=("appearance",))
    assert not (tmp_path / "res").exists()
    monkeypatch.setattr(G, "_has_module", lambda name: name not in ("met3r", "moge"))
    with pytest.raises(G.ModelLoadError, match=r"MEt3R needs .*not installed: met3r.*; delta needs .*moge"):
        G.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto(), models=G.Models(device="cpu"),
                   device="cpu", verbose=False, metrics=("met3r", "delta"), reference_dir=tmp_path / "ref")
    assert not (tmp_path / "res").exists() and not (tmp_path / "ref").exists()


def test_unknown_depth_anchor_raises(bench):
    """A depth_anchor that is no anchor mode (e.g. the typo 'gt-depth') must not fall back to an unanchored lift."""
    p = proto()
    p.depth_anchor = {"Real-Robot": "gt-depth"}
    with pytest.raises(G.SettingsError, match="depth_anchor must be one of gt_depth, gt_points, none, got 'gt-depth'"):
        G.subset_settings(p, "rr")
    with pytest.raises(ValueError):
        G.protocol_hash(p)
    assert issubclass(G.SettingsError, ValueError) and G.subset_settings(p, "Simul-Robot")["depth_anchor"] == "gt_depth"
    s = bench.sample("Real-Robot/r000")
    st = dict(G.subset_settings(proto(), s.subset), depth_anchor="bogus")
    with pytest.raises(ValueError, match="depth_anchor must be one of"):
        G.SampleCase(s, proto(), st).base_anchor()
    st["depth_anchor"] = "none"
    assert G.SampleCase(s, proto(), st).base_anchor().mode == "none"


@needs_torch
def test_cli_exit_status_and_one_line_errors(bench, tmp_path, monkeypatch, capsys):
    p = proto()
    p.save(tmp_path / "p.json")
    s = bench.sample("Simul-Robot/a000")
    write_pred(tmp_path / "pred", s, real_window(s, p))
    monkeypatch.setattr(G, "get_models", lambda *a, **k: fake_models())
    common = ["--root", str(bench.root), "--protocol", str(tmp_path / "p.json"), "--quiet", "--metrics", "appearance"]
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"),
                   "--ids", "sr/a000"]) == 0
    assert json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())["status"] == "ok"
    capsys.readouterr()
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "nopred"), "--out", str(tmp_path / "res2"),
                   "--subsets", "sr"]) == 1
    assert "every evaluated sample failed (missing 2)" in capsys.readouterr().err
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res5"),
                   "--ids", "nope"]) == 1
    assert "no sample matches the selection" in capsys.readouterr().err
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res5"),
                   "--ids", "a000", "--shard", "1/2"]) == 0
    other_root = ["evaluate", *common, "--pred", str(tmp_path / "nopred"), "--out", str(tmp_path / "res"), "--ids", "a000"]
    assert G.main(other_root) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: ") and "predictions in" in err and len(err.strip().splitlines()) == 1
    from vpm_worldbench import cli
    monkeypatch.setenv("VPMWB_TRACEBACK", "0")
    assert cli.main(["generation", *other_root]) == 2
    assert capsys.readouterr().err.startswith("vpmwb: error: ")
    monkeypatch.setenv("VPMWB_TRACEBACK", "1")
    with pytest.raises(G.RunDirError, match="predictions in"):
        G.main(other_root)
    with pytest.raises(G.RunDirError):
        cli.main(["generation", *other_root])
    with pytest.raises(FileNotFoundError, match="not a generation run directory"):
        G.main(["summarize", "--run", f"m={tmp_path / 'nodir'}", "--out", str(tmp_path / "sum")])
    monkeypatch.delenv("VPMWB_TRACEBACK")
    assert isinstance(G.RunDirError("x"), ValueError) and isinstance(G.ModelLoadError("x"), RuntimeError)

    def unloadable(*a, **k):
        m = fake_models()
        m.load = lambda *a, **k: (_ for _ in ()).throw(ImportError("LPIPS needs torch and lpips:\n pip install x"))
        return m
    monkeypatch.setattr(G, "get_models", unloadable)
    for flags in ([], ["--raise-errors"]):
        assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res3"),
                       *flags]) == 2
        err = capsys.readouterr().err
        assert err.startswith("vpmwb: error: loading the metric models failed: ImportError: LPIPS") and \
            err.count("\n") == 1
    monkeypatch.setattr(G, "get_models", lambda *a, **k: fake_models())
    monkeypatch.setattr(G, "_has_module", lambda name: name != "skimage")
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res4")]) == 2
    assert "SSIM needs scikit-image" in capsys.readouterr().err
    assert G.main(["evaluate", *common, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res4"),
                   "--no-ssim"]) == 0
    base = ["evaluate", "--root", str(bench.root), "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "x")]
    bogus = proto()
    bogus.depth_anchor = {"Real-Robot": "gt-depth"}
    (tmp_path / "bogus.json").write_text(json.dumps(bogus.to_dict()))
    for bad in (["--protocol", str(tmp_path / "nope.yaml")], ["--subsets", "qq"], ["--metrics", "fvdd"],
                ["--shard", "3/2"], ["--shard", "x"], ["--track-size", "12x"], ["--track-size", "qq=64x48"],
                ["--root", str(tmp_path / "noroot")], ["--protocol", str(tmp_path / "bogus.json")]):
        capsys.readouterr()
        with pytest.raises(SystemExit) as e:
            G.main(base + bad)
        assert e.value.code == 2, bad
        assert "vpmwb generation evaluate: error: " in capsys.readouterr().err, bad
    assert not (tmp_path / "x").exists()
    with pytest.raises(SystemExit) as e:
        G.main(["evaluate", "--root", str(bench.root), "--pred", f"_x={tmp_path / 'pred'}", "--out", str(tmp_path)])
    assert e.value.code == 2
    capsys.readouterr()
    assert G.main(["summarize", "--run", f"m={tmp_path / 'nodir'}", "--out", str(tmp_path / "sum")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: ") and "not a generation run directory" in err and err.count("\n") == 1


RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")


@pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT to the VP-WorldBench data")
def test_released_data_cases():
    """vpm-v1 on the released data (CPU): every sample evaluable, queries / GT / cameras / anchors of 2 samples per
    subset, the query count of index.json, the shared key of configs/vpm-v1.yaml."""
    bench = Benchmark(RELEASED)
    p = Protocol()
    n = len(p.window_indices)
    for sub in SUBSETS:
        recs = bench.records(sub)
        assert recs and all(r["n_frames"] - 1 >= p.last_frame for r in recs)
        st = G.subset_settings(p, sub)
        for r in recs[:2]:
            s = bench.sample(sub, r["id"])
            c = G.SampleCase(s, p, st)
            assert c.ok and c.n_queries == r["queries_at_ref_frame"] > 0
            assert c.gt.shape == (n, c.n_queries, 3) and np.isfinite(c.gt[0]).all() and c.gt_vis[0].all()
            assert c.K.shape == (s.n_views, 3, 3) and c.w2c.shape == (s.n_views, n, 4, 4)
            assert c.main_group in c.groups and c.groups[c.main_group].any()
            a = c.base_anchor()
            if st["depth_anchor"] == "gt_depth":
                assert a.depth.shape == (s.n_views, st["track_size"][1], st["track_size"][0])
                assert (a.depth > 0).mean() > 0.5
            else:
                assert a.points_vis.shape == (s.n_views, s.tracks.n_points) and a.points_vis.any(1).all()
    cfg = Path(__file__).resolve().parents[1] / "configs" / "vpm-v1.yaml"
    if cfg.exists():
        assert G.protocol_hash(str(cfg)) == G.protocol_hash(p)


GEN_ROOT = os.environ.get("VPMWB_GEN_ROOT") or RELEASED
GEN_PRED = os.environ.get("VPMWB_GEN_PRED")


@pytest.mark.skipif(not (GEN_ROOT and GEN_PRED), reason="set VPMWB_GEN_ROOT (or VPMWB_RELEASED_ROOT) and "
                                                         "VPMWB_GEN_PRED (GPU, real models)")
def test_gpu_end_to_end_with_oracle(tmp_path):
    """Real models: a model's predictions, a lossless copy of the real frames ("oracle", must reproduce the reference
    rows exactly), uniform noise (scored without failure, flagged by the I2V check) and a repeated evaluation
    (identical records apart from "runtime")."""
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    out = Path(os.environ.get("VPMWB_GEN_OUT", tmp_path))
    ids = os.environ.get("VPMWB_GEN_IDS", "Real-Robot/episode_000").split(",")
    lk = json.loads(os.environ.get("VPMWB_GEN_LIFTER_KW", "{}"))
    bench = Benchmark(GEN_ROOT)
    p = Protocol()
    for uid in ids:
        s = bench.sample(uid)
        write_pred(out / "oracle", s, real_window(s, p))
    s0 = bench.sample(ids[0])
    ref0 = real_window(s0, p)
    write_pred(out / "noise", s0, np.random.default_rng(0).integers(0, 256, ref0.shape, dtype=np.uint8))
    kw = dict(reference_dir=out / "ref", ids=ids, lifter_kwargs=lk, verbose=True)
    G.evaluate(bench, GEN_PRED, out / "model", p, **kw)
    G.evaluate(bench, out / "oracle", out / "oracle_res", p, **kw)
    G.evaluate(bench, out / "noise", out / "noise_res", p, **dict(kw, ids=ids[:1]))
    G.evaluate(bench, GEN_PRED, out / "model_again", p, **dict(kw, ids=ids[:1], reference_dir=out / "ref_again"))
    h = G.protocol_hash(p)
    for uid in ids:
        sub, sid = uid.split("/")
        m = json.loads((out / "model" / sub / f"{sid}.json").read_text())
        o = json.loads((out / "oracle_res" / sub / f"{sid}.json").read_text())
        r = json.loads((out / "ref" / h / sub / f"{sid}.json").read_text())
        assert m["status"] == "ok" and o["status"] == "ok" and r["status"] == "ok"
        assert o["met3r"]["cross"] == r["met3r"]["cross"]
        assert o["delta"]["groups"] == r["delta"]["real"]["groups"]
        assert o["cond_psnr_ok"] and o["appearance"]["lpips"] < 1e-6
    sub, sid = ids[0].split("/")
    noise = json.loads((out / "noise_res" / sub / f"{sid}.json").read_text())
    assert noise["status"] == "ok" and not noise["cond_psnr_ok"] and math.isfinite(noise["met3r"]["mean"])
    assert noise["appearance"]["psnr"] < 15 and noise["delta"]["groups"][noise["delta"]["main_group"]]["n_queries"] > 0
    for a, b in ((out / "model", out / "model_again"), (out / "ref" / h, out / "ref_again" / h)):
        x, y = (json.loads((d / sub / f"{sid}.json").read_text()) for d in (a, b))
        x.pop("runtime"), y.pop("runtime")
        assert x == y, f"{a} vs {b}: records differ"
    summ = G.summarize({"model": out / "model", "oracle": out / "oracle_res"}, reference_dir=out / "ref",
                       out=out / "summary")
    for s in summ["subsets"].values():
        assert s["table"]["rows"]["oracle"]["met3r"]["abs_delta"] == 0.0
        assert s["n_not_evaluated"] == 0
    assert summ["warnings"] == [] and summ["n_not_evaluated"] == 0
    m = json.loads((out / "model" / sub / f"{sid}.json").read_text())
    r = json.loads((out / "ref" / h / sub / f"{sid}.json").read_text())
    assert m["data_version"] and m["data_version"] == r["data_version"]
    assert m["models"]["met3r"]["weights"] and m["models"]["fvd"]["weights_md5"]
    assert m["models"]["delta"]["tracker_weights"] and m["models"]["delta"]["moge_weights"]
    assert m["models"]["appearance"]["lpips_weights"]
    for k in ("fvd", "met3r", "delta"):
        assert m["models"][k] == r["models"][k]
    cfg = Path(__file__).resolve().parents[1] / "configs" / "square_center_crop.yaml"
    if cfg.exists():
        pc = Protocol.load(cfg)
        cid = next((u for u in ids if not u.startswith("Simul")), ids[0])
        csub, csid = cid.split("/")
        cs = bench.sample(cid)
        write_pred(out / "oracle_crop", cs, real_window(cs, pc))
        G.evaluate(bench, out / "oracle_crop", out / "crop_res", pc, **dict(kw, ids=[cid]))
        c = json.loads((out / "crop_res" / csub / f"{csid}.json").read_text())
        rc = json.loads((out / "ref" / G.protocol_hash(pc) / csub / f"{csid}.json").read_text())
        assert c["status"] == "ok" and rc["status"] == "ok" and c["eval_size"] == [480, 480]
        assert c["delta"]["groups"] == rc["delta"]["real"]["groups"] and c["met3r"]["cross"] == rc["met3r"]["cross"]
        case = G.SampleCase(cs, pc, G.subset_settings(pc, cs.subset))
        assert c["delta"]["n_queries"] == case.n_queries and case.in_grid[0][case.query_index].any(-1).all()
