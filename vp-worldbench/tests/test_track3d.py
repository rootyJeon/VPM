"""Tests for vpm_worldbench.tasks.track3d.

Synthetic mini benchmark (written to tmp_path) for everything; optional check on real data:
  VPMWB_RELEASED_ROOT=<VP-WorldBench root>      oracle / frozen on one sample per subset
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.constants import KIND_NAMES, SCHEMA_SAMPLE
from vpm_worldbench.data.benchmark import Benchmark
from vpm_worldbench.geometry import inside, project
from vpm_worldbench.metrics import tracking as trk
from vpm_worldbench.protocol import Protocol, legacy_prerelease
from vpm_worldbench.tasks import track3d as t3
from vpm_worldbench.tasks.common import InvalidPrediction, RunDirError
from vpm_worldbench.utils import UsageError

TQ, HZ = 4, 12


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


def make_sample(root: Path, subset: str, sid: str, V=2, T=24, N=12, W=64, H=48, groups=("object", "robot"),
                main="object", moving=False, invalid=False, depth=False, seed=0, video=True):
    """Static (or moving) look-at cameras around the origin, points moving linearly near the origin, random
    occlusions; `invalid` adds unknown positions and a stricter default_gt; `depth` adds a depth.zip."""
    import cv2
    from vpm_worldbench.io.video import write_video
    rng = np.random.default_rng(seed)
    d = root / subset / sid
    (d / "videos").mkdir(parents=True)
    K = np.array([[40.0, 0, (W - 1) / 2], [0, 40.0, (H - 1) / 2], [0, 0, 1]])
    centres = np.array([(3.0, 0.2, 1.0), (0.3, 3.0, 1.2), (-3.0, -0.4, 0.9)])[:V]
    w2c = np.zeros((V, T, 4, 4))
    for v in range(V):
        for t in range(T):
            w2c[v, t] = _look_at(centres[v] + (np.array([0.03 * t, -0.02 * t, 0.0]) if moving and v == 0 else 0))
    xyz0 = rng.uniform(-0.4, 0.4, (N, 3))
    vel = rng.normal(0, 0.012, (N, 3))
    vel[: N // 4] = 0.0
    xyz = (xyz0[None] + vel[None] * np.arange(T)[:, None, None]).astype(np.float32)
    valid = np.ones((T, N), bool)
    if invalid:
        valid[18:, -2:] = False
        valid[:3, -3] = False
    xyz[~valid] = np.nan
    infr = np.zeros((T, N, V), bool)
    for v in range(V):
        for t in range(T):
            uv, z = project(xyz[t], K, w2c[v, t])
            infr[t, :, v] = valid[t] & (z > 0.1) & inside(uv, (W, H))
    vis = infr & (rng.random((T, N, V)) > 0.2)
    vis[TQ, 0, :] = True
    vis[TQ, 1, :] = False
    vis[TQ, 1, 0] = False
    dgt = valid.copy()
    if invalid:
        dgt[8:11, -1] = False
    gi = np.array([0] * (N // 2) + [1] * (N - N // 2), np.int16) if len(groups) > 1 else np.zeros(N, np.int16)
    np.savez(d / "tracks3d.npz", xyz=xyz, valid=valid, visible=vis, in_frustum=infr, default_gt=dgt, group=gi,
             group_names=np.array(groups), part=np.zeros(N, np.int16), part_names=np.array(["p"]),
             kind=np.zeros(N, np.int8), kind_names=np.array(KIND_NAMES), side=np.full(N, -1, np.int8))
    np.savez(d / "cameras.npz", view_names=np.array([f"view{v}" for v in range(V)]),
             source_view_names=np.array([f"cam{v}" for v in range(V)]), image_size=np.array([[W, H]] * V, np.int32),
             K=np.repeat(K[None], V, 0), w2c=w2c, static=np.array([not (moving and v == 0) for v in range(V)]))
    (d / "captions.json").write_text(json.dumps({"caption": "points move."}))
    views = [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W, "height": H, "source_name": f"cam{v}"}
             for v in range(V)]
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid, "n_views": V,
            "n_frames": T, "fps": 10.0, "views": views,
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": None, "labels": {}},
            "tracks": {"n_points": N, "groups": list(groups), "default_report_group": main},
            "caption": "points move.", "attributes": {"difficulty": "normal"}, "flags": [], "depth": None}
    if depth:
        import zipfile
        (d / "depth").mkdir()
        with zipfile.ZipFile(d / "depth" / "depth.zip", "w") as z:
            for v in range(V):
                for t in range(T):
                    img = np.full((H, W), 3000 + 10 * v + t, np.uint16)
                    z.writestr(f"cam{v}/frame_{t:04d}.png", cv2.imencode(".png", img)[1].tobytes())
        meta["depth"] = {"file": "depth/depth.zip", "member": "{source_view}/frame_{t:04d}.png",
                         "view_member_names": [f"cam{v}" for v in range(V)], "dtype": "uint16", "unit_scale": 0.001,
                         "invalid_values": [0], "width": W, "height": H, "kind": "rendered (synthetic GT)"}
        meta["files"]["depth"] = "depth/depth.zip"
    (d / "sample.json").write_text(json.dumps(meta))
    if video:
        for v in range(V):
            fr = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
            write_video(d / "videos" / f"view{v}.mp4", fr, fps=10.0)
    return d


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("vpmwb")
    make_sample(root, "Simul-Robot", "a000", seed=1, depth=True)
    make_sample(root, "Simul-Robot", "a001", seed=2)
    make_sample(root, "Simul-General", "c000", V=3, groups=("target",), main="target", moving=True, seed=3)
    make_sample(root, "Real-General", "b000", V=3, groups=("dynamic_surface", "human_body"), main="all",
                invalid=True, seed=4)
    return Benchmark(root)


@pytest.fixture(scope="module")
def proto():
    return t3.Track3DProtocol(ref_frame=TQ, horizon=HZ)


def _approx(x):
    return pytest.approx(x, abs=1e-9)


def test_protocol_window_roundtrip_and_presets(bench, proto, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    assert proto.window(s).tolist() == list(range(TQ, TQ + HZ + 1))
    p = t3.Track3DProtocol(ref_frame=2, horizon=None, frame_step=5)
    assert p.window(s).tolist() == [2, 7, 12, 17, 22]
    assert t3.Track3DProtocol(ref_frame=10, horizon=20).evaluable(s)[0] is False
    assert t3.Track3DProtocol(mode="monocular", mono_view=5).evaluable(s)[0] is False
    assert proto.ladder_for("Real-Robot") == (0.01, 0.02, 0.05, 0.10, 0.20)
    assert proto.survival_for("rg") == 1.0
    q = t3.Track3DProtocol(ladder_m={"Real-Robot": [0.1, 0.2]}, flow_strict=[0.01, 0.02], views=[0, 1])
    q.save(tmp_path / "p.json")
    r = t3.Track3DProtocol.load(tmp_path / "p.json")
    assert r == q and r.ladder_for("Real-Robot") == (0.1, 0.2) and r.flow_strict == (0.01, 0.02)
    with pytest.raises(KeyError):
        t3.Track3DProtocol.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        t3.Track3DProtocol(mode="stereo")
    leg = t3.Track3DProtocol.legacy_prerelease()
    assert leg.get("query_requires_visible", "Simul-Robot") is False
    assert leg.get("query_requires_visible", "Real-General") is True
    assert leg.get("gt_visibility", "Simul-General") == "in_frustum"
    assert leg.get("gt_visibility", "Simul-Robot") == "visible"
    g = t3.Track3DProtocol.from_protocol(legacy_prerelease())
    assert g.get("query_requires_visible", "sg") is False and g.get("gt_visibility", "sg") == "in_frustum"
    d = t3.Track3DProtocol.from_protocol(Protocol(ref_frame=8, horizon=32, delta_ladder_m={"Real-Robot": (0.5,)}))
    assert (d.ref_frame, d.horizon) == (8, 32) and d.ladder_for("Real-Robot") == (0.5,)
    assert d.get("query_requires_visible", "Simul-Robot") is True
    assert t3.get_protocol("legacy-prerelease").name == leg.name


def test_protocol_subset_aliases_flow_stride_hash_and_overrides():
    p = t3.Track3DProtocol(ladder_m={"rr": (0.5, 1.0)}, survival_m={"real-robot": 2.0},
                           gt_visibility={"sg": "in_frustum"}, flow_min_motion_m={"Real_General": 0.05})
    assert p.ladder_for("Real-Robot") == (0.5, 1.0) and p.survival_for("rr") == 2.0
    assert p.get("gt_visibility", "Simul-General") == "in_frustum" and set(p.ladder_m) == {"Real-Robot"}
    assert p.get("flow_min_motion_m", "rg") == 0.05 and p.get("flow_min_motion_m", "sr") == t3.FLOW_MIN_MOTION_M
    for bad in ({"ladder_m": {"nope": (1.0,)}}, {"survival_m": {"rr": 1.0, "Real-Robot": 2.0}}):
        with pytest.raises(ValueError):
            t3.Track3DProtocol(**bad)
    d = t3.Track3DProtocol()
    assert (d.flow_stride_for(20.0), d.flow_stride_for(15.0), d.flow_stride_for(10.0)) == (10, 8, 5)
    assert t3.Track3DProtocol(frame_step=2).flow_stride_for(20.0) == 5
    assert t3.Track3DProtocol(flow_stride=1).flow_stride_for(20.0) == 1
    with pytest.raises(ValueError):
        t3.Track3DProtocol(flow_stride=0)
    assert t3.Track3DProtocol(name="x").hash() == d.hash() != t3.Track3DProtocol(horizon=32).hash()
    assert t3.Track3DProtocol.from_dict(json.loads(json.dumps(p.to_dict()))).hash() == p.hash()
    o = t3.override_protocol(d, {"mode": "monocular", "mono_view": 1, "horizon": None, "ref_frame": 16})
    assert o.name == "track3d-v1+mode=monocular,mono_view=1,horizon=None" and o.mode == "monocular"
    assert t3.override_protocol(d, {"ref_frame": 16}) is d


def test_task_case_queries_and_gt(bench, proto):
    s = bench.sample("Simul-Robot/a000")
    tr = s.tracks
    c = t3.task_case(s, proto)
    expect = np.nonzero(tr.default_gt[TQ] & tr.visible[TQ].any(-1))[0]
    assert np.array_equal(c["query_index"], expect) and 0 in expect and 1 not in expect
    win = proto.window(s)
    np.testing.assert_array_equal(c["gt"], tr.xyz[win][:, expect].astype(np.float64))
    np.testing.assert_array_equal(c["gt_vis"], tr.visible[win][:, expect].any(-1))
    np.testing.assert_array_equal(c["xyz_q"], tr.xyz[TQ, expect])
    assert c["main_group"] == "object" and list(c["groups"]) == ["object", "robot", "all"]
    c2 = t3.task_case(s, t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, query_requires_visible=False))
    assert np.array_equal(c2["query_index"], np.arange(tr.n_points))
    c3 = t3.task_case(s, t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, gt_visibility={"Simul-Robot": "in_frustum"}))
    np.testing.assert_array_equal(c3["gt_vis"], tr.in_frustum[win][:, c3["query_index"]].any(-1))
    m = t3.task_case(s, t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode="monocular", mono_view=1))
    assert np.array_equal(m["query_index"], np.nonzero(tr.visible[TQ, :, 1])[0])
    E = s.cameras.w2c[1][win]
    X = tr.xyz[win][:, m["query_index"]].astype(np.float64)
    np.testing.assert_allclose(m["gt"], np.einsum("tij,tnj->tni", E[:, :3, :3], X) + E[:, None, :3, 3])
    r = t3.task_case(bench.sample("Real-General/b000"), proto)
    assert np.isnan(r["gt"]).any() and not (r["gt_vis"] & np.isnan(r["gt"]).any(-1)).any()
    assert list(r["groups"]) == ["all", "dynamic_surface", "human_body"]
    assert t3.task_case(s, t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, views=[0]))["views"] == [0]


def test_no_query_sample_is_skipped(bench, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, main_group={"Simul-Robot": "robot"})
    assert t3.task_case(s, p)["main_group"] == "robot"
    with pytest.raises(KeyError):
        t3.task_case(s, t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, main_group={"Simul-Robot": "nope"}))
    r = t3.evaluate_sample(s, tmp_path, t3.Track3DProtocol(ref_frame=30, horizon=2))
    assert r["status"] == "skipped" and "last frame" in r["reason"]


def _check_perfect(m):
    for k in ("any/delta_avg", "any/aj", "any/oa", "any/survival"):
        assert m[k] == _approx(100.0), k
    for k in ("any/ade", "any/mte", "ade_all", "fde", "flow/epe3d", "allframes/ade", "allframes/fde"):
        assert m[k] == _approx(0.0), k
    if m.get("flow/moving/n"):
        assert m["flow/moving/acc_strict"] == m["flow/moving/acc_relax"] == _approx(100.0)
        assert m["flow/moving/epe3d"] == _approx(0.0) and m["flow/moving/outliers"] == 0.0


@pytest.mark.parametrize("mode", ["world", "monocular"])
def test_oracle_is_perfect(bench, tmp_path, mode):
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode=mode, mono_view=1)
    assert t3.run_baseline(bench, tmp_path / "pred", "oracle", p) == {"written": 4, "skipped": 0}
    summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, verbose=False)
    for sub in ("Simul-Robot", "Simul-General", "Real-General"):
        s = summ["subsets"][sub]
        assert s["n_ok"] == s["n_samples"] and s["n_missing"] == 0
        for g, agg in s["groups"].items():
            _check_perfect({k: v["mean"] for k, v in agg.items()})
    rec = json.loads((tmp_path / "res/Real-General/b000.json").read_text())
    assert rec["status"] == "ok" and rec["prediction"]["has_vis"] and rec["mode"] == mode
    assert (tmp_path / "res/summary.md").read_text().count("|") > 20


def test_frozen_matches_direct_metrics(bench, proto, tmp_path):
    t3.run_baseline(bench, tmp_path / "pred", "frozen", proto)
    for uid in ("Simul-Robot/a000", "Real-General/b000"):
        s = bench.sample(uid)
        rec = t3.evaluate_sample(s, tmp_path / "pred", proto)
        c = t3.task_case(s, proto)
        for g, m in c["groups"].items():
            gt, vis = c["gt"][:, m], c["gt_vis"][:, m]
            fr = trk.frozen_prediction(gt, np.zeros(m.sum(), int))
            ref = trk.track3d_metrics(gt, vis, fr, None, np.zeros(m.sum(), int), proto.ladder_for(s.subset),
                                      proto.survival_for(s.subset))
            for k, v in ref.items():
                assert rec["groups"][g][k] == pytest.approx(v, nan_ok=True), (uid, g, k)
            assert rec["groups"][g]["allframes/ade"] == pytest.approx(trk.ade(gt, fr))
            stride = proto.flow_stride_for(s.fps)
            assert stride == rec["flow_stride"] == 5
            fl = t3.scene_flow_metrics(gt, vis, fr, stride=stride, min_motion_m=t3.FLOW_MIN_MOTION_M)
            for k, v in fl.items():
                assert rec["groups"][g][f"flow/{k}"] == pytest.approx(v, nan_ok=True), (uid, g, k)
    assert not rec["prediction"]["has_vis"]


def test_frozen_monocular_moving_camera(bench, tmp_path):
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode="monocular", mono_view=0)
    s = bench.sample("Simul-General/c000")
    c = t3.task_case(s, p)
    f = t3.frozen_prediction(s, p, c)["xyz_cam"]
    E = s.cameras.w2c[0][c["window"]]
    np.testing.assert_allclose(f, np.einsum("tij,nj->tni", E[:, :3, :3], c["xyz_q"]) + E[:, None, :3, 3])
    assert not np.allclose(f[0], f[-1])
    np.testing.assert_allclose(f[0], c["gt"][0])
    t3.write_prediction(tmp_path, s, xyz=np.repeat(c["xyz_q"][None], len(c["window"]), 0))
    pred, _, info = t3.load_prediction(tmp_path, s, p, c)
    assert info["key"] == "xyz"
    np.testing.assert_allclose(pred, f)


def test_query_index_permutation_and_format_errors(bench, proto, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    c = t3.task_case(s, proto)
    o = t3.oracle_prediction(s, proto, c)
    perm = np.random.default_rng(0).permutation(len(c["query_index"]))
    t3.write_prediction(tmp_path, s, xyz=o["xyz"][:, perm], vis=o["vis"][:, perm], query_index=c["query_index"][perm])
    rec = t3.evaluate_sample(s, tmp_path, proto)
    _check_perfect(rec["groups"]["object"])
    bad = c["query_index"].copy()
    bad[0] = 10_000
    t3.write_prediction(tmp_path, s, xyz=o["xyz"], query_index=bad)
    with pytest.raises(ValueError, match="permutation"):
        t3.evaluate_sample(s, tmp_path, proto)
    t3.write_prediction(tmp_path, s, xyz=o["xyz"][:-1])
    with pytest.raises(ValueError, match="shape"):
        t3.evaluate_sample(s, tmp_path, proto)
    t3.write_prediction(tmp_path, s, xyz=o["xyz"], vis=np.ones((3, 3), bool))
    with pytest.raises(ValueError, match="vis"):
        t3.evaluate_sample(s, tmp_path, proto)
    t3.write_prediction(tmp_path, s, xyz_cam=o["xyz"])
    with pytest.raises(ValueError, match="xyz"):
        t3.evaluate_sample(s, tmp_path, proto)
    with pytest.raises(ValueError):
        t3.write_prediction(tmp_path, s)
    vv = np.zeros(o["vis"].shape + (2,), np.float32)
    vv[..., 1] = o["vis"]
    vv[0, 0, 0] = np.nan
    t3.write_prediction(tmp_path, s, xyz=o["xyz"], vis=vv)
    _check_perfect(t3.evaluate_sample(s, tmp_path, proto)["groups"]["object"])


def test_nonfinite_predictions_are_failures(bench, proto, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    c = t3.task_case(s, proto)
    o = t3.oracle_prediction(s, proto, c)
    xyz = o["xyz"].copy()
    j = int(np.nonzero(c["gt_vis"][1:, 0])[0][0]) + 1
    xyz[j, 0] = np.nan
    t3.write_prediction(tmp_path, s, xyz=xyz, vis=o["vis"])
    rec = t3.evaluate_sample(s, tmp_path, proto)
    m = rec["groups"]["object"]
    assert rec["prediction"]["nonfinite_frac"] == pytest.approx(1 / xyz[..., 0].size)
    assert m["any/delta_avg"] < 100.0 and m["any/survival"] < 100.0 and math.isinf(m["any/ade"])
    assert math.isinf(m["flow/epe3d"]) and m["any/mte"] == 0.0
    n_sc = c["gt_vis"][1:, 0].sum()
    assert m["any/delta_avg"] == pytest.approx(100.0 - 100.0 / n_sc / m["any/n_tracks"])
    t3.write_json(tmp_path / "r.json", rec)
    assert json.loads((tmp_path / "r.json").read_text())["groups"]["object"]["any/ade"] == "inf"


def test_missing_predictions(bench, proto, tmp_path):
    s = bench.sample("Simul-Robot/a001")
    t3.write_prediction(tmp_path / "pred", s, **t3.frozen_prediction(s, proto))
    summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto, subsets=["Simul-Robot"], verbose=False)
    sr = summ["subsets"]["Simul-Robot"]
    assert (sr["n_samples"], sr["n_ok"], sr["n_missing"]) == (2, 1, 1)
    assert json.loads((tmp_path / "res/Simul-Robot/a000.json").read_text())["status"] == "missing"
    with pytest.raises(FileNotFoundError):
        t3.evaluate(bench, tmp_path / "pred", tmp_path / "res2", proto, subsets=["Simul-Robot"], on_missing="error",
                    verbose=False)


def test_scene_flow_hand_computed():
    gt = np.zeros((3, 2, 3))
    gt[:, 0, 0] = [0.0, 0.1, 0.2]
    gt[:, 1] = 1.0
    vis = np.ones((3, 2), bool)
    vis[1, 1] = False
    pred = gt.copy()
    pred[2, 0, 0] = 0.45
    r = t3.scene_flow_metrics(gt, vis, pred)
    assert r["n"] == r["moving/n"] == 3
    assert r["epe3d"] == r["moving/epe3d"] == pytest.approx(0.25 / 3)
    assert r["moving/acc_strict"] == pytest.approx(200 / 3) and r["moving/acc_relax"] == pytest.approx(200 / 3)
    assert r["moving/outliers"] == pytest.approx(100 / 3)
    r = t3.scene_flow_metrics(gt, vis, pred, min_motion_m=0.05)
    assert (r["n"], r["moving/n"]) == (3, 2) and r["moving/epe3d"] == pytest.approx(0.25 / 2)
    assert r["moving/outliers"] == pytest.approx(50.0) and r["epe3d"] == pytest.approx(0.25 / 3)
    r = t3.scene_flow_metrics(gt, vis, pred, min_motion_m=1.0)
    assert r["moving/n"] == 0 and math.isnan(r["moving/epe3d"]) and r["n"] == 3
    r2 = t3.scene_flow_metrics(gt, vis, pred, stride=2)
    assert r2["n"] == 2 and r2["epe3d"] == pytest.approx(0.25 / 2)
    r3 = t3.scene_flow_metrics(gt, vis, pred, query_t=np.array([1, 0]))
    assert r3["n"] == 2
    pred[1, 1] = np.nan
    assert math.isinf(t3.scene_flow_metrics(gt, vis, pred)["epe3d"])
    assert t3.scene_flow_metrics(gt[:1], vis[:1], pred[:1])["n"] == 0


def test_monocular_median_scale(bench, tmp_path):
    s = bench.sample("Simul-Robot/a000")
    for scale_mode, perfect in (("median", True), ("none", False)):
        p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode="monocular", mono_view=0, mono_scale=scale_mode)
        c = t3.task_case(s, p)
        o = t3.oracle_prediction(s, p, c)
        t3.write_prediction(tmp_path, s, xyz_cam=o["xyz_cam"] * 0.5, vis=o["vis"])
        rec = t3.evaluate_sample(s, tmp_path, p)
        assert rec["prediction"]["scale"] == (pytest.approx(2.0) if perfect else 1.0)
        assert (rec["groups"]["object"]["any/delta_avg"] == pytest.approx(100.0)) is perfect
    assert t3.median_scale(np.ones((2, 1, 3)), np.ones((2, 1), bool), np.full((2, 1, 3), np.nan)) == 1.0


def test_mean_se_and_summary(bench, proto, tmp_path):
    a = t3.mean_se([1.0, 3.0, None, float("nan")])
    assert a == {"mean": 2.0, "se": pytest.approx(np.std([1, 3], ddof=1) / np.sqrt(2)), "n": 2, "n_inf": 0}
    b = t3.mean_se([1.0, "inf"])
    assert math.isinf(b["mean"]) and math.isnan(b["se"]) and b["n"] == 2 and b["n_inf"] == 1
    assert t3.mean_se([])["n"] == 0 and math.isnan(t3.mean_se([5.0])["se"])
    t3.run_baseline(bench, tmp_path / "pred", "frozen", proto)
    summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto, verbose=False)
    recs = [json.loads((tmp_path / f"res/Simul-Robot/{i}.json").read_text()) for i in ("a000", "a001")]
    vals = [r["groups"]["object"]["any/delta_avg"] for r in recs]
    got = summ["subsets"]["Simul-Robot"]["main"]["any/delta_avg"]
    assert got["mean"] == pytest.approx(np.mean(vals)) and got["n"] == 2
    assert got["se"] == pytest.approx(np.std(vals, ddof=1) / np.sqrt(2))
    assert summ["subsets"]["Simul-Robot"]["groups"]["robot"]["any/delta_avg"]["n"] == 2
    again = t3.summarize(tmp_path / "res")
    assert json.loads(json.dumps(t3.jsonable(again))) == json.loads((tmp_path / "res/summary.json").read_text())
    md = (tmp_path / "res/summary.md").read_text()
    assert "object (main)" in md and "Simul-General" in md and "target (main)" in md


def test_sample_inputs_and_export(bench, tmp_path):
    from vpm_worldbench.io.video import read_frames
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, depth="query")
    s = bench.sample("Simul-Robot/a000")
    inp = t3.sample_inputs(s, p, size=(32, 24))
    n, N = HZ + 1, len(inp["query_index"])
    assert inp["frames"].shape == (2, n, 24, 32, 3) and inp["w2c"].shape == (2, n, 4, 4)
    np.testing.assert_allclose(inp["K"][1], s.K(1, (32, 24)))
    assert inp["lifter_queries"].shape == (N, 4) and (inp["lifter_queries"][:, 0] == 0).all()
    for i in range(2):
        uv, _ = project(inp["queries_xyz"], inp["K"][i], s.cameras.w2c[i, TQ])
        vis = inp["queries_visible"][i]
        np.testing.assert_allclose(inp["queries_uv"][i][vis], uv[vis])
        assert np.isnan(inp["queries_uv"][i][~vis]).all()
        g_uv, g_vis, _ = s.tracks2d(i, size=(32, 24), indices=[TQ])
        np.testing.assert_allclose(inp["queries_uv"][i][vis], g_uv[0, inp["query_index"]][vis])
        assert np.array_equal(vis, g_vis[0, inp["query_index"]])
    assert inp["depth"].shape == (2, 24, 32) and inp["depth"][0, 0, 0] == pytest.approx(3.0 + TQ * 0.001)

    res = t3.export_inputs(bench, tmp_path / "png", p, subsets=["Simul-Robot"], frames="png", size=(32, 24))
    assert res == {"exported": 2, "skipped": 0, "existing": 0}
    again = t3.export_inputs(bench, tmp_path / "png", p, subsets="sr", frames="png", size=(32, 24))
    assert again == {"exported": 2, "skipped": 0, "existing": 2}
    with pytest.raises(ValueError, match="different protocol"):
        t3.export_inputs(bench, tmp_path / "png", p, subsets=["Simul-Robot"], frames="mp4")
    d = tmp_path / "png/Simul-Robot/a000"
    rec = json.loads((d / "inputs.json").read_text())
    assert rec["prediction"]["shape"] == [n, N, 3] and rec["window_indices"] == list(range(TQ, TQ + n))
    assert rec["depth"] == "depth_query.npz" and (d / "depth_query.npz").exists()
    assert len(list((d / "frames/view1").glob("*.png"))) == n
    q = np.load(d / "queries.npz")
    np.testing.assert_array_equal(q["query_index"], inp["query_index"])
    np.testing.assert_allclose(q["uv"], inp["queries_uv"])
    np.testing.assert_allclose(rec["cameras"]["view0"]["K"], inp["K"][0])
    assert json.loads((tmp_path / "png/protocol.json").read_text())["depth"] == "query"
    assert len((tmp_path / "png/manifest.jsonl").read_text().splitlines()) == 2
    assert not (tmp_path / "png/Simul-Robot/a001/depth_query.npz").exists()

    t3.export_inputs(bench, tmp_path / "mp4", t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode="monocular",
                                                                 mono_view=1), ids=["Simul-Robot/a001"], frames="mp4")
    d = tmp_path / "mp4/Simul-Robot/a001"
    fr = read_frames(d / "frames/view1.mp4")
    np.testing.assert_array_equal(fr, bench.sample("Simul-Robot/a001").frames(1, range(TQ, TQ + n)))
    q = np.load(d / "queries.npz")
    assert q["uv"].shape[0] == 1 and "xyz_cam" in q.files
    assert json.loads((d / "inputs.json").read_text())["prediction"]["key"] == "xyz_cam"

    t3.export_inputs(bench, tmp_path / "none", p, ids=["Simul-Robot/a000"])
    rec = json.loads((tmp_path / "none/Simul-Robot/a000/inputs.json").read_text())
    assert rec["frames"]["files"]["view0"]["video"] == "Simul-Robot/a000/videos/view0.mp4"
    with pytest.raises(ValueError):
        t3.export_inputs(bench, tmp_path / "x", p, frames="none", size=(32, 24))
    with pytest.raises(ValueError):
        t3.sample_inputs(s, t3.Track3DProtocol(ref_frame=30, horizon=2))


class _EchoLifter:
    """Stands in for a Lifter: records its inputs and returns the query points (the frozen prediction)."""

    def __init__(self):
        self.calls = []

    def prepare_anchor(self, anchor, frames, K, w2c, **kw):
        return anchor

    def track(self, frames, K, w2c, queries, anchor, **kw):
        self.calls.append({"frames": frames.shape, "K": K.shape, "w2c": w2c.shape, "queries": queries.copy(),
                           "anchor": anchor.mode, "depth": None if anchor.depth is None else anchor.depth.shape})
        n = frames.shape[1]
        return np.repeat(queries[None, :, 1:], n, 0).astype(np.float32), np.ones((n, len(queries)), bool)


def test_lifter_baseline_with_a_fake_lifter(bench, tmp_path):
    from vpm_worldbench.lifters.base import register_lifter
    s = bench.sample("Simul-Robot/a000")
    L = _EchoLifter()
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, depth="query")
    out = t3.lifter_prediction(s, p, L, size=(32, 24))
    c = L.calls[-1]
    N = len(t3.task_case(s, p)["query_index"])
    assert c["frames"] == (2, HZ + 1, 24, 32, 3) and c["K"] == (2, HZ + 1, 3, 3) and c["w2c"] == (2, HZ + 1, 4, 4)
    assert c["anchor"] == "gt_depth" and c["depth"] == (2, 24, 32) and c["queries"].shape == (N, 4)
    np.testing.assert_allclose(out["xyz"], t3.frozen_prediction(s, p)["xyz"])
    t3.lifter_prediction(bench.sample("Simul-Robot/a001"), p, L, size=None)
    assert L.calls[-1]["anchor"] == "gt_points" and L.calls[-1]["frames"][2:4] == (48, 64)
    pm = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, mode="monocular", mono_view=1)
    out = t3.lifter_prediction(s, pm, L)
    assert L.calls[-1]["frames"] == (1, HZ + 1, 480, 480, 3) and L.calls[-1]["anchor"] == "gt_points"
    np.testing.assert_allclose(out["xyz_cam"], t3.frozen_prediction(s, pm)["xyz_cam"], atol=1e-6)
    register_lifter("echo_test", _EchoLifter)
    r = t3.run_lifter_baseline(bench, tmp_path / "pred", p, "echo_test", size=(32, 24), subsets=["Simul-Robot"],
                               verbose=False)
    assert r == {"written": 2, "skipped": 0, "existing": 0}
    assert t3.run_lifter_baseline(bench, tmp_path / "pred", p, L, subsets=["Simul-Robot"], skip_existing=True,
                                  verbose=False)["existing"] == 2
    rec = t3.evaluate_sample(s, tmp_path / "pred", p)
    assert rec["prediction"]["has_vis"] and rec["groups"]["object"]["any/oa"] < 100.0
    with pytest.raises(ValueError):
        t3.lifter_prediction(s, p, L, size="huge")


def test_export_parallel_jobs(bench, proto, tmp_path):
    assert t3.export_inputs(bench, tmp_path / "e", proto, jobs=2) == {"exported": 4, "skipped": 0, "existing": 0}


def test_evaluate_jobs_reuse_overwrite_and_stale(bench, proto, tmp_path):
    assert t3.run_baseline(bench, tmp_path / "pred", "frozen", proto, jobs=2) == {"written": 4, "skipped": 0}
    a = t3.evaluate(bench, tmp_path / "pred", tmp_path / "r1", proto, verbose=False)
    b = t3.evaluate(bench, tmp_path / "pred", tmp_path / "r2", proto, verbose=False, jobs=2)
    assert json.dumps(t3.jsonable(a)) == json.dumps(t3.jsonable(b))
    f = tmp_path / "r1/Simul-Robot/a000.json"
    stamp = f.stat().st_mtime_ns
    t3.evaluate(bench, tmp_path / "pred", tmp_path / "r1", proto, verbose=False)
    assert f.stat().st_mtime_ns == stamp
    s = bench.sample("Simul-Robot/a000")
    t3.write_prediction(tmp_path / "pred", s, **t3.oracle_prediction(s, proto))
    summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "r1", proto, verbose=False)
    assert json.loads(f.read_text())["groups"]["object"]["any/delta_avg"] == _approx(100.0)
    assert summ["subsets"]["Simul-Robot"]["n_stale"] == 0
    other = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ - 2)
    with pytest.raises(ValueError, match="different protocol"):
        t3.evaluate(bench, tmp_path / "pred", tmp_path / "r1", other, verbose=False)
    t3.run_baseline(bench, tmp_path / "pred_other", "frozen", other, ids=["Simul-Robot/a000"])
    summ = t3.evaluate(bench, tmp_path / "pred_other", tmp_path / "r1", other, ids=["Simul-Robot/a000"],
                       overwrite=True, verbose=False)
    assert summ["subsets"]["Simul-Robot"]["n_samples"] == 1 and summ["subsets"]["Simul-Robot"]["n_stale"] == 0
    assert json.loads(f.read_text())["n_window"] == HZ - 1
    assert "Real-General" not in summ["subsets"] and not (tmp_path / "r1/Simul-Robot/a001.json").exists()
    rec = json.loads(f.read_text())
    t3.write_json(tmp_path / "r1/Simul-Robot/zz.json", dict(rec, uid="Simul-Robot/zz", id="zz", run_id="other"))
    with pytest.warns(UserWarning, match="another protocol or prediction root"):
        summ = t3.summarize(tmp_path / "r1")
    assert summ["subsets"]["Simul-Robot"]["n_samples"] == 1 and summ["subsets"]["Simul-Robot"]["n_stale"] == 1
    assert "1 stale" in (tmp_path / "r1/summary.md").read_text()


def _as_older_version(out: Path):
    """Make `out` look like an out dir written before run.json existed (no run.json, records without run_id)."""
    (out / "run.json").unlink()
    for f in out.glob("*/*.json"):
        r = json.loads(f.read_text())
        r.pop("run_id")
        f.write_text(json.dumps(r))


def test_out_dir_holds_one_prediction_root(bench, proto, tmp_path):
    oracle, frozen, out = tmp_path / "oracle", tmp_path / "frozen", tmp_path / "res"
    t3.run_baseline(bench, oracle, "oracle", proto)
    t3.run_baseline(bench, frozen, "frozen", proto)
    a = t3.evaluate(bench, oracle, out, proto, verbose=False)
    run = json.loads((out / "run.json").read_text())
    assert run["pred_root"] == str(oracle.resolve()) == a["pred_root"] and run["protocol_hash"] == proto.hash()
    assert {json.loads(f.read_text())["run_id"] for f in out.glob("*/*.json")} == {run["run_id"]}
    ev = a["evaluated"]
    assert (ev["status"], ev["n_failed"], ev["all_failed"], ev["unmatched_ids"]) == ({"ok": 4}, 0, False, [])
    with pytest.raises(t3.RunDirError, match="predictions in"):
        t3.evaluate(bench, frozen, out, proto, subsets=["sr"], verbose=False)
    assert t3.RunDirError is RunDirError and issubclass(RunDirError, (UsageError, ValueError))
    assert json.loads((out / "run.json").read_text()) == run
    b = t3.evaluate(bench, frozen, out, proto, subsets="sr", overwrite=True, verbose=False)
    assert list(b["subsets"]) == ["Simul-Robot"] and b["pred_root"] == str(frozen.resolve())
    assert sorted(f.name for f in out.glob("*/*.json")) == ["a000.json", "a001.json"]
    assert b["subsets"]["Simul-Robot"]["main"]["any/delta_avg"]["mean"] < 100.0
    assert json.loads((out / "summary.json").read_text())["pred_root"] == str(frozen.resolve())
    old = tmp_path / "old"
    t3.evaluate(bench, oracle, old, proto, verbose=False)
    _as_older_version(old)
    s = t3.summarize(old)
    assert len(s["subsets"]) == 3 and s["pred_root"] is None
    c = t3.evaluate(bench, oracle, old, proto, subsets=["sr"], verbose=False)
    assert len(c["subsets"]) == 3 and c["evaluated"]["status"] == {"ok": 2} and c["pred_root"] == str(oracle.resolve())
    rid = json.loads((old / "run.json").read_text())["run_id"]
    assert {json.loads(f.read_text())["run_id"] for f in old.glob("*/*.json")} == {rid}
    _as_older_version(old)
    with pytest.raises(ValueError, match="do not match the prediction files"):
        t3.evaluate(bench, frozen, old, proto, subsets=["sr"], verbose=False)
    d = t3.evaluate(bench, frozen, old, proto, verbose=False)
    assert all(s["n_ok"] == s["n_samples"] for s in d["subsets"].values()) and len(d["subsets"]) == 3
    assert d["subsets"]["Simul-Robot"]["main"]["any/delta_avg"]["mean"] < 100.0


def test_malformed_prediction_is_an_invalid_record(bench, proto, tmp_path, capsys):
    pred, out = tmp_path / "pred", tmp_path / "res"
    t3.run_baseline(bench, pred, "oracle", proto)
    s = bench.sample("Simul-Robot/a000")
    o = t3.oracle_prediction(s, proto)
    t3.write_prediction(pred, s, xyz=o["xyz"][:-1])
    (pred / "Real-General/b000.npz").write_bytes(b"not an npz archive")
    with pytest.warns(UserWarning, match=r"2 sample\(s\) failed \(status invalid / error\), e.g. Simul-Robot/a000"):
        summ = t3.evaluate(bench, pred, out, proto, verbose=False)
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    assert rec["status"] == "invalid" and "'xyz' has shape" in rec["reason"] and "a000.npz" in rec["reason"]
    assert rec["run_id"] == json.loads((out / "run.json").read_text())["run_id"]
    assert "not a readable .npz" in json.loads((out / "Real-General/b000.json").read_text())["reason"]
    sr = summ["subsets"]["Simul-Robot"]
    assert (sr["n_samples"], sr["n_ok"], sr["n_invalid"], sr["n_error"]) == (2, 1, 1, 0)
    assert summ["subsets"]["Simul-General"]["n_ok"] == 1
    ev = summ["evaluated"]
    assert (ev["status"], ev["n_failed"], ev["all_failed"]) == ({"invalid": 2, "ok": 2}, 2, False)
    md = (out / "summary.md").read_text()
    assert "| Simul-Robot (1 / 2, 0 missing, 1 invalid) |" in md
    assert "| Real-General (0 / 1, 0 missing, 1 invalid) |" in md
    for bad, msg in (({"xyz": o["xyz"].astype(str)}, "must hold real numbers"),
                     ({"xyz": o["xyz"], "query_index": o["query_index"] + 0.5}, "integer ids"),
                     ({"xyz": o["xyz"], "query_index": o["query_index"][::-1] + 1}, "not a permutation"),
                     ({"xyz": o["xyz"], "vis": o["vis"][:, :-1], "query_index": o["query_index"]}, "'vis' has shape"),
                     ({"pts": o["xyz"]}, "needs 'xyz'")):
        np.savez(pred / "Simul-Robot/a000.npz", **bad)
        with pytest.raises(InvalidPrediction, match=msg):
            t3.evaluate_sample(s, pred, proto)
    with pytest.raises(ValueError, match="needs 'xyz'"):
        t3.evaluate(bench, pred, out, proto, ids=[s.uid], strict=True, verbose=False)
    bad_group = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ, main_group={"sr": "nope"})
    with pytest.warns(UserWarning, match="failed"):
        summ = t3.evaluate(bench, pred, tmp_path / "err", bad_group, subsets="sr,sg", verbose=False)
    rec = json.loads((tmp_path / "err/Simul-Robot/a001.json").read_text())
    assert rec["status"] == "error" and rec["reason"].startswith("KeyError") and "task_case" in rec["traceback"]
    assert summ["evaluated"]["status"] == {"error": 2, "ok": 1} and summ["subsets"]["Simul-Robot"]["n_error"] == 2
    t3.write_prediction(pred, s, xyz=o["xyz"][:-1])
    args = ["evaluate", "--root", str(bench.root), "--pred", str(pred), "--ref-frame", str(TQ), "--horizon", str(HZ)]
    with pytest.warns(UserWarning, match="failed"):
        assert t3.main(args + ["--out", str(tmp_path / "c1")]) == 0
    with pytest.warns(UserWarning, match="failed"):
        assert t3.main(args + ["--out", str(tmp_path / "c2"), "--ids", "sr/a000", "rg/b000"]) == 1
    assert "every evaluated sample failed (invalid 2)" in capsys.readouterr().err
    with pytest.raises(ValueError, match="shape"):
        t3.main(args + ["--out", str(tmp_path / "c3"), "--strict"])
    t3.write_prediction(pred, s, **o)
    t3.run_baseline(bench, pred, "oracle", proto, ids=["rg/b000"])
    summ = t3.evaluate(bench, pred, out, proto, verbose=False)
    assert (summ["subsets"]["Simul-Robot"]["n_ok"], summ["subsets"]["Simul-Robot"]["n_invalid"]) == (2, 0)
    assert summ["evaluated"]["status"] == {"ok": 4}


def test_ids_with_subset_aliases_and_unmatched_ids(bench, proto, tmp_path, capsys):
    t3.run_baseline(bench, tmp_path / "pred", "frozen", proto, ids=["sr/a001"])
    assert (tmp_path / "pred/Simul-Robot/a001.npz").exists() and not (tmp_path / "pred/Simul-Robot/a000.npz").exists()
    with pytest.warns(UserWarning) as w:
        summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto, ids=["sr/a001", "rg/nope"],
                           verbose=False)
    msgs = [str(x.message) for x in w if "match no" in str(x.message)]
    assert msgs == ["track3d: 1 requested id(s) match no selected sample (check --subsets, --ids and the protocol's "
                    "clip_filter): ['Real-General/nope']"]
    assert (summ["evaluated"]["status"], summ["evaluated"]["unmatched_ids"]) == ({"ok": 1}, ["Real-General/nope"])
    with pytest.warns(UserWarning, match="match no selected sample"):
        t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", proto, subsets=["sg"], ids=["Simul-Robot/a001"],
                    verbose=False)
    with pytest.warns(UserWarning, match=r"track3d baseline oracle: 1 requested id\(s\) match no selected sample"):
        assert t3.run_baseline(bench, tmp_path / "p2", "oracle", proto, ids=["a000", "nope"])["written"] == 1
    with pytest.warns(UserWarning, match="track3d export: 1 requested id"):
        t3.export_inputs(bench, tmp_path / "in", proto, ids=["nope"])
    root, common = str(bench.root), ["--ref-frame", str(TQ), "--horizon", str(HZ), "--ids", "nope"]
    with pytest.warns(UserWarning, match="match no selected sample"):
        assert t3.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "r3")]
                       + common) == 1
    with pytest.warns(UserWarning, match="match no selected sample"):
        assert t3.main(["baseline", "--root", root, "--pred", str(tmp_path / "p3")] + common) == 1
    assert capsys.readouterr().err.count("no sample matches the selection") == 2


def test_group_rows_report_their_sample_counts(bench, proto, tmp_path):
    s = bench.sample("Real-General/b000")
    p = t3.Track3DProtocol(ref_frame=TQ, horizon=HZ)
    t3.run_baseline(bench, tmp_path / "pred", "frozen", p)
    summ = t3.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, verbose=False)
    g = summ["subsets"]["Real-General"]
    assert g["group_n_samples"] == {"all": 1, "dynamic_surface": 1, "human_body": 1}
    md = (tmp_path / "res/summary.md").read_text()
    assert "| Real-General (1 / 1, 0 missing) | all (main) | 1 |" in md and "| | human_body | 1 |" in md
    assert "EPE3D mov" in md and "Acc3DS" not in md and s.uid


def test_cli(bench, tmp_path, capsys):
    from vpm_worldbench import cli
    root, pred, out = str(bench.root), str(tmp_path / "pred"), str(tmp_path / "res")
    common = ["--ref-frame", str(TQ), "--horizon", str(HZ)]
    assert t3.main(["baseline", "--root", root, "--pred", pred, "--kind", "oracle", "--jobs", "2"] + common) == 0
    assert cli.main(["track3d", "evaluate", "--root", root, "--pred", pred, "--out", out, "--jobs", "2"] + common) == 0
    s = json.loads(Path(out, "summary.json").read_text())
    assert s["subsets"]["Simul-Robot"]["main"]["any/delta_avg"]["mean"] == pytest.approx(100.0)
    assert s["protocol"]["name"] == f"track3d-v1+ref_frame={TQ},horizon={HZ}"
    rec = json.loads(Path(out, "Simul-Robot/a000.json").read_text())
    assert rec["protocol"] == s["protocol"]["name"]
    with pytest.raises(ValueError, match="different protocol"):
        t3.main(["evaluate", "--root", root, "--pred", pred, "--out", out, "--ref-frame", str(TQ), "--horizon", "8"])
    capsys.readouterr()
    assert cli.main(["track3d", "evaluate", "--root", root, "--pred", pred, "--out", out, "--ref-frame", str(TQ),
                     "--horizon", "8"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: ") and "holds results of a different protocol" in err
    assert len(err.strip().splitlines()) == 1
    assert t3.main(["evaluate", "--root", root, "--pred", pred, "--out", out, "--overwrite", "--ids",
                    "Simul-Robot/a000,Simul-Robot/a001", "--subsets", "sr", "rg"] + common) == 0
    assert t3.main(["summarize", "--out", out]) == 0
    assert "track3d" in capsys.readouterr().out
    assert t3.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--mode", "monocular", "--view", "1",
                    "--subsets", "sr,sg"] + common) == 0
    pj = json.loads(Path(tmp_path, "in/protocol.json").read_text())
    assert pj["mono_view"] == 1 and pj["name"].startswith("track3d-v1+")
    assert sorted(p.name for p in Path(tmp_path, "in").iterdir() if p.is_dir()) == ["Simul-General", "Simul-Robot"]
    with pytest.raises(SystemExit):
        t3.main(["baseline", "--root", root, "--pred", pred, "oracle"])


def test_cli_parser_errors_and_horizon_to_clip_end(bench, tmp_path, capsys):
    root, pred, out = str(bench.root), str(tmp_path / "pred"), tmp_path / "res"
    base = ["evaluate", "--root", root, "--pred", pred, "--out", str(out)]
    for args, msg in ((base + ["--protocol", "vpm-v1"], "unknown track3d protocol 'vpm-v1': not a preset (track3d-v1, "
                                                        "legacy-prerelease)"),
                      (base + ["--subsets", "qq"], "argument --subsets: unknown subset 'qq'"),
                      (base + ["--frame-step", "0"], "frame_step >= 1"),
                      (base + ["--protocol", str(tmp_path)], "not a preset"),
                      (["evaluate", "--root", str(tmp_path / "nope"), "--pred", pred, "--out", str(out)],
                       "--root " + str(tmp_path / "nope") + ": no such directory"),
                      (["export", "--root", root, "--out", str(out), "--frames", "png", "--size", "64by48"],
                       "--size must be WxH"),
                      (["export", "--root", root, "--out", str(out), "--frames", "png", "--size", "0x48"],
                       "--size must be WxH (e.g. 640x480), got '0x48'"),
                      (["export", "--root", root, "--out", str(out), "--size", "64x48"], "--size needs --frames png"),
                      (["baseline", "--root", root, "--pred", pred, "--kind", "lifter", "--lifter", "nope"],
                       "--lifter: unknown lifter 'nope'"),
                      (["summarize", "--out", str(out)], "--out " + str(out) + ": no such directory")):
        with pytest.raises(SystemExit) as e:
            t3.main(args)
        assert e.value.code == 2 and msg in capsys.readouterr().err, args
    assert not out.exists() and not Path(pred).exists()
    assert t3.main(["baseline", "--root", root, "--pred", pred, "--kind", "oracle", "--ref-frame", "20", "--horizon",
                    "0", "--subsets", "sr"]) == 0
    assert t3.main(base + ["--ref-frame", "20", "--horizon", "-1", "--subsets", "sr"]) == 0
    pj = json.loads((out / "protocol.json").read_text())
    assert pj["horizon"] is None and pj["name"] == "track3d-v1+ref_frame=20,horizon=None"
    rec = json.loads((out / "Simul-Robot/a000.json").read_text())
    assert rec["status"] == "ok" and rec["window"] == [20, 23, 1] and rec["n_window"] == 4


def test_cli_lifter_device_defaults_to_auto(bench, tmp_path, capsys):
    from vpm_worldbench.lifters.base import register_lifter
    seen = {}

    def factory(**kw):
        seen.update(kw)
        return _EchoLifter()

    register_lifter("echo_device_test", factory)
    assert t3._device("cpu") == "cpu" and t3._device("cuda:1") == "cuda:1"
    try:
        import torch
        auto = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        auto = "cpu"
    assert t3._device("auto") == auto
    assert t3.main(["baseline", "--root", str(bench.root), "--pred", str(tmp_path / "pred"), "--kind", "lifter",
                    "--lifter", "echo_device_test", "--lifter-size", "32x24", "--subsets", "sr", "--ref-frame",
                    str(TQ), "--horizon", str(HZ)]) == 0
    assert seen == {"device": auto} and "'written': 2" in capsys.readouterr().out
    assert len(list((tmp_path / "pred/Simul-Robot").glob("*.npz"))) == 2


def test_python_m_is_the_vpmwb_command(bench, tmp_path):
    """`python -m vpm_worldbench.tasks.track3d` runs through `vpmwb`: usage errors such as an out dir of another run
    end with one line and exit status 2."""
    import subprocess
    import sys
    root, out = str(bench.root), str(tmp_path / "res")
    common = ["--root", root, "--ref-frame", str(TQ), "--horizon", str(HZ), "--subsets", "sr"]
    for kind in ("oracle", "frozen"):
        t3.main(["baseline", "--pred", str(tmp_path / kind), "--kind", kind] + common)
    assert t3.main(["evaluate", "--pred", str(tmp_path / "oracle"), "--out", out] + common) == 0
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(t3.__file__).resolve().parents[2])] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    r = subprocess.run([sys.executable, "-m", "vpm_worldbench.tasks.track3d", "evaluate", "--pred",
                        str(tmp_path / "frozen"), "--out", out] + common, capture_output=True, text=True, env=env)
    assert r.returncode == 2 and r.stderr.startswith("vpmwb: error: ") and "--overwrite" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr and len(r.stderr.strip().splitlines()) == 1


def test_protocol_files_presets_and_from_protocol_crop_warning(tmp_path):
    (tmp_path / "short.yaml").write_text("horizon: 32\n")
    p = t3.get_protocol(str(tmp_path / "short.yaml"))
    assert p.name == "short" and p.horizon == 32
    (tmp_path / "named.json").write_text(json.dumps({"name": "mine", "horizon": 8}))
    assert t3.Track3DProtocol.load(tmp_path / "named.json").name == "mine"
    (tmp_path / "empty.yaml").write_text("")
    assert t3.get_protocol(tmp_path / "empty.yaml") == t3.Track3DProtocol(name="empty")
    for bad in ("vpm-v1", str(tmp_path / "missing.yaml")):
        with pytest.raises(ValueError, match="not a preset"):
            t3.get_protocol(bad)
    with pytest.warns(UserWarning, match=r"centre-crops \['Real-Robot'\]"):
        t3.Track3DProtocol.from_protocol(Protocol(resize_mode={"rr": "center_crop"}))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t3.Track3DProtocol.from_protocol(Protocol())


RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")
released = pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT (a VP-WorldBench data root)")


@released
def test_released_oracle_and_frozen(tmp_path):
    bench = Benchmark(RELEASED)
    ids = [f"{sub}/{bench.ids(sub)[0]}" for sub in bench.subsets()]
    p = t3.Track3DProtocol()
    t3.run_baseline(bench, tmp_path / "oracle", "oracle", p, ids=ids)
    summ = t3.evaluate(bench, tmp_path / "oracle", tmp_path / "res_o", p, ids=ids, verbose=False)
    for sub, s in summ["subsets"].items():
        assert s["n_ok"] == 1
        for agg in s["groups"].values():
            if agg.get("any/delta_avg", {}).get("n"):
                _check_perfect({k: v["mean"] for k, v in agg.items()})
    t3.run_baseline(bench, tmp_path / "frozen", "frozen", p, ids=ids)
    summ = t3.evaluate(bench, tmp_path / "frozen", tmp_path / "res_f", p, ids=ids, verbose=False)
    for s in summ["subsets"].values():
        assert 0.0 < s["main"]["any/delta_avg"]["mean"] < 100.0
