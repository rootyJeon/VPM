"""Tests for vpm_worldbench.tasks.track2d (TAP-Vid style per-view 2D tracking + cross-view correspondence).

Synthetic mini benchmark (written to tmp_path); optional: VPMWB_RELEASED_ROOT=<VP-WorldBench root> runs the oracle /
frozen baselines on one real sample per subset.
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
from vpm_worldbench.metrics.tracking import track2d_metrics
from vpm_worldbench.tasks import track2d as t2
from vpm_worldbench.tasks.common import InvalidPrediction, RunDirError

T0, HZ = 4, 12


def _look_at(C):
    C = np.asarray(C, float)
    f = -C / np.linalg.norm(C)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    E = np.eye(4)
    E[:3, :3] = np.stack([r, np.cross(f, r), f])
    E[:3, 3] = -E[:3, :3] @ C
    return E


def make_sample(root: Path, subset: str, sid: str, V=3, T=24, N=16, W=64, H=48, invalid=False, seed=0):
    """Static look-at cameras around the origin, points moving near the origin, random occlusions."""
    from vpm_worldbench.io.video import write_video
    rng = np.random.default_rng(seed)
    d = root / subset / sid
    (d / "videos").mkdir(parents=True)
    K = np.array([[40.0, 0, (W - 1) / 2], [0, 40.0, (H - 1) / 2], [0, 0, 1]])
    centres = np.array([(3.0, 0.2, 1.0), (0.3, 3.0, 1.2), (-3.0, -0.4, 0.9)])[:V]
    w2c = np.repeat(np.stack([_look_at(c) for c in centres])[:, None], T, 1)
    xyz0 = rng.uniform(-0.5, 0.5, (N, 3))
    vel = rng.normal(0, 0.015, (N, 3))
    vel[:3] = 0.0
    xyz = (xyz0[None] + vel[None] * np.arange(T)[:, None, None]).astype(np.float32)
    valid = np.ones((T, N), bool)
    if invalid:
        valid[15:, -2:] = False
        xyz[~valid] = np.nan
    infr = np.zeros((T, N, V), bool)
    for v in range(V):
        for t in range(T):
            uv, z = project(xyz[t], K, w2c[v, t])
            infr[t, :, v] = valid[t] & (z > 0.1) & inside(uv, (W, H))
    vis = infr & (rng.random((T, N, V)) > 0.25)
    vis[:T0 + 3, 2, :] = False
    vis[:, 3, :] = False
    dgt = valid.copy()
    if invalid:
        dgt[8:10, 0] = False
    np.savez(d / "tracks3d.npz", xyz=xyz, valid=valid, visible=vis, in_frustum=infr, default_gt=dgt,
             group=np.zeros(N, np.int16), group_names=np.array(["target"]), part=np.zeros(N, np.int16),
             part_names=np.array(["p"]), kind=np.zeros(N, np.int8), kind_names=np.array(KIND_NAMES),
             side=np.full(N, -1, np.int8))
    np.savez(d / "cameras.npz", view_names=np.array([f"view{v}" for v in range(V)]),
             source_view_names=np.array([f"cam{v}" for v in range(V)]), image_size=np.array([[W, H]] * V, np.int32),
             K=np.repeat(K[None], V, 0), w2c=w2c, static=np.ones(V, bool))
    (d / "captions.json").write_text(json.dumps({"caption": "points move."}))
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid, "n_views": V,
            "n_frames": T, "fps": 10.0,
            "views": [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W, "height": H} for v in range(V)],
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": None, "labels": {}},
            "tracks": {"n_points": N, "groups": ["target"], "default_report_group": "target"},
            "caption": "points move.", "attributes": {}, "flags": [], "depth": None}
    (d / "sample.json").write_text(json.dumps(meta))
    for v in range(V):
        write_video(d / "videos" / f"view{v}.mp4", rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8), fps=10.0)
    return d


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("vpmwb2d")
    make_sample(root, "Simul-General", "g000", seed=1)
    make_sample(root, "Simul-General", "g001", seed=2, invalid=True)
    make_sample(root, "Real-Robot", "r000", V=2, W=80, H=48, seed=3)
    return Benchmark(root)


def P(**kw):
    return t2.Track2DProtocol(ref_frame=T0, horizon=HZ, **kw)


def test_protocol(bench, tmp_path):
    s = bench.sample("Simul-General/g000")
    p = P(eval_size={"rr": [96, 54]}, resize_mode={"real_robot": "letterbox"}, corr_frames=[1, 20])
    assert p.eval_size_for("Real-Robot") == (96, 54) and p.eval_size_for("Simul-General") == (256, 256)
    assert p.resize_mode_for("Real-Robot") == "letterbox" and p.resize_mode_for("sg") == "resize"
    assert set(p.eval_size) == {"Real-Robot"}
    assert p.window(s).tolist() == list(range(T0, T0 + HZ + 1)) and p.corr_frames_for() == [1, 20]
    assert P().corr_frames_for() == [T0 // 2] and t2.Track2DProtocol().corr_frames_for() == [8]
    assert t2.Track2DProtocol(ref_frame=0, horizon=8).corr_frames_for() == []
    assert t2.Track2DProtocol(ref_frame=20, horizon=None).window(s).tolist() == [20, 21, 22, 23]
    q = P(corr_frames=[2, 30])
    assert q.evaluable(s)[0] is True and q.corr_frames_for(s) == [2]
    assert t2.corr_case(s, q)["frames_dropped"] == [30]
    assert P(views=[7]).evaluable(s)[0] is False
    assert t2.Track2DProtocol(ref_frame=T0, horizon=40).evaluable(s)[0] is False
    p.save(tmp_path / "p.json")
    assert t2.Track2DProtocol.load(tmp_path / "p.json") == p
    assert t2.Track2DProtocol.load(tmp_path / "p.json").hash() == p.hash() != P().hash()
    for bad in ({"query_mode": "random"}, {"max_points": 0}, {"corr_max_points": 0}, {"resize_mode": "stretch"},
                {"query_stride": 0}, {"eval_size": {"nope": (8, 8)}},
                {"corr_frames": [T0]}, {"corr_frames": [T0 + 3]}):
        with pytest.raises(ValueError):
            P(**bad)
    for bad in ((0, 64), (64,), {"rr": (64, -1)}):
        with pytest.raises(ValueError, match="eval_size must be"):
            P(eval_size=bad)
    assert P(corr_frames=[T0 + 3], corr=False).corr is False
    assert t2.Track2DProtocol(ref_frame=4, frame_step=2, horizon=4, corr_frames=[5, 14]).corr_frames == [5, 14]
    with pytest.raises(ValueError):
        t2.Track2DProtocol(ref_frame=4, horizon=None, corr_frames=[50])
    with pytest.raises(KeyError):
        t2.Track2DProtocol.from_dict({"nope": 1})


def test_view_gt_matches_sample_tracks2d(bench):
    s = bench.sample("Simul-General/g001")
    p = P(eval_size=(128, 96))
    win = p.window(s)
    uv, vis = t2.view_gt(s, 1, p, win)
    ruv, rvis, _ = s.tracks2d(1, size=(128, 96), indices=win)
    known = s.tracks.default_gt[win]
    np.testing.assert_array_equal(vis, rvis & known)
    np.testing.assert_allclose(uv[known], ruv[known])
    assert np.isnan(uv[~known]).all()


def test_query_modes(bench):
    s = bench.sample("Simul-General/g000")
    win = P().window(s)
    uv, vis = t2.view_gt(s, 0, P(), win)
    c = t2.view_case(s, 0, P(query_mode="fixed"))
    assert sorted(c["point_index"]) == np.nonzero(vis[0])[0].tolist() and (c["query_t"] == 0).all()
    c = t2.view_case(s, 0, P(query_mode="first"))
    assert sorted(c["point_index"]) == np.nonzero(vis.any(0))[0].tolist() and 3 not in c["point_index"]
    assert not np.all(np.diff(c["point_index"]) > 0)
    for p_, q_ in zip(c["point_index"], c["query_t"]):
        assert vis[q_, p_] and not vis[:q_, p_].any()
    assert c["query_t"][list(c["point_index"]).index(2)] >= 3
    np.testing.assert_allclose(c["query_uv"], uv[c["query_t"], c["point_index"]])
    np.testing.assert_allclose(c["gt_uv"], uv[:, c["point_index"]])
    c = t2.view_case(s, 0, P(query_mode="strided", query_stride=5))
    expect = {(j, p_) for j in (0, 5, 10) for p_ in np.nonzero(vis[j])[0]}
    got = list(zip(c["query_t"].tolist(), c["point_index"].tolist()))
    assert set(got) == expect and len(got) == len(expect) and got != sorted(got)
    assert c["gt_uv"].shape == (len(win), len(expect), 2)
    assert np.array_equal(t2.view_case(s, 0, P(query_mode="strided"))["point_index"], c["point_index"])
    a = t2.view_case(s, 0, P(query_mode="fixed", max_points=4))
    b = t2.view_case(s, 0, P(query_mode="fixed", max_points=4))
    assert len(a["point_index"]) == 4 and np.array_equal(a["point_index"], b["point_index"])
    assert set(a["point_index"]) <= set(np.nonzero(vis[0])[0])
    assert not np.array_equal(t2.view_case(s, 0, P(query_mode="fixed", seed=1))["point_index"],
                              t2.view_case(s, 0, P(query_mode="fixed"))["point_index"])


def test_corr_case_one_source_view_per_point(bench):
    s = bench.sample("Simul-General/g000")
    p = P(corr_frames=[1, 2])
    c = t2.corr_case(s, p)
    P_ = len(c["frame"])
    assert P_ > 20 and set(c["frame"].tolist()) == {1, 2} and c["frames"] == [1, 2] and c["frames_dropped"] == []
    assert not np.all(np.diff(c["frame"]) >= 0)
    seen = {}
    for t in (1, 2):
        vis = np.stack([t2.view_gt(s, v, p, [t])[1][0] for v in range(3)])
        rows = np.nonzero(c["frame"] == t)[0]
        pts = set(c["point_index"][rows].tolist())
        assert pts == set(np.nonzero(vis.sum(0) >= 2)[0].tolist())
        for pt in pts:
            r = rows[c["point_index"][rows] == pt]
            src = set(c["view_a"][r].tolist())
            assert len(src) == 1
            a = src.pop()
            assert sorted(c["view_b"][r].tolist()) == [b for b in range(3) if vis[b, pt] and b != a]
            seen[a] = seen.get(a, 0) + 1
    assert len(seen) == 3
    for i in np.random.default_rng(0).choice(P_, 10, replace=False):
        t, a, b, n = c["frame"][i], c["view_a"][i], c["view_b"][i], c["point_index"][i]
        for v, key in ((a, "uv_a"), (b, "uv_b")):
            uv, vis, _ = s.tracks2d(v, size=(256, 256), indices=[t])
            assert vis[0, n]
            np.testing.assert_allclose(c[key][i], uv[0, n])
    cc = t2.corr_case(s, P(corr_max_points=3))
    assert len(set(cc["point_index"].tolist())) == 3
    assert len(t2.corr_case(s, P(views=[1]))["frame"]) == 0


def test_corr_duplicate_points_count_once(tmp_path):
    """Two GT points at the same position must not both become correspondence points (with different source views,
    one would reveal the other's target)."""
    d = make_sample(tmp_path, "Real-Robot", "dup", seed=5)
    z = dict(np.load(d / "tracks3d.npz"))
    for k in ("xyz", "valid", "visible", "in_frustum", "default_gt"):
        z[k][:, 6] = z[k][:, 5]
    np.savez(d / "tracks3d.npz", **z)
    s = Benchmark(tmp_path).sample("Real-Robot/dup")
    c = t2.corr_case(s, P())
    assert 5 in c["point_index"] and 6 not in c["point_index"]
    tgt = {tuple(x) for x in np.round(c["uv_b"], 9)}
    src = {(int(b), tuple(x)) for b, x in zip(c["view_a"], np.round(c["uv_a"], 9))}
    assert not {(int(b), tuple(x)) for b, x in zip(c["view_b"], np.round(c["uv_b"], 9))} & src and tgt


@pytest.mark.parametrize("mode", ["first", "strided", "fixed"])
def test_oracle_perfect_and_frozen_direct(bench, tmp_path, mode):
    p = P(query_mode=mode)
    assert t2.run_baseline(bench, tmp_path / "oracle", "oracle", p) == {"written": 3, "skipped": 0}
    summ = t2.evaluate(bench, tmp_path / "oracle", tmp_path / "res_o", p, verbose=False)
    for sub, s in summ["subsets"].items():
        assert s["n_ok"] == s["n_samples"] and s["n_views_missing"] == 0 and s["n_track_samples"] == s["n_samples"]
        for k in ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy"):
            assert s["track"][k]["mean"] == pytest.approx(100.0), (sub, k)
        assert s["corr"]["avg_pct_within"]["mean"] == 100.0 and s["corr"]["mean_err_px"]["mean"] == 0.0
    t2.run_baseline(bench, tmp_path / "frozen", "frozen", p, jobs=2)
    s = bench.sample("Simul-General/g001")
    rec = t2.evaluate_sample(s, tmp_path / "frozen", p)
    qm = "strided" if mode == "strided" else "first"
    per = []
    for v in range(3):
        c = t2.view_case(s, v, p)
        fr = np.repeat(c["query_uv"][None], c["gt_uv"].shape[0], 0)
        ref = track2d_metrics(c["gt_uv"], c["gt_vis"], fr, None, c["query_t"], (256, 256), query_mode=qm)
        for k, val in ref.items():
            assert rec["views"][f"view{v}"][k] == val or (math.isnan(val) and math.isnan(rec["views"][f"view{v}"][k])), \
                (v, k)
        per.append(ref["average_jaccard"])
    assert rec["track"]["average_jaccard"] == pytest.approx(np.mean(per)) and rec["track"]["n_views"] == 3
    assert np.load(tmp_path / "frozen/Simul-General/g001/view0.npz")["uv"].dtype == np.float64
    cc = t2.corr_case(s, p)
    ref = t2.correspondence_metrics(cc["uv_a"], cc["uv_b"], (256, 256))
    assert rec["corr"]["avg_pct_within"] == pytest.approx(ref["avg_pct_within"]) and rec["corr"]["n"] == len(cc["frame"])
    assert set(rec["corr"]["pairs"]) <= {"0->1", "0->2", "1->0", "1->2", "2->0", "2->1"}
    assert sum(v["n"] for v in rec["corr"]["pairs"].values()) == len(cc["frame"]) and rec["corr"]["frames"] == [2]


def test_eval_grid_and_normalisation(bench, tmp_path):
    p = P(eval_size=(128, 96), resize_mode="center_crop", query_mode="fixed")
    t2.run_baseline(bench, tmp_path / "o", "oracle", p)
    summ = t2.evaluate(bench, tmp_path / "o", tmp_path / "r", p, verbose=False)
    assert summ["subsets"]["Real-Robot"]["track"]["average_jaccard"]["mean"] == pytest.approx(100.0)
    s = bench.sample("Real-Robot/r000")
    c = t2.view_case(s, 1, p)
    gtuv, _, _ = s.tracks2d(1, size=(128, 96), mode="center_crop", indices=p.window(s))
    np.testing.assert_allclose(c["gt_uv"][np.isfinite(c["gt_uv"])],
                               gtuv[:, c["point_index"]][np.isfinite(c["gt_uv"])])
    pred = c["gt_uv"] + np.array([1.5, 0.0])
    t2.write_prediction(tmp_path / "x", s, 1, np.where(np.isfinite(pred), pred, 0.0), c["gt_vis"])
    r = t2.evaluate_sample(s, tmp_path / "x", p)["views"]["view1"]
    assert r["pts_within_1"] == 0.0 and r["pts_within_2"] == 0.0 and r["pts_within_4"] == 100.0
    p2 = P(eval_size=(128, 96), resize_mode="center_crop", query_mode="fixed", normalize_to=None)
    r = t2.evaluate_sample(s, tmp_path / "x", p2)["views"]["view1"]
    assert r["pts_within_1"] == 0.0 and r["pts_within_2"] == 100.0


def test_prediction_ids_errors_and_failures(bench, tmp_path):
    p = P(query_mode="strided")
    s = bench.sample("Simul-General/g000")
    c = t2.view_case(s, 0, p)
    o = t2.oracle_prediction(s, 0, p, c)
    Nt = len(c["query_t"])
    perm = np.random.default_rng(1).permutation(Nt)
    t2.write_prediction(tmp_path, s, 0, o["uv"][:, perm], o["vis"][:, perm], track_id=perm)
    assert t2.evaluate_sample(s, tmp_path, p)["views"]["view0"]["average_jaccard"] == pytest.approx(100.0)
    bad = perm.copy()
    bad[0] = Nt + 3
    t2.write_prediction(tmp_path, s, 0, o["uv"], o["vis"], track_id=bad)
    with pytest.raises(ValueError, match="permutation"):
        t2.evaluate_sample(s, tmp_path, p)
    t2.write_prediction(tmp_path, s, 0, o["uv"][:, :-1])
    with pytest.raises(ValueError, match="shape"):
        t2.evaluate_sample(s, tmp_path, p)
    uv = o["uv"].copy()
    j = int(np.nonzero(c["gt_vis"][:, 0] & (np.arange(len(uv)) != c["query_t"][0]))[0][0])
    uv[j, 0] = np.nan
    vis = o["vis"].astype(np.float32)
    t2.write_prediction(tmp_path, s, 0, uv, vis)
    r = t2.evaluate_sample(s, tmp_path, p)["views"]["view0"]
    assert r["nonfinite_frac"] > 0 and r["pts_within_16"] == pytest.approx(100.0 * (1 - 1 / r["n_visible"]))
    assert r["occlusion_accuracy"] == pytest.approx(100.0)
    vis[j, 0] = np.nan
    t2.write_prediction(tmp_path, s, 0, o["uv"], vis)
    r = t2.evaluate_sample(s, tmp_path, p)["views"]["view0"]
    assert r["occlusion_accuracy"] < 100.0 and r["pts_within_1"] == pytest.approx(100.0)
    t2.write_corr_prediction(tmp_path, s, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="corr"):
        t2.evaluate_sample(s, tmp_path, p)
    cc = t2.corr_case(s, p)
    qp = np.random.default_rng(2).permutation(len(cc["frame"]))
    t2.write_corr_prediction(tmp_path, s, cc["uv_b"][qp], query_id=qp)
    assert t2.evaluate_sample(s, tmp_path, p)["corr"]["avg_pct_within"] == 100.0


def test_missing_views_make_the_sample_missing(bench, tmp_path):
    p = P()
    s = bench.sample("Simul-General/g000")
    c = t2.view_case(s, 1, p)
    t2.write_prediction(tmp_path / "pred", s, 1, **t2.oracle_prediction(s, 1, p, c))
    rec = t2.evaluate_sample(s, tmp_path / "pred", p)
    assert rec["status"] == "missing" and rec["track"]["status"] == "missing"
    assert rec["track"]["missing_views"] == ["view0", "view2"] and "average_jaccard" not in rec["track"]
    assert rec["views"]["view1"]["average_jaccard"] == pytest.approx(100.0)
    assert rec["corr"]["status"] == "missing" and "view0, view2, corr" in rec["reason"]
    summ = t2.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, verbose=False)
    sg = summ["subsets"]["Simul-General"]
    assert (sg["n_ok"], sg["n_missing"], sg["n_track_samples"], sg["n_track_missing"], sg["n_views_missing"],
            sg["n_corr_missing"]) == (0, 2, 0, 2, 5, 2)
    assert "| Simul-General | 0 / 2 | 2 (5) |" in (tmp_path / "res/summary.md").read_text()
    for kw in ({}, {"corr": False}):
        with pytest.raises(FileNotFoundError, match="missing predictions"):
            t2.evaluate(bench, tmp_path / "pred", tmp_path / f"err{len(kw)}", P(**kw), on_missing="error",
                        verbose=False)
    for v in (0, 2):
        t2.write_prediction(tmp_path / "pred", s, v, **t2.frozen_prediction(s, v, p))
    rec = t2.evaluate_sample(s, tmp_path / "pred", p)
    assert rec["status"] == "partial" and rec["track"]["status"] == "ok" and rec["corr"]["status"] == "missing"
    assert t2.evaluate_sample(s, tmp_path / "pred", P(corr=False))["status"] == "ok"
    with pytest.raises(FileNotFoundError):
        t2.evaluate(bench, tmp_path / "pred", tmp_path / "err2", p, ids=[s.uid], on_missing="error", verbose=False)
    summ = t2.evaluate(bench, tmp_path / "pred", tmp_path / "ok", P(corr=False), ids=[s.uid], on_missing="error",
                       verbose=False)
    assert summ["subsets"]["Simul-General"]["n_track_samples"] == 1


def test_correspondence_metrics_hand_computed():
    gt = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pred = gt + np.array([[0.5, 0.0], [0.0, 3.0], [12.0, 0.0], [np.nan, 0.0]])
    r = t2.correspondence_metrics(pred, gt, (256, 256))
    assert (r["pct_within_1"], r["pct_within_2"], r["pct_within_4"], r["pct_within_8"], r["pct_within_16"]) == \
        (25.0, 25.0, 50.0, 50.0, 75.0)
    assert r["avg_pct_within"] == pytest.approx(45.0) and math.isinf(r["mean_err_px"])
    assert r["median_err_px"] == pytest.approx(7.5) and r["n"] == 4
    r = t2.correspondence_metrics(pred[:3], gt[:3], (128, 128))
    assert r["pct_within_1"] == 0.0 and r["pct_within_2"] == pytest.approx(100 / 3)
    assert r["mean_err_px"] == pytest.approx(31 / 3)
    assert t2.correspondence_metrics(np.zeros((0, 2)), np.zeros((0, 2)), (8, 8))["n"] == 0


def test_sample_inputs_and_export(bench, tmp_path):
    from vpm_worldbench.io.video import read_frames
    p = P(eval_size=(32, 24), corr_frames=[1])
    s = bench.sample("Real-Robot/r000")
    inp = t2.sample_inputs(s, p)
    n = HZ + 1
    assert inp["views"] == [0, 1] and inp["frames"][1].shape == (n, 24, 32, 3) and inp["w2c"].shape == (2, n, 4, 4)
    np.testing.assert_allclose(inp["K"][0], s.K(0, (32, 24)))
    c = t2.view_case(s, 1, p)
    q = inp["queries"][1]
    assert set(q) == {"track_id", "query_t", "query_frame", "query_uv"}
    np.testing.assert_allclose(q["query_uv"], c["query_uv"])
    assert q["track_id"].tolist() == list(range(len(c["query_t"])))
    assert q["query_frame"].tolist() == (T0 + c["query_t"]).tolist()
    cr = inp["corr"]
    assert set(cr) == {"query_id", "frame", "view_a", "view_b", "uv_a", "frames", "frames_dropped", "w2c", "images"}
    assert cr["images"][0].shape == (1, 24, 32, 3) and cr["w2c"].shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(cr["images"][1][0], s.frames(1, [1], size=(32, 24))[0])
    assert t2.export_inputs(bench, tmp_path / "mp4", p) == {"exported": 3, "skipped": 0, "existing": 0}
    d = tmp_path / "mp4/Real-Robot/r000"
    rec = json.loads((d / "inputs.json").read_text())
    np.testing.assert_array_equal(read_frames(d / "frames/view1.mp4"), inp["frames"][1])
    z = np.load(d / "queries_view1.npz")
    assert sorted(z.files) == ["query_frame", "query_t", "query_uv", "track_id"]
    np.testing.assert_allclose(z["query_uv"], c["query_uv"])
    assert rec["prediction"]["view1"]["shape"] == [n, len(c["query_t"]), 2] and rec["n_tracks"]["view1"] == len(z["query_t"])
    cq = np.load(d / "corr/queries.npz")
    assert sorted(cq.files) == ["frame", "query_id", "uv_a", "view_a", "view_b"]
    assert len(cq["frame"]) == rec["corr"]["n"] and (d / "corr/view0_frame_0001.png").exists()
    assert rec["corr"]["frames"] == [1] and np.asarray(rec["corr"]["cameras"]["view1"]["w2c"]).shape == (1, 4, 4)
    assert len((tmp_path / "mp4/manifest.jsonl").read_text().splitlines()) == 3
    assert t2.export_inputs(bench, tmp_path / "mp4", p)["existing"] == 3
    t2.export_inputs(bench, tmp_path / "png", p, ids=["Real-Robot/r000"], frames="png")
    assert len(list((tmp_path / "png/Real-Robot/r000/frames/view0").glob("*.png"))) == n
    t2.export_inputs(bench, tmp_path / "none", p, ids="Real-Robot/r000", frames="none")
    f = json.loads((tmp_path / "none/Real-Robot/r000/inputs.json").read_text())["frames"]["files"]["view0"]
    m = f["native_to_eval"]
    assert f["video"] == "Real-Robot/r000/videos/view0.mp4" and (m["sx"], m["sy"]) == (32 / 80, 24 / 48)
    assert not list((tmp_path / "none/Real-Robot/r000/corr").glob("*.png"))
    assert t2.export_inputs(bench, tmp_path / "par", P(corr=False), frames="none", jobs=2)["exported"] == 3
    assert json.loads((tmp_path / "par/Simul-General/g000/inputs.json").read_text())["corr"] is None


def _exported(d: Path) -> tuple:
    rec = json.loads((d / "inputs.json").read_text())
    tq = {k: dict(np.load(d / f"queries_view{k}.npz")) for k in rec["views"]}
    cq = dict(np.load(d / "corr/queries.npz")) if rec["corr"] else None
    return rec, tq, cq


@pytest.mark.parametrize("mode", ["first", "strided", "fixed"])
def test_exported_inputs_do_not_reveal_targets(bench, tmp_path, mode):
    """Lookup-only predictors built from the exported inputs (no frames, no GT files) stay at the frozen / identity
    level: no point ids, no reverse correspondence pairs, no tracking query at a corr frame."""
    p = P(query_mode=mode, corr_frames=[2, 20])
    t2.export_inputs(bench, tmp_path / "in", p, frames="none")
    t2.run_baseline(bench, tmp_path / "frozen", "frozen", p)
    for line in (tmp_path / "in/manifest.jsonl").read_text().splitlines():
        m = json.loads(line)
        s = bench.sample(m["uid"])
        rec, tq, cq = _exported(tmp_path / "in" / m["dir"])
        corr_frames = set(cq["frame"].tolist())
        assert corr_frames <= {2, 20} and not corr_frames & set(rec["window_indices"])
        for k, q in tq.items():
            assert set(q) == {"track_id", "query_t", "query_frame", "query_uv"}
            assert q["track_id"].tolist() == list(range(len(q["track_id"])))
            assert not set(q["query_frame"].tolist()) & corr_frames
        assert set(cq) == {"query_id", "frame", "view_a", "view_b", "uv_a"}
        assert cq["query_id"].tolist() == list(range(len(cq["frame"])))
        c = t2.corr_case(s, p)
        for t in corr_frames:
            sel = c["frame"] == t
            src = {}
            for a, pt in zip(c["view_a"][sel], c["point_index"][sel]):
                src.setdefault(int(pt), set()).add(int(a))
            assert all(len(v) == 1 for v in src.values())
        tgt = {tuple(np.round(x, 9)) for x in c["uv_b"]}
        same_t = [tuple(np.round(x, 9)) for x in cq["uv_a"]]
        assert not tgt & set(same_t)
        for k, q in tq.items():
            n = rec["n_window"]
            ids, qt, quv = q["track_id"], q["query_t"], q["query_uv"]
            uv = np.zeros((n, len(ids), 2))
            for tid in np.unique(ids):
                sel = np.nonzero(ids == tid)[0]
                o = np.argsort(qt[sel])
                ts, us = qt[sel][o], quv[sel][o]
                uv[:, sel] = np.stack([np.interp(np.arange(n), ts, us[:, 0]),
                                       np.interp(np.arange(n), ts, us[:, 1])], -1)[:, None]
            t2.write_prediction(tmp_path / "cheat", s, k, uv)
        t2.write_corr_prediction(tmp_path / "cheat", s, cq["uv_a"])
    a = t2.evaluate(bench, tmp_path / "cheat", tmp_path / "rc", p, verbose=False)
    b = t2.evaluate(bench, tmp_path / "frozen", tmp_path / "rf", p, verbose=False)
    for sub in a["subsets"]:
        for part in ("track", "corr"):
            for k, v in b["subsets"][sub][part].items():
                assert a["subsets"][sub][part][k]["mean"] == pytest.approx(v["mean"], nan_ok=True), (sub, part, k)


def test_cli(bench, tmp_path, capsys):
    from vpm_worldbench import cli
    root, pred, out = str(bench.root), str(tmp_path / "pred"), str(tmp_path / "res")
    common = ["--ref-frame", str(T0), "--horizon", str(HZ), "--query-mode", "strided", "--size", "64x64"]
    assert t2.main(["baseline", "--root", root, "--pred", pred, "--kind", "oracle", "--jobs", "2"] + common) == 0
    assert cli.main(["track2d", "evaluate", "--root", root, "--pred", pred, "--out", out, "--jobs", "2"] + common) == 0
    s = json.loads(Path(out, "summary.json").read_text())
    assert s["protocol"]["eval_size"] == [64, 64] and s["protocol"]["query_mode"] == "strided"
    assert s["protocol"]["name"].startswith("track2d-v1+") and "query_mode=strided" in s["protocol"]["name"]
    assert s["subsets"]["Real-Robot"]["track"]["average_jaccard"]["mean"] == pytest.approx(100.0)
    assert t2.main(["summarize", "--out", out]) == 0 and "track2d" in capsys.readouterr().out
    with pytest.raises(ValueError, match="different protocol"):
        t2.main(["evaluate", "--root", root, "--pred", pred, "--out", out, "--ref-frame", str(T0), "--horizon",
                 str(HZ), "--size", "64x64"])
    assert t2.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--frames", "none", "--no-corr",
                    "--ids", "Simul-General/g000", "Real-Robot/r000", "--subsets", "sg,rr"] + common) == 0
    assert len((tmp_path / "in/manifest.jsonl").read_text().splitlines()) == 2
    assert t2.main(["export", "--root", root, "--out", str(tmp_path / "in2"), "--frames", "none", "--corr-frames", "1",
                    "20", "--ids", "Simul-General/g000"] + common) == 0
    assert json.loads((tmp_path / "in2/protocol.json").read_text())["corr_frames"] == [1, 20]
    with pytest.raises(SystemExit) as e:
        t2.main(["export", "--root", root, "--out", str(tmp_path / "in3"), "--corr-frames", str(T0 + 1)] + common)
    assert e.value.code == 2 and "tracking-window frames" in capsys.readouterr().err


def test_out_dir_run_identity_and_failed_records(bench, tmp_path, capsys):
    from vpm_worldbench import cli
    p = P()
    oracle, frozen, out = tmp_path / "oracle", tmp_path / "frozen", tmp_path / "res"
    t2.run_baseline(bench, oracle, "oracle", p)
    t2.run_baseline(bench, frozen, "frozen", p)
    a = t2.evaluate(bench, oracle, out, p, verbose=False)
    assert a["pred_root"] == str(oracle.resolve()) and a["evaluated"]["status"] == {"ok": 3}
    with pytest.raises(RunDirError, match="predictions in"):
        t2.evaluate(bench, frozen, out, p, subsets=["rr"], verbose=False)
    capsys.readouterr()
    assert cli.main(["track2d", "evaluate", "--root", str(bench.root), "--pred", str(frozen), "--out", str(out),
                     "--ref-frame", str(T0), "--horizon", str(HZ), "--subsets", "rr"]) == 2
    assert capsys.readouterr().err.startswith("vpmwb: error: ")
    b = t2.evaluate(bench, frozen, out, p, subsets=["rr"], overwrite=True, verbose=False)
    assert list(b["subsets"]) == ["Real-Robot"] and b["pred_root"] == str(frozen.resolve())
    assert [f.relative_to(out).as_posix() for f in out.glob("*/*.json")] == ["Real-Robot/r000.json"]
    assert b["subsets"]["Real-Robot"]["track"]["average_jaccard"]["mean"] < 100.0
    bad = tmp_path / "bad"
    t2.run_baseline(bench, bad, "oracle", p)
    s = bench.sample("Simul-General/g000")
    c = t2.view_case(s, 0, p)
    o = t2.oracle_prediction(s, 0, p, c)
    t2.write_prediction(bad, s, 0, o["uv"][:, :-1])
    (bad / "Real-Robot/r000/corr.npz").write_bytes(b"garbage")
    with pytest.warns(UserWarning, match=r"2 sample\(s\) failed \(status invalid / error\), e.g. Simul-General/g000"):
        summ = t2.evaluate(bench, bad, tmp_path / "rb", p, verbose=False)
    rec = json.loads((tmp_path / "rb/Simul-General/g000.json").read_text())
    assert rec["status"] == "invalid" and "'uv' has shape" in rec["reason"] and "traceback" not in rec
    assert "not a readable .npz" in json.loads((tmp_path / "rb/Real-Robot/r000.json").read_text())["reason"]
    sg = summ["subsets"]["Simul-General"]
    assert (sg["n_samples"], sg["n_ok"], sg["n_invalid"], sg["n_track_samples"]) == (2, 1, 1, 1)
    assert summ["evaluated"]["status"] == {"invalid": 2, "ok": 1} and not summ["evaluated"]["all_failed"]
    assert "| Simul-General | 1 / 2 | 0 (0); 1 invalid |" in (tmp_path / "rb/summary.md").read_text()
    for kw, msg in (({"uv": o["uv"].astype(str)}, "must hold real numbers"),
                    ({"uv": o["uv"], "track_id": np.arange(o["uv"].shape[1]) + 0.5}, "integer ids"),
                    ({"uv": o["uv"], "vis": o["vis"][:-1]}, "'vis' has shape"),
                    ({"xy": o["uv"]}, "needs 'uv'")):
        np.savez(bad / "Simul-General/g000/view0.npz", **kw)
        with pytest.raises(InvalidPrediction, match=msg):
            t2.evaluate_sample(s, bad, p)
    with pytest.raises(ValueError, match="needs 'uv'"):
        t2.evaluate(bench, bad, tmp_path / "rb", p, ids=[s.uid], strict=True, verbose=False)
    args = ["evaluate", "--root", str(bench.root), "--pred", str(bad), "--ref-frame", str(T0), "--horizon", str(HZ)]
    with pytest.warns(UserWarning, match="failed"):
        assert t2.main(args + ["--out", str(tmp_path / "c1"), "--ids", "sg/g000", "rr/r000"]) == 1
    assert "every evaluated sample failed (invalid 2)" in capsys.readouterr().err
    with pytest.warns(UserWarning, match="failed"):
        assert t2.main(args + ["--out", str(tmp_path / "c2")]) == 0


def test_cli_frame_step_horizon_and_parser_errors(bench, tmp_path, capsys):
    root, pred, out = str(bench.root), str(tmp_path / "pred"), tmp_path / "res"
    base = ["evaluate", "--root", root, "--pred", pred, "--out", str(out)]
    for args, msg in ((base + ["--protocol", "vpm-v1"], "unknown track2d protocol 'vpm-v1': not a preset (track2d-v1)"),
                      (base + ["--subsets", "qq"], "argument --subsets: unknown subset 'qq'"),
                      (base + ["--size", "64by64"], "--size must be WxH"),
                      (base + ["--size", "0x64"], "--size must be WxH (e.g. 256x256), got '0x64'"),
                      (base + ["--frame-step", "0"], "frame_step >= 1"),
                      (["evaluate", "--root", str(tmp_path / "nope"), "--pred", pred, "--out", str(out)],
                       "no such directory"),
                      (["summarize", "--out", str(out)], "--out " + str(out) + ": no such directory")):
        with pytest.raises(SystemExit) as e:
            t2.main(args)
        assert e.value.code == 2 and msg in capsys.readouterr().err, args
    assert not out.exists()
    common = ["--ref-frame", str(T0), "--frame-step", "3", "--horizon", "0", "--subsets", "sg"]
    assert t2.main(["baseline", "--root", root, "--pred", pred, "--kind", "oracle"] + common) == 0
    assert t2.main(base + common) == 0
    pj = json.loads((out / "protocol.json").read_text())
    assert pj["frame_step"] == 3 and pj["horizon"] is None
    assert pj["name"] == f"track2d-v1+ref_frame={T0},horizon=None,frame_step=3"
    rec = json.loads((out / "Simul-General/g000.json").read_text())
    assert rec["status"] == "ok" and rec["window"] == [T0, 22, 3] and rec["n_window"] == 7
    assert rec["track"]["average_jaccard"] == pytest.approx(100.0) and rec["corr"]["avg_pct_within"] == 100.0
    with pytest.warns(UserWarning, match=r"track2d: 1 requested id\(s\) match no selected sample"):
        assert t2.main(base + common + ["--ids", "nope"]) == 1
    assert "no sample matches the selection" in capsys.readouterr().err


RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")


@pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT (a VP-WorldBench data root)")
def test_released_oracle_and_frozen(tmp_path):
    bench = Benchmark(RELEASED)
    ids = [f"{sub}/{bench.ids(sub)[0]}" for sub in bench.subsets()]
    p = t2.Track2DProtocol()
    t2.run_baseline(bench, tmp_path / "o", "oracle", p, ids=ids)
    summ = t2.evaluate(bench, tmp_path / "o", tmp_path / "ro", p, ids=ids, verbose=False)
    for sub, s in summ["subsets"].items():
        assert s["n_ok"] == 1 and s["track"]["average_jaccard"]["mean"] == pytest.approx(100.0), sub
        assert s["corr"]["avg_pct_within"]["mean"] == pytest.approx(100.0), sub
    t2.run_baseline(bench, tmp_path / "f", "frozen", p, ids=ids)
    summ = t2.evaluate(bench, tmp_path / "f", tmp_path / "rf", p, ids=ids, verbose=False)
    for sub, s in summ["subsets"].items():
        assert 0.0 < s["track"]["average_pts_within_thresh"]["mean"] < 100.0, sub
