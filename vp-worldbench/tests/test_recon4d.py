"""Tests of vpm_worldbench.tasks.recon4d on the synthetic benchmark of test_common.py (plane z = 0, static and moving
rigs, a sample without depth, a sample with a distortion border)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.geometry import camera_center
from vpm_worldbench.tasks import recon4d as R

_spec = importlib.util.spec_from_file_location("_vpmwb_test_common", Path(__file__).with_name("test_common.py"))
TC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TC)

GRID = {s: (64, 48) for s in TC.SPECS}


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return TC.make_bench(tmp_path_factory.mktemp("synthbench_r4d"))


def _proto(**kw):
    kw.setdefault("eval_size", GRID)
    return R.Recon4DProtocol(ref_frame=2, horizon=8, frame_step=2, **kw)


def _rand_sim(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return 2.5, Rm, rng.normal(size=3) * 5.0


def _transform(points, w2c, s, Rm, t):
    """Express a reconstruction in the world X' = s R X + t (points and world-to-camera poses)."""
    P = s * np.asarray(points, np.float64) @ Rm.T + t
    W = np.array(w2c, np.float64, copy=True)
    Rc = W[..., :3, :3] @ Rm.T
    W[..., :3, :3] = Rc
    W[..., :3, 3] = s * np.asarray(w2c)[..., :3, 3] - np.einsum("...ij,j->...i", Rc, t)
    return P, W


def _rec(out, uid):
    sub, sid = uid.split("/")
    return json.loads((Path(out) / sub / f"{sid}.json").read_text())


def _write(root, uid, **arrays):
    d = Path(root) / uid
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "recon4d.npz", **arrays)


def _means(summary, sub):
    return {k: v["mean"] for k, v in summary["subsets"][sub]["metrics"].items()}


def test_protocol_defaults():
    p = R.Recon4DProtocol()
    assert p.window_indices == list(range(16, 81, 4))
    assert p.eval_size_for("Real-General") == (256, 192) and p.max_depth_for("Simul-Robot") == 3.0
    assert p.fscore_for("Real-General") == (0.02, 0.05, 0.10) and p.ladder_for("Real-Robot")[0] == 0.01
    assert p.tolerance_for("sr") == (0.02, 0.05) and p.voxel_for("sg") is None and p.track_interp == "nearest"
    q = R.Recon4DProtocol.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.to_dict() == p.to_dict()
    a = R.Recon4DProtocol(eval_size={"rr": (32, 18)}, voxel_size={"sg": 0.1}, track_consistency=None)
    assert a.eval_size_for("Real-Robot") == (32, 18) and a.voxel_for("Simul-General") == 0.1
    assert a.tolerance_for("Simul-Robot") is None
    for bad in (dict(align="affine"), dict(track_interp="cubic"), dict(fscore_m={"xx": (0.1,)})):
        with pytest.raises(ValueError):
            R.Recon4DProtocol(**bad)


def test_robust_similarity_recovers_transform_with_outliers():
    rng = np.random.default_rng(0)
    G = rng.uniform(-1, 1, (2000, 3))
    s, Rm, t = _rand_sim(1)
    P = (G - t) @ Rm / s
    P_bad = P.copy()
    P_bad[:30] *= 500.0
    P_bad[30:200] += rng.normal(scale=0.3, size=(170, 3))
    for robust, ok in ((True, True), (False, False)):
        s2, R2, t2, n = R.robust_similarity(P_bad, G, "sim3", robust=robust)
        err = np.linalg.norm(s2 * P @ R2.T + t2 - G, axis=1).max()
        assert (err < 1e-6) == ok, (robust, err)
    with pytest.raises(ValueError):
        R.robust_similarity(P[:2], G[:2], "sim3")


def test_oracle_is_perfect(bench, tmp_path):
    p = _proto()
    R.run_baseline(bench, tmp_path / "pred", kind="oracle", protocol=p, progress=False)
    s = R.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True)
    for sub, r in s["subsets"].items():
        m = {k: v["mean"] for k, v in r["metrics"].items()}
        assert r["status"] == {"ok": 1}
        assert m["camera/rra@5"] == 100.0 and m["camera/rta@5"] == 100.0 and m["camera/auc@30"] == 100.0
        assert m["camera/rpe_trans_rmse"] < 1e-9 and m["camera/focal_rel_err"] < 1e-9
        assert m["camera/coverage"] == 100.0 and m["align/failed_pct"] == 0.0
        assert m["tracks/coverage"] == 100.0 and m["tracks/err_median"] < 0.05
        assert m[f"tracks/delta@{p.ladder_for(sub)[-1]:g}"] == 100.0
        if sub == "Simul-General":
            assert m["camera/ate_rmse"] < 1e-9 and m["camera/are_mean_deg"] < 1e-6
        else:
            assert "camera/ate_rmse" not in m and "camera/are_mean_deg" not in m
        if TC.SPECS[sub][2]:
            assert m["points/chamfer_mean"] < 1e-5 and m[f"points/fscore@{p.fscore_for(sub)[0]:g}"] == 100.0
            assert m["depth/abs_rel"] < 1e-6 and m["depth/delta1"] == 100.0
            assert m["points/coverage"] == 100.0
        else:
            assert "points/chamfer_mean" not in m and m["tracks/err_median"] < 1e-6
    rec = _rec(tmp_path / "res", "Simul-General/s000")
    assert rec["camera"]["cameras_move"] and rec["camera"]["centres_move"] and rec["align"]["on"] == "depth"
    assert not rec["camera"]["gt_degenerate"] and not rec["camera"]["pred_degenerate"]
    assert not _rec(tmp_path / "res", "Simul-Robot/s000")["camera"]["cameras_move"]
    rr = _rec(tmp_path / "res", "Real-Robot/s000")
    assert rr["align"]["on"] == "tracks" and rr["camera"]["gt_degenerate"]
    assert "## Simul-General" in (tmp_path / "res" / "summary.md").read_text()


def test_similarity_invariance_and_depth_input(bench, tmp_path):
    """A similarity-transformed oracle scores like the oracle; so does depth + K + w2c at another resolution."""
    p = _proto()
    R.run_baseline("oracle", bench, tmp_path / "oracle", p, progress=False)
    s_, Rm, t_ = _rand_sim(3)
    ids = ["Simul-Robot/s000", "Simul-General/s000", "Real-Robot/s000"]
    for uid in ids:
        z = np.load(tmp_path / "oracle" / uid / "recon4d.npz")
        P, W = _transform(z["points"], z["w2c"], s_, Rm, t_)
        _write(tmp_path / "sim", uid, points=P, w2c=W, K=z["K"])
    s = R.evaluate(bench, tmp_path / "sim", tmp_path / "res_sim", p, progress=False, strict=True, ids=ids)
    for sub, r in s["subsets"].items():
        m = {k: v["mean"] for k, v in r["metrics"].items()}
        assert m["camera/rra@5"] == 100.0 and m["camera/rpe_trans_rmse"] < 1e-6
        assert m["tracks/err_median"] < 0.05
        if sub != "Real-Robot":
            assert m["points/chamfer_mean"] < 1e-5 and m["depth/abs_rel"] < 1e-6
        if sub == "Simul-General":
            assert m["camera/ate_rmse"] < 1e-6 and m["camera/are_mean_deg"] < 1e-6
    smp = bench.sample("Simul-General/s000")
    win = p.window_indices
    dep = np.stack([[TC.plane_depth(smp.K(v, (128, 96)), smp.cameras.w2c[v, t], 128, 96) for v in range(3)]
                    for t in win])
    W = _transform(np.zeros((1, 3)), smp.cameras.w2c[:, win], s_, Rm, t_)[1]
    _write(tmp_path / "dep", "Simul-General/s000", depth=dep * s_, w2c=W,
           K=np.stack([smp.K(v, (128, 96)) for v in range(3)]))
    err = {}
    for interp in ("nearest", "bilinear"):
        q = _proto(track_interp=interp)
        s = R.evaluate(bench, tmp_path / "dep", tmp_path / f"res_dep_{interp}", q, progress=False, strict=True,
                       ids=["Simul-General/s000"])
        m = _means(s, "Simul-General")
        assert m["camera/ate_rmse"] < 1e-6 and m["camera/focal_rel_err"] < 1e-9
        assert m["points/fscore@0.1"] == 100.0 and m["points/acc_mean"] < 0.03
        assert m["depth/delta1"] == 100.0 and m["tracks/coverage"] == 100.0
        err[interp] = m["tracks/err_median"]
    assert err["bilinear"] < err["nearest"] < 0.05


def test_cameras_only_and_points_only(bench, tmp_path):
    p = _proto()
    smp = bench.sample("Simul-Robot/s000")
    _write(tmp_path / "cam", "Simul-Robot/s000", w2c=smp.cameras.w2c[:, 0])
    s = R.evaluate(bench, tmp_path / "cam", tmp_path / "res_cam", p, progress=False, strict=True,
                   ids=["Simul-Robot/s000"])
    m = s["subsets"]["Simul-Robot"]["metrics"]
    assert m["camera/rra@5"]["mean"] == 100.0 and not any(k.startswith(("points", "tracks", "align")) for k in m)
    R.run_baseline("oracle", bench, tmp_path / "orc", p, ids=["Simul-Robot/s000"], progress=False)
    z = np.load(tmp_path / "orc" / "Simul-Robot" / "s000" / "recon4d.npz")
    _write(tmp_path / "pts", "Simul-Robot/s000", points=z["points"])
    s = R.evaluate(bench, tmp_path / "pts", tmp_path / "res_pts", p, progress=False, strict=True,
                   ids=["Simul-Robot/s000"])
    m = s["subsets"]["Simul-Robot"]["metrics"]
    assert "camera/rra@5" not in m and m["points/chamfer_mean"]["mean"] < 1e-5
    _write(tmp_path / "pts", "Simul-Robot/s000", points=z["points"][:, :2])
    with pytest.raises(ValueError, match="points"):
        R.load_prediction(tmp_path / "pts", smp, p)
    with pytest.warns(UserWarning, match="1 sample"):
        s = R.evaluate(bench, tmp_path / "pts", tmp_path / "res_bad", p, progress=False, ids=["Simul-Robot/s000"])
    assert s["subsets"]["Simul-Robot"]["status"] == {"invalid": 1} and s["evaluated"]["all_failed"]
    _write(tmp_path / "pts", "Simul-Robot/s000", points=z["points"].astype(str))
    with pytest.raises(R.InvalidPrediction, match="real numbers"):
        R.load_prediction(tmp_path / "pts", smp, p)
    _write(tmp_path / "tr", "Simul-Robot/s000", points=np.swapaxes(z["points"], 2, 3))
    with pytest.warns(UserWarning, match="with warnings"):
        s = R.evaluate(bench, tmp_path / "tr", tmp_path / "res_tr", p, progress=False, strict=True,
                       ids=["Simul-Robot/s000"])
    rec = _rec(tmp_path / "res_tr", "Simul-Robot/s000")
    assert rec["status"] == "ok" and len(rec["warnings"]) == 3 and "swapped" in rec["warnings"][0]
    assert s["subsets"]["Simul-Robot"]["n_warnings"] == 1


def test_degenerate_predicted_cameras_are_scored(bench, tmp_path):
    """Moving rig (Simul-General): identity poses (coincident centres) and collinear centres are degenerate
    PREDICTIONS; ATE and ARE are still summarised (ARE = the failure value 180 deg), so a collapsed camera estimate is
    penalised instead of leaving the mean."""
    p = _proto()
    uid = "Simul-General/s000"
    R.run_baseline("oracle", bench, tmp_path / "orc", p, ids=[uid], progress=False)
    z = np.load(tmp_path / "orc" / uid / "recon4d.npz")
    V, n = z["w2c"].shape[:2]
    _write(tmp_path / "ident", uid, w2c=np.tile(np.eye(4), (V, n, 1, 1)))
    W = z["w2c"].copy()
    Cc = camera_center(W)
    C2 = np.zeros_like(Cc)
    C2[..., 0] = Cc[..., 0]
    W[..., :3, 3] = -np.einsum("...ij,...j->...i", W[..., :3, :3], C2)
    _write(tmp_path / "line", uid, w2c=W)
    gt_c = camera_center(bench.sample(uid).cameras.w2c[:, p.window_indices]).reshape(-1, 3)
    spread = float(np.sqrt(np.mean(np.sum((gt_c - gt_c.mean(0)) ** 2, -1))))
    for name in ("ident", "line"):
        s = R.evaluate(bench, tmp_path / name, tmp_path / f"res_{name}", p, ids=[uid], progress=False, strict=True)
        m = _means(s, "Simul-General")
        rec = _rec(tmp_path / f"res_{name}", uid)["camera"]
        assert rec["pred_degenerate"] and not rec["gt_degenerate"], name
        assert m["camera/are_mean_deg"] == R.ARE_FAIL_DEG and m["camera/ate_rmse"] > 0.05, (name, m)
        if name == "ident":
            assert m["camera/ate_rmse"] == pytest.approx(spread, rel=1e-9)
            assert m["camera/rra@5"] < 100.0 and m["camera/rta@5"] == 0.0


def test_gappy_tracks_are_split_by_bridged_path(tmp_path):
    """Moving GT tracks with a validity gap on every other frame still move (bridged path), so they are dynamic."""
    b = TC.make_bench(tmp_path / "bench", subsets=("Simul-Robot",), gappy=True)
    p = _proto()
    R.run_baseline("oracle", b, tmp_path / "orc", p, progress=False)
    R.evaluate(b, tmp_path / "orc", tmp_path / "res", p, progress=False, strict=True)
    rec = _rec(tmp_path / "res", "Simul-Robot/s000")["tracks"]
    s = b.sample("Simul-Robot/s000")
    obs_moving = sum(int(s.tracks.visible[p.window_indices][:, 30:, v].sum()) for v in range(3))
    assert rec["dynamic"]["n_obs"] > 0 and rec["dynamic"]["n_obs"] <= obs_moving
    assert rec["static"]["n_obs"] + rec["dynamic"]["n_obs"] == rec["all"]["n_obs"]


def test_failed_alignment_scores_as_failure(bench, tmp_path):
    """Fewer than 3 finite points: the alignment fails and every block reports explicit failure values."""
    p = _proto()
    uid = "Simul-Robot/s000"
    R.run_baseline("oracle", bench, tmp_path / "orc", p, ids=[uid], progress=False)
    z = dict(np.load(tmp_path / "orc" / uid / "recon4d.npz"))
    pts = np.full_like(z["points"], np.nan)
    pts[0, 0, 0, :2] = z["points"][0, 0, 0, :2]
    _write(tmp_path / "two", uid, points=pts)
    s = R.evaluate(bench, tmp_path / "two", tmp_path / "res", p, ids=[uid], progress=False, strict=True)
    m = s["subsets"]["Simul-Robot"]["metrics"]
    assert m["align/failed_pct"]["mean"] == 100.0
    assert m["points/acc_mean"]["n_inf"] == 1 and m["points/coverage"]["mean"] == 0.0
    assert m["depth/abs_rel"]["n_inf"] == 1 and m["depth/delta1"]["mean"] == 0.0 and m["depth/coverage"]["mean"] == 0
    assert m["tracks/delta_avg"]["mean"] == 0.0 and m["tracks/err_median"]["n_inf"] == 1
    assert "(1 inf)" in (tmp_path / "res" / "summary.md").read_text()


def test_frozen_baseline_fails_on_dynamic_points(bench, tmp_path):
    p = _proto()
    out = {}
    for name in ("oracle", "frozen"):
        R.run_baseline(name, bench, tmp_path / name, p, progress=False)
        out[name] = R.evaluate(bench, tmp_path / name, tmp_path / f"res_{name}", p, progress=False, strict=True)
    m, o = _means(out["frozen"], "Simul-Robot"), _means(out["oracle"], "Simul-Robot")
    assert m["tracks_static/delta_avg"] == pytest.approx(o["tracks_static/delta_avg"])
    assert m["tracks_static/err_median"] == pytest.approx(o["tracks_static/err_median"])
    m, o = _means(out["frozen"], "Real-Robot"), _means(out["oracle"], "Real-Robot")
    assert o["tracks_dynamic/delta_avg"] > 90.0, o
    assert m["tracks_dynamic/delta_avg"] < 50.0, m
    assert m["tracks_static/delta_avg"] > m["tracks_dynamic/delta_avg"]
    g = _means(out["frozen"], "Simul-General")
    assert g["camera/rra@5"] == 100.0 and g["camera/ate_rmse"] > 0.05 and g["camera/rpe_trans_rmse"] > 0.05
    assert 0.0 <= g["camera/are_mean_deg"] < 1.0


def test_gross_outliers_do_not_break_alignment(bench, tmp_path):
    p = _proto()
    R.run_baseline("oracle", bench, tmp_path / "orc", p, ids=["Simul-Robot/s000"], progress=False)
    z = dict(np.load(tmp_path / "orc" / "Simul-Robot" / "s000" / "recon4d.npz"))
    pts = z["points"].astype(np.float64)
    rng = np.random.default_rng(0)
    bad = rng.random(pts.shape[:-1]) < 0.01
    pts[bad] *= 300.0
    z["points"] = pts
    _write(tmp_path / "bad", "Simul-Robot/s000", **z)
    s = R.evaluate(bench, tmp_path / "bad", tmp_path / "res", p, ids=["Simul-Robot/s000"], progress=False,
                   strict=True)
    m = _means(s, "Simul-Robot")
    assert m["depth/delta1"] > 98.0 and m["points/comp_median"] < 1e-6
    assert m["points/acc_mean"] > 1.0


def test_missing_export_and_cli(bench, tmp_path):
    p = _proto()
    s = R.evaluate(bench, tmp_path / "nothing", tmp_path / "res", p, progress=False, ids=["Real-Robot/s000"])
    assert s["subsets"]["Real-Robot"]["status"] == {"missing": 1}
    r = R.export_inputs(bench, tmp_path / "in", p, frames="none", progress=False)
    assert r == {"exported": 4, "skipped": 0}
    rec = json.loads((tmp_path / "in" / "Simul-General" / "s000" / "inputs.json").read_text())
    assert "K" in rec["frames"]["view0"] and "w2c" not in rec["frames"]["view0"]
    q = _proto(provide_intrinsics=False, provide_cameras=True)
    R.export_inputs(bench, tmp_path / "in2", q, frames="none", ids=["Simul-General/s000"], progress=False)
    rec = json.loads((tmp_path / "in2" / "Simul-General" / "s000" / "inputs.json").read_text())
    assert "K" not in rec["frames"]["view0"] and np.array(rec["frames"]["view0"]["w2c"]).shape == (9, 4, 4)
    p.save(tmp_path / "p.json")
    root = str(bench.root)
    assert R.main(["baseline", "--kind", "frozen", "--root", root, "--pred", str(tmp_path / "fz"), "--protocol",
                   str(tmp_path / "p.json"), "--subsets", "rr"]) == 0
    assert R.main(["evaluate", "--root", root, "--pred", str(tmp_path / "fz"), "--out", str(tmp_path / "res_fz"),
                   "--protocol", str(tmp_path / "p.json"), "--subsets", "rr"]) == 0
    assert json.loads((tmp_path / "res_fz" / "summary.json").read_text())["subsets"]["Real-Robot"]["status"] == \
        {"ok": 1}
