"""Tests of vpm_worldbench.tasks.depth on the synthetic benchmark of test_common.py (`make_bench`: a ground plane seen
by static or moving cameras, analytic depth, GT tracks on the plane)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.metrics import depth as dm
from vpm_worldbench.tasks import common as C
from vpm_worldbench.tasks import depth as D

_spec = importlib.util.spec_from_file_location("_vpmwb_test_common", Path(__file__).with_name("test_common.py"))
TC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TC)

SMALL = TC.SMALL
GRID = {s: (64, 48) for s in TC.SPECS}


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return TC.make_bench(tmp_path_factory.mktemp("synthbench_depth"))


def _proto(**kw):
    kw.setdefault("eval_size", GRID)
    return D.DepthProtocol(**SMALL, **kw)


def _rec(out, uid):
    sub, sid = uid.split("/")
    return json.loads((Path(out) / sub / f"{sid}.json").read_text())


def test_protocol_defaults_and_roundtrip(tmp_path):
    p = D.DepthProtocol(eval_size={"Simul-Robot": (32, 32)}, align=("none", "scale_shift"), **SMALL)
    assert p.window_indices == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    p.save(tmp_path / "p.json")
    q = D.DepthProtocol.load(tmp_path / "p.json")
    assert q.to_dict() == p.to_dict() and q.eval_size_for("Simul-Robot") == (32, 32)
    d = D.DepthProtocol()
    assert d.eval_size_for("Real-General") == (640, 480) and d.max_depth_for("Simul-Robot") == 80.0
    assert d.tolerance_for("Simul-Robot") == (0.02, 0.05) and d.tolerance_for("rg") == (0.05, 0.05)
    assert D.DepthProtocol(sparse_consistency=None).tolerance_for("sr") is None
    assert D.DepthProtocol(sparse_consistency=[0.1, 0.0]).tolerance_for("sg") == (0.1, 0.0)
    assert D.DepthProtocol(sparse_consistency={"sg": None}).tolerance_for("Simul-General") is None
    with pytest.raises(ValueError):
        D.DepthProtocol(align=("bogus",))
    with pytest.raises(ValueError):
        D.DepthProtocol(sparse_consistency={"nowhere": (0.1, 0.1)})


def test_score_depth_matches_metrics_depth():
    """The single-pass reduction equals metrics.depth.depth_metrics; delta1_all counts missing pixels as failures."""
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 5.0, (4, 10, 12))
    gt[1] = 0.0
    pred = gt * rng.uniform(0.7, 1.4, gt.shape)
    pred[2, :5] = np.nan
    pred[3] = np.nan
    mask = rng.random(gt.shape) > 0.1
    for reduce in ("frames", "pixels"):
        ref = dm.depth_metrics(pred, gt, mask, max_depth=80.0, reduce=reduce)
        got = D.score_depth(pred, gt, mask, dm.DEFAULT_MIN_DEPTH, 80.0, reduce)
        for k in dm.METRIC_KEYS + ("n_valid", "n_scored", "coverage", "n_frames"):
            assert got[k] == pytest.approx(ref[k], rel=1e-12, abs=1e-12), (reduce, k)
    per = dm.depth_metrics_per_frame(pred, gt, mask, max_depth=80.0)
    v = mask & (gt > dm.DEFAULT_MIN_DEPTH)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(pred / gt, gt / pred)
        good = v & np.isfinite(pred) & (ratio < 1.25)
    frames = [f for f in range(4) if v[f].any()]
    exp = np.mean([100.0 * good[f].sum() / v[f].sum() for f in frames])
    assert D.score_depth(pred, gt, mask, dm.DEFAULT_MIN_DEPTH, 80.0, "frames")["delta1_all"] == pytest.approx(exp)
    assert D.score_depth(pred, gt, mask, dm.DEFAULT_MIN_DEPTH, 80.0, "pixels")["delta1_all"] == pytest.approx(
        100.0 * good.sum() / v.sum())
    assert per["n_valid"][3] > 0


def test_oracle_is_perfect(bench, tmp_path):
    p = _proto()
    D.run_baseline(bench, tmp_path / "pred", kind="oracle", protocol=p, progress=False)
    s = D.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True)
    for sub in ("Simul-Robot", "Simul-General", "Real-General"):
        m = s["subsets"][sub]["metrics"]
        for mode in D.ALIGN_DEFAULT:
            assert m[f"dense/{mode}/abs_rel"]["mean"] == pytest.approx(0.0, abs=1e-6)
            assert m[f"dense/{mode}/delta1"]["mean"] == 100.0 and m[f"dense/{mode}/delta1_all"]["mean"] == 100.0
            assert m[f"dense/{mode}/coverage"]["mean"] == 100.0
        assert m["sparse/none/delta1"]["mean"] == 100.0 and m["sparse/none/abs_rel"]["mean"] < 0.01
    rr = s["subsets"]["Real-Robot"]["metrics"]
    assert "dense/none/abs_rel" not in rr
    assert rr["sparse/none/abs_rel"]["mean"] < 1e-2 and rr["sparse/none/delta1"]["mean"] == 100.0
    assert all(r["status"] == {"ok": 1} for r in s["subsets"].values())
    md = (tmp_path / "res" / "summary.md").read_text()
    assert "## Simul-Robot" in md and "dense / scale_shift / AbsRel" in md and "constant prediction" in md
    rec = _rec(tmp_path / "res", "Simul-Robot/s000")
    assert rec["status"] == "ok" and rec["window"] == {"first": 2, "last": 18, "step": 2, "n": 9}
    assert set(rec["per_view"]) == {"view0", "view1", "view2"} and rec["sparse_tolerance"] == [0.02, 0.05]


def test_constant_baseline_matches_direct_computation(bench, tmp_path):
    p = _proto(align=("none", "median", "scale_shift"))
    D.run_baseline("constant", bench, tmp_path / "pred", p, progress=False, value=2.0)
    D.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True)
    rec = _rec(tmp_path / "res", "Real-General/s000")
    s = bench.sample("Real-General/s000")
    for v in range(3):
        gt = C.gt_depth(s, v, p.window_indices, (64, 48))
        mask = np.broadcast_to(C.fov_mask(s, v, (64, 48)), gt.shape)
        pred = np.full(gt.shape, 2.0)
        for mode in ("none", "median", "scale_shift"):
            ref = dm.evaluate_depth(pred, gt, mask, align_mode=mode, per="sequence", space=D.ALIGN_SPACE[mode],
                                    max_depth=20.0)
            got = rec["per_view"][f"view{v}"]["dense"][mode]
            for k in ("abs_rel", "rmse", "delta1", "coverage"):
                assert got[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-12), (v, mode, k)
            assert got["delta1_all"] == pytest.approx(ref["delta1"], rel=1e-9)
    assert rec["metrics"]["dense/none/abs_rel"] > 0.1


def test_missing_pixels_and_views_are_penalised(bench, tmp_path):
    """A prediction with an all-NaN view and holes: AbsRel / delta1 cover the predicted pixels only, coverage and
    delta1_all count the missing ones as failures (the missing view enters the sample mean with 0)."""
    p = _proto(align=("none", "scale_shift"))
    D.run_baseline("oracle", bench, tmp_path / "orc", p, ids=["Simul-General/s000"], progress=False)
    d = tmp_path / "holes" / "Simul-General" / "s000"
    d.mkdir(parents=True)
    for k in range(3):
        x = np.load(tmp_path / "orc" / "Simul-General" / "s000" / f"view{k}.npz")["depth"].copy()
        if k == 1:
            x[:] = np.nan
        if k == 2:
            x[:, :24] = np.nan
        np.savez(d / f"view{k}.npz", depth=x)
    D.evaluate(bench, tmp_path / "holes", tmp_path / "res", p, ids=["Simul-General/s000"], progress=False,
               strict=True)
    rec = _rec(tmp_path / "res", "Simul-General/s000")
    m = rec["metrics"]
    assert m["dense/none/abs_rel"] == pytest.approx(0.0, abs=1e-6) and m["dense/none/delta1"] == 100.0
    pv = rec["per_view"]
    assert pv["view1"]["dense"]["none"]["coverage"] == 0.0 and pv["view1"]["dense"]["none"]["delta1_all"] == 0.0
    assert 30.0 < pv["view2"]["dense"]["none"]["delta1_all"] < 70.0
    v2 = pv["view2"]["dense"]["none"]["delta1_all"]
    assert m["dense/none/delta1_all"] == pytest.approx(np.mean([100.0, 0.0, v2]))
    assert m["dense/none/coverage"] < 60.0 and m["dense/coverage"] == m["dense/none/coverage"]


def test_sparse_block_uses_the_dense_fit(bench, tmp_path):
    """A prediction 2x the GT: the scene-level scale fit (dense) also corrects the sparse samples; without dense GT
    (Real-Robot) the sparse points are fitted themselves."""
    p = _proto(align=("none", "scale", "scale_shift"))
    D.run_baseline("oracle", bench, tmp_path / "orc", p, subsets=["Simul-Robot", "Real-Robot"], progress=False)
    for sub in ("Simul-Robot", "Real-Robot"):
        for f in (tmp_path / "orc" / sub / "s000").glob("view*.npz"):
            d = np.load(f)["depth"]
            dst = tmp_path / "x2" / sub / "s000" / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, depth=2.0 * d)
    D.evaluate(bench, tmp_path / "x2", tmp_path / "res", p, subsets=["Simul-Robot", "Real-Robot"], progress=False,
               strict=True)
    for sub in ("Simul-Robot", "Real-Robot"):
        rec = _rec(tmp_path / "res", f"{sub}/s000")
        m = rec["metrics"]
        assert m["sparse/none/abs_rel"] == pytest.approx(1.0, abs=0.02)
        assert m["sparse/scale/abs_rel"] < 0.02 and m["sparse/scale_shift/abs_rel"] < 0.02
        fitted = rec["per_view"]["view0"]["sparse"]["scale"]["align"]["fitted_on"]
        assert fitted == ("dense" if sub == "Simul-Robot" else "self")
        if sub == "Simul-Robot":
            sc = rec["per_view"]["view0"]["dense"]["scale"]["align"]["scale"]
            assert rec["per_view"]["view0"]["sparse"]["scale"]["align"]["scale"] == sc == pytest.approx(0.5)


def test_prediction_formats(bench, tmp_path):
    """full-clip maps, frame_indices, disparity, other resolutions, missing and malformed files."""
    p = _proto(align=("none",))
    s = bench.sample("Simul-Robot/s000")
    root = tmp_path / "pred"
    d = root / "Simul-Robot" / "s000"
    d.mkdir(parents=True)
    full = C.gt_depth(s, 0, range(24))
    np.savez(d / "view0.npz", depth=full)
    sel = [18, 2, 4, 6, 8, 10, 12, 14, 16, 0]
    np.savez(d / "view1.npz", disparity=1.0 / np.maximum(C.gt_depth(s, 1, sel), 1e-6),
             frame_indices=np.array(sel))
    up = np.repeat(np.repeat(C.gt_depth(s, 2, p.window_indices), 2, 1), 2, 2)
    np.savez(d / "view2.npz", depth=up)
    D.evaluate(bench, root, tmp_path / "res", p, ids=["Simul-Robot/s000"], progress=False, strict=True)
    rec = _rec(tmp_path / "res", "Simul-Robot/s000")
    assert rec["status"] == "ok" and rec["prediction"]["keys"] == ["depth", "disparity"]
    for v in ("view0", "view1"):
        assert rec["per_view"][v]["dense"]["none"]["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert rec["per_view"]["view2"]["pred_size"] == [128, 96]
    assert rec["per_view"]["view2"]["dense"]["none"]["delta1"] > 99.0
    assert rec["per_view"]["view2"]["dense"]["none"]["coverage"] == 100.0
    np.savez(d / "view2.npz", depth=up[:3])
    with pytest.warns(UserWarning, match="1 sample"):
        sm = D.evaluate(bench, root, tmp_path / "res2", p, ids=["Simul-Robot/s000", "Simul-General/s000"],
                        progress=False)
    r1, r2 = _rec(tmp_path / "res2", "Simul-Robot/s000"), _rec(tmp_path / "res2", "Simul-General/s000")
    assert r1["status"] == "invalid" and "3 maps" in r1["reason"]
    assert r2["status"] == "missing"
    assert sm["evaluated"]["all_failed"] and sm["evaluated"]["status"] == {"invalid": 1, "missing": 1}
    for bad in ({"depth": up.astype(bool)}, {"depth": up, "frame_indices": np.full(len(up), 2.5)}, {"other": up}):
        np.savez(d / "view2.npz", **bad)
        with pytest.raises(D.InvalidPrediction):
            D.load_prediction(root, s, 2, p)
    (d / "view2.npz").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="not a readable .npz"):
        D.load_prediction(root, s, 2, p)
    with pytest.raises(ValueError):
        D.evaluate(bench, root, tmp_path / "res2", _proto(), progress=False)


def test_native_resolution_oracle_with_holes(bench, tmp_path):
    """A native-resolution prediction with holes, resized to a coarser eval grid: bilinear keeps full coverage of the
    predicted area (holes do not grow) and stays near-exact; nearest is exact."""
    s = bench.sample("Simul-General/s000")
    for interp, tol in (("auto", 1e-6), ("nearest", 1e-6), ("bilinear", 0.02)):
        p = _proto(align=("none",), eval_size={k: (40, 30) for k in GRID}, resize_interp=interp)
        d = tmp_path / f"nat_{interp}" / "Simul-General" / "s000"
        d.mkdir(parents=True)
        for k in range(3):
            x = C.gt_depth(s, k, p.window_indices).astype(np.float32)
            x[:, 10:12, 20:22] = np.nan
            np.savez(d / f"view{k}.npz", depth=x)
        D.evaluate(bench, tmp_path / f"nat_{interp}", tmp_path / f"res_{interp}", p, ids=["Simul-General/s000"],
                   progress=False, strict=True)
        rec = _rec(tmp_path / f"res_{interp}", "Simul-General/s000")
        m = rec["metrics"]
        assert m["dense/none/abs_rel"] < tol and m["dense/none/coverage"] > 99.0, (interp, m)
        assert rec["per_view"]["view0"]["resize_interp"] == ("nearest" if interp == "auto" else interp)
    assert D.DepthProtocol().resize_interp_for((8, 8), (40, 30)) == "bilinear"
    assert D.DepthProtocol().resize_interp_for((64, 30), (40, 30)) == "nearest"


def test_not_evaluable_window(bench, tmp_path):
    p = D.DepthProtocol(eval_size=GRID, ref_frame=16, horizon=64)
    s = D.evaluate(bench, tmp_path / "none", tmp_path / "res", p, ids=["Simul-Robot/s000"], progress=False)
    assert s["subsets"]["Simul-Robot"]["status"] == {"skipped": 1}
    assert not s["evaluated"]["all_failed"]


def test_transposed_prediction_is_flagged(bench, tmp_path):
    """Maps saved as (n, W, H) are scored as given (any resolution is accepted) but flagged."""
    p = _proto(align=("none",))
    s = bench.sample("Real-General/s000")
    d = tmp_path / "pred" / "Real-General" / "s000"
    d.mkdir(parents=True)
    for k in range(3):
        g = C.gt_depth(s, k, p.window_indices)
        np.savez(d / f"view{k}.npz", depth=np.swapaxes(g, 1, 2) if k == 1 else g)
    with pytest.warns(UserWarning, match="with warnings.*view1.*swapped"):
        sm = D.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, ids=["rg/s000"], progress=False, strict=True)
    rec = _rec(tmp_path / "res", "Real-General/s000")
    assert rec["status"] == "ok" and len(rec["warnings"]) == 1 and "48x64" in rec["warnings"][0]
    assert "warning" in rec["per_view"]["view1"] and "warning" not in rec["per_view"]["view0"]
    assert sm["subsets"]["Real-General"]["n_warnings"] == 1
    assert "1 sample(s) with warnings" in (tmp_path / "res" / "summary.md").read_text()


def test_export_inputs(bench, tmp_path):
    from vpm_worldbench.io.video import read_frames
    p = _proto()
    for frames in ("png", "mp4", "none"):
        out = tmp_path / frames
        r = D.export_inputs(bench, out, p, subsets=["Simul-Robot"], frames=frames, progress=False)
        assert r == {"exported": 1, "skipped": 0}
        rec = json.loads((out / "Simul-Robot" / "s000" / "inputs.json").read_text())
        assert rec["frame_indices"] == p.window_indices and rec["views"] == ["view0", "view1", "view2"]
        assert np.allclose(rec["frames"]["view1"]["K"], bench.sample("Simul-Robot/s000").cameras.K[1])
        assert len((out / "manifest.jsonl").read_text().splitlines()) == 1
        assert json.loads((out / "protocol.json").read_text())["name"] == "depth-v1"
    s = bench.sample("Simul-Robot/s000")
    import cv2
    png = cv2.imread(str(tmp_path / "png" / "Simul-Robot" / "s000" / "view0" / "frame_0004.png"))[..., ::-1]
    assert np.array_equal(png, s.frames(0, [4])[0])
    assert read_frames(tmp_path / "mp4" / "Simul-Robot" / "s000" / "view0.mp4").shape == (9, 48, 64, 3)
    assert json.loads((tmp_path / "none" / "Simul-Robot" / "s000" / "inputs.json").read_text())["frames"]["view0"][
        "file"] is None


def test_cli_roundtrip(bench, tmp_path):
    p = _proto(align=("none",))
    p.save(tmp_path / "proto.json")
    root = str(bench.root)
    assert D.main(["baseline", "--kind", "oracle", "--root", root, "--pred", str(tmp_path / "pred"), "--protocol",
                   str(tmp_path / "proto.json"), "--subsets", "Simul-Robot"]) == 0
    assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"),
                   "--protocol", str(tmp_path / "proto.json"), "--subsets", "Simul-Robot", "--jobs", "2"]) == 0
    assert D.main(["summarize", "--out", str(tmp_path / "res")]) == 0
    sm = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert sm["subsets"]["Simul-Robot"]["metrics"]["dense/none/delta1"]["mean"] == 100.0
