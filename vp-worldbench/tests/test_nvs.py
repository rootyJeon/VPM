"""Tests of vpm_worldbench.tasks.nvs on the synthetic benchmark of test_common.py. SSIM is scored only when
scikit-image is installed and LPIPS only when torch + lpips are (the [metrics] extra), so the tests pass on a plain
[dev] install."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.geometry import camera_center
from vpm_worldbench.io.video import read_frames, write_video
from vpm_worldbench.metrics.appearance import PSNR_MSE_FLOOR, appearance
from vpm_worldbench.predictions import load_reference
from vpm_worldbench.tasks import nvs as N

_spec = importlib.util.spec_from_file_location("_vpmwb_test_common", Path(__file__).with_name("test_common.py"))
TD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TD)

GRID = {s: (64, 48) for s in TD.SPECS}
PSNR_MAX = 10 * np.log10(255.0 ** 2 / PSNR_MSE_FLOOR)
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return TD.make_bench(tmp_path_factory.mktemp("synthbench_nvs"))


def _proto(**kw):
    kw.setdefault("eval_size", GRID)
    kw.setdefault("lpips", False)
    kw.setdefault("ssim", HAS_SKIMAGE)
    return N.NVSProtocol(ref_frame=2, horizon=8, frame_step=2, **kw)


def _rec(out, uid):
    sub, sid = uid.split("/")
    return json.loads((Path(out) / sub / f"{sid}.json").read_text())


def test_protocol_targets(bench):
    p = N.NVSProtocol()
    s = bench.sample("Simul-General/s000")
    assert p.target_for(s) == 2 and p.sources_for(s) == [0, 1]
    q = N.NVSProtocol(target_view={"sg": 0})
    assert q.target_view == {"Simul-General": 0}
    assert q.target_for(s) == 0 and q.sources_for(s) == [1, 2]
    with pytest.raises(ValueError):
        N.NVSProtocol(target_view={"Simul-Generall": 0})
    assert q.target_for(bench.sample("Real-Robot/s000")) == 1
    assert not N.NVSProtocol(target_view=5).evaluable(s)[0]
    assert not N.NVSProtocol(views=[2], target_view=2).evaluable(s)[0]
    g = _proto().generation_protocol()
    assert g.window_indices == [2, 4, 6, 8, 10, 12, 14, 16, 18] and g.eval_size_for("Real-Robot") == (64, 48)
    with pytest.raises(ValueError):
        N.NVSProtocol(horizon=0)


def test_oracle_reaches_the_psnr_ceiling(bench, tmp_path):
    p = _proto()
    N.run_baseline(bench, tmp_path / "pred", kind="oracle", protocol=p, progress=False)
    s = N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True)
    for sub, r in s["subsets"].items():
        assert r["status"] == {"ok": 1}
        assert r["metrics"]["psnr"]["mean"] == pytest.approx(PSNR_MAX)
        if HAS_SKIMAGE:
            assert r["metrics"]["ssim"]["mean"] == pytest.approx(1.0)
        else:
            assert "ssim" not in r["metrics"]
        assert "lpips" not in r["metrics"]
    rec = _rec(tmp_path / "res", "Real-General/s000")
    assert rec["n_frames"] == 9 and len(rec["per_frame"]["psnr"]) == 9 and rec["fov_masked_px"] > 0
    assert "## Real-Robot" in (tmp_path / "res" / "summary.md").read_text()


def test_nearest_view_matches_direct_computation(bench, tmp_path):
    p = _proto()
    s = bench.sample("Simul-General/s000")
    win = p.window_indices
    choice = N.nearest_source_views(s, p)
    ct = camera_center(s.cameras.w2c[2, win])
    d = [np.linalg.norm(camera_center(s.cameras.w2c[k, win]) - ct, axis=-1) for k in (0, 1)]
    assert choice == [int(np.argmin([d[0][j], d[1][j]])) for j in range(len(win))]
    N.run_baseline("nearest_view", bench, tmp_path / "pred", p, ids=["Simul-General/s000"], progress=False,
                   lossless=True)
    N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True, ids=["Simul-General/s000"])
    rec = _rec(tmp_path / "res", "Simul-General/s000")
    fake = np.stack([s.frames(k, [t])[0] for k, t in zip(choice, win)])
    real = load_reference(s, 2, p.generation_protocol())
    ref = appearance(real, fake, lpips=None, ssim=HAS_SKIMAGE)
    assert rec["metrics"]["psnr"] == pytest.approx(ref["psnr"], rel=1e-12)
    if HAS_SKIMAGE:
        assert rec["metrics"]["ssim"] == pytest.approx(ref["ssim"], rel=1e-12)
    assert rec["metrics"]["psnr"] < 40.0


def test_fov_border_is_not_scored(bench, tmp_path):
    """A prediction that is right inside the valid field of view and wrong outside it scores perfectly."""
    p = _proto()
    s = bench.sample("Real-General/s000")
    real = load_reference(s, 2, p.generation_protocol())
    m = N.fov_mask(s, 2, (64, 48))
    assert m is not None and (~m).sum() > 0
    fake = real.copy()
    fake[:, ~m] = 255 - fake[:, ~m]
    write_video(tmp_path / "pred" / "Real-General" / "s000" / "view2.mp4", fake, fps=5.0, lossless=True)
    N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True, ids=["Real-General/s000"])
    assert _rec(tmp_path / "res", "Real-General/s000")["metrics"]["psnr"] == pytest.approx(PSNR_MAX)
    N.evaluate(bench, tmp_path / "pred", tmp_path / "res_nomask", _proto(use_fov_mask=False), progress=False,
               strict=True, ids=["Real-General/s000"])
    assert _rec(tmp_path / "res_nomask", "Real-General/s000")["metrics"]["psnr"] < 40.0


def test_prediction_meta_and_errors(bench, tmp_path):
    p = _proto()
    s = bench.sample("Simul-Robot/s000")
    real = load_reference(s, 2, p.generation_protocol())
    d = tmp_path / "pred" / "Simul-Robot" / "s000"
    win = p.window_indices
    dense = s.frames(2, list(range(win[0], win[-1] + 1)), size=(128, 96))
    write_video(d / "view2.mp4", dense, fps=10.0, lossless=True)
    (d / "meta.json").write_text(json.dumps({"fps": 10.0}))
    N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True, ids=["Simul-Robot/s000"])
    rec = _rec(tmp_path / "res", "Simul-Robot/s000")
    assert rec["status"] == "ok" and rec["metrics"]["psnr"] > 30.0
    for i, meta in enumerate(({"first_frame": "future"}, {"first_frame": "Future"}, {"fps": 0}, {"fps": -5},
                              {"fps": "abc"}, {"fps": "10"}, {"fps": True}, [1, 2], "{not json")):
        (d / "meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta))
        with pytest.warns(UserWarning, match="1 sample"):
            sm = N.evaluate(bench, tmp_path / "pred", tmp_path / f"res2_{i}", p, progress=False,
                            ids=["Simul-Robot/s000"])
        rec = _rec(tmp_path / f"res2_{i}", "Simul-Robot/s000")
        assert rec["status"] == "invalid" and "meta.json" in rec["reason"], meta
        assert ("not valid JSON" in rec["reason"]) == isinstance(meta, str), rec["reason"]
        assert sm["evaluated"]["all_failed"]
    (d / "meta.json").write_text(json.dumps({"fps": None, "first_frame": "ref"}))
    write_video(d / "view2.mp4", s.frames(2, win[:3]), fps=5.0, lossless=True)
    with pytest.warns(UserWarning, match="1 sample"):
        N.evaluate(bench, tmp_path / "pred", tmp_path / "res4", p, progress=False, ids=["Simul-Robot/s000"])
    assert _rec(tmp_path / "res4", "Simul-Robot/s000")["status"] == "invalid"
    s2 = N.evaluate(bench, tmp_path / "pred", tmp_path / "res3", p, progress=False, ids=["Real-Robot/s000"])
    assert s2["subsets"]["Real-Robot"]["status"] == {"missing": 1} and s2["evaluated"]["all_failed"]
    assert real.shape == (9, 48, 64, 3)


def test_export_excludes_the_target(bench, tmp_path):
    p = _proto()
    r = N.export_inputs(bench, tmp_path / "in", p, frames="mp4", progress=False)
    assert r == {"exported": 4, "skipped": 0}
    d = tmp_path / "in" / "Real-General" / "s000"
    rec = json.loads((d / "inputs.json").read_text())
    assert rec["target"]["view"] == "view2" and sorted(rec["sources"]) == ["view0", "view1"]
    assert not (d / "view2.mp4").exists() and (d / "view0.mp4").exists() and (d / "view1.mp4").exists()
    assert rec["target"]["fov_mask"] == "target_fov_mask.png" and (d / "target_fov_mask.png").exists()
    s = bench.sample("Real-General/s000")
    assert np.allclose(rec["target"]["w2c"], s.cameras.w2c[2, p.window_indices])
    assert np.allclose(rec["target"]["K"], s.K(2, (64, 48)))
    assert np.array(rec["sources"]["view1"]["w2c"]).shape == (9, 4, 4)
    assert read_frames(d / "view0.mp4").shape == (9, 48, 64, 3)
    man = [json.loads(x) for x in (tmp_path / "in" / "manifest.jsonl").read_text().splitlines()]
    assert {m["target_view"] for m in man} == {"view2", "view1"}
    assert json.loads((tmp_path / "in" / "Simul-Robot" / "s000" / "inputs.json").read_text())["target"][
        "fov_mask"] is None


def test_missing_optional_dependencies(bench, tmp_path, monkeypatch, capsys):
    """SSIM / LPIPS without their packages: one message before the first sample (no per-sample errors); `vpmwb`
    prints it as one line `vpmwb: error: ...` and exits with status 2 (VPMWB_TRACEBACK=1: the traceback);
    --no-ssim / --no-lpips turn them off."""
    from vpm_worldbench import cli
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    monkeypatch.setattr(N, "_has_module", lambda name: False)
    p = _proto(ssim=True, lpips=True)
    msg = N.missing_dependencies(p)
    assert "scikit-image" in msg and "torch and lpips" in msg and "vpm-worldbench[metrics]" in msg
    assert "--no-ssim --no-lpips" in msg and N.missing_dependencies(_proto(ssim=False, lpips=False)) is None
    with pytest.raises(ImportError, match="SSIM needs scikit-image"):
        N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False)
    assert not (tmp_path / "res").exists()
    root = str(bench.root)
    p.save(tmp_path / "p.json")
    N.run_baseline(bench, tmp_path / "pred", kind="oracle", protocol=p, ids=["Real-Robot/s000"], progress=False)
    common = ["--root", root, "--protocol", str(tmp_path / "p.json"), "--ids", "Real-Robot/s000", "--pred",
              str(tmp_path / "pred")]
    capsys.readouterr()
    assert cli.main(["nvs", "evaluate", "--out", str(tmp_path / "res"), *common]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: SSIM needs scikit-image") and len(err.strip().splitlines()) == 1
    with pytest.raises(ImportError, match="SSIM needs scikit-image"):
        N.main(["evaluate", "--out", str(tmp_path / "res"), *common])
    monkeypatch.setenv("VPMWB_TRACEBACK", "1")
    with pytest.raises(ImportError, match="SSIM needs scikit-image"):
        cli.main(["nvs", "evaluate", "--out", str(tmp_path / "res"), *common])
    monkeypatch.delenv("VPMWB_TRACEBACK")
    assert not (tmp_path / "res").exists()
    assert N.main(["evaluate", "--out", str(tmp_path / "res"), "--no-ssim", "--no-lpips", *common]) == 0
    sm = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(sm["subsets"]["Real-Robot"]["metrics"]) == {"psnr"}
    assert sm["protocol"]["name"] == "nvs-v1+lpips=False,ssim=False" and not sm["protocol"]["ssim"]


def test_lpips_on_cpu(bench, tmp_path):
    pytest.importorskip("lpips")
    pytest.importorskip("torch")
    p = _proto(lpips=True)
    N.run_baseline("oracle", bench, tmp_path / "pred", p, ids=["Simul-Robot/s000"], progress=False)
    try:
        s = N.evaluate(bench, tmp_path / "pred", tmp_path / "res", p, progress=False, strict=True,
                       ids=["Simul-Robot/s000"], lpips_device="cpu")
    except Exception as e:
        pytest.skip(f"LPIPS unavailable: {e}")
    assert s["subsets"]["Simul-Robot"]["metrics"]["lpips"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_cli_target_view_and_kind(bench, tmp_path, capsys):
    root = str(bench.root)
    _proto().save(tmp_path / "p.json")
    common = ["--root", root, "--protocol", str(tmp_path / "p.json"), "--ids", "Simul-General/s000,Real-Robot/s000",
              "--target-view", "0"]
    assert N.main(["baseline", "--kind", "oracle", "--pred", str(tmp_path / "pred"), *common]) == 0
    assert (tmp_path / "pred" / "Simul-General" / "s000" / "view0.mp4").exists()
    assert N.main(["evaluate", "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"), *common]) == 0
    sm = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert sm["protocol"]["target_view"] == 0 and sm["subsets"]["Real-Robot"]["metrics"]["psnr"]["mean"] == \
        pytest.approx(PSNR_MAX)
    assert sm["protocol"]["name"] == "nvs-v1+target_view=0"
    assert _rec(tmp_path / "res", "Real-Robot/s000")["source_views"] == [1]
    assert N.main(["export", "--out", str(tmp_path / "in"), "--frames", "none", *common]) == 0
    assert json.loads((tmp_path / "in" / "Simul-General" / "s000" / "inputs.json").read_text())["target"][
        "view"] == "view0"
    capsys.readouterr()
    assert N.main(["evaluate", "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res_last"),
                   *common[:-2]]) == 1
    assert "every evaluated sample failed (missing 2)" in capsys.readouterr().err
