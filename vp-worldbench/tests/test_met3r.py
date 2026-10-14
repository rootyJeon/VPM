"""Tests for vpm_worldbench.metrics.met3r.

The default tests run on the CPU: `delta_table` with numpy only, `prepare` / the scorer bookkeeping / the handling of
process-wide torch state / `cross_view` with torch (the official MEt3R stack is replaced by small stand-ins with the
same interface). Opt-in:

  VPMWB_TEST_MET3R=1     CUDA + the official MEt3R stack + weights: cached-feature scores == official forward, the
                         caller's TF32 flags / RNG states survive construction and scoring, scores do not depend on
                         the caller's flags.
"""
import math
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from vpm_worldbench.metrics import met3r as M
from vpm_worldbench.metrics.met3r import ASPECTS, MEt3R, clip_mean, cross_view, delta_table, prepare

try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed (pip install 'vpm-worldbench[met3r]')")


def _frames(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _reference_to_tensor(v, img=256):
    """The pre-release preprocessing: centre square crop, cv2.INTER_AREA to img, x / 127.5 - 1."""
    V, T, h, w, _ = v.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    v = v[:, :, y0:y0 + s, x0:x0 + s]
    out = np.stack([[cv2.resize(f, (img, img), interpolation=cv2.INTER_AREA) for f in vv] for vv in v])
    return torch.from_numpy(out).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0


@pytest.fixture
def restore_backend():
    """Put the process-wide TF32 / cuDNN flags back after a test that changes them on purpose."""
    if torch is None:
        yield
        return
    saved = M._backend_state()
    try:
        yield
    finally:
        M._restore_backend_state(saved)


def _odd_caller_state():
    """A non-default caller state: TF32 off for matmuls and convolutions, cuDNN autotuning on."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    return M._backend_state()


def _expected_scoring_state(tf32):
    new_api = M._backend_state()["fp32_precision"] is not None
    v = "tf32" if tf32 else "ieee"
    return {"benchmark": False, "matmul_precision": "high" if tf32 else "highest", "cudnn_allow_tf32": tf32,
            "fp32_precision": (v,) * len(M._PRECISION_KNOBS) if new_api else None}


def _autocast_cpu_enabled():
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


@pytest.mark.parametrize("hw", [(48, 85), (85, 48), (64, 64), (20, 30)])
@needs_torch
def test_prepare_center_crop_matches_prerelease(hw):
    v = _frames((2, 3, *hw, 3))
    got = prepare(v, size=32, aspect="center_crop")
    ref = _reference_to_tensor(v, 32)
    assert got.shape == (2, 3, 3, 32, 32) and got.dtype == torch.float32
    assert torch.equal(got, ref)


@needs_torch
def test_prepare_resize_squashes_whole_frame():
    v = _frames((1, 2, 48, 64, 3), 1)
    got = prepare(v, size=32, aspect="resize")
    ref = np.stack([cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA) for f in v[0]])
    ref = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    assert torch.equal(got[0], ref)


@needs_torch
def test_prepare_value_range_and_identity_size():
    v = np.zeros((1, 1, 16, 16, 3), np.uint8)
    v[..., 0], v[..., 1], v[..., 2] = 0, 255, 128
    t = prepare(v, size=16)
    assert torch.all(t[0, 0, 0] == -1.0) and torch.all(t[0, 0, 1] == 1.0)
    assert torch.equal(t[0, 0, 2], torch.full((16, 16), 128.0) / 127.5 - 1.0)
    single = prepare(v[0, 0], size=16)
    assert single.shape == (3, 16, 16) and torch.equal(single, t[0, 0])


@needs_torch
def test_prepare_rejects_bad_inputs():
    with pytest.raises(TypeError):
        prepare(np.zeros((1, 1, 8, 8, 3), np.float32))
    with pytest.raises(ValueError):
        prepare(np.zeros((1, 1, 8, 8, 4), np.uint8))
    with pytest.raises(ValueError):
        prepare(np.zeros((1, 1, 8, 8, 3), np.uint8), aspect="letterbox")
    assert ASPECTS == ("center_crop", "resize")


class _StandIn:
    """Mimics met3r.MEt3R: forward(images (b,2,3,S,S) in [-1,1]) maps to [0,1], computes features with
    _get_features/_interpolate on the (b k) batch and returns an ASYMMETRIC pair score + an overlap mask. Records the
    torch state it runs under."""

    def __init__(self):
        self.feature_calls = 0
        self.seen = []
        self.fail = False

    def _record(self):
        self.seen.append({"state": M._backend_state(), "grad": torch.is_grad_enabled(),
                          "autocast": _autocast_cpu_enabled()})

    def _get_features(self, images):
        self.feature_calls += 1
        self._record()
        return torch.cat([images * 2.0, images.mean(1, keepdim=True)], 1)

    def _interpolate(self, lr, images):
        return lr + images[:, :1]

    def __call__(self, images, return_overlap_mask=False):
        self._record()
        if self.fail:
            raise RuntimeError("boom")
        b, k = images.shape[:2]
        im = (images.flatten(0, 1) + 1) / 2
        f = self._interpolate(self._get_features(im), im).reshape(b, k, -1, *im.shape[-2:])
        score = (f[:, 0] - f[:, 1]).abs().mean((1, 2, 3)) + 0.25 * f[:, 0].mean((1, 2, 3))
        mask = (f[:, 1, 0] > 0.5).float()
        return (score, mask) if return_overlap_mask else (score,)


def _standin_scorer(img_size=8, chunk=3):
    s = MEt3R.__new__(MEt3R)
    s.device, s.img_size, s.feature_chunk, s.tf32 = torch.device("cpu"), img_size, chunk, True
    s.raster, s.backbone, s.model = "naive", "mast3r", _StandIn()
    return s


@needs_torch
def test_score_pairs_cached_equals_uncached_forward_and_is_symmetric():
    sc = _standin_scorer()
    x = prepare(_frames((3, 4, 8, 8, 3), 3), size=8)
    pairs = [((0, 0), (1, 0)), ((1, 2), (2, 2)), ((0, 3), (0, 1)), ((2, 1), (0, 0))]
    s, o = sc.score_pairs(x, pairs, batch=3)
    assert s.shape == (4,) and s.dtype == np.float64 and o.shape == (4,)
    calls_after_cache = sc.model.feature_calls
    fwd = torch.stack([torch.stack([x[a], x[b]]) for a, b in pairs])
    bwd = torch.stack([torch.stack([x[b], x[a]]) for a, b in pairs])
    s1, m1 = sc.forward_pairs(fwd)
    s2, m2 = sc.forward_pairs(bwd)
    np.testing.assert_allclose(s, ((s1 + s2) / 2).numpy(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(o, ((m1.mean((-1, -2)) + m2.mean((-1, -2))) / 2).numpy(), rtol=0, atol=1e-7)
    assert sc.model.feature_calls == calls_after_cache + 2
    rev, _ = sc.score_pairs(x, [(b, a) for a, b in pairs], batch=8)
    np.testing.assert_allclose(rev, s, rtol=0, atol=1e-7)
    assert "_get_features" not in vars(sc.model) and "_interpolate" not in vars(sc.model)


@needs_torch
def test_score_pairs_features_only_for_used_images_and_validation():
    sc = _standin_scorer(chunk=100)
    x = prepare(_frames((2, 5, 8, 8, 3), 4), size=8)
    sc.score_pairs(x, [((0, 1), (1, 1))])
    assert sc.model.feature_calls == 1
    assert sc.score_pairs(x, [])[0].shape == (0,)
    with pytest.raises(IndexError):
        sc.score_pairs(x, [((0, 5), (1, 1))])
    with pytest.raises(IndexError):
        sc.score_pairs(x, [((0, -1), (1, 1))])
    with pytest.raises(ValueError):
        sc.score_pairs(x[..., :4], [((0, 1), (1, 1))])


@needs_torch
def test_score_pairs_rejects_images_outside_the_input_format():
    sc = _standin_scorer()
    x = prepare(_frames((2, 2, 8, 8, 3), 10), size=8)
    p = [((0, 0), (1, 0))]
    with pytest.raises(TypeError, match="prepare"):
        sc.score_pairs(((x + 1) * 127.5).round().to(torch.uint8), p)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        sc.score_pairs(x * 2, p)
    bad = x.clone()
    bad[1, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        sc.score_pairs(bad, p)
    with pytest.raises(ValueError):
        sc.score_pairs(x, p, batch=0)
    s64, _ = sc.score_pairs(x.numpy().astype(np.float64), p)
    np.testing.assert_allclose(s64, sc.score_pairs(x, p)[0], rtol=0, atol=1e-7)
    x_edge = x.clone()
    x_edge[0, 0, 0, 0, 0] = 1.0 + 0.5 * M.RANGE_TOL
    sc.score_pairs(x_edge, p)
    with pytest.raises(TypeError):
        sc.features(torch.zeros((1, 3, 8, 8), dtype=torch.uint8))
    with pytest.raises(ValueError):
        sc.features(torch.zeros((0, 3, 8, 8)))
    with pytest.raises(ValueError):
        sc.forward_pairs(torch.full((1, 2, 3, 8, 8), 3.0))
    with pytest.raises(ValueError):
        sc.forward_pairs(torch.zeros((1, 3, 3, 8, 8)))


@needs_torch
def test_features_fill_a_preallocated_cache_in_chunks():
    sc = _standin_scorer(chunk=3)
    x = prepare(_frames((7, 8, 8, 3), 9), size=8)
    f = sc.features(x)
    assert sc.model.feature_calls == 3
    im = (x + 1) / 2
    ref = sc.model._interpolate(sc.model._get_features(im), im)
    assert f.shape == (7, 4, 8, 8) and f.dtype == torch.float32 and torch.equal(f, ref)


@needs_torch
def test_scoring_runs_with_fixed_numerics_and_restores_the_caller_state(restore_backend):
    sc = _standin_scorer()
    x = prepare(_frames((2, 2, 8, 8, 3), 8), size=8)
    pairs = [((0, 0), (1, 0)), ((0, 1), (1, 1))]
    caller = _odd_caller_state()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        s_ac, _ = sc.score_pairs(x, pairs)
    assert M._backend_state() == caller and torch.is_grad_enabled()
    seen = sc.model.seen
    assert seen and all(d["state"] == _expected_scoring_state(True) for d in seen)
    assert not any(d["grad"] or d["autocast"] for d in seen)
    s_plain, _ = sc.score_pairs(x, pairs)
    np.testing.assert_array_equal(s_ac, s_plain)
    for fn in (lambda: sc.features(x[0]), lambda: sc.forward_pairs(torch.stack([x[0], x[1]], 1))):
        sc.model.seen.clear()
        fn()
        assert sc.model.seen and all(d["state"] == _expected_scoring_state(True) for d in sc.model.seen)
        assert M._backend_state() == caller
    sc.tf32 = False
    sc.model.seen.clear()
    sc.score_pairs(x, pairs)
    assert all(d["state"] == _expected_scoring_state(False) for d in sc.model.seen)
    assert M._backend_state() == caller
    sc.model.fail = True
    with pytest.raises(RuntimeError, match="boom"):
        sc.score_pairs(x, pairs)
    assert M._backend_state() == caller
    assert "_get_features" not in vars(sc.model)


def _fake_official_module(fail=None):
    """A stand-in for `met3r.met3r` whose MEt3R constructor does what the real stack does to the process when it is
    first loaded: CroCo's import enables TF32 matmuls (legacy flag), FeatUp's import calls torch.manual_seed(42)."""

    class FakeOfficialMEt3R(torch.nn.Module):
        inside = []

        def __init__(self, img_size=256, **kwargs):
            super().__init__()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.manual_seed(42)
            FakeOfficialMEt3R.inside.append(M._backend_state())
            if fail is not None:
                raise fail
            self.kwargs = kwargs
            self.lin = torch.nn.Linear(2, 2)
            self.set_rasterizer(img_size)

        def set_rasterizer(self, image_size, points_per_pixel=10, bin_size=0, **kwargs):
            self.rasterizer = types.SimpleNamespace(raster_settings=types.SimpleNamespace(
                image_size=image_size, points_per_pixel=points_per_pixel, bin_size=bin_size))

    return types.SimpleNamespace(MEt3R=FakeOfficialMEt3R, backbone_to_weights={"mast3r": "naver/MASt3R"})


@needs_torch
def test_constructor_keeps_the_caller_tf32_flags_and_rng(monkeypatch, restore_backend):
    fake = _fake_official_module()
    monkeypatch.setattr(M, "_official", lambda: fake)
    caller = _odd_caller_state()
    torch.manual_seed(1234)
    rng = torch.random.get_rng_state()
    sc = MEt3R(device="cpu", raster="binned", featup_repo="someone/FeatUp", backbone_weights="/weights/mast3r")
    assert fake.MEt3R.inside and fake.MEt3R.inside[0] != caller
    assert M._backend_state() == caller
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert fake.backbone_to_weights == {"mast3r": "naver/MASt3R"}
    assert sc.model.kwargs["upsampler"] == "featup" and sc.model.kwargs["distance"] == "cosine"
    assert sc.model.rasterizer.raster_settings.bin_size is None and not sc.model.training
    assert sc.tf32 and sc.feature_chunk == 8 and "tf32=True" in repr(sc)
    with pytest.raises(ValueError):
        MEt3R(device="cpu", feature_chunk=0)


@needs_torch
def test_constructor_protects_the_rng_from_the_featup_import(monkeypatch, tmp_path, restore_backend):
    """featup_repo='auto' imports the installed `featup` package, whose import seeds the global RNG (as FeatUp's
    adaptive-conv module does); that import must happen inside the protected block. The package sits in a checkout
    with a hubconf.py, so 'auto' resolves to that checkout."""
    (tmp_path / "featup").mkdir()
    (tmp_path / "featup" / "__init__.py").write_text("import torch\ntorch.manual_seed(42)\n")
    (tmp_path / "hubconf.py").write_text("")
    monkeypatch.setattr(M, "_official", lambda: _fake_official_module())
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "featup" or k.startswith("featup.")}
    try:
        torch.manual_seed(1234)
        rng = torch.random.get_rng_state()
        sc = MEt3R(device="cpu")
        assert str(Path(sys.modules["featup"].__file__).resolve()).startswith(str(tmp_path.resolve()))
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert sc.featup_source == (str(tmp_path.resolve()), "local")
    finally:
        for k in [k for k in sys.modules if k == "featup" or k.startswith("featup.")]:
            del sys.modules[k]
        sys.modules.update(saved)


@needs_torch
def test_constructor_failure_restores_state_and_explains_a_models_package_conflict(monkeypatch, restore_backend):
    err = ModuleNotFoundError("No module named 'models.croco'", name="models.croco")
    monkeypatch.setattr(M, "_official", lambda: _fake_official_module(fail=err))
    foreign = types.ModuleType("models")
    foreign.__file__ = "/project/models/__init__.py"
    monkeypatch.setitem(sys.modules, "models", foreign)
    caller = _odd_caller_state()
    with pytest.raises(ImportError, match="CroCo"):
        MEt3R(device="cpu", featup_repo="someone/FeatUp")
    assert M._backend_state() == caller


class _FakeScorer:
    """score of ((v,t),(v2,t2)) = 100*v + 10*v2 + t + t2/100 (decodes the pair), overlap = 0.5."""
    img_size = 16

    def __init__(self):
        self.calls = []

    def score_pairs(self, images, pairs, batch=16):
        self.calls.append((tuple(images.shape), list(pairs), batch))
        s = np.array([100 * a[0] + 10 * b[0] + a[1] + b[1] / 100 for a, b in pairs], np.float64)
        return s, np.full(len(pairs), 0.5)


@needs_torch
def test_cross_view_layout():
    fs = _FakeScorer()
    frames = _frames((3, 9, 24, 20, 3), 5)
    r = cross_view(fs, frames, [4, 8], batch=7)
    assert r["pairs"] == [[0, 1], [0, 2], [1, 2]] and r["positions"] == [4, 8]
    assert fs.calls[0][0] == (3, 2, 3, 16, 16) and fs.calls[0][2] == 7
    np.testing.assert_allclose(r["cross"], [[10.0, 20.0, 120.0], [11.01, 21.01, 121.01]], rtol=0, atol=1e-12)
    assert r["overlap"] == [[0.5] * 3] * 2
    assert r["mean"] == pytest.approx(np.mean(r["cross"]))
    assert r["img_size"] == 16 and r["aspect"] == "center_crop" and "temporal" not in r


@needs_torch
def test_cross_view_temporal_list_and_tensor_inputs():
    fs = _FakeScorer()
    views = [_frames((6, 24, 32, 3), 6), _frames((6, 30, 30, 3), 7)]
    r = cross_view(fs, views, [1, 3, 5], temporal=True)
    np.testing.assert_allclose(r["cross"], [[10.0], [11.01], [12.02]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["temporal"], [[0.01, 1.02], [110.01, 111.02]], rtol=0, atol=1e-12)
    assert np.shape(r["temporal_overlap"]) == (2, 2)
    x = prepare(np.stack([views[0][:, :24, :24], views[1][:, :24, :24]]), size=16)
    r2 = cross_view(fs, x, [0, 2])
    assert fs.calls[-1][0] == (2, 2, 3, 16, 16) and r2["positions"] == [0, 2]
    assert r2["aspect"] is None and r2["img_size"] == 16


@needs_torch
def test_cross_view_validates_positions_aspect_and_size():
    fs = _FakeScorer()
    frames = _frames((2, 5, 16, 16, 3), 11)
    with pytest.raises(ValueError, match="non-negative"):
        cross_view(fs, frames, [2, -1])
    with pytest.raises(ValueError):
        cross_view(fs, frames, [1], aspect="bogus")
    x = prepare(frames, size=16)
    with pytest.raises(ValueError):
        cross_view(fs, x, [1], aspect="bogus")
    with pytest.raises(ValueError, match="not resampled"):
        cross_view(fs, x, [1], size=32)
    assert cross_view(fs, x, [1], size=16)["img_size"] == 16
    assert cross_view(fs, frames, [1, 2], aspect="resize")["aspect"] == "resize"
    with pytest.raises(IndexError):
        cross_view(fs, frames, [5])


@needs_torch
def test_cross_view_single_view_has_no_pairs():
    r = cross_view(_FakeScorer(), _frames((1, 3, 16, 16, 3)), [1, 2])
    assert r["pairs"] == [] and r["cross"] == [[], []] and r["mean"] is None


def _prerelease_table(d, sources):
    """The pre-release report formulas, verbatim (clips = sorted, arrays without the conditioning row)."""
    clips = sorted(d)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    real = np.array([d[c]["real"].mean() for c in clips])
    x = np.array([d[c]["real"].mean(1) for c in clips])
    xp = x - x.mean(0, keepdims=True)
    floor = np.abs(xp[:, :8].mean(1) - xp[:, 8:].mean(1)) / np.sqrt(2)
    noise = np.abs(x[:, 0::2].mean(1) - x[:, 1::2].mean(1)) / np.sqrt(2)
    rows = {"real": dict(raw=float(real.mean()), raw_se=se(real))}
    for s in sources:
        g = np.array([d[c][s].mean() for c in clips])
        dl = g - real
        rows[s] = dict(raw=float(g.mean()), raw_se=se(g), delta=float(dl.mean()), delta_se=se(dl),
                       abs_delta=float(np.abs(dl).mean()), abs_delta_se=se(np.abs(dl)), below=float((dl < 0).mean()))
    return rows, float(floor.mean()), se(floor), float(noise.mean())


def _synthetic(n_clips=30, T=16, P=3, seed=0):
    rng = np.random.default_rng(seed)
    prof = np.linspace(0.40, 0.45, T)[:, None]
    out = {}
    for i in range(n_clips):
        base = rng.uniform(0.3, 0.6) + prof + rng.normal(0, 0.01, (T, P))
        out[f"clip_{i:03d}"] = {"real": base, "a": base + rng.normal(0.02, 0.02, (T, P)),
                                "b": base - rng.uniform(0, 0.05) + rng.normal(0, 0.01, (T, P))}
    return out


def test_delta_table_matches_prerelease_formulas():
    d = _synthetic()
    t = delta_table(d, ["a", "b"])
    rows, floor, floor_se, noise = _prerelease_table(d, ["a", "b"])
    for s in ("a", "b"):
        for k, v in rows[s].items():
            assert t[s][k] == v, (s, k)
        assert t[s]["n"] == 30
    assert t["real"]["raw"] == rows["real"]["raw"] and t["real"]["raw_se"] == rows["real"]["raw_se"]
    assert t["_floor"] == {"mean": floor, "se": floor_se, "n": 30}
    assert t["_noise_floor"]["mean"] == noise and t["_noise_floor"]["n"] == 30
    assert t["_clips"] == sorted(d)


def test_delta_table_pairs_clips_and_accepts_lists():
    d = _synthetic(6)
    d["clip_000"]["b"] = None
    del d["clip_001"]["a"]
    d["clip_002"] = {k: v.tolist() for k, v in d["clip_002"].items()}
    d["clip_003"]["real"][2, 1] = np.nan
    t = delta_table(d, ["a", "b"])
    assert t["_clips"] == ["clip_002", "clip_004", "clip_005"] and t["_skipped"] == ["clip_000", "clip_001", "clip_003"]
    assert t["a"]["n"] == t["b"]["n"] == t["real"]["n"] == 3
    only_a = delta_table(d, ["a"])
    assert only_a["_clips"] == ["clip_000", "clip_002", "clip_004", "clip_005"]


def test_delta_table_accepts_a_single_source_name():
    d = _synthetic(5)
    one = delta_table(d, "a")
    assert one == delta_table(d, ["a"]) and set(one) == {"real", "a", "_floor", "_noise_floor", "_clips", "_skipped"}


def test_delta_table_identity_and_sign():
    d = _synthetic(10)
    for c in d.values():
        c["same"] = c["real"].copy()
        c["worse"] = c["real"] + 0.1
    t = delta_table(d, ["same", "worse", "real"])
    assert t["same"]["abs_delta"] == 0.0 and t["same"]["below"] == 0.0 and t["same"]["delta_se"] == 0.0
    assert t["worse"]["delta"] == pytest.approx(0.1) and t["worse"]["abs_delta"] == pytest.approx(0.1)
    assert t["worse"]["below"] == 0.0
    assert set(t) == {"real", "same", "worse", "_floor", "_noise_floor", "_clips", "_skipped"}


def test_delta_table_floor_removes_shared_profile():
    T, P = 16, 6
    prof = np.sin(np.linspace(0, 3, T))[:, None] * 0.05
    d = {f"c{i}": {"real": np.full((T, P), 0.3 + 0.01 * i) + prof, "g": np.full((T, P), 0.5)} for i in range(5)}
    t = delta_table(d, ["g"])
    assert t["_floor"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert t["_noise_floor"]["mean"] > 0.0
    flat = {c: {"real": np.full((T, P), 0.4), "g": np.full((T, P), 0.5)} for c in "xyz"}
    tf = delta_table(flat, ["g"])
    assert tf["_noise_floor"]["mean"] == 0.0 and tf["_floor"]["mean"] == 0.0
    odd = {c: {"real": np.arange(5, dtype=float)[:, None] * (1 + i), "g": np.zeros((5, 1))} for i, c in enumerate("ab")}
    x = np.array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]], float)
    xp = x - x.mean(0)
    want = np.abs(xp[:, :2].mean(1) - xp[:, 2:].mean(1)) / math.sqrt(2)
    assert delta_table(odd, ["g"])["_floor"]["mean"] == pytest.approx(want.mean())


def test_delta_table_errors():
    d = _synthetic(3)
    with pytest.raises(ValueError):
        delta_table(d, ["missing"])
    bad = _synthetic(3)
    bad["clip_000"]["a"] = bad["clip_000"]["a"][:, :2]
    with pytest.raises(ValueError):
        delta_table(bad, ["a"])
    short = {"c": {"real": np.ones((1, 3)), "a": np.ones((1, 3))}}
    with pytest.raises(ValueError):
        delta_table(short, ["a"])
    with pytest.raises(ValueError):
        delta_table(_synthetic(3), ["_x"])
    assert clip_mean([[1.0, 2.0], [3.0, 6.0]]) == 3.0
    with pytest.raises(ValueError):
        clip_mean([[], []])


gpu = pytest.mark.skipif(os.environ.get("VPMWB_TEST_MET3R") != "1",
                         reason="set VPMWB_TEST_MET3R=1 (CUDA + official MEt3R stack + cached weights)")


def _rng_snapshot():
    return torch.random.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]


def _same_rng(a, b):
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(torch.equal(u, v) for u, v in zip(a[1], b[1]))


def _legacy_flags():
    try:
        return bool(torch.backends.cuda.matmul.allow_tf32), torch.get_float32_matmul_precision()
    except RuntimeError as e:
        return f"RuntimeError: {e}"


@pytest.fixture(scope="module")
def gpu_scorer():
    """The real scorer, plus the caller state right before and after its construction."""
    if torch is None or not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    pytest.importorskip("met3r")
    torch.manual_seed(1234)
    before = (M._backend_state(), _legacy_flags(), _rng_snapshot())
    sc = MEt3R(device="cuda")
    after = (M._backend_state(), _legacy_flags(), _rng_snapshot())
    return sc, before, after


def _gpu_frames():
    """(2 views, 3 steps, 256, 256, 3) uint8: two overlapping crops of one smooth random texture, shifted per step."""
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8), (320, 320), interpolation=cv2.INTER_CUBIC)
    return np.stack([[base[s:s + 256, s:s + 256] for s in (0, 24, 48)],
                     [base[64 - s:320 - s, s:s + 256] for s in (0, 24, 48)]])


@gpu
@needs_torch
def test_construction_keeps_the_caller_state_gpu(gpu_scorer):
    sc, before, after = gpu_scorer
    assert after[0] == before[0] and after[1] == before[1]
    assert _same_rng(before[2], after[2])
    assert sc.model.rasterizer.raster_settings.bin_size == 0 and sc.feature_chunk == 8 and sc.tf32
    assert not sc.model.training


@gpu
@needs_torch
def test_cached_equals_official_forward_gpu(gpu_scorer):
    sc = gpu_scorer[0]
    x = prepare(_gpu_frames())
    pairs = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 0), (0, 1))]
    fwd = torch.stack([torch.stack([x[a], x[b]]) for a, b in pairs])
    bwd = torch.stack([torch.stack([x[b], x[a]]) for a, b in pairs])
    s1, m1 = sc.forward_pairs(fwd)
    s2, m2 = sc.forward_pairs(bwd)
    official = ((s1 + s2) / 2).float().cpu().numpy()
    official_ovl = ((m1.mean((-1, -2)) + m2.mean((-1, -2))) / 2).float().cpu().numpy()
    s, o = sc.score_pairs(x, pairs, batch=len(pairs))
    assert np.all(np.isfinite(s)) and np.all((o > 0) & (o <= 1))
    np.testing.assert_allclose(s, official, rtol=0, atol=1e-4)
    np.testing.assert_allclose(o, official_ovl, rtol=0, atol=1e-7)
    s16, _ = sc.score_pairs(x, pairs)
    np.testing.assert_allclose(s16, official, rtol=0, atol=2e-3)
    chunk = sc.feature_chunk
    sc.feature_chunk = 2 * len(pairs)
    try:
        feats = sc.features(fwd.flatten(0, 1)).reshape(len(pairs), 2, -1, 256, 256)
    finally:
        sc.feature_chunk = chunk
    s3, m3 = sc.forward_pairs(fwd, feats)
    assert torch.equal(s3, s1) and torch.equal(m3, m1)


@gpu
@needs_torch
def test_scores_do_not_depend_on_the_caller_flags_gpu(gpu_scorer, restore_backend):
    sc = gpu_scorer[0]
    x = prepare(_gpu_frames())
    pairs = [((0, 0), (1, 0)), ((0, 2), (1, 2)), ((1, 0), (1, 1))]
    results = []
    for caller_tf32, autotune in ((False, False), (True, True)):
        M._set_scoring_backend(caller_tf32)
        torch.backends.cudnn.benchmark = autotune
        state, rng = M._backend_state(), _rng_snapshot()
        results.append(sc.score_pairs(x, pairs)[0])
        assert M._backend_state() == state and _same_rng(rng, _rng_snapshot())
    np.testing.assert_array_equal(results[0], results[1])
    sc.tf32 = False
    try:
        full = sc.score_pairs(x, pairs)[0]
    finally:
        sc.tf32 = True
    diff = np.abs(full - results[0]).max()
    assert diff < 1e-2
    if torch.cuda.get_device_capability()[0] >= 8:
        assert diff > 0
