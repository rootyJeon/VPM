"""Tests for vpm_worldbench.metrics.appearance and vpm_worldbench.metrics.fvd.

CPU only and synthetic data. LPIPS / I3D tests run on the CPU when torch + lpips / cd-fvd (and their weights) are
available and are skipped otherwise.
"""
import importlib.util
import math

import numpy as np
import pytest

from vpm_worldbench.metrics.appearance import PSNR_MSE_FLOOR, appearance, psnr_frames, ssim_frames
from vpm_worldbench.metrics.fvd import (COV_MODES, I3D_MD5, frechet, fvd_from_features, fvd_windows, pooled_fvd,
                                        window_starts)


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _video(n=5, h=32, w=40, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8)


def test_psnr_identical_frames_hit_the_mse_floor():
    v = _video()
    p = psnr_frames(v, v)
    assert p.shape == (5,) and p.dtype == np.float64
    np.testing.assert_allclose(p, 10 * np.log10(255.0 ** 2 / PSNR_MSE_FLOOR))


def test_psnr_known_mse():
    a = np.full((2, 8, 8, 3), 100, np.uint8)
    b = a.copy()
    b[0] += 10
    b[1, :4] += 20
    np.testing.assert_allclose(psnr_frames(a, b), [10 * np.log10(65025 / 100), 10 * np.log10(65025 / 200)],
                               rtol=0, atol=1e-12)


def test_psnr_matches_float32_prerelease_formula():
    a, b = _video(8, 48, 64, 1), _video(8, 48, 64, 2)
    mse32 = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).reshape(len(a), -1).mean(1)
    p32 = 10 * np.log10(255.0 ** 2 / np.maximum(mse32, 1e-8))
    np.testing.assert_allclose(psnr_frames(a, b), p32, rtol=0, atol=1e-4)
    exact = 10 * np.log10(65025.0 / ((a.astype(np.int64) - b) ** 2).reshape(len(a), -1).mean(1))
    np.testing.assert_allclose(psnr_frames(a, b), exact, rtol=0, atol=1e-12)


def test_psnr_is_mean_of_per_frame_values():
    a = np.full((2, 4, 4, 3), 50, np.uint8)
    b = a.copy()
    b[0] += 1
    b[1] += 30
    res = appearance(a, b, ssim=False)
    per = psnr_frames(a, b)
    assert res["psnr"] == pytest.approx(per.mean())
    pooled = 10 * np.log10(65025 / np.mean([1.0, 900.0]))
    assert abs(res["psnr"] - pooled) > 1.0


def test_input_validation():
    v = _video()
    with pytest.raises(TypeError):
        psnr_frames(v.astype(np.float32), v)
    with pytest.raises(ValueError):
        psnr_frames(v, v[:4])
    with pytest.raises(ValueError):
        psnr_frames(v[:0], v[:0])
    with pytest.raises(ValueError):
        psnr_frames(v[..., :2], v[..., :2])
    assert psnr_frames(v[0], v[0]).shape == (1,)


@pytest.mark.skipif(not _has("skimage"), reason="scikit-image not installed")
def test_ssim_matches_skimage_defaults():
    from skimage.metrics import structural_similarity
    a, b = _video(3, 40, 40, 3), _video(3, 40, 40, 4)
    s = ssim_frames(a, b)
    assert s.shape == (3,) and s.dtype == np.float64
    ref = [structural_similarity(x, y, channel_axis=2, data_range=255) for x, y in zip(a, b)]
    np.testing.assert_array_equal(s, ref)
    np.testing.assert_allclose(ssim_frames(a, a), 1.0)
    np.testing.assert_allclose(ssim_frames(b, a), s, rtol=0, atol=1e-12)


def test_appearance_keys_and_lpips_callable():
    a, b = _video(4, 16, 16, 5), _video(4, 16, 16, 6)
    res = appearance(a, b, lpips=lambda r, f: np.full(len(r), 0.25), ssim=False)
    assert set(res) >= {"psnr", "ssim", "lpips", "psnr_per_frame", "n_frames"}
    assert res["ssim"] is None and res["lpips"] == pytest.approx(0.25) and res["n_frames"] == 4
    assert len(res["psnr_per_frame"]) == 4 and res["psnr"] == pytest.approx(np.mean(res["psnr_per_frame"]))
    assert appearance(a, b, ssim=False)["lpips"] is None
    with pytest.raises(TypeError):
        appearance(a, b, lpips=True)
    with pytest.raises(ValueError):
        appearance(a, b, lpips=lambda r, f: np.zeros(3), ssim=False)
    if _has("skimage"):
        res = appearance(a, b)
        assert res["ssim"] == pytest.approx(float(np.mean(ssim_frames(a, b))))


@pytest.mark.skipif(not (_has("torch") and _has("lpips")), reason="torch / lpips not installed")
def test_lpips_cpu():
    from vpm_worldbench.metrics.appearance import LPIPS
    try:
        lp = LPIPS(device="cpu")
    except Exception as e:
        pytest.skip(f"LPIPS weights unavailable: {e}")
    a, b = _video(5, 64, 64, 7), _video(5, 64, 64, 8)
    d = lp(a, b)
    assert d.shape == (5,) and d.dtype == np.float64 and (d > 0).all()
    np.testing.assert_allclose(lp(a, a), 0.0, atol=1e-6)
    np.testing.assert_allclose(lp(a, b, batch=2), d, rtol=1e-5, atol=1e-6)
    res = appearance(a, b, lpips=lp, ssim=False)
    assert res["lpips"] == pytest.approx(float(d.mean()), rel=1e-5)


def test_window_starts():
    assert window_starts(65) == [1, 17, 33, 49]
    assert window_starts(65, n_windows=4) == [1, 17, 33, 49]
    assert window_starts(65, n_windows=2) == [1, 33]
    assert window_starts(65, n_windows=1) == [1]
    assert window_starts(65, n_windows=3) == [1, 17, 33]
    assert window_starts(64, start=0) == [0, 16, 32, 48]
    assert window_starts(70) == [1, 17, 33, 49]
    assert window_starts(33, window=8, start=1) == [1, 9, 17, 25]
    with pytest.raises(ValueError):
        window_starts(65, n_windows=5)
    with pytest.raises(ValueError):
        window_starts(16)
    with pytest.raises(ValueError):
        window_starts(65, n_windows=0)


def test_fvd_windows_match_prerelease_cuts():
    import cv2
    v = _video(65, 72, 96, 9)
    full = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in v])
    w4 = fvd_windows(v)
    assert w4.shape == (4, 16, 224, 224, 3) and w4.dtype == np.uint8
    np.testing.assert_array_equal(w4, full[1:65].reshape(4, 16, 224, 224, 3))
    np.testing.assert_array_equal(fvd_windows(v, n_windows=2), np.stack([full[1:17], full[33:49]]))
    assert fvd_windows(v, size=32).shape == (4, 16, 32, 32, 3)
    assert fvd_windows(v, window=8, n_windows=3, start=0, size=16).shape == (3, 8, 16, 16, 3)
    with pytest.raises(ValueError):
        fvd_windows(v.astype(np.float32))
    with pytest.raises(ValueError):
        fvd_windows(v[:10])


def _feats(n, d, seed, shift=0.0, scale=1.0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(d, d)) / np.sqrt(d)
    return (r.normal(size=(n, d)) @ A) * scale + shift


def _frechet_reference(a, b, biased):
    from scipy import linalg
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    if biased:
        ca = a.T @ a / len(a) - np.outer(mu_a, mu_a)
        cb = b.T @ b / len(b) - np.outer(mu_b, mu_b)
    else:
        ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = linalg.sqrtm(ca @ cb)
    return float(np.real(np.square(mu_a - mu_b).sum() + np.trace(ca + cb - 2 * s)))


@pytest.mark.parametrize("cov", COV_MODES)
def test_frechet_matches_reference_formulas(cov):
    a, b = _feats(300, 12, 0), _feats(200, 12, 1, shift=0.3, scale=1.4)
    np.testing.assert_allclose(frechet(a, b, cov=cov), _frechet_reference(a, b, cov == "biased"), rtol=1e-12)
    np.testing.assert_allclose(fvd_from_features(a, b, cov=cov), frechet(a, b, cov=cov), rtol=0)


@pytest.mark.parametrize("cov", COV_MODES)
def test_frechet_identity_and_shift(cov):
    a = _feats(400, 8, 2)
    assert abs(frechet(a, a, cov=cov)) < 1e-8
    c = np.linspace(-1, 1, 8)
    np.testing.assert_allclose(frechet(a, a + c, cov=cov), float(c @ c), rtol=1e-8)


def test_frechet_1d_closed_form():
    r = np.random.default_rng(3)
    a, b = r.normal(1.0, 2.0, (500, 1)), r.normal(-0.5, 0.5, (300, 1))
    for cov, ddof in (("unbiased", 1), ("biased", 0)):
        va, vb = a.var(ddof=ddof), b.var(ddof=ddof)
        expect = (a.mean() - b.mean()) ** 2 + (math.sqrt(va) - math.sqrt(vb)) ** 2
        np.testing.assert_allclose(frechet(a, b, cov=cov), expect, rtol=1e-10)


def test_frechet_unbiased_vs_biased_differ():
    a, b = _feats(50, 6, 4), _feats(50, 6, 5, shift=0.2)
    assert frechet(a, b, cov="unbiased") != pytest.approx(frechet(a, b, cov="biased"), rel=1e-6)


def test_frechet_pooling_and_counts():
    parts_a = [_feats(12, 10, s) for s in range(5)]
    parts_b = [_feats(12, 10, s + 10, shift=0.1) for s in range(5)]
    A, B = np.concatenate(parts_a), np.concatenate(parts_b)
    assert frechet(parts_a, parts_b) == pytest.approx(frechet(A, B), rel=1e-12)
    res = pooled_fvd(iter(parts_a), iter(parts_b), cov="biased")
    assert res["n_real"] == 60 and res["n_fake"] == 60 and res["cov"] == "biased"
    assert res["fvd"] == pytest.approx(frechet(A, B, cov="biased"), rel=1e-12)
    assert frechet(A.astype(np.float32), B) == pytest.approx(frechet(A.astype(np.float32).astype(np.float64), B))


def test_frechet_validation():
    a = _feats(20, 4, 6)
    with pytest.raises(ValueError):
        frechet(a, _feats(20, 5, 7))
    with pytest.raises(ValueError):
        frechet(a[:1], a)
    with pytest.raises(ValueError):
        frechet(a, a, cov="sample")
    bad = a.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        frechet(bad, a)
    with pytest.raises(ValueError):
        frechet([], a)


@pytest.mark.skipif(not (_has("torch") and _has("cdfvd")), reason="torch / cd-fvd not installed")
def test_i3d_matches_cdfvd_cpu():
    from vpm_worldbench.metrics.fvd import I3D
    try:
        i3d = I3D(device="cpu")
    except Exception as e:
        pytest.skip(f"I3D weights unavailable: {e}")
    import torch
    from cdfvd.third_party.i3d.utils import preprocess_i3d
    x = _video(16, 224, 224, 11)[None]
    f = i3d.features(x)
    assert f.shape == (1, 400) and f.dtype == np.float64 and np.isfinite(f).all()
    assert i3d.weights_md5 == I3D_MD5 and i3d.info["weights_md5"] == I3D_MD5 and i3d.info["device"] == "cpu"
    with torch.no_grad():
        ref = i3d.model(preprocess_i3d(x).to(device="cpu", dtype=torch.float32)).numpy()
    np.testing.assert_array_equal(f.astype(np.float32), ref)
    np.testing.assert_array_equal(i3d.features(x[0]), f)
    with pytest.raises(TypeError):
        i3d.features(x.astype(np.float32))
