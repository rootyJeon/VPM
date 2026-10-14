"""Unit tests of vpm_worldbench.metrics.{depth, pose, pointcloud, segmentation, qa} on synthetic data with known
answers (CPU only, numpy / scipy / cv2). Where a metric follows a published evaluation script, the script's
computation is transcribed to numpy here (`_*_reference`) and compared on random data."""
import math

import numpy as np
import pytest

from vpm_worldbench.metrics import depth as D
from vpm_worldbench.metrics import pointcloud as PC
from vpm_worldbench.metrics import pose as PO
from vpm_worldbench.metrics import qa as QA
from vpm_worldbench.metrics import segmentation as SG


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _rand_rot(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _w2c(R_c2w, centre):
    """World-to-camera 4x4 from a camera orientation (camera-to-world rotation) and centre."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_c2w).T
    T[:3, 3] = -np.asarray(R_c2w).T @ np.asarray(centre, float)
    return T


def _rand_rig(rng, n):
    return np.stack([_w2c(_rand_rot(rng), rng.normal(size=3) * 2.0) for _ in range(n)])


def _line_rig(n=4):
    """n cameras at (k, 0, 0), identity orientation (looking along +z)."""
    return np.stack([_w2c(np.eye(3), (k, 0.0, 0.0)) for k in range(n)])


def _perturbed(rng, G, max_deg=20.0, sigma=0.3):
    """Every camera of G rotated about its optical axis by 2..max_deg degrees and moved by N(0, sigma) metres."""
    P = []
    for g in G:
        c2w = PO.invert_rigid(g)
        P.append(_w2c(c2w[:3, :3] @ _rz(rng.uniform(2, max_deg)), c2w[:3, 3] + rng.normal(scale=sigma, size=3)))
    return np.stack(P)


def test_valid_gt_mask():
    gt = np.array([0.0, 0.5, np.nan, 3.0, 100.0])
    m = np.array([True, True, True, False, True])
    assert D.valid_gt_mask(gt).tolist() == [False, True, False, True, True]
    assert D.valid_gt_mask(gt, m, max_depth=50).tolist() == [False, True, False, False, False]


def test_depth_metrics_perfect():
    gt = np.random.default_rng(0).uniform(0.5, 20, (3, 6, 7))
    for reduce in ("frames", "pixels"):
        m = D.depth_metrics(gt, gt, reduce=reduce)
        for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"):
            assert m[k] == pytest.approx(0.0, abs=1e-9)
        assert m["delta1"] == m["delta2"] == m["delta3"] == 100.0
        assert (m["n_valid"], m["n_scored"], m["coverage"], m["n_frames"]) == (126, 126, 100.0, 3)
        assert m["reduce"] == reduce


def test_depth_metrics_known_values():
    gt = np.array([[[1.0, 2.0], [4.0, 8.0]]])
    m = D.depth_metrics(1.1 * gt, gt)
    assert m["abs_rel"] == pytest.approx(0.1)
    assert m["sq_rel"] == pytest.approx(0.01 * gt.mean())
    assert m["rmse"] == pytest.approx(0.1 * math.sqrt(np.mean(gt ** 2)))
    assert m["rmse_log"] == pytest.approx(math.log(1.1))
    assert m["log10"] == pytest.approx(math.log10(1.1))
    assert m["silog"] == pytest.approx(0.0, abs=1e-6)
    assert m["delta1"] == 100.0
    m = D.depth_metrics(gt * np.exp(np.array([[[0.1, -0.1], [0.1, -0.1]]])), gt)
    assert m["silog"] == pytest.approx(10.0)


def test_depth_delta_thresholds_are_strict():
    gt = np.full((1, 4), 4.0)
    assert D.depth_metrics(np.full((1, 4), 5.0), gt)["delta1"] == 0.0
    m = D.depth_metrics(np.full((1, 4), 5.0), gt)
    assert m["delta2"] == 100.0
    m = D.depth_metrics(np.full((1, 4), 8.0), gt)
    assert (m["delta1"], m["delta2"], m["delta3"]) == (0.0, 0.0, 0.0)
    m = D.depth_metrics(np.array([[4.0, 4.0, 6.0, 8.0]]), gt)
    assert (m["delta1"], m["delta2"], m["delta3"]) == (50.0, 75.0, 75.0)


def test_depth_metrics_mask_range_and_coverage():
    gt = np.array([[0.0, 1.0, 2.0, 100.0, 3.0]])
    pred = np.array([[5.0, np.nan, 2.0, 2.0, 0.0]])
    m = D.depth_metrics(pred, gt, max_depth=50.0)
    assert (m["n_valid"], m["n_scored"]) == (3, 1)
    assert m["coverage"] == pytest.approx(100.0 / 3)
    assert m["abs_rel"] == 0.0
    mask = np.array([[True, True, False, True, True]])
    m = D.depth_metrics(pred, gt, mask, max_depth=50.0)
    assert (m["n_valid"], m["n_scored"], m["n_frames"]) == (2, 0, 0)
    assert math.isnan(m["abs_rel"]) and m["coverage"] == 0.0
    m = D.depth_metrics(np.array([[1e-6, 80.0]]), np.array([[1.0, 10.0]]), min_depth=0.5, max_depth=20.0)
    assert m["abs_rel"] == pytest.approx(((1.0 - 0.5) / 1.0 + (20.0 - 10.0) / 10.0) / 2)


def test_depth_reduce_frames_vs_pixels():
    gt = np.ones((2, 1, 4))
    pred = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, np.nan, np.nan]])[:, None]
    mf = D.depth_metrics(pred, gt)
    assert mf["abs_rel"] == pytest.approx(0.5) and mf["n_frames"] == 2 and mf["reduce"] == "frames"
    assert mf["rmse"] == pytest.approx(0.5) and mf["delta1"] == pytest.approx(50.0)
    mp = D.depth_metrics(pred, gt, reduce="pixels")
    assert mp["abs_rel"] == pytest.approx(2 / 6) and mp["rmse"] == pytest.approx(math.sqrt(2 / 6))
    assert mp["delta1"] == pytest.approx(400 / 6)
    per = D.depth_metrics_per_frame(np.vstack([pred, [[[np.nan] * 4]]]), np.vstack([gt, [[[1.0] * 4]]]))
    assert per["abs_rel"][:2].tolist() == [0.0, 1.0] and math.isnan(per["abs_rel"][2])
    assert per["n_scored"].tolist() == [4, 2, 0] and per["coverage"].tolist() == [100.0, 50.0, 0.0]
    assert D.depth_metrics(np.vstack([pred, [[[np.nan] * 4]]]), np.vstack([gt, [[[1.0] * 4]]]))["abs_rel"] == 0.5
    with pytest.raises(ValueError):
        D.depth_metrics(pred, gt, reduce="median")


def test_depth_two_dim_input_is_one_frame():
    rng = np.random.default_rng(14)
    gt = rng.uniform(1, 9, (6, 8))
    pred = gt * rng.uniform(0.8, 1.25, gt.shape)
    m2 = D.depth_metrics(pred, gt)
    m3 = D.depth_metrics(pred[None], gt[None])
    mp = D.depth_metrics(pred, gt, reduce="pixels")
    assert m2["n_frames"] == 1 and m2["abs_rel"] == pytest.approx(m3["abs_rel"]) == pytest.approx(mp["abs_rel"])
    _, p = D.align(pred, gt, mode="scale", per="frame", space="depth", return_params=True)
    _, q = D.align(pred, gt, mode="scale", per="sequence", space="depth", return_params=True)
    assert p["scale"].shape == (1,) and p["scale"][0] == pytest.approx(q["scale"])
    sp_gt = np.array([[1.0, 2.0], [1.0, 2.0]])
    sp_pred = np.array([[1.0, 2.0], [2.0, 2.0]])
    assert D.depth_metrics(sp_pred[..., None], sp_gt[..., None])["n_frames"] == 2
    assert D.depth_metrics(sp_pred, sp_gt)["n_frames"] == 1


def test_align_median_and_scale_depth_space():
    gt = np.random.default_rng(1).uniform(1, 9, (2, 5, 5))
    for mode in ("median", "scale"):
        out, p = D.align(3.0 * gt, gt, mode=mode, space="depth", return_params=True)
        assert np.allclose(out, gt) and p["scale"] == pytest.approx(1 / 3) and p["shift"] == 0.0
        assert p["n_pixels"] == 50 and p["per"] == "sequence"


def test_align_scale_shift_recovers_affine_disparity():
    gt = np.random.default_rng(2).uniform(0.5, 30, (3, 4, 6))
    disp = 0.5 / gt + 0.2
    out, p = D.align(disp, gt, pred_space="disparity", return_params=True)
    assert p["scale"] == pytest.approx(2.0) and p["shift"] == pytest.approx(-0.4)
    assert np.allclose(out, gt, rtol=1e-9)
    out = D.align(1.0 / (3.0 / gt + 0.05), gt, mode="scale_shift", space="disparity")
    assert np.allclose(out, gt, rtol=1e-9)
    assert D.evaluate_depth(disp, gt, pred_space="disparity")["abs_rel"] == pytest.approx(0.0, abs=1e-9)


def test_align_scale_shift_depth_space():
    gt = np.random.default_rng(3).uniform(1, 5, (2, 3, 3))
    out, p = D.align(2.0 * gt + 1.0, gt, space="depth", return_params=True)
    assert p["scale"] == pytest.approx(0.5) and p["shift"] == pytest.approx(-0.5)
    assert np.allclose(out, gt)


def test_align_per_frame_vs_sequence():
    g = np.random.default_rng(4).uniform(1, 5, (4, 4))
    gt = np.stack([g, g])
    pred = np.stack([2.0 * g, 4.0 * g])
    out, p = D.align(pred, gt, mode="scale", per="frame", space="depth", return_params=True)
    assert np.allclose(out, gt) and np.allclose(p["scale"], [0.5, 0.25]) and p["n_pixels"].tolist() == [16, 16]
    seq, ps = D.align(pred, gt, mode="scale", per="sequence", space="depth", return_params=True)
    assert ps["scale"] == pytest.approx(6 / 20)
    assert D.depth_metrics(seq, gt)["abs_rel"] == pytest.approx(0.3)


def test_align_clamps_aligned_disparity():
    gt = np.array([[[1.0, 2.0], [4.0, 0.0]]])
    disp = np.array([[[1.0, 0.5], [0.25, -1.0]]])
    out = D.align(disp, gt, pred_space="disparity")
    assert np.allclose(out, [[[1.0, 2.0], [4.0, 1.0 / D.DISP_FLOOR]]])


def test_align_pred_min_clamps_before_the_fit():
    gt = np.array([[[1.0, 2.0, 4.0, 8.0]]])
    disp = np.array([[[1.0, 0.5, 0.25, -0.5]]])
    out, p = D.align(disp, gt, mode="scale", pred_space="disparity", pred_min=1e-3, return_params=True)
    x, y = np.array([1.0, 0.5, 0.25, 1e-3]), 1.0 / gt.ravel()
    assert p["scale"] == pytest.approx(float(x @ y / (x @ x)))
    assert out[0, 0, 3] == pytest.approx(1.0 / (p["scale"] * 1e-3))
    _, q = D.align(disp, gt, mode="scale", pred_space="disparity", return_params=True)
    assert q["scale"] != pytest.approx(p["scale"])
    nan = D.align(np.array([[[np.nan, 1.0]]]), np.array([[[1.0, 1.0]]]), mode="none", pred_space="disparity",
                  pred_min=1e-3)
    assert np.isnan(nan[0, 0, 0]) and nan[0, 0, 1] == 1.0


def test_align_missing_predictions_and_unfittable_frames():
    gt = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])[:, None]
    pred = np.array([[2.0, 0.0, 6.0], [1.0, 1.0, 1.0]])[:, None]
    out, p = D.align(pred, gt, mode="scale", per="frame", space="depth", return_params=True)
    assert out[0, 0, 0] == pytest.approx(1.0) and np.isnan(out[0, 0, 1]) and out[0, 0, 2] == pytest.approx(3.0)
    assert np.isnan(out[1]).all() and np.isnan(p["scale"][1]) and p["n_pixels"].tolist() == [2, 0]
    m = D.evaluate_depth(pred, gt, align_mode="scale", per="frame", space="depth")
    assert m["coverage"] == pytest.approx(200 / 3) and m["abs_rel"] == pytest.approx(0.0, abs=1e-12)
    assert isinstance(m["align"]["scale"], list) and len(m["align"]["scale"]) == 2 and m["n_frames"] == 1


def test_align_none_converts_and_validates():
    gt = np.full((1, 3), 2.0)
    assert np.allclose(D.align(np.full((1, 3), 0.5), gt, mode="none", pred_space="disparity"), 2.0)
    assert np.allclose(D.align(np.full((1, 3), 7.0), gt, mode="none", space="depth"), 7.0)
    with pytest.raises(ValueError):
        D.align(gt, gt, mode="affine")
    with pytest.raises(ValueError):
        D.align(gt, np.ones((1, 4)))
    with pytest.raises(ValueError):
        D.depth_metrics(gt, gt, mask=np.ones((3,), bool))


def test_align_constant_prediction_falls_back_to_scale():
    gt = np.array([[1.0, 2.0, 3.0]])
    out, p = D.align(np.full((1, 3), 4.0), gt, space="depth", return_params=True)
    assert p["shift"] == 0.0 and p["scale"] == pytest.approx(6.0 / 12.0)
    assert np.allclose(out, 2.0)


def test_align_keeps_every_fitted_pixel_scored():
    rng = np.random.default_rng(26)
    g = rng.uniform(1, 10, (3, 8, 8))
    p = np.abs(1.0 / g + rng.normal(0, 0.05, g.shape)) + 1e-3
    aligned, prm = D.align(p, g, space="depth", min_depth=0.0, return_params=True)
    raw = prm["scale"] * p + prm["shift"]
    assert (raw <= 0).any()
    assert np.array_equal(aligned, np.maximum(raw, D.MIN_PRED_DEPTH))
    m = D.evaluate_depth(p, g, space="depth", min_depth=0.0)
    assert m["coverage"] == 100.0 and m["n_scored"] == m["n_valid"] == g.size
    a2 = D.align(p, g, space="depth", min_depth=0.5)
    assert a2.min() == 0.5


def _monst3r_reference(pred, gt, max_depth, lstsq):
    """numpy transcription of MonST3R `dust3r/depth_eval.py::depth_evaluation` (align_with_lstsq or median scaling,
    post_clip_max=max_depth, no custom_mask; pooled pixels)."""
    mask = (gt > 0) & (gt < max_depth)
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    if lstsq:
        s, t = np.linalg.lstsq(np.stack([p, np.ones_like(p)], 1), g, rcond=None)[0]
        p = s * p + t
    else:
        p = p * (np.median(g) / np.median(p))
    p = np.minimum(p, max_depth)
    out = {"abs_rel": np.mean(np.abs(p - g) / g), "sq_rel": np.mean((p - g) ** 2 / g),
           "rmse": np.sqrt(np.mean((p - g) ** 2))}
    p = np.maximum(p, 1e-5)
    r = np.maximum(p / g, g / p)
    out.update(rmse_log=np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)), delta1=100 * np.mean(r < 1.25),
               delta2=100 * np.mean(r < 1.25 ** 2), delta3=100 * np.mean(r < 1.25 ** 3))
    return out


def test_matches_monst3r_reference_code():
    rng = np.random.default_rng(27)
    gt = rng.uniform(0.3, 30.0, (9, 24, 32))
    gt[rng.uniform(size=gt.shape) < 0.1] = 0.0
    pred = 2.0 * gt.clip(0.3) * np.exp(rng.normal(0, 0.15, gt.shape)) + 0.05
    for lstsq, mode in ((True, "scale_shift"), (False, "median")):
        ref = _monst3r_reference(pred, gt, 25.0, lstsq)
        m = D.evaluate_depth(pred, gt, align_mode=mode, space="depth", min_depth=0.0, max_depth=25.0, reduce="pixels")
        for k, v in ref.items():
            assert m[k] == pytest.approx(v, rel=1e-9), (mode, k)
    g = rng.uniform(1, 10, (3, 8, 8))
    bad = np.abs(1.0 / g + rng.normal(0, 0.05, g.shape)) + 1e-3
    ref = _monst3r_reference(bad, g, 80.0, True)
    m = D.evaluate_depth(bad, g, space="depth", min_depth=0.0, max_depth=80.0, reduce="pixels")
    for k in ("rmse_log", "delta1", "delta2", "delta3"):
        assert m[k] == pytest.approx(ref[k], rel=1e-9), k
    s, t = np.linalg.lstsq(np.stack([bad.ravel(), np.ones(bad.size)], 1), g.ravel(), rcond=None)[0]
    assert (s * bad + t <= 0).any()
    pf = np.maximum(s * bad.ravel() + t, D.MIN_PRED_DEPTH)
    assert m["abs_rel"] == pytest.approx(np.mean(np.abs(pf - g.ravel()) / g.ravel()), rel=1e-9)
    assert m["abs_rel"] < ref["abs_rel"] and m["coverage"] == 100.0


def test_align_none_is_a_pure_conversion():
    far = np.array([[[2000.0, 5.0, 0.0, np.nan]]])
    gt = np.array([[[2000.0, 5.0, 3.0, 3.0]]])
    out = D.align(far, gt, mode="none")
    assert out[0, 0, 0] == 2000.0 and out[0, 0, 1] == 5.0 and np.isnan(out[0, 0, 2:]).all()
    disp = np.array([[[0.5, -0.2, 0.0]]])
    out = D.align(disp, np.ones((1, 1, 3)), mode="none", pred_space="disparity")
    assert out[0, 0, 0] == 2.0 and np.isnan(out[0, 0, 1:]).all()
    out = D.align(disp, np.ones((1, 1, 3)), mode="none", pred_space="disparity", pred_min=0.1)
    assert np.allclose(out, [[[2.0, 10.0, 10.0]]])
    m = D.evaluate_depth(far, gt, align_mode="none")
    assert m["abs_rel"] == 0.0 and (m["n_valid"], m["n_scored"]) == (4, 2)


def test_depth_empty_inputs_and_range_checks():
    for shape in ((0, 4, 5), (0,), (3, 0, 4)):
        z = np.ones(shape)
        m = D.depth_metrics(z, z)
        assert math.isnan(m["abs_rel"]) and math.isnan(m["coverage"])
        assert (m["n_valid"], m["n_scored"], m["n_frames"]) == (0, 0, 0)
        out, p = D.align(z, z, return_params=True)
        assert out.shape == shape and p["n_pixels"] == 0 and math.isnan(p["scale"])
        assert math.isnan(D.evaluate_depth(z, z, per="frame")["rmse"])
    per = D.depth_metrics_per_frame(np.ones((0, 2, 2)), np.ones((0, 2, 2)))
    assert per["abs_rel"].shape == (0,) and per["coverage"].shape == (0,)
    gt = np.ones((1, 2, 2))
    with pytest.raises(ValueError):
        D.depth_metrics(gt, gt, min_depth=-1.0)
    with pytest.raises(ValueError):
        D.align(gt, gt, min_depth=1.0, max_depth=0.5)
    with pytest.raises(ValueError):
        D.valid_gt_mask(gt, min_depth=np.nan)


def _depthcrafter_reference(pred_disp, gt_depth, max_depth):
    """numpy transcription of DepthCrafter `benchmark/eval/eval.py::eval_single(domain="depth")` (identical to Video
    Depth Anything's `eval_depthcrafter`) with the `metric.py` functions (per-frame sum / n, then mean over frames)."""
    valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth < max_depth)
    pred_disp = np.clip(pred_disp, a_min=1e-3, a_max=None)
    pred_disp_masked = pred_disp[valid_mask].reshape((-1, 1))
    gt_disp_masked = 1.0 / (gt_depth[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
    A = np.concatenate([pred_disp_masked, np.ones_like(pred_disp_masked)], axis=-1)
    scale, shift = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    aligned_pred = np.clip(scale * pred_disp + shift, a_min=1e-3, a_max=None)
    pred_depth = np.zeros_like(aligned_pred)
    pred_depth[aligned_pred > 0] = 1.0 / aligned_pred[aligned_pred > 0]
    pred_depth = np.clip(pred_depth, a_min=1e-3, a_max=max_depth)
    n = valid_mask.sum((-1, -2))
    keep = n > 0
    p, g, m, n = pred_depth[keep], gt_depth[keep], valid_mask[keep], n[keep]

    def frame_mean(x):
        return (np.where(m, x, 0.0).sum((-1, -2)) / n)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(p / g, g / p)
        out = {"abs_rel": frame_mean(np.abs(p - g) / g).mean(),
               "sq_rel": frame_mean(np.abs(p - g) ** 2 / g).mean(),
               "rmse": np.sqrt(frame_mean((p - g) ** 2)).mean(),
               "rmse_log": np.sqrt(frame_mean((np.log(p) - np.log(g)) ** 2)).mean()}
        for k in (1, 2, 3):
            out[f"delta{k}"] = 100.0 * frame_mean((ratio < 1.25 ** k).astype(float)).mean()
    return out, float(scale[0]), float(shift[0])


def test_matches_depthcrafter_reference_code():
    rng = np.random.default_rng(15)
    T, H, W, max_depth = 7, 12, 16, 70.0
    gt = rng.uniform(0.5, 90.0, (T, H, W))
    gt[:, :2] = 0.0
    gt[3] = 0.0
    disp = 0.8 / gt.clip(0.5) - 0.004 + rng.normal(0, 0.01, gt.shape)
    assert (disp < 1e-3).mean() > 0.05
    ref, s, b = _depthcrafter_reference(disp, gt, max_depth)
    m = D.evaluate_depth(disp, gt, pred_space="disparity", max_depth=max_depth, pred_min=1e-3)
    assert m["align"]["scale"] == pytest.approx(s, rel=1e-6)
    assert m["align"]["shift"] == pytest.approx(b, rel=1e-5, abs=1e-7)
    for k, v in ref.items():
        assert m[k] == pytest.approx(v, rel=1e-6), k
    assert m["n_frames"] == T - 1


def _monodepth2_reference(pred_disps, gt_depths, crop):
    """numpy transcription of Monodepth2 `evaluate_depth.py` (eigen split, monocular: median scaling per image,
    clamp to [1e-3, 80], `compute_errors`, mean over images)."""
    errors = []
    for pred_disp, gt_depth in zip(pred_disps, gt_depths):
        pred_depth = 1 / pred_disp
        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80) & crop
        pred_depth, gt_depth = pred_depth[mask], gt_depth[mask]
        pred_depth *= np.median(gt_depth) / np.median(pred_depth)
        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80
        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
        errors.append([np.mean(np.abs(gt_depth - pred_depth) / gt_depth),
                       np.mean(((gt_depth - pred_depth) ** 2) / gt_depth),
                       np.sqrt(((gt_depth - pred_depth) ** 2).mean()),
                       np.sqrt(((np.log(gt_depth) - np.log(pred_depth)) ** 2).mean()),
                       (thresh < 1.25).mean(), (thresh < 1.25 ** 2).mean(), (thresh < 1.25 ** 3).mean()])
    return dict(zip(("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"),
                    np.array(errors).mean(0) * np.array([1, 1, 1, 1, 100, 100, 100])))


def test_matches_monodepth2_reference_code():
    rng = np.random.default_rng(16)
    gt = rng.uniform(0.5, 95.0, (5, 20, 30))
    gt[rng.uniform(size=gt.shape) < 0.3] = 0.0
    pred_disp = rng.uniform(0.2, 2.0) / gt.clip(0.5) * np.exp(rng.normal(0, 0.2, gt.shape))
    crop = np.zeros((20, 30), bool)
    crop[8:19, 1:29] = True
    ref = _monodepth2_reference(pred_disp, gt, crop)
    m = D.evaluate_depth(1 / pred_disp, gt, np.broadcast_to(crop, gt.shape), align_mode="median", per="frame",
                         space="depth", max_depth=80.0)
    for k, v in ref.items():
        assert m[k] == pytest.approx(v, rel=1e-10, abs=1e-12), k


def test_sample_depth():
    d = np.arange(12.0).reshape(3, 4) + 1.0
    uv = np.array([[1.4, 0.6], [0.5, 0.0], [1.25, 1.5], [4.0, 0.0], [-0.6, 0.0], [-0.4, 0.0], [3.4, 2.4],
                   [np.nan, 1.0]])
    near = D.sample_depth(d, uv, "nearest")
    assert near[0] == 6.0 and near[5] == 1.0 and near[6] == 12.0
    assert np.isnan(near[[3, 4, 7]]).all()
    bil = D.sample_depth(d, uv, "bilinear")
    assert bil[1] == pytest.approx(1.5) and bil[2] == pytest.approx(8.25) and bil[5] == 1.0 and bil[6] == 12.0
    assert np.isnan(bil[[3, 4, 7]]).all()
    d[0, 1] = 0.0
    assert np.isnan(D.sample_depth(d, [[0.5, 0.0]], "bilinear")[0])
    assert D.sample_depth(d, [[0.0, 0.0]], "bilinear")[0] == 1.0
    assert np.isnan(D.sample_depth(d, [[1.0, 0.0]], "nearest")[0])
    assert D.sample_depth(d, np.zeros((2, 3, 2)), "nearest").shape == (2, 3)


def test_pose_basic_helpers():
    T = np.eye(4)[:3]
    assert PO.as_w2c(T).shape == (1, 4, 4) and PO.as_w2c(np.tile(np.eye(4), (2, 3, 1, 1))).shape == (6, 4, 4)
    assert PO.as_w2c(np.zeros((2, 3, 4, 4)), check=False).shape == (6, 4, 4)
    with pytest.raises(ValueError):
        PO.as_w2c(np.zeros((3, 3)))
    A = _w2c(_rand_rot(np.random.default_rng(0)), [1.0, 2.0, 3.0])
    assert np.allclose(PO.invert_rigid(A) @ A, np.eye(4))
    assert PO.all_pairs(4).tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    assert PO.rotation_error_deg(np.eye(3), _rz(37.0)) == pytest.approx(37.0)
    assert PO.rotation_error_deg(np.eye(3), _rz(179.9)) == pytest.approx(179.9)
    assert abs(PO.rotation_error_deg(np.eye(3), _rz(1e-6)) - 1e-6) < 1e-12
    assert PO.direction_error_deg([1, 0, 0], [0, 0, 0]) == 90.0
    assert PO.direction_error_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert PO.direction_error_deg([1, 0, 0], [-1, 0, 0], sign_invariant=True) == pytest.approx(0.0)


def test_pose_gauge_invariance_and_alignment():
    rng = np.random.default_rng(5)
    G = _rand_rig(rng, 6)
    s, R, t = 2.5, _rand_rot(rng), rng.normal(size=3)
    P = PO.transform_w2c(G, s, R, t)
    for conv in PO.CONVENTIONS:
        m = PO.pose_metrics(P, G, convention=conv)
        assert m["ate_rmse"] == pytest.approx(0.0, abs=1e-9) and m["scale"] == pytest.approx(1 / s)
        assert m["are_mean_deg"] == pytest.approx(0.0, abs=1e-6) and not m["degenerate"]
        assert m["rpe_trans_rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["rpe_rot_rmse_deg"] == pytest.approx(0.0, abs=1e-6)
        for k in ("rra@5", "rta@5", "auc@3", "auc@30"):
            assert m[k] == 100.0
        assert (m["n_pairs"], m["n_pairs_translation"]) == (15, 15)
    assert PO.ate(P, G, align="se3")["ate_rmse"] > 0.1
    X = rng.normal(size=3)
    xc = G[0, :3, :3] @ X + G[0, :3, 3]
    xc2 = P[0, :3, :3] @ (s * R @ X + t) + P[0, :3, 3]
    assert np.allclose(xc2, s * xc)
    with pytest.raises(ValueError):
        PO.relative_pose_errors(P, G, convention="colmap")


def test_relative_pose_known_errors():
    G = _line_rig(4)
    P = G.copy()
    P[2] = _w2c(_rz(10.5), (2.0, 0.0, 0.0))
    e = PO.relative_pose_errors(P, G)
    assert np.allclose(e["rot_err_deg"], [0, 10.5, 0, 10.5, 0, 10.5], atol=1e-9)
    assert np.allclose(e["trans_err_deg"], [0, 10.5, 0, 10.5, 0, 0], atol=1e-9)
    assert np.allclose(e["gt_baseline"], [1, 2, 3, 1, 2, 1])
    m = PO.relative_pose_metrics(P, G)
    assert m["rra@5"] == pytest.approx(50.0) and m["rta@5"] == pytest.approx(400 / 6) and m["rra@15"] == 100.0
    assert m["auc@3"] == pytest.approx(50.0)
    assert m["auc@15"] == pytest.approx(100 * (10 * 0.5 + 5) / 15)
    assert m["auc@30"] == pytest.approx(100 * (10 * 0.5 + 20) / 30)
    assert m["rre_median_deg"] == pytest.approx(5.25)
    pairs = np.array([[0, 2], [2, 3]])
    e2 = PO.relative_pose_errors(P, G, pairs)
    assert np.allclose(e2["rot_err_deg"], [10.5, 10.5]) and np.allclose(e2["trans_err_deg"], [10.5, 0], atol=1e-9)
    with pytest.raises(IndexError):
        PO.relative_pose_errors(P, G, [[0, 4]])
    ev = PO.relative_pose_errors(P, G, convention="vggt")
    assert np.allclose(ev["rot_err_deg"], e["rot_err_deg"], atol=1e-9)
    assert np.allclose(ev["trans_err_deg"], [0, 0, 0, 0, 0, 10.5], atol=1e-9)


def _quat_angle_deg(Rg, Rp):
    """VGGT / Depth-Anything-3 `rotation_angle` (quaternions: arccos(1 - 2 (1 - <q_p, q_g>^2)))."""
    from scipy.spatial.transform import Rotation
    d = np.sum(Rotation.from_matrix(Rp).as_quat() * Rotation.from_matrix(Rg).as_quat(), 1)
    return np.degrees(np.arccos(np.clip(1 - 2 * np.clip(1 - d ** 2, 1e-15, None), -1, 1)))


def _translation_angle_deg(tg, tp):
    """`compare_translation_by_angle` + the ambiguity min(a, |180 - a|) of the reference code."""
    tp = tp / (np.linalg.norm(tp, axis=1, keepdims=True) + 1e-15)
    tg = tg / (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-15)
    err = np.degrees(np.arccos(np.sqrt(1 - np.clip(1.0 - np.sum(tp * tg, 1) ** 2, 1e-15, None))))
    return np.minimum(err, np.abs(180 - err))


def _reference_pair_errors(P, G, convention):
    """Transcriptions of the reference pair transforms (np.linalg.inv instead of the closed-form inverse):
    relative     = PoseDiffusion `camera_to_rel_deg` on pytorch3d row-vector matrices M = w2c^T: inv(M_i) M_j,
                   translation = row 3;
    vggt         = VGGT `evaluation/test_co3d.py::se3_to_relative_pose_error` (evaluation branch): w2c_i inv(w2c_j);
    first_camera = Depth-Anything-3 `bench/utils.py::compute_pose`: align_to_first_camera, then inv(w2c_i) w2c_j."""
    inv = np.linalg.inv
    i, j = np.triu_indices(len(G), 1)
    if convention == "relative":
        Mg, Mp = np.transpose(G, (0, 2, 1)), np.transpose(P, (0, 2, 1))
        rg, rp = inv(Mg[i]) @ Mg[j], inv(Mp[i]) @ Mp[j]
        return _quat_angle_deg(rg[:, :3, :3], rp[:, :3, :3]), _translation_angle_deg(rg[:, 3, :3], rp[:, 3, :3])
    if convention == "vggt":
        rg, rp = G[i] @ inv(G[j]), P[i] @ inv(P[j])
    else:
        G0, P0 = G @ inv(G[0]), P @ inv(P[0])
        rg, rp = inv(G0[i]) @ G0[j], inv(P0[i]) @ P0[j]
    return _quat_angle_deg(rg[:, :3, :3], rp[:, :3, :3]), _translation_angle_deg(rg[:, :3, 3], rp[:, :3, 3])


def _auc_reference(r_error, t_error, max_threshold):
    """`calculate_auc_np` of PoseDiffusion / VGGT / Depth-Anything-3, in percent."""
    max_errors = np.max(np.concatenate((r_error[:, None], t_error[:, None]), axis=1), axis=1)
    histogram, _ = np.histogram(max_errors, bins=np.arange(max_threshold + 1))
    return 100.0 * np.mean(np.cumsum(histogram.astype(float) / float(len(max_errors))))


@pytest.mark.parametrize("convention", PO.CONVENTIONS)
def test_relative_pose_matches_reference_code(convention):
    rng = np.random.default_rng(6)
    G = _rand_rig(rng, 7)
    P = PO.transform_w2c(_perturbed(rng, G), 1.7, _rand_rot(rng), rng.normal(size=3))
    e = PO.relative_pose_errors(P, G, convention=convention)
    rot, tr = _reference_pair_errors(P, G, convention)
    assert np.allclose(e["rot_err_deg"], rot, atol=1e-5)
    assert np.allclose(e["trans_err_deg"], tr, atol=1e-5)
    m = PO.relative_pose_metrics(P, G, convention=convention)
    for tau in PO.AUC_THRESHOLDS:
        assert m[f"auc@{tau}"] == pytest.approx(_auc_reference(rot, tr, tau), abs=1e-9)
    for tau in PO.RA_THRESHOLDS:
        assert m[f"rra@{tau}"] == pytest.approx(100 * np.mean(rot < tau))
        assert m[f"rta@{tau}"] == pytest.approx(100 * np.mean(tr < tau))


def test_conventions_relations():
    rng = np.random.default_rng(17)
    G = _rand_rig(rng, 6)
    P = _perturbed(rng, G)
    e = {c: PO.relative_pose_errors(P, G, convention=c) for c in PO.CONVENTIONS}
    assert np.allclose(e["vggt"]["rot_err_deg"], e["relative"]["rot_err_deg"], atol=1e-9)
    assert not np.allclose(e["vggt"]["trans_err_deg"], e["relative"]["trans_err_deg"], atol=1e-3)
    assert not np.allclose(e["first_camera"]["rot_err_deg"], e["relative"]["rot_err_deg"], atol=1e-3)
    Q = np.stack([_w2c(PO.invert_rigid(g)[:3, :3], PO.invert_rigid(g)[:3, 3] + rng.normal(scale=0.3, size=3))
                  for g in G])
    a = PO.relative_pose_errors(Q, G)
    b = PO.relative_pose_errors(Q, G, convention="vggt")
    assert np.allclose(a["trans_err_deg"], b["trans_err_deg"], atol=1e-9)
    assert np.allclose(a["rot_err_deg"], 0, atol=1e-6)
    for c in PO.CONVENTIONS:
        assert np.allclose(e[c]["gt_baseline"], e["relative"]["gt_baseline"])


def test_relative_pose_errors_blocked_computation(monkeypatch):
    rng = np.random.default_rng(24)
    G = _rand_rig(rng, 9)
    P = _perturbed(rng, G)
    full = {c: PO.relative_pose_errors(P, G, convention=c) for c in PO.CONVENTIONS}
    monkeypatch.setattr(PO, "_PAIR_CHUNK", 5)
    for c in PO.CONVENTIONS:
        e = PO.relative_pose_errors(P, G, convention=c)
        assert np.allclose(e["rot_err_deg"], full[c]["rot_err_deg"], rtol=0, atol=1e-10)
        assert np.allclose(e["trans_err_deg"], full[c]["trans_err_deg"], rtol=0, atol=1e-10)
    assert PO.relative_pose_errors(P[:1], G[:1])["rot_err_deg"].shape == (0,)


def test_rig_pairs():
    V, T = 3, 4
    sf = PO.rig_pairs(V, T, "same_frame")
    assert len(sf) == T * V * (V - 1) // 2 and (sf[:, 0] < sf[:, 1]).all()
    assert all(a % T == b % T and a // T != b // T for a, b in sf)
    sv = PO.rig_pairs(V, T, "same_view")
    assert len(sv) == V * T * (T - 1) // 2 and all(a // T == b // T for a, b in sv)
    assert np.array_equal(PO.rig_pairs(V, T, "all"), PO.all_pairs(V * T))
    assert len({tuple(p) for p in sf} | {tuple(p) for p in sv}) == len(sf) + len(sv)
    with pytest.raises(ValueError):
        PO.rig_pairs(V, T, "cross")
    G = np.stack([np.stack([_w2c(np.eye(3), (v, 0.0, 0.0))] * T) for v in range(V)])
    P = G.copy()
    P[2] = _w2c(_rz(10.0), (2.0, 0.0, 0.0))
    m = PO.pose_metrics(P, G, pairs=PO.rig_pairs(V, T), rpe_delta=None)
    assert m["n_pairs"] == T * 3 and m["rra@5"] == pytest.approx(100 / 3)
    assert PO.pose_metrics(P, G, rpe_delta=None)["rra@5"] > m["rra@5"]


def test_translation_sign_invariance():
    G = np.stack([_w2c(np.eye(3), (0, 0, 0)), _w2c(np.eye(3), (1, 0, 0))])
    P = np.stack([_w2c(np.eye(3), (0, 0, 0)), _w2c(np.eye(3), (-1, 0, 0))])
    assert PO.relative_pose_errors(P, G)["trans_err_deg"][0] == pytest.approx(0.0)
    assert PO.relative_pose_errors(P, G, sign_invariant=False)["trans_err_deg"][0] == pytest.approx(180.0)


def test_pose_auc_and_rra_rta_formulas():
    assert PO.pose_auc([0.5, 1.5, 2.5], [0.0, 0.0, np.nan], 3) == pytest.approx(100 * (1 / 3 + 2 / 3 + 1) / 3)
    assert PO.pose_auc([3.0], [0.0], 3) == pytest.approx(100 / 3)
    assert PO.pose_auc([0.5, np.inf], [0.0, 0.0], 2) == pytest.approx(50.0)
    assert PO.pose_auc([0.5, np.nan], [0.0, np.nan], 2) == pytest.approx(100.0)
    assert math.isnan(PO.pose_auc([np.nan], [np.nan], 5))
    with pytest.raises(ValueError):
        PO.pose_auc([1.0], [1.0], 0)
    r = PO.rra_rta([1.0, 6.0, np.inf, np.nan], [1.0, np.nan, 20.0, np.nan], thresholds=(5, 15))
    assert r == {"rra@5": pytest.approx(100 / 3), "rta@5": 50.0, "rra@15": pytest.approx(200 / 3), "rta@15": 50.0}
    rng = np.random.default_rng(18)
    for _ in range(50):
        rr, tt = np.round(rng.uniform(0, 35, 40)), rng.uniform(0, 35, 40)
        tt[:5] = 30.0
        for tau in (3, 5, 15, 30):
            assert PO.pose_auc(rr, tt, tau) == pytest.approx(_auc_reference(rr, tt, tau), abs=1e-9)


def test_static_rig_zero_baseline_pairs():
    G = np.stack([np.stack([_w2c(np.eye(3), (v, 0.0, 0.0))] * 3) for v in range(2)])
    for conv in PO.CONVENTIONS:
        m = PO.relative_pose_metrics(G, G, convention=conv)
        assert (m["n_pairs"], m["n_pairs_translation"]) == (15, 9)
        assert m["rra@5"] == m["rta@5"] == m["auc@30"] == 100.0
    e = PO.relative_pose_errors(G, G)
    same = np.array([a // 3 == b // 3 for a, b in e["pairs"]])
    assert np.isnan(e["trans_err_deg"][same]).all() and not np.isnan(e["trans_err_deg"][~same]).any()
    a = PO.ate(G, G)
    assert a["degenerate"] and a["ate_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_missing_predicted_pose():
    G = _line_rig(4)
    P = G.copy()
    P[1] = np.nan
    m = PO.relative_pose_metrics(P, G)
    assert m["rra@5"] == pytest.approx(50.0) and m["rta@5"] == pytest.approx(50.0) and m["n_pairs"] == 6
    assert m["rre_mean_deg"] == np.inf
    a = PO.ate(P, G)
    assert (a["n_poses"], a["n_missing"]) == (3, 1) and a["ate_rmse"] == pytest.approx(0.0, abs=1e-12)
    G2 = G.copy()
    G2[3] = np.nan
    assert PO.relative_pose_metrics(G, G2)["n_pairs"] == 3
    P0 = G.copy()
    P0[0] = np.nan
    assert PO.relative_pose_metrics(P0, G, convention="first_camera")["rra@30"] == 0.0
    assert PO.relative_pose_metrics(P0, G)["rra@30"] == 50.0


def test_all_poses_missing_does_not_raise():
    G = np.stack([_w2c(np.eye(3), (t, 0.0, 0.0)) for t in range(5)])
    P = np.full_like(G, np.nan)
    m = PO.pose_metrics(P, G)
    assert math.isnan(m["ate_rmse"]) and math.isnan(m["rpe_trans_rmse"]) and m["n_rpe_pairs"] == 0
    assert math.isnan(m["rpe_scale"]) and m["rra@5"] == 0.0 and m["auc@30"] == 0.0 and m["n_missing"] == 5


def test_ate_known_offset_and_degeneracy():
    rng = np.random.default_rng(7)
    G = _rand_rig(rng, 5)
    P = np.stack([_w2c(PO.invert_rigid(g)[:3, :3], PO.invert_rigid(g)[:3, 3] + [0.3, 0.0, 0.4]) for g in G])
    a = PO.ate(P, G, align="none")
    assert a["ate_rmse"] == pytest.approx(0.5) and a["ate_max"] == pytest.approx(0.5) and a["scale"] == 1.0
    assert PO.ate(P, G, align="se3")["ate_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert PO.ate(P, G, align="sim3")["ate_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert PO.ate(_line_rig(4), _line_rig(4))["degenerate"]
    assert PO.ate(P, G, valid=[True, True, False, True, True])["n_poses"] == 4
    nan = PO.ate(np.full((2, 4, 4), np.nan), _line_rig(2))
    assert math.isnan(nan["ate_rmse"]) and nan["n_missing"] == 2
    with pytest.raises(ValueError):
        PO.ate(G, G, align="sim2")


def test_ate_rmse_after_sim3_matches_umeyama_residual():
    rng = np.random.default_rng(19)
    G = _rand_rig(rng, 8)
    P = PO.transform_w2c(_perturbed(rng, G, sigma=0.2), 0.4, _rand_rot(rng), rng.normal(size=3))
    from vpm_worldbench.geometry import camera_center, umeyama
    cg, cp = camera_center(G), camera_center(P)
    s, R, t = umeyama(cp, cg)
    res = np.linalg.norm(s * cp @ R.T + t - cg, axis=1)
    a = PO.ate(P, G)
    assert a["ate_rmse"] == pytest.approx(np.sqrt(np.mean(res ** 2))) and a["scale"] == pytest.approx(s)
    assert a["ate_rmse"] < PO.ate(P, G, align="se3")["ate_rmse"]


def test_rpe_known_values():
    T = 6
    G = np.stack([_w2c(np.eye(3), (t, 0.0, 0.0)) for t in range(T)])
    P = np.stack([_w2c(np.eye(3), (2.0 * t, 0.0, 0.0)) for t in range(T)])
    r = PO.rpe(P, G, align="none")
    assert r["rpe_trans_rmse"] == pytest.approx(1.0) and r["rpe_rot_rmse_deg"] == pytest.approx(0.0, abs=1e-9)
    assert r["n_rpe_pairs"] == T - 1
    r = PO.rpe(P, G, align="sim3")
    assert r["rpe_scale"] == pytest.approx(0.5) and r["rpe_trans_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert PO.rpe(P, G, delta=2, align="none")["rpe_trans_mean"] == pytest.approx(2.0)
    G = np.stack([np.eye(4)] * T)
    P = np.stack([_w2c(_rz(3.0 * t), (0.0, 0.0, 0.0)) for t in range(T)])
    r = PO.rpe(P, G, align="none")
    assert r["rpe_rot_mean_deg"] == pytest.approx(3.0) and r["rpe_trans_rmse"] == pytest.approx(0.0, abs=1e-12)
    r = PO.rpe(np.stack([P, P]), np.stack([G, G]), align="none")
    assert r["n_rpe_pairs"] == 2 * (T - 1)
    assert math.isnan(PO.rpe(P[:1], G[:1], align="none")["rpe_trans_rmse"])
    with pytest.raises(ValueError):
        PO.rpe(P, G, delta=0)


def test_pose_inputs_must_be_rigid():
    rng = np.random.default_rng(25)
    G = _rand_rig(rng, 5)
    P = _perturbed(rng, G)
    scaled = P.copy()
    scaled[:, :3, :] *= 2.0
    mirrored = P.copy()
    mirrored[1, :3, 0] *= -1.0
    for bad in (scaled, mirrored):
        for f in (PO.pose_metrics, PO.relative_pose_metrics, PO.ate, PO.rpe, PO.align_cameras):
            with pytest.raises(ValueError, match="not rigid"):
                f(bad, G)
        with pytest.raises(ValueError):
            PO.pose_metrics(G, bad)
    noisy = P + rng.normal(scale=1e-6, size=P.shape)
    assert PO.pose_metrics(noisy, G)["n_pairs"] == 10
    missing = P.copy()
    missing[2] = np.nan
    assert PO.pose_metrics(missing, G)["n_missing"] == 1
    zero_bottom = P.copy()
    zero_bottom[:, 3, :] = 0.0
    before = zero_bottom.copy()
    for conv in PO.CONVENTIONS:
        a = PO.relative_pose_errors(zero_bottom, G, convention=conv)
        b = PO.relative_pose_errors(P, G, convention=conv)
        assert np.array_equal(a["rot_err_deg"], b["rot_err_deg"])
        assert np.array_equal(a["trans_err_deg"], b["trans_err_deg"])
    assert np.array_equal(zero_bottom, before)


def test_pose_valid_and_thresholds_are_checked():
    G = np.stack([_w2c(_rz(5.0 * t), (t, 0.1 * t * t, 0.0)) for t in range(6)])
    with pytest.raises(ValueError):
        PO.ate(G, G, valid=[True])
    with pytest.raises(ValueError):
        PO.align_cameras(G, G, valid=np.ones(5, bool))
    with pytest.raises(TypeError):
        PO.ate(G, G, valid=np.ones(6))
    vt = np.ones((2, 3), bool)
    vt[1, 2] = False
    assert PO.ate(G.reshape(2, 3, 4, 4), G.reshape(2, 3, 4, 4), valid=vt)["n_poses"] == 5
    assert PO.ate(G, G, valid=np.array([1, 1, 0, 1, 1, 1]))["n_poses"] == 5
    with pytest.raises(ValueError):
        PO.pose_auc([1.0], [1.0], 7.5)
    with pytest.raises(ValueError):
        PO.relative_pose_metrics(G, G, auc_thresholds=(7.5,))
    assert PO.pose_auc([1.0], [1.0], 30.0) == PO.pose_auc([1.0], [1.0], 30)
    assert "auc@30" in PO.relative_pose_metrics(G, G, auc_thresholds=(30.0,))


def test_single_and_empty_pose_sets():
    one = _w2c(_rz(10.0), (1.0, 2.0, 3.0))
    for P in (one, one[:3]):
        m = PO.pose_metrics(P, P)
        assert m["n_poses"] == 1 and m["ate_rmse"] == pytest.approx(0.0, abs=1e-12) and m["degenerate"]
        assert m["n_rpe_pairs"] == 0 and math.isnan(m["rpe_trans_rmse"])
        assert m["n_pairs"] == 0 and math.isnan(m["rra@5"]) and math.isnan(m["auc@30"])
    m = PO.pose_metrics(np.zeros((0, 4, 4)), np.zeros((0, 4, 4)))
    assert m["n_poses"] == 0 and math.isnan(m["ate_rmse"]) and m["n_pairs"] == 0 and m["n_rpe_pairs"] == 0


def test_align_cameras_and_transform_roundtrip():
    rng = np.random.default_rng(8)
    G = _rand_rig(rng, 7)
    s, R, t = 0.7, _rand_rot(rng), rng.normal(size=3)
    P = PO.transform_w2c(G, s, R, t)
    s2, R2, t2 = PO.align_cameras(P, G, "sim3")
    assert np.allclose(PO.transform_w2c(P, s2, R2, t2), G, atol=1e-9)
    assert PO.align_cameras(P, G, "none")[0] == 1.0
    with pytest.raises(ValueError):
        PO.align_cameras(np.full_like(G, np.nan), G)


def _grid(n=10):
    return np.stack(np.meshgrid(np.arange(float(n)), np.arange(float(n)), [0.0], indexing="ij"), -1).reshape(-1, 3)


def test_pointcloud_identical_and_shifted():
    x = _grid()
    m = PC.pointcloud_metrics(x, x, thresholds=(0.01,))
    assert m["chamfer_mean"] == 0.0 and m["fscore@0.01"] == 100.0 and (m["n_pred"], m["n_gt"]) == (100, 100)
    m = PC.pointcloud_metrics(x + [0.0, 0.0, 0.1], x, thresholds=(0.05, 0.2))
    for k in ("acc_mean", "acc_median", "comp_mean", "comp_median", "chamfer_mean", "chamfer_median"):
        assert m[k] == pytest.approx(0.1)
    assert m["chamfer_sum"] == pytest.approx(0.2)
    assert (m["precision@0.05"], m["recall@0.05"], m["fscore@0.05"]) == (0.0, 0.0, 0.0)
    assert (m["precision@0.2"], m["recall@0.2"], m["fscore@0.2"]) == (100.0, 100.0, 100.0)


def test_pointcloud_partial_reconstruction():
    gt = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], -1)
    m = PC.pointcloud_metrics(gt[:5], gt, thresholds=(0.5,))
    assert m["acc_mean"] == 0.0 and m["comp_mean"] == pytest.approx(1.5) and m["comp_median"] == pytest.approx(0.5)
    assert m["chamfer_mean"] == pytest.approx(0.75) and m["chamfer_median"] == pytest.approx(0.25)
    assert m["precision@0.5"] == 100.0 and m["recall@0.5"] == 50.0
    assert m["fscore@0.5"] == pytest.approx(200 / 3)


def _recon_reference(pred, gt, threshold):
    """Depth-Anything-3 `bench/utils.py::evaluate_3d_reconstruction` (numpy arrays, no downsampling)."""
    from scipy.spatial import KDTree
    d_pg = KDTree(gt).query(pred)[0].reshape(-1)
    d_gp = KDTree(pred).query(gt)[0].reshape(-1)
    acc, comp = float(np.mean(d_pg)), float(np.mean(d_gp))
    precision = float(np.mean((d_pg < threshold).astype(float)))
    recall = float(np.mean((d_gp < threshold).astype(float)))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"acc": acc, "comp": comp, "overall": (acc + comp) / 2, "precision": precision, "recall": recall,
            "fscore": fscore}


def test_pointcloud_matches_reference_code():
    rng = np.random.default_rng(20)
    gt = rng.normal(size=(1500, 3))
    pred = gt[:1100] + rng.normal(scale=0.04, size=(1100, 3))
    ref = _recon_reference(pred, gt, 0.05)
    m = PC.pointcloud_metrics(pred, gt, thresholds=(0.05,))
    assert m["acc_mean"] == pytest.approx(ref["acc"]) and m["comp_mean"] == pytest.approx(ref["comp"])
    assert m["chamfer_mean"] == pytest.approx(ref["overall"])
    assert m["precision@0.05"] == pytest.approx(100 * ref["precision"])
    assert m["recall@0.05"] == pytest.approx(100 * ref["recall"])
    assert m["fscore@0.05"] == pytest.approx(100 * ref["fscore"])


def test_pointcloud_empty_nan_and_mask():
    x = _grid(3)
    m = PC.pointcloud_metrics(np.full((4, 3), np.nan), x)
    assert m["n_pred"] == 0 and m["acc_mean"] == np.inf and m["chamfer_sum"] == np.inf and m["fscore@0.05"] == 0.0
    y = x.copy()
    y[0] = np.nan
    assert PC.pointcloud_metrics(y, x)["n_pred"] == 8
    pred = x + [0.0, 0.0, 1.0]
    pred[:4] = x[:4]
    mask = np.zeros(len(x), bool)
    mask[:4] = True
    m = PC.pointcloud_metrics(pred, x, mask=mask)
    assert m["chamfer_mean"] == 0.0 and m["n_pred"] == m["n_gt"] == 4
    with pytest.raises(ValueError):
        PC.pointcloud_metrics(x[:3], x, mask=mask)


def test_pointcloud_unalignable_prediction_is_a_failure():
    x = _grid(3)
    for pred in (np.full_like(x, np.nan), np.where(np.arange(len(x))[:, None] < 2, x, np.nan)):
        m = PC.pointcloud_metrics(pred, x, thresholds=(0.1,), align="sim3")
        assert m["align"]["failed"] and m["align"]["mode"] == "sim3" and "error" in m["align"]
        assert m["acc_mean"] == m["chamfer_mean"] == np.inf and m["fscore@0.1"] == 0.0
        assert m["n_gt"] == 9
    m = PC.pointcloud_metrics(np.zeros_like(x), x, align="scale")
    assert m["align"]["failed"] and m["recall@0.05"] == 0.0
    assert not PC.pointcloud_metrics(x, x, align="sim3")["align"]["failed"]


@pytest.mark.parametrize("mode", ["sim3", "se3", "scale"])
def test_pointcloud_alignment(mode):
    rng = np.random.default_rng(9)
    gt = rng.normal(size=(60, 3))
    s = 1.7 if mode in ("sim3", "scale") else 1.0
    R = _rand_rot(rng) if mode in ("sim3", "se3") else np.eye(3)
    t = rng.normal(size=3) if mode in ("sim3", "se3") else np.zeros(3)
    pred = ((gt - t) @ R) / s
    pred_map = pred.reshape(6, 10, 3).copy()
    gt_map = gt.reshape(6, 10, 3).copy()
    gt_map[0, 0] = np.nan
    m = PC.pointcloud_metrics(pred_map, gt_map, align=mode)
    assert m["align"]["scale"] == pytest.approx(s) and (m["n_gt"], m["n_pred"]) == (59, 60)
    assert m["comp_mean"] == pytest.approx(0.0, abs=1e-6) and m["acc_median"] == pytest.approx(0.0, abs=1e-6)
    assert m["acc_mean"] > 1e-4
    valid = np.isfinite(gt_map).all(-1)
    m = PC.pointcloud_metrics(pred_map, gt_map, align=mode, mask=valid)
    assert m["chamfer_mean"] == pytest.approx(0.0, abs=1e-6) and (m["n_gt"], m["n_pred"]) == (59, 59)
    assert PC.pointcloud_metrics(pred_map, gt_map)["chamfer_mean"] > 0.1
    with pytest.raises(ValueError):
        PC.pointcloud_metrics(pred[:10], gt, align=mode)


def test_pointcloud_mask_must_match_the_point_grid():
    gt = np.random.default_rng(28).normal(size=(4, 6, 3))
    with pytest.raises(ValueError):
        PC.pointcloud_metrics(gt, gt, mask=np.ones((6, 4), bool))
    with pytest.raises(ValueError):
        PC.pointcloud_metrics(gt, gt, mask=np.ones(24, bool))
    with pytest.raises(ValueError):
        PC.fit_similarity(gt, gt, "sim3", mask=np.ones((6, 4), bool))
    with pytest.raises(TypeError):
        PC.pointcloud_metrics(gt, gt, mask=np.ones((4, 6)))
    m = PC.pointcloud_metrics(gt, gt, mask=np.ones((4, 6), np.uint8))
    assert m["n_pred"] == m["n_gt"] == 24 and m["chamfer_mean"] == 0.0


def test_fit_and_apply_similarity():
    rng = np.random.default_rng(10)
    a = rng.normal(size=(20, 3))
    R = _rand_rot(rng)
    b = PC.apply_similarity(a, 2.0, R, [1.0, 2.0, 3.0])
    s, R2, t2 = PC.fit_similarity(a, b, "sim3")
    assert s == pytest.approx(2.0) and np.allclose(R2, R) and np.allclose(t2, [1, 2, 3])
    assert PC.apply_similarity(a.reshape(4, 5, 3), 1.0, np.eye(3), np.zeros(3)).shape == (4, 5, 3)
    assert PC.fit_similarity(a, b, "none")[0] == 1.0
    with pytest.raises(ValueError):
        PC.fit_similarity(a[:2], b[:2], "sim3")
    with pytest.raises(ValueError):
        PC.fit_similarity(a, b, "affine")


def test_voxel_downsample():
    p = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.6, 0.1, 0.1]])
    assert np.allclose(PC.voxel_downsample(p, 1.0), [[0.15, 0.15, 0.15], [1.6, 0.1, 0.1]])
    assert np.allclose(PC.voxel_downsample(p, 1.0, "first"), [p[0], p[2]])
    assert np.allclose(PC.voxel_downsample(p, 1.0, "center"), [[0.1, 0.1, 0.1], [2.1, 0.1, 0.1]])
    assert np.allclose(PC.voxel_downsample(p, 1.0, "center", origin=(0, 0, 0)), [[0.5] * 3, [1.5, 0.5, 0.5]])
    assert np.allclose(PC.voxel_downsample(p[::-1], 1.0), PC.voxel_downsample(p, 1.0))
    assert np.allclose(PC.voxel_downsample(p, 0.0), p)
    rng = np.random.default_rng(11)
    q = rng.uniform(0, 4, (500, 3))
    out = PC.voxel_downsample(q, 1.0)
    assert len(out) == len(np.unique(np.floor((q - (q.min(0) - 0.5)) / 1.0), axis=0))
    idx = np.floor((q - (q.min(0) - 0.5)) / 1.0).astype(int)
    ref = {}
    for k, pt in zip(map(tuple, idx), q):
        ref.setdefault(k, []).append(pt)
    ref = np.array([np.mean(ref[k], 0) for k in sorted(ref)])
    assert np.allclose(out, ref)
    huge = PC.voxel_downsample(np.array([[0.0, 0.0, 0.0], [1e6, 1e6, 1e6], [1e6, 1e6, 1e6]]), 1e-6)
    assert np.allclose(huge, [[0, 0, 0], [1e6, 1e6, 1e6]])
    with pytest.raises(ValueError):
        PC.voxel_downsample(p, 1.0, "median")


def test_nn_distance_and_subsample():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(40, 3)), rng.normal(size=(30, 3))
    brute = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(PC.nn_distance(a, b), brute) and np.allclose(PC.nn_distance(a, b, workers=1), brute)
    assert np.isinf(PC.nn_distance(a, np.zeros((0, 3)))).all() and PC.nn_distance(np.zeros((0, 3)), b).size == 0
    a2, b2 = a.copy(), np.vstack([b, [[np.nan, 0.0, 0.0]]])
    a2[3] = np.inf
    d = PC.nn_distance(a2, b2)
    assert np.isnan(d[3]) and np.allclose(np.delete(d, 3), np.delete(brute, 3))
    assert np.isinf(PC.nn_distance(a[:2], np.full((2, 3), np.nan))).all()
    s1, s2 = PC.subsample(a, 10, seed=3), PC.subsample(a, 10, seed=3)
    assert len(s1) == 10 and np.array_equal(s1, s2)
    idx = [int(np.nonzero((a == r).all(1))[0][0]) for r in s1]
    assert idx == sorted(idx)
    assert np.array_equal(PC.subsample(a, None), a) and np.array_equal(PC.subsample(a, 100), a)
    assert PC.as_points(np.array([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]])).shape == (1, 3)
    m = PC.pointcloud_metrics(a, b, max_points=5, voxel_size=0.01)
    assert m["n_pred"] == m["n_gt"] == 5


def test_jaccard():
    g = np.zeros((6, 6), bool)
    g[1:5, 1:5] = True
    p = np.zeros_like(g)
    p[2:4, 2:4] = True
    assert SG.jaccard(p, g) == pytest.approx(0.25)
    assert SG.jaccard(np.zeros_like(g), np.zeros_like(g)) == 1.0
    void = np.zeros_like(g)
    void[1:5, 1:3] = True
    assert SG.jaccard(p, g, void) == pytest.approx(2 / 8)
    v = SG.jaccard(np.stack([p, g, np.zeros_like(g)]), np.stack([g, g, g]))
    assert v.shape == (3,) and np.allclose(v, [0.25, 1.0, 0.0])
    assert SG.jaccard(p.astype(np.uint8) * 255, g.astype(np.float32)) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        SG.jaccard(p, g[:5])


def test_seg2bmap_single_pixel():
    seg = np.zeros((4, 4), bool)
    seg[1, 1] = True
    expect = np.zeros((4, 4), bool)
    expect[0:2, 0:2] = True
    assert np.array_equal(SG.seg2bmap(seg), expect)
    full = np.ones((3, 3), bool)
    assert not SG.seg2bmap(full).any()


def test_boundary_radius_and_disk():
    assert SG.boundary_radius((480, 854)) == 8 and SG.boundary_radius((480, 480)) == 6
    assert SG.boundary_radius((480, 640)) == 7 and SG.boundary_radius((768, 1024)) == 11
    assert SG.boundary_radius((10, 10), bound_th=3) == 3
    skm = pytest.importorskip("skimage.morphology")
    for r in (1, 2, 5, 8):
        assert np.array_equal(SG._disk(r), skm.disk(r).astype(np.uint8))


def test_dilation_backends_agree():
    rng = np.random.default_rng(13)
    b = rng.uniform(size=(37, 41)) > 0.97
    for r in (1, 3, 6):
        assert np.array_equal(SG._dilate(b, r, "cv2"), SG._dilate(b, r, "scipy"))


def _f_reference(pred, gt, r):
    """Brute force: boundary pixels matched when within Euclidean distance r of the other boundary."""
    bp, bg = np.argwhere(SG.seg2bmap(pred)), np.argwhere(SG.seg2bmap(gt))
    d = np.sqrt(((bp[:, None, :] - bg[None, :, :]) ** 2).sum(-1))
    P, R = np.mean(d.min(1) <= r), np.mean(d.min(0) <= r)
    return 0.0 if P + R == 0 else 2 * P * R / (P + R)


def test_boundary_f_against_brute_force():
    yy, xx = np.mgrid[:30, :40]
    gt = ((yy - 15) / 9.0) ** 2 + ((xx - 20) / 13.0) ** 2 <= 1
    pred = ((yy - 13) / 8.0) ** 2 + ((xx - 23) / 14.0) ** 2 <= 1
    for th in (1, 2, 4):
        assert SG.boundary_f(pred, gt, bound_th=th) == pytest.approx(_f_reference(pred, gt, th))
    r = SG.boundary_radius(gt.shape)
    assert r == 1 and SG.boundary_f(pred, gt) == pytest.approx(_f_reference(pred, gt, 1))


def _davis_seg2bmap(seg):
    """davis2017-evaluation `metrics.py::_seg2bmap` (same-size case)."""
    seg = seg.astype(bool)
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _davis_f_measure(fg, gt, void=None, bound_th=0.008):
    """davis2017-evaluation `metrics.py::f_measure` (cv2.dilate with skimage's disk)."""
    import cv2
    skm = pytest.importorskip("skimage.morphology")
    void = np.zeros_like(fg).astype(bool) if void is None else void.astype(bool)
    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(fg.shape))
    fg_b = _davis_seg2bmap(fg * np.logical_not(void))
    gt_b = _davis_seg2bmap(gt * np.logical_not(void))
    fg_dil = cv2.dilate(fg_b.astype(np.uint8), skm.disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(gt_b.astype(np.uint8), skm.disk(bound_pix).astype(np.uint8))
    n_fg, n_gt = np.sum(fg_b), np.sum(gt_b)
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1, 0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0, 1
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1, 1
    else:
        precision, recall = np.sum(fg_b * gt_dil) / float(n_fg), np.sum(gt_b * fg_dil) / float(n_gt)
    return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _davis_statistics(v):
    """davis2017-evaluation `utils.py::db_statistics` (uint8 bin edges: fine below 256 frames)."""
    M, O = np.nanmean(v), np.nanmean(v > 0.5)
    ids = (np.round(np.linspace(1, len(v), 5) + 1e-10) - 1).astype(np.uint8)
    bins = [v[ids[i]:ids[i + 1] + 1] for i in range(4)]
    return M, O, np.nanmean(bins[0]) - np.nanmean(bins[3])


def test_j_and_f_matches_davis_reference_code():
    rng = np.random.default_rng(21)
    for trial in range(12):
        H, W, T = int(rng.integers(30, 90)), int(rng.integers(30, 90)), int(rng.integers(3, 25))
        yy, xx = np.mgrid[:H, :W]
        gt, pr = np.zeros((T, H, W), bool), np.zeros((T, H, W), bool)
        for t in range(T):
            cy, cx = rng.uniform(0.3, 0.7) * H, rng.uniform(0.3, 0.7) * W
            ry, rx = rng.uniform(4, H / 3), rng.uniform(4, W / 3)
            gt[t] = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            pr[t] = ((yy - cy - rng.normal(0, 2)) / (ry * rng.uniform(0.8, 1.2))) ** 2 + \
                ((xx - cx - rng.normal(0, 2)) / (rx * rng.uniform(0.8, 1.2))) ** 2 <= 1
            pr[t] ^= rng.uniform(size=(H, W)) > 0.995
            if rng.uniform() < 0.15:
                pr[t] = False
        void = (rng.uniform(size=(T, H, W)) > 0.98) if trial % 2 else None
        th = 0.008 if trial % 3 else 2
        f_ref = np.array([_davis_f_measure(pr[t], gt[t], None if void is None else void[t], th) for t in range(T)])
        assert np.allclose(SG.boundary_f(pr, gt, void, th), f_ref, rtol=0, atol=1e-12)
        s = SG.j_and_f(pr, gt, void, th)
        J = SG.jaccard(pr, gt, void)
        for key, vals in (("J", J), ("F", f_ref)):
            M, O, Dd = _davis_statistics(vals)
            assert s[key] == pytest.approx(100 * M) and s[f"{key}_recall"] == pytest.approx(100 * O)
            assert s[f"{key}_decay"] == pytest.approx(100 * Dd)


def test_boundary_f_special_cases():
    g = np.zeros((20, 20), bool)
    g[5:15, 5:15] = True
    z = np.zeros_like(g)
    assert SG.boundary_f(g, g) == 1.0
    assert SG.boundary_f(z, g) == 0.0
    assert SG.boundary_f(g, z) == 0.0
    assert SG.boundary_f(z, z) == 1.0
    shifted = np.roll(g, 1, axis=1)
    assert SG.boundary_f(shifted, g, bound_th=1) == 1.0
    assert SG.boundary_f(np.stack([g, z]), np.stack([g, g])).tolist() == [1.0, 0.0]
    void = np.zeros_like(g)
    void[:, 10:] = True
    half = g.copy()
    half[:, 12:] = False
    assert SG.boundary_f(half, g, void, bound_th=1) == 1.0


def test_davis_statistics():
    s = SG.davis_statistics([1, 1, 1, 1, 0, 0, 0, 0])
    assert s == {"mean": 0.5, "recall": 0.5, "decay": 1.0}
    s = SG.davis_statistics([0.2, 0.6, np.nan, 0.8])
    assert s["mean"] == pytest.approx(1.6 / 3) and s["recall"] == 0.5
    long = np.r_[np.ones(150), np.zeros(150)]
    assert SG.davis_statistics(long)["decay"] == 1.0
    assert math.isnan(SG.davis_statistics([])["mean"])


def test_j_and_f_summary_and_frames():
    g = np.zeros((4, 20, 20), bool)
    g[:, 5:15, 5:15] = True
    p = g.copy()
    p[2:] = False
    s = SG.j_and_f(p, g)
    assert (s["J"], s["F"], s["J&F"], s["n_frames"]) == (50.0, 50.0, 50.0, 4)
    assert s["J_recall"] == 50.0 and s["J_decay"] == 100.0
    s = SG.j_and_f(p, g, frames=[0, 1])
    assert s["J&F"] == 100.0 and s["n_frames"] == 2 and "per_frame" not in s
    pf = SG.j_and_f(p, g, frames=[1, 2], per_frame=True)["per_frame"]
    assert pf == {"frame": [1, 2], "J": [100.0, 0.0], "F": [100.0, 0.0], "J&F": [100.0, 0.0]}
    assert SG.j_and_f(p, g, per_frame=True)["per_frame"]["frame"] == [0, 1, 2, 3]
    assert SG.j_and_f(p, g, frames=np.array([False, False, True, True]))["J"] == 0.0
    assert SG.j_and_f(g[0], g[0])["J&F"] == 100.0
    with pytest.raises(ValueError):
        SG.j_and_f(p, g, frames=[])


def test_j_and_f_objects():
    gt = np.zeros((3, 20, 20), np.int32)
    gt[:, 2:8, 2:8] = 1
    gt[:, 10:18, 10:18] = 2
    pred = np.where(gt == 1, 1, 0)
    s = SG.j_and_f_objects(pred, gt)
    assert s["n_objects"] == 2 and s["J"] == 50.0 and s["F"] == 50.0 and s["J&F"] == 50.0
    assert s["per_object"][1]["J&F"] == 100.0 and s["per_object"][2]["J&F"] == 0.0
    assert SG.j_and_f_objects(pred, gt, object_ids=[1])["J&F"] == 100.0
    with pytest.raises(ValueError):
        SG.j_and_f_objects(np.zeros_like(gt), np.zeros_like(gt))


def test_point_label_metrics():
    mask = np.zeros((4, 4), bool)
    mask[:, :2] = True
    uv = np.array([[0, 0], [1.4, 3], [2.6, 1], [0.2, 2], [3, 3], [5, 0], [np.nan, 0], [0, 0]], float)
    lab = np.array([True, True, True, False, False, True, True, True])
    valid = np.array([True] * 7 + [False])
    m = SG.point_label_metrics(mask, uv, lab, valid)
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["n_points"], m["n_unscored"]) == (2, 1, 1, 1, 5, 2)
    assert m["precision"] == pytest.approx(200 / 3) and m["recall"] == pytest.approx(200 / 3)
    assert m["iou"] == 50.0 and m["f1"] == pytest.approx(200 / 3) and m["accuracy"] == 60.0
    val, ins = SG.sample_mask(mask, uv)
    assert ins.tolist() == [True] * 5 + [False, False, True] and val.tolist()[:5] == [True, True, False, True, False]
    mv = np.stack([mask, np.zeros_like(mask)])
    uvv = np.stack([uv[:5], uv[:5]])
    lv = np.stack([lab[:5], np.zeros(5, bool)])
    m = SG.point_label_metrics(mv, uvv, lv, per_frame=True)
    assert m["per_frame"]["iou"] == [50.0, 100.0] and m["tn"] == 1 + 5
    e = SG.point_label_metrics(np.zeros((4, 4), bool), uv[:5], lab[:5])
    assert math.isnan(e["precision"]) and e["recall"] == 0.0 and e["iou"] == 0.0 and math.isnan(e["f1"])
    with pytest.raises(ValueError):
        SG.point_label_metrics(mv, uv, lab)


def test_point_label_metrics_without_evidence():
    mask = np.ones((4, 4), bool)
    for uv, lab, valid in ((np.zeros((0, 2)), np.zeros(0, bool), None),
                           (np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([True, True]), np.array([False, False])),
                           (np.array([[9.0, 9.0]]), np.array([True]), None)):
        m = SG.point_label_metrics(mask, uv, lab, valid)
        assert m["n_points"] == 0
        for k in ("precision", "recall", "iou", "f1", "accuracy"):
            assert math.isnan(m[k]), k
    m = SG.point_label_metrics(np.zeros((4, 4), bool), [[1.0, 1.0], [2.0, 2.0]], np.array([False, False]))
    assert m["iou"] == 100.0 and m["accuracy"] == 100.0 and math.isnan(m["precision"]) and m["n_points"] == 2
    m = SG.point_label_metrics(np.stack([mask, mask]), np.ones((2, 1, 2)), np.array([[True], [True]]),
                               np.array([[True], [False]]), per_frame=True)
    assert m["per_frame"]["iou"][0] == 100.0 and math.isnan(m["per_frame"]["iou"][1]) and m["n_points"] == 1


def test_point_label_metrics_label_values():
    uv = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mask = np.zeros((4, 4), bool)
    mask[1, 1] = True
    m = SG.point_label_metrics(mask, uv, np.array([1.0, np.nan, 0.0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["n_points"]) == (1, 0, 0, 1, 2)
    assert SG.point_label_metrics(mask, uv, np.array([1, 0, 0], np.uint8))["n_points"] == 3
    assert SG.point_label_metrics(mask, uv, np.array([True] * 3), valid=np.array([1, 1, 0]))["n_points"] == 2
    with pytest.raises(ValueError):
        SG.point_label_metrics(mask, uv, np.array([255, 0, 0]))
    with pytest.raises(ValueError):
        SG.point_label_metrics(mask, uv, np.array([True, False, True]), valid=np.ones(2, bool))
    with pytest.raises(TypeError):
        SG.point_label_metrics(mask, uv, np.array(["a", "b", "c"]))


def test_j_and_f_frames_without_evidence():
    g = np.zeros((4, 12, 12), bool)
    g[:, 3:9, 3:9] = True
    p = g.copy()
    p[3] = False
    void = np.zeros_like(g)
    void[1] = True
    J, F = SG.jaccard(p, g, void), SG.boundary_f(p, g, void)
    assert math.isnan(J[1]) and math.isnan(F[1]) and J[0] == F[0] == 1.0
    s = SG.j_and_f(p, g, void, per_frame=True)
    ref = SG.j_and_f(p[[0, 2, 3]], g[[0, 2, 3]])
    for k in ("J", "J_recall", "J_decay", "F", "F_recall", "F_decay", "J&F"):
        assert s[k] == pytest.approx(ref[k]), k
    assert (s["n_frames"], s["n_frames_skipped"]) == (3, 1) and math.isnan(s["per_frame"]["J"][1])
    assert s["per_frame"]["frame"] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        SG.j_and_f(p, g, np.ones_like(g))
    e = np.zeros((2, 0, 0), bool)
    assert np.isnan(SG.jaccard(e, e)).all() and np.isnan(SG.boundary_f(e, e)).all()
    assert SG.seg2bmap(np.zeros((0, 5), bool)).shape == (0, 5)


def test_j_and_f_frame_selection_is_checked():
    g = np.zeros((4, 10, 10), bool)
    g[:, 2:6, 2:6] = True
    for frames in ([-1], [4], [0, 9]):
        with pytest.raises(ValueError):
            SG.j_and_f(g, g, frames=frames)
    with pytest.raises(ValueError):
        SG.j_and_f(g, g, frames=np.array([True, False]))
    with pytest.raises(TypeError):
        SG.j_and_f(g, g, frames=np.array([0.0, 1.0]))
    assert SG.j_and_f(g, g, frames=range(1, 4))["n_frames"] == 3


def test_point_label_iou_estimates_mask_iou():
    rng = np.random.default_rng(22)
    yy, xx = np.mgrid[:60, :80]
    gt = ((yy - 30) / 20.0) ** 2 + ((xx - 40) / 25.0) ** 2 <= 1
    pred = ((yy - 33) / 18.0) ** 2 + ((xx - 44) / 26.0) ** 2 <= 1
    uv = np.stack(np.meshgrid(np.arange(80.0), np.arange(60.0)), -1).reshape(-1, 2)
    m = SG.point_label_metrics(pred, uv, gt.reshape(-1))
    assert m["iou"] == pytest.approx(100 * SG.jaccard(pred, gt))
    sub = rng.choice(len(uv), 1500, replace=False)
    assert abs(SG.point_label_metrics(pred, uv[sub], gt.reshape(-1)[sub])["iou"] - m["iou"]) < 5.0


def test_normalize_text_and_exact_match():
    assert QA.normalize_text("The Three red cups!") == "3 red cups"
    assert QA.normalize_text("It costs 1,000 dollars.") == "it costs 1000 dollars"
    assert QA.normalize_text("3.5 m.") == "3.5 m"
    assert QA.normalize_text("Left-hand") == "left hand"
    assert QA.normalize_text(None) == ""
    assert QA.normalize_text("The Cup", lowercase=False, remove_articles=False) == "The Cup"
    assert QA.exact_match("two", ["2", "a pair"]) and QA.exact_match("The left hand.", "left hand")
    assert not QA.exact_match("left", "left hand")
    assert not QA.exact_match("The cup", "cup", normalize=False) and QA.exact_match(" cup ", "cup", normalize=False)


@pytest.mark.parametrize("response,n,choices,expect", [
    ("B", 4, None, "B"),
    ("(c)", 4, None, "C"),
    ("b.", 4, None, "B"),
    ("B. a red cup", 4, None, "B"),
    ("(d) the blue one", 4, None, "D"),
    ("The answer is C.", 4, None, "C"),
    ("Answer: (d)", 4, None, "D"),
    ("I think the correct option is B because it moves.", 4, None, "B"),
    ("Option A is wrong; the answer is B", 4, None, "B"),
    ("I'd go with (C)", 4, None, "C"),
    ("**B**", 4, None, "B"),
    ("The answer is **C**.", 4, None, "C"),
    ("**Answer:** `D`", 4, None, "D"),
    ("B\nBecause the cup moves to the left.", 4, None, "B"),
    ("B\r\n\r\nThe robot picks it up.", 4, None, "B"),
    ("A person is walking", 4, None, None),
    ("A\nperson", 4, None, "A"),
    ("the answer is a cup", 4, None, None),
    ("E", 4, None, None),
    ("E\nbecause", 4, None, None),
    ("Both (A) and (B) look right", 4, None, None),
    ("a red cup", None, ["a blue cup", "a red cup"], "B"),
    ("I see a red cup on the table", None, ["blue cup", "red cup"], "B"),
    ("the red cup is on the left", None, ["cup", "red cup"], "B"),
    ("cup", None, ["cup", "red cup"], "A"),
    ("a green cup", None, ["blue cup", "red cup"], None),
    (None, 4, None, None),
    ("(A) or (B)?", 4, None, None),
    ("B or C", 4, None, None),
    ("The answer is B or C", 4, None, None),
    ("It is B or C", 4, None, None),
    ("A, B and C are all plausible", 4, None, None),
    ("B. (A) would also work", 4, None, None),
    ("(A) or (B)? The answer is B.", 4, None, "B"),
    ("The answer is A person walking", 4, None, None),
    ("The answer is: I don't know", None, None, None),
    ("The answer is A because the cup moves", 4, None, "A"),
    ("The answer is A, the red cup", 4, None, "A"),
    ("The answer is option A since it moves", 4, None, "A"),
    ("Option A is correct.", 4, None, "A"),
    ("B, because the robot moves left", 4, None, "B"),
    ("B - the cup moves", 4, None, "B"),
    ("C) The robot moves.", 4, None, "C"),
    ("answer: b", 4, None, "B"),
    ("answer: b\nbecause it moves", 4, None, "B"),
    ("My choice: b", 4, None, "B"),
    ("I think it's B.", 4, None, "B"),
    ("it is (C) because the cup falls", 4, None, "C"),
    ("['B']", 4, None, "B"),
    ("'c'", 4, None, "C"),
    ("Among the options, B is correct", None, None, None),
    ("B-52 bombers", 4, None, None),
    ("I answered B", 4, None, None),
])
def test_extract_choice(response, n, choices, expect):
    assert QA.extract_choice(response, n, choices) == expect


@pytest.mark.parametrize("text,expect", [
    ("twenty-one", "21"), ("Twenty one cups", "21 cups"), ("thirty", "30"), (".5", "0.5"), ("-.5", "-0.5"),
    ("U.S.", "us"), ("e.g. the cup", "eg cup"), ("-3 degrees", "-3 degrees"), ("−3", "-3"), ("2-3", "2 3"),
])
def test_normalize_text_numbers_and_abbreviations(text, expect):
    assert QA.normalize_text(text) == expect


def test_exact_match_number_formats():
    assert QA.exact_match(".5", "0.5") and QA.exact_match("U.S.", "US") and QA.exact_match("twenty-one", "21")
    assert not QA.exact_match("-3", "3") and not QA.exact_match("3.50", "3.5")
    assert QA.normalize_text("twenty-one", remove_punctuation=False) == "21"


def test_multiple_choice_rejects_a_flat_choices_list():
    with pytest.raises(ValueError):
        QA.score_multiple_choice(["red", "B", "C"], [0, 1, 2], choices=["red", "blue", "green"])
    with pytest.raises(TypeError):
        QA.score_multiple_choice(["red"], [0], choices="red")
    shared = ["red", "blue", "green"]
    s = QA.score_multiple_choice(["red", "B", "a green one"], [0, 1, 2], choices=[shared] * 3)
    assert s["accuracy"] == 100.0 and s["extracted"] == ["A", "B", "C"]


def test_retrieval_empty_inputs():
    for sim, gt in ((np.zeros((0, 0)), None), (np.zeros((0, 4)), np.zeros(0, np.int64))):
        m = QA.retrieval_metrics(sim, gt)
        assert m["n_queries"] == 0 and math.isnan(m["R@1"]) and math.isnan(m["median_rank"])
    r = QA.bidirectional_retrieval(np.zeros((0, 0)))
    assert r["text_to_video"]["n_queries"] == r["video_to_text"]["n_queries"] == 0
    with pytest.raises(TypeError):
        QA.retrieval_ranks(np.eye(3), np.array([0.0, 1.7, 2.0]))


def test_answer_letter_and_scoring():
    assert QA.answer_letter(1) == "B" and QA.answer_letter("(c)") == "C" and QA.answer_letter(np.int64(0)) == "A"
    with pytest.raises(ValueError):
        QA.answer_letter("blue")
    with pytest.raises(ValueError):
        QA.answer_letter(30)
    correct, ext = QA.multiple_choice_correct(["A", "B", "nonsense", "C"], ["A", 2, "C", "(c)"], n_choices=4)
    assert correct.tolist() == [True, False, False, True] and ext == ["A", "B", None, "C"]
    s = QA.score_multiple_choice(["A", "B", "nonsense", "C"], ["A", 2, "C", "(c)"], n_choices=4,
                                 categories=["x", "x", "x", "y"])
    assert s["accuracy"] == 50.0 and s["n"] == 4 and s["n_correct"] == 2 and s["n_unparsed"] == 1
    assert s["per_category"] == {"x": {"accuracy": pytest.approx(100 / 3), "n": 3}, "y": {"accuracy": 100.0, "n": 1}}
    assert s["macro_accuracy"] == pytest.approx((100 / 3 + 100) / 2)
    s = QA.score_multiple_choice(["red cup", "B"], [1, 0], choices=[["blue cup", "red cup"], ["x", "y"]])
    assert s["accuracy"] == 50.0
    a = QA.accuracy([True, False, True, True])
    assert a == {"accuracy": 75.0, "n": 4, "n_correct": 3}
    assert math.isnan(QA.accuracy([])["accuracy"])
    e = QA.score_exact_match(["Two", "left", "3.50"], [["2"], "right", "3.5"])
    assert e["accuracy"] == pytest.approx(100 / 3)
    with pytest.raises(ValueError):
        QA.accuracy([True], categories=["a", "b"])


def _clip4clip_reference(x):
    """CLIP4Clip `metrics.py::compute_metrics` (text i <-> video i; no ties)."""
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)[:, np.newaxis]
    ind = np.where(sx - d == 0)[1]
    return {"R@1": float(np.sum(ind == 0)) * 100 / len(ind), "R@5": float(np.sum(ind < 5)) * 100 / len(ind),
            "R@10": float(np.sum(ind < 10)) * 100 / len(ind), "median_rank": np.median(ind) + 1,
            "mean_rank": np.mean(ind) + 1}


def test_retrieval_matches_reference_code():
    rng = np.random.default_rng(23)
    sim = rng.normal(size=(200, 200)) + 3 * np.eye(200) * rng.uniform(0, 1, 200)[:, None]
    ref = _clip4clip_reference(sim)
    m = QA.retrieval_metrics(sim)
    for k, v in ref.items():
        assert m[k] == pytest.approx(v), k


def test_retrieval():
    sim = np.array([[0.9, 0.1, 0.2, 0.3],
                    [0.8, 0.5, 0.1, 0.2],
                    [0.9, 0.8, 0.5, 0.1],
                    [0.9, 0.8, 0.7, 0.1]])
    assert QA.retrieval_ranks(sim).tolist() == [1, 2, 3, 4]
    m = QA.retrieval_metrics(sim, ks=(1, 2, 3))
    assert m == {"R@1": 25.0, "R@2": 50.0, "R@3": 75.0, "median_rank": 2.5, "mean_rank": 2.5, "n_queries": 4}
    ones = np.ones((5, 5))
    assert QA.retrieval_ranks(ones).tolist() == [5] * 5
    assert QA.retrieval_ranks(ones, ties="optimistic").tolist() == [1] * 5
    assert QA.retrieval_metrics(np.eye(5))["R@1"] == 100.0
    nan = np.eye(3)
    nan[0, 0] = np.nan
    assert QA.retrieval_ranks(nan).tolist() == [3, 1, 1]
    multi = np.array([[0.9, 0.5, 0.95, 0.7], [0.1, 0.2, 0.3, 0.4]])
    gt = np.array([[False, True, False, True], [False, False, False, True]])
    assert QA.retrieval_ranks(multi, gt).tolist() == [3, 1]
    assert QA.retrieval_ranks(multi, np.array([3, 0])).tolist() == [3, 4]
    with pytest.raises(ValueError):
        QA.retrieval_ranks(multi)
    with pytest.raises(ValueError):
        QA.retrieval_ranks(multi, np.zeros((2, 4), bool))
    with pytest.raises(ValueError):
        QA.retrieval_ranks(sim, ties="random")


def test_bidirectional_retrieval():
    sim = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    r = QA.bidirectional_retrieval(sim, gt=np.array([0, 0, 1]), ks=(1,))
    assert r["text_to_video"]["R@1"] == pytest.approx(200 / 3) and r["text_to_video"]["n_queries"] == 3
    assert r["video_to_text"]["R@1"] == 100.0 and r["video_to_text"]["n_queries"] == 2
    r = QA.bidirectional_retrieval(np.array([[0.9, 0.1, 0.3], [0.4, 0.6, 0.2]]), gt=np.array([0, 1]), ks=(1,))
    assert r["video_to_text"]["n_queries"] == 2
