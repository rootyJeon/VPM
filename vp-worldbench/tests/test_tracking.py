"""Tests for vpm_worldbench.metrics.tracking (pure numpy; the MVTracker cross-check runs only when it is importable)."""
import os
import sys
import warnings

import numpy as np
import pytest

from vpm_worldbench.metrics import tracking as trk


def _two_tracks():
    """T=5, N=2, queries at frame 0, ladder (1 cm, 2 cm), survival 10 cm (answers worked out in the tests)."""
    T = 5
    gt = np.zeros((T, 2, 3))
    gt[:, 0, 0] = [0.0, 0.1, 0.2, 0.3, 0.4]
    gt[:, 1] = 1.0
    gt_vis = np.ones((T, 2), bool)
    gt_vis[[2, 4], 1] = False
    pred = gt.copy()
    pred[:, 0, 1] += [0.0, 0.005, 0.015, 0.03, 0.2]
    pred[:, 1, 1] += [0.0, 0.012, 5.0, 0.001, 0.5]
    pred_vis = np.ones((T, 2), bool)
    pred_vis[2, 0] = False
    return gt, gt_vis, pred, pred_vis, np.zeros(2, int)


def test_track3d_hand_computed():
    gt, gt_vis, pred, pred_vis, q = _two_tracks()
    r = trk.track3d_metrics(gt, gt_vis, pred, pred_vis, q, (0.01, 0.02), 0.10)
    a = pytest.approx
    assert r["any/n_tracks"] == 2
    assert r["any/delta_avg"] == a(56.25)
    assert r["any/delta@0.01"] == a(37.5)
    assert r["any/delta@0.02"] == a(75.0)
    assert r["any/jaccard@0.01"] == a(100 * (1 / 6 + 0.2) / 2)
    assert r["any/jaccard@0.02"] == a(100 * (1 / 6 + 0.5) / 2)
    assert r["any/aj"] == a(100 * (1 / 6 + 0.35) / 2)
    assert r["any/oa"] == a(62.5)
    assert r["any/survival"] == a(90.0)
    assert r["any/ade"] == a(0.263 / 6)
    assert r["any/mte"] == a(0.0135)
    assert (r["static/n_tracks"], r["dynamic/n_tracks"]) == (1, 1)
    assert r["static/delta_avg"] == a(75.0) and r["static/aj"] == a(35.0)
    assert r["static/oa"] == a(50.0) and r["static/survival"] == a(100.0)
    assert r["static/ade"] == a(0.0065)
    assert r["dynamic/delta_avg"] == a(37.5) and r["dynamic/aj"] == a(100 / 6)
    assert r["dynamic/oa"] == a(75.0) and r["dynamic/survival"] == a(80.0)
    assert r["ade_all"] == a(0.263 / 6) and r["mte_all"] == a(0.0135)
    assert r["fde"] == a((0.2 + 0.001) / 2)
    assert r["n_scored"] == 6


def test_track3d_query_frames_and_two_frame_rule():
    T = 6
    gt = np.zeros((T, 3, 3))
    gt_vis = np.zeros((T, 3), bool)
    gt_vis[[0, 1, 2, 4], 0] = True
    gt_vis[[0, 3], 1] = True
    gt_vis[[5], 2] = True
    pred = gt.copy()
    pred[:, :, 0] = 0.05
    q = np.array([2, 1, 0])
    r = trk.track3d_metrics(gt, gt_vis, pred, None, q, (0.01, 0.1), 1.0, static_m=None, dynamic_m=None)
    assert r["any/n_tracks"] == 1
    assert r["any/delta@0.01"] == 0.0 and r["any/delta@0.1"] == 100.0
    assert r["any/oa"] == pytest.approx(100 / 3)
    assert r["n_scored"] == 3 and r["ade_all"] == pytest.approx(0.05) and r["fde"] == pytest.approx(0.05)
    assert "static/n_tracks" not in r and "dynamic/delta_avg" not in r


def test_track3d_survival_counts_query_frame_and_ignores_occluded_failures():
    T = 5
    gt = np.zeros((T, 2, 3))
    gt_vis = np.ones((T, 2), bool)
    gt_vis[2, 1] = False
    pred = gt.copy()
    pred[0, 0, 0] = 1.0
    pred[2, 1, 0] = 1.0
    pred[3, 1, 0] = 1.0
    r = trk.track3d_metrics(gt, gt_vis, pred, None, [0, 0], (0.5,), 0.5, None, None)
    assert r["any/survival"] == pytest.approx(100 * (0 / 5 + 3 / 5) / 2)


def test_track3d_nan_gt_is_invisible_and_values_there_do_not_matter():
    gt, gt_vis, pred, pred_vis, q = _two_tracks()
    ref_vis = gt_vis.copy()
    ref_vis[3, 0] = False
    ref = trk.track3d_metrics(gt, ref_vis, pred, pred_vis, q, (0.01, 0.02), 0.10)
    gt2 = gt.copy()
    gt2[3, 0] = np.nan
    r = trk.track3d_metrics(gt2, gt_vis, pred, pred_vis, q, (0.01, 0.02), 0.10)
    gt3 = gt.copy()
    gt3[3, 0] = 123.0
    r3 = trk.track3d_metrics(gt3, ref_vis, pred, pred_vis, q, (0.01, 0.02), 0.10)
    for k in ref:
        np.testing.assert_equal(r[k], ref[k], err_msg=k)
        np.testing.assert_equal(r3[k], ref[k], err_msg=k)


def test_track3d_nonfinite_prediction_semantics():
    T = 4
    gt = np.zeros((T, 1, 3))
    pred = gt.copy()
    pred[2, 0] = np.nan
    r = trk.track3d_metrics(gt, np.ones((T, 1), bool), pred, None, [0], (0.1,), 0.05, None, None)
    assert r["any/delta_avg"] == pytest.approx(200 / 3)
    assert r["any/survival"] == 100.0
    assert np.isnan(r["any/ade"]) and np.isnan(r["ade_all"])


def test_track3d_pred_vis_forms_agree():
    gt, gt_vis, pred, pred_vis, q = _two_tracks()
    a = trk.track3d_metrics(gt, gt_vis, pred, None, q, (0.01, 0.02), 0.10)
    b = trk.track3d_metrics(gt, gt_vis, pred, np.ones_like(gt_vis), q, (0.01, 0.02), 0.10)
    assert a == b
    scores = np.where(pred_vis, 0.9, 0.1).astype(np.float32)
    c = trk.track3d_metrics(gt, gt_vis, pred, scores, q, (0.01, 0.02), 0.10)
    d = trk.track3d_metrics(gt, gt_vis, pred, pred_vis, q, (0.01, 0.02), 0.10)
    assert c == d


def test_track3d_empty_is_nan_without_warnings():
    T = 4
    gt = np.zeros((T, 2, 3))
    gt_vis = np.zeros((T, 2), bool)
    gt_vis[1, 0] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = trk.track3d_metrics(gt, gt_vis, gt, None, [0, 0], (0.01,), 0.1)
    for s in ("any", "static", "dynamic"):
        assert r[f"{s}/n_tracks"] == 0
        assert np.isnan(r[f"{s}/delta_avg"]) and np.isnan(r[f"{s}/aj"]) and np.isnan(r[f"{s}/survival"])
    assert r["n_scored"] == 1 and r["ade_all"] == 0.0


def test_track3d_input_validation():
    gt = np.zeros((4, 2, 3))
    vis = np.ones((4, 2), bool)
    with pytest.raises(ValueError):
        trk.track3d_metrics(gt, vis, gt[:, :1], None, [0, 0], (0.1,), 0.5)
    with pytest.raises(ValueError):
        trk.track3d_metrics(gt, np.ones((4, 2, 3), bool), gt, None, [0, 0], (0.1,), 0.5)
    with pytest.raises(ValueError):
        trk.track3d_metrics(gt, vis, gt, None, [0, 4], (0.1,), 0.5)
    with pytest.raises(ValueError):
        trk.track3d_metrics(gt, vis, gt, None, [0.5, 0], (0.1,), 0.5)
    with pytest.raises(ValueError):
        trk.track3d_metrics(gt, vis, gt, None, [0, 0], (), 0.5)


def test_threshold_key():
    assert [trk.threshold_key(t) for t in (0.01, 0.10, 0.2, 1.6, 16, 2.0)] == ["0.01", "0.1", "0.2", "1.6", "16", "2"]


def test_frozen_prediction():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(6, 4, 3)).astype(np.float32)
    q = np.array([0, 2, 5, 1])
    f = trk.frozen_prediction(gt, q)
    assert f.shape == gt.shape and f.dtype == gt.dtype
    for n in range(4):
        assert np.array_equal(f[:, n], np.repeat(gt[q[n], n][None], 6, 0))
    assert np.array_equal(trk.frozen_prediction(gt, 0), np.repeat(gt[:1], 6, 0))
    r = trk.track3d_metrics(gt, np.ones((6, 4), bool), f, None, q, (1e-6,), 1e-6, None, None)
    assert r["any/n_tracks"] == 3


def test_error_helpers():
    gt = np.zeros((3, 2, 3))
    gt[1, 1] = np.nan
    pred = np.zeros((3, 2, 3))
    pred[:, 0, 0] = [0.0, 1.0, 2.0]
    pred[:, 1, 0] = [3.0, 9.0, 4.0]
    e = trk.point_errors(gt, pred)
    assert e.shape == (3, 2) and np.isnan(e[1, 1]) and e[2, 1] == 4.0
    assert trk.ade(gt, pred) == pytest.approx((0 + 1 + 2 + 3 + 4) / 5)
    assert trk.mte(gt, pred) == pytest.approx(2.0)
    assert trk.fde(gt, pred) == pytest.approx(3.0)
    vis = np.array([[1, 1], [1, 1], [1, 0]], bool)
    m = trk.scored_mask(vis, [0, 0], gt)
    assert m.tolist() == [[False, False], [True, False], [True, False]]
    assert trk.ade(gt, pred, m) == pytest.approx(1.5)
    assert trk.fde(gt, pred, vis) == pytest.approx((2.0 + 3.0) / 2)
    assert np.isnan(trk.ade(gt, pred, np.zeros((3, 2), bool)))


def _tapvid_case():
    """Image 512x512 (factor 0.5 to the 256 grid). Track 0 queried at t=0, track 1 at t=1 (answers in the tests)."""
    gt = np.zeros((4, 2, 2))
    gt[:, 0] = [100, 100]
    gt[:, 1] = [200, 300]
    gt_vis = np.array([[1, 1], [1, 1], [1, 1], [1, 0]], bool)
    pred = gt.copy()
    pred[1, 0] = [101, 100]
    pred[2, 0] = [104, 100]
    pred[3, 0] = [100, 140]
    pred[0, 1] = [0, 0]
    pred[2, 1] = [200, 301]
    pred[3, 1] = [250, 300]
    pred_vis = np.array([[1, 1], [1, 1], [1, 1], [0, 1]], bool)
    return gt, gt_vis, pred, pred_vis, np.array([0, 1])


def test_track2d_first_hand_computed():
    gt, gt_vis, pred, pred_vis, q = _tapvid_case()
    r = trk.track2d_metrics(gt, gt_vis, pred, pred_vis, q, (512, 512))
    a = pytest.approx
    assert r["occlusion_accuracy"] == a(60.0)
    assert [r[f"pts_within_{x}"] for x in (1, 2, 4, 8, 16)] == [a(50), a(50), a(75), a(75), a(75)]
    assert [r[f"jaccard_{x}"] for x in (1, 2, 4, 8, 16)] == [a(100 / 3), a(100 / 3), a(60), a(60), a(60)]
    assert r["average_pts_within_thresh"] == a(65.0)
    assert r["average_jaccard"] == a((200 / 3 + 180) / 5)
    assert r["n_visible"] == 4 and r["n_tracks"] == 2


def test_track2d_strided_hand_computed():
    gt, gt_vis, pred, pred_vis, q = _tapvid_case()
    r = trk.track2d_metrics(gt, gt_vis, pred, pred_vis, q, (512, 512), query_mode="strided")
    a = pytest.approx
    assert r["occlusion_accuracy"] == a(400 / 6)
    assert [r[f"pts_within_{x}"] for x in (1, 2, 4, 8, 16)] == [a(40), a(40), a(60), a(60), a(60)]
    assert [r[f"jaccard_{x}"] for x in (1, 2, 4, 8, 16)] == [a(25), a(25), a(300 / 7), a(300 / 7), a(300 / 7)]
    assert r["average_pts_within_thresh"] == a(52.0)
    assert r["average_jaccard"] == a((50 + 900 / 7) / 5)


def test_track2d_native_pixels_and_aspect_scaling():
    gt, gt_vis, pred, pred_vis, q = _tapvid_case()
    r = trk.track2d_metrics(gt, gt_vis, pred, pred_vis, q, (512, 512), thresholds_px=(1, 2), normalize_to=None)
    assert r["pts_within_1"] == 0.0 and r["pts_within_2"] == pytest.approx(50.0)
    g = np.zeros((2, 1, 2))
    p = g.copy()
    p[1, 0] = [3.9, 0.0]
    r = trk.track2d_metrics(g, np.ones((2, 1), bool), p, None, [0], (1024, 256), thresholds_px=(1,))
    assert r["pts_within_1"] == 100.0
    p[1, 0] = [0.0, 1.0]
    r = trk.track2d_metrics(g, np.ones((2, 1), bool), p, None, [0], (1024, 256), thresholds_px=(1,))
    assert r["pts_within_1"] == 0.0


def test_track2d_no_visible_points_is_nan():
    g = np.zeros((3, 1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = trk.track2d_metrics(g, np.zeros((3, 1), bool), g, np.zeros((3, 1), bool), [0], (256, 256))
    assert np.isnan(r["average_pts_within_thresh"]) and np.isnan(r["average_jaccard"])
    assert r["occlusion_accuracy"] == 100.0


def _tapnet_compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode,
                                   get_trackwise_metrics=False):
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)
    metrics = {}
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0
    occ_acc = np.sum(np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=summing_axis) / np.sum(
        evaluation_points, axis=summing_axis)
    metrics['occlusion_accuracy'] = occ_acc
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible_points = np.sum(visible & evaluation_points, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)
        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    return metrics


@pytest.mark.parametrize("mode", ["first", "strided"])
@pytest.mark.parametrize("seed", range(6))
def test_track2d_matches_tapnet_reference(mode, seed):
    rng = np.random.default_rng(seed)
    T, N = int(rng.integers(3, 40)), int(rng.integers(1, 60))
    W, H = int(rng.integers(64, 1300)), int(rng.integers(64, 1300))
    gt = rng.uniform(0, 1, (T, N, 2)) * [W, H]
    pred = gt + rng.normal(0, rng.uniform(0.5, 40), (T, N, 2))
    gt_vis = rng.random((T, N)) < 0.8
    pred_vis = rng.random((T, N)) < 0.8
    q = rng.integers(0, T, N)
    r = trk.track2d_metrics(gt, gt_vis, pred, pred_vis, q, (W, H), query_mode=mode)
    s = np.array([256 / W, 256 / H])
    qp = np.stack([q, gt[q, np.arange(N), 1] * s[1], gt[q, np.arange(N), 0] * s[0]], -1)[None]
    ref = _tapnet_compute_tapvid_metrics(qp, ~gt_vis.T[None], (gt * s).transpose(1, 0, 2)[None], ~pred_vis.T[None],
                                         (pred * s).transpose(1, 0, 2)[None], mode)
    for k, v in ref.items():
        if np.isnan(v[0]):
            assert np.isnan(r[k])
        else:
            assert r[k] == pytest.approx(100 * v[0], abs=1e-9), k


def _mvtracker_metrics():
    repo = os.environ.get("VPMWB_MVTRACKER_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    return pytest.importorskip("mvtracker.evaluation.metrics")


@pytest.mark.parametrize("seed", range(4))
def test_track3d_matches_mvtracker(seed):
    """Same numbers as mvtracker.evaluation.metrics.evaluate_predictions (unrounded; MVTracker accumulates in
    float32, so the tolerance is 1e-4 percentage points). Set VPMWB_MVTRACKER_REPO to a clone of ethz-vlg/mvtracker."""
    mvm = _mvtracker_metrics()
    import pandas as pd
    rng = np.random.default_rng(100 + seed)
    T, N = int(rng.integers(5, 70)), int(rng.integers(5, 300))
    sc = rng.choice([0.0, 1e-4, 0.01, 0.03], N)
    gt = rng.normal(0, 1, (1, N, 3)) + np.cumsum(rng.normal(0, 1, (T, N, 3)) * sc[None, :, None], 0)
    gt_vis = rng.random((T, N)) < rng.uniform(0.3, 1.0)
    gt[rng.random((T, N)) < 0.03] = np.nan
    pred = np.nan_to_num(gt) + rng.normal(0, 1, (T, N, 3)) * rng.uniform(0.002, 0.5, N)[None, :, None]
    pred_vis = rng.random((T, N)) < 0.7
    q = rng.integers(0, T // 2, N) if seed % 2 else np.zeros(N, int)
    ladder, surv = [(0.01, 0.02, 0.05, 0.10, 0.20), (0.05, 0.10, 0.20, 0.40)][seed % 2], [0.1, 1.0][seed % 2]
    mine = trk.track3d_metrics(gt, gt_vis, pred, pred_vis, q, ladder, surv)

    ok = np.isfinite(gt).all(-1)
    g = np.nan_to_num(gt)
    qp = np.concatenate([q[:, None].astype(float), g[q, np.arange(N)]], 1)
    orig = pd.DataFrame.round
    pd.DataFrame.round = lambda self, *a, **k: self
    try:
        df, _ = mvm.evaluate_predictions(g, gt_vis & ok, pred, ~pred_vis, query_points=qp,
                                         distance_thresholds=list(ladder), survival_distance_threshold=surv,
                                         static_threshold=0.01, dynamic_threshold=0.10, very_dynamic_threshold=None)
    finally:
        pd.DataFrame.round = orig
    for split in ("any", "static", "dynamic"):
        col = f"all_{split}"
        if col not in df.columns:
            assert mine[f"{split}/n_tracks"] == 0
            continue
        pairs = {"average_pts_within_thresh": "delta_avg", "average_jaccard": "aj", "occlusion_accuracy": "oa",
                 "survival": "survival"}
        for t in ladder:
            pairs[f"pts_within_{t:.2f}"] = f"delta@{trk.threshold_key(t)}"
            pairs[f"jaccard_{t:.2f}"] = f"jaccard@{trk.threshold_key(t)}"
        for ref_key, key in pairs.items():
            assert mine[f"{split}/{key}"] == pytest.approx(float(df[col][ref_key]), abs=1e-4), (split, key)
        assert mine[f"{split}/n_tracks"] == round(float(df[col]["n"]) * N / 100)


def test_moving_delta_frozen_zero_and_oracle_full():
    from vpm_worldbench.metrics.tracking import frozen_prediction, moving_delta
    T, N = 10, 3
    gt = np.zeros((T, N, 3))
    gt[:, 0, 0] = np.linspace(0.0, 0.5, T)
    gt[:, 1, 0] = np.linspace(0.0, 0.05, T)
    vis = np.ones((T, N), bool)
    q = np.zeros(N, int)
    ladder = (0.01, 0.02, 0.04, 0.08, 0.16)
    fr = moving_delta(gt, vis, frozen_prediction(gt, q), q, ladder)
    assert fr["moving/delta_avg"] == 0.0 and fr["moving/n_tracks"] == 1
    assert fr["moving/n_entries"] == int((np.linspace(0, 0.5, T) >= 0.16).sum())
    assert moving_delta(gt, vis, gt, q, ladder)["moving/delta_avg"] == 100.0
    bad = gt.copy()
    bad[5:, 0] = np.nan
    assert moving_delta(gt, vis, bad, q, ladder)["moving/delta_avg"] < 100.0
    none = moving_delta(gt * 0, vis, gt * 0, q, ladder)
    assert none["moving/n_entries"] == 0 and np.isnan(none["moving/delta_avg"])
