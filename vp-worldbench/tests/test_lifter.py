"""Tests for vpm_worldbench.lifters (base + mvtracker).

CPU tests (numpy only) check the anchoring / normalisation geometry and the lifter pipeline with fake models; the
MVTracker wrapper tests stand in for the hub model and MoGe-2 (torch needed, no weights).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vpm_worldbench.geometry import camera_center, project
from vpm_worldbench.lifters.base import (Anchor, DepthAnchoredLifter, Lifter, anchor_depth, apply_normalization,
                                         fit_scale_shift, fov_x_deg, get_lifter, invert_normalization, map_K,
                                         median_ratio_scale, register_lifter, resample_inputs, scene_normalization,
                                         source_world_transform)
from vpm_worldbench.lifters.mvtracker import MVTrackerLifter


def look_at(center, target, up=(0.0, 0.0, 1.0)):
    """OpenCV world->camera 4x4 for a camera at `center` looking at `target` (z-up world)."""
    c, t, u = (np.asarray(x, np.float64) for x in (center, target, up))
    z = t - c
    z /= np.linalg.norm(z)
    x = np.cross(z, u)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ c
    return E


def make_scene(V=2, n=3, H=40, W=48, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[50.0, 0, W / 2 - 0.5], [0, 50.0, H / 2 - 0.5], [0, 0, 1]])
    centers = [(4.0, 0.0, 1.5), (0.0, 4.0, 1.8), (-4.0, 0.5, 1.2)][:V]
    w2c = np.stack([np.repeat(look_at(c, (0, 0, 0.5))[None], n, 0) for c in centers])
    Ks = np.broadcast_to(K, (V, n, 3, 3)).copy()
    depth = rng.uniform(2.0, 6.0, (V, n, H, W)).astype(np.float32)
    frames = np.zeros((V, n, H, W, 3), np.uint8)
    for v in range(V):
        frames[v] = v
    return frames, Ks, w2c, depth


def legacy_scene_norm(depth0, K0, w2c0, radius=6.3):
    """Oracle: the pre-release scene normalisation, verbatim apart from float64 inputs."""
    pts = []
    for v in range(depth0.shape[0]):
        d = depth0[v]
        m = d > 0
        if m.sum() < 100:
            continue
        yy, xx = np.nonzero(m)
        uv1 = np.stack([xx, yy, np.ones_like(xx)], 0).astype(np.float64)
        cam = (np.linalg.inv(K0[v]) @ uv1) * d[m][None]
        E = np.eye(4)
        E[:3] = w2c0[v][:3]
        pts.append((np.linalg.inv(E) @ np.concatenate([cam, np.ones((1, cam.shape[1]))]))[:3].T)
    pts = np.concatenate(pts)
    centroid = pts.mean(0)
    floor_z = np.quantile(pts[:, 2] - centroid[2], 0.12)
    cam_c = np.stack([np.linalg.inv(np.vstack([w2c0[v][:3], [0, 0, 0, 1]]))[:3, 3] for v in range(len(w2c0))])
    cc = cam_c - centroid
    cc[:, 2] -= floor_z
    scale = float(radius / max(np.median(np.linalg.norm(cc, axis=1)), 1e-3))
    tr = -scale * centroid
    tr[2] -= scale * floor_z
    return scale, tr


class FakeLifter(DepthAnchoredLifter):
    """Depth = true depth / depth_div; tracker = frozen normalised queries (records its inputs)."""

    name = "fake"

    def __init__(self, true_depth, depth_div=2.0, **kw):
        super().__init__(**kw)
        self.true_depth = true_depth
        self.depth_div = depth_div
        self.depth_calls = []
        self.tracker_inputs = None

    def estimate_depth(self, frames, fov_x):
        v = int(frames[0, 0, 0, 0])
        self.depth_calls.append((v, len(frames), np.asarray(fov_x).copy()))
        d = self.true_depth[v, :len(frames)]
        if d.shape[1:] != frames.shape[1:3]:
            import cv2
            d = np.stack([cv2.resize(x, frames.shape[2:0:-1], interpolation=cv2.INTER_NEAREST) for x in d])
        return (d / self.depth_div).astype(np.float32)

    def run_tracker(self, frames, depths, K, w2c, queries):
        self.tracker_inputs = dict(frames=frames, depths=depths, K=K, w2c=w2c, queries=queries)
        n = frames.shape[1]
        return np.repeat(queries[None, :, 1:], n, 0).astype(np.float32), np.ones((n, len(queries)), bool)


def test_anchor_validation():
    with pytest.raises(ValueError):
        Anchor("bogus")
    with pytest.raises(ValueError):
        Anchor("gt_depth")
    with pytest.raises(ValueError):
        Anchor("gt_points", points=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        Anchor("gt_points", points=np.zeros((3, 3)), points_vis=np.ones((2, 4), bool))
    with pytest.raises(ValueError):
        Anchor("gt_depth", depth=np.ones((4, 4)))
    with pytest.raises(ValueError):
        Anchor("none", far_clip_m=0)
    a = Anchor("none", far_clip_m=20)
    assert a.far_clip_m == 20.0 and a.depth is None


def test_fit_scale_shift_recovers_affine_map():
    rng = np.random.default_rng(1)
    pred = rng.uniform(1.0, 10.0, (50, 50)).astype(np.float32)
    gt = 2.5 * pred.astype(np.float64) + 0.3
    a, b, n = fit_scale_shift(pred, gt)
    assert a == pytest.approx(2.5, rel=1e-6) and b == pytest.approx(0.3, abs=1e-5)
    assert 0.9 * 2500 < n < 2500
    pred2, gt2 = pred.copy(), gt.copy()
    pred2[:5] = np.nan
    gt2[5:10] = 500.0
    gt2[10:12] = 0.05
    a2, b2, n2 = fit_scale_shift(pred2, gt2, far_clip_m=80.0)
    assert a2 == pytest.approx(2.5, rel=1e-6) and b2 == pytest.approx(0.3, abs=1e-5) and n2 < n


def test_fit_scale_shift_degenerate_cases():
    pred = np.full((20, 20), 3.0, np.float32)
    assert fit_scale_shift(pred, pred * 2) == (1.0, 0.0, 400)
    rng = np.random.default_rng(2)
    pred = rng.uniform(1.0, 10.0, (40, 40))
    a, b, _ = fit_scale_shift(pred, 50.0 - 2.0 * pred)
    keep = (pred > np.quantile(pred, 0.02)) & (pred < np.quantile(pred, 0.98))
    assert b == 0.0 and a == pytest.approx(np.median((50.0 - 2.0 * pred)[keep]) / np.median(pred[keep]))
    a, b, _ = fit_scale_shift(np.full((40, 40), 2.0), np.full((40, 40), 5.0))
    assert a == pytest.approx(2.5) and b == 0.0


def test_median_ratio_scale():
    frames, K, w2c, depth = make_scene(V=1, n=1)
    d0 = depth[0, 0].astype(np.float64)
    H, W = d0.shape
    rng = np.random.default_rng(3)
    px = np.stack([rng.integers(0, W, 30), rng.integers(0, H, 30)], 1)
    z = d0[px[:, 1], px[:, 0]]
    cam = np.stack([(px[:, 0] - K[0, 0, 0, 2]) / K[0, 0, 0, 0] * z,
                    (px[:, 1] - K[0, 0, 1, 2]) / K[0, 0, 1, 1] * z, z], 1)
    c2w = np.linalg.inv(w2c[0, 0])
    X = cam @ c2w[:3, :3].T + c2w[:3, 3]
    vis = np.ones(len(X), bool)
    a, n = median_ratio_scale(d0 / 3.0, X, vis, K[0, 0], w2c[0, 0])
    assert n == 30 and a == pytest.approx(3.0, rel=1e-9)
    vis[4:] = False
    assert median_ratio_scale(d0 / 3.0, X, vis, K[0, 0], w2c[0, 0]) == (1.0, 4)
    X2 = X.copy()
    X2[0] = np.nan
    a, n = median_ratio_scale(d0 / 3.0, X2, np.ones(len(X), bool), K[0, 0], w2c[0, 0])
    assert n == 29 and a == pytest.approx(3.0)


def test_anchor_depth_clamps_and_reports():
    _, K, w2c, depth = make_scene(V=1, n=2)
    est = depth[0] / 2.0
    est[1, :3] = np.nan
    anchor = Anchor("gt_depth", depth=depth[:, 0], far_clip_m=5.0)
    out, info = anchor_depth(est, 0, K[0, 0], w2c[0, 0], anchor)
    assert info["mode"] == "gt_depth" and info["scale"] == pytest.approx(2.0, rel=1e-5)
    assert np.all(out[1, :3] == 0)
    assert out.max() <= 5.0 and np.all(out[depth[0] > 5.001] == 0)
    keep = (depth[0] <= 4.999) & np.isfinite(est)
    np.testing.assert_allclose(out[keep], depth[0][keep], rtol=1e-4)
    out2, info2 = anchor_depth(est, 0, K[0, 0], w2c[0, 0], Anchor("none", far_clip_m=80.0))
    assert info2["scale"] == 1.0 and np.allclose(out2[0], est[0])


def test_scene_normalization_matches_prerelease_formula():
    _, K, w2c, depth = make_scene(V=3, n=1)
    s, T, info = scene_normalization(depth[:, 0], K[:, 0], w2c[:, 0])
    s_ref, T_ref = legacy_scene_norm(depth[:, 0], K[:, 0], w2c[:, 0])
    assert info["source"] == "depth" and info["n_points"] == depth[:, 0].size
    assert s == pytest.approx(s_ref, rel=1e-12)
    np.testing.assert_allclose(T, T_ref, rtol=1e-10, atol=1e-12)
    d = depth[:, 0].copy()
    d[2] = 0
    d[2, :5, :5] = 3.0
    s2, T2, _ = scene_normalization(d, K[:, 0], w2c[:, 0, :3])
    s2_ref, T2_ref = legacy_scene_norm(d, K[:, 0], w2c[:, 0])
    assert s2 == pytest.approx(s2_ref, rel=1e-12)
    np.testing.assert_allclose(T2, T2_ref, rtol=1e-10, atol=1e-12)
    d3 = depth[:, 0].copy()
    d3[0, :10] = 500.0
    s3, _, i3 = scene_normalization(d3, K[:, 0], w2c[:, 0], far_clip_m=80.0)
    assert i3["n_points"] == depth[:, 0].size - 10 * depth.shape[-1]
    pts = np.array([[0.0, 0, 0], [1, 1, 1], [np.nan, 0, 0]])
    _, _, i4 = scene_normalization(np.zeros_like(d), K[:, 0], w2c[:, 0], fallback_points=pts)
    assert i4 == {"source": "points", "n_points": 2}
    assert scene_normalization(np.zeros_like(d), K[:, 0], w2c[:, 0])[2]["source"] == "identity"


def test_normalization_preserves_pixels_and_scales_depth():
    _, K, w2c, depth = make_scene(V=2, n=2)
    s, T, _ = scene_normalization(depth[:, 0], K[:, 0], w2c[:, 0])
    X = np.random.default_rng(4).uniform(-1, 1, (25, 3))
    dn, wn, Xn = apply_normalization(s, T, depth, w2c, X)
    assert dn.dtype == np.float32 and np.allclose(dn, depth * s, rtol=1e-6)
    for v in range(2):
        for t in range(2):
            uv, z = project(X, K[v, t], w2c[v, t])
            uvn, zn = project(Xn, K[v, t], wn[v, t])
            np.testing.assert_allclose(uvn, uv, atol=1e-9)
            np.testing.assert_allclose(zn, s * z, rtol=1e-12)
    np.testing.assert_allclose(invert_normalization(Xn, s, T), X, atol=1e-12)
    np.testing.assert_allclose(camera_center(wn[:, 0]), s * camera_center(w2c[:, 0]) + T, atol=1e-9)


def test_map_K_and_resample_inputs():
    K = np.array([[100.0, 0, 239.5], [0, 133.0, 239.5], [0, 0, 1]])
    Kp = map_K(K, (480, 480), (640, 480), plain=True)
    np.testing.assert_allclose(Kp, [[100 * 4 / 3, 0, 239.5 * 4 / 3], [0, 133.0, 239.5], [0, 0, 1]])
    Kc = map_K(K, (480, 480), (640, 480), plain=False)
    assert Kc[0, 2] == pytest.approx(4 / 3 * 240 - 0.5) and Kc[0, 0] == pytest.approx(100 * 4 / 3)
    frames = np.random.default_rng(5).integers(0, 255, (2, 3, 48, 48, 3), dtype=np.uint8)
    d = np.arange(2 * 48 * 48, dtype=np.float32).reshape(2, 48, 48)
    fr, K2, (d2, none, same) = resample_inputs(frames, np.broadcast_to(K, (2, 3, 3, 3)), (64, 48), True,
                                               (d, None, np.ones((2, 48, 64))))
    assert fr.shape == (2, 3, 48, 64, 3) and K2.shape == (2, 3, 3, 3) and none is None and same.shape == (2, 48, 64)
    assert d2.shape == (2, 48, 64) and set(np.unique(d2)) <= set(np.unique(d))
    import cv2
    np.testing.assert_array_equal(fr[1, 2], cv2.resize(frames[1, 2], (64, 48), interpolation=cv2.INTER_CUBIC))
    with pytest.raises(ValueError):
        resample_inputs(frames, K2, (64, 48), True, (np.ones((2, 30, 30)),))
    assert fov_x_deg(np.array([[50.0, 0, 0], [0, 50, 0], [0, 0, 1]]), 100) == pytest.approx(90.0)


@pytest.mark.parametrize("mode", ["gt_depth", "gt_points", "none"])
def test_pipeline_round_trip(mode):
    frames, K, w2c, depth = make_scene(V=3, n=4)
    rng = np.random.default_rng(6)
    q_xyz = rng.uniform(-1, 1, (17, 3))
    queries = np.concatenate([rng.integers(0, 4, (17, 1)), q_xyz], 1)
    div = {"gt_depth": 2.0, "gt_points": 3.0, "none": 1.0}[mode]
    lifter = FakeLifter(depth, depth_div=div)
    if mode == "gt_depth":
        anchor = Anchor(mode, depth=depth[:, 0], far_clip_m=80.0)
    elif mode == "gt_points":
        Xs, vis = [], []
        for v in range(3):
            px = np.stack([rng.integers(0, 48, 20), rng.integers(0, 40, 20)], 1)
            z = depth[v, 0][px[:, 1], px[:, 0]].astype(np.float64)
            cam = np.stack([(px[:, 0] - K[v, 0, 0, 2]) / K[v, 0, 0, 0] * z,
                            (px[:, 1] - K[v, 0, 1, 2]) / K[v, 0, 1, 1] * z, z], 1)
            c2w = np.linalg.inv(w2c[v, 0])
            Xs.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
            vv = np.zeros((3, 20), bool)
            vv[v] = True
            vis.append(vv)
        anchor = Anchor(mode, points=np.concatenate(Xs), points_vis=np.concatenate(vis, 1), far_clip_m=80.0)
    else:
        anchor = Anchor(mode, far_clip_m=80.0)
    xyz, vis, info = lifter.track(frames, K, w2c, queries, anchor, return_info=True)
    assert xyz.shape == (4, 17, 3) and xyz.dtype == np.float32 and vis.shape == (4, 17) and vis.dtype == bool
    np.testing.assert_allclose(xyz, np.broadcast_to(q_xyz, (4, 17, 3)), atol=1e-5)
    assert [c[1] for c in lifter.depth_calls] == [4, 4, 4]
    np.testing.assert_allclose(lifter.depth_calls[0][2], fov_x_deg(K[0], 48))
    for a in info["depth_align"]:
        assert a["mode"] == mode and a["scale"] == pytest.approx(div, rel=1e-5)
        assert a["shift"] == pytest.approx(0, abs=1e-4)
    s = info["scene_norm"]["scale"]
    ti = lifter.tracker_inputs
    np.testing.assert_allclose(ti["depths"], depth * s, rtol=2e-4)
    np.testing.assert_array_equal(ti["queries"][:, 0], queries[:, 0])
    np.testing.assert_allclose(ti["queries"][:, 1:], s * q_xyz + np.array(info["scene_norm"]["translation"]), atol=1e-5)
    assert ti["queries"].dtype == np.float32 and ti["depths"].dtype == np.float32
    assert ti["K"].dtype == np.float32 and ti["w2c"].dtype == np.float32 and ti["w2c"].shape == (3, 4, 3, 4)
    assert info["scene_norm"]["from"] == ("depth" if mode == "gt_depth" else "own")
    assert lifter.last_info is info and json.dumps(info)


def test_prepare_anchor_and_norm_depth_sources():
    frames, K, w2c, depth = make_scene(V=2, n=3)
    lifter = FakeLifter(depth, depth_div=2.0)
    a = lifter.prepare_anchor(Anchor("gt_depth", depth=depth[:, 0]), frames, K, w2c)
    assert a.norm_depth is a.depth and lifter.depth_calls == []
    a = lifter.prepare_anchor(Anchor("none", far_clip_m=80.0), frames, K, w2c)
    assert [c[1] for c in lifter.depth_calls] == [1, 1]
    np.testing.assert_allclose(a.norm_depth, depth[:, 0] / 2.0, rtol=1e-6)
    assert lifter.prepare_anchor(a, frames, K, w2c) is a
    _, _, i1 = lifter.track(frames, K, w2c, np.zeros((1, 4)), a, return_info=True)
    lifter.depth_div = 4.0
    _, _, i2 = lifter.track(frames, K, w2c, np.zeros((1, 4)), a, return_info=True)
    assert i1["scene_norm"] == i2["scene_norm"] and i1["scene_norm"]["from"] == "norm_depth"
    xyz, vis = lifter.track(frames, K[:, 0], w2c[:, 0], np.zeros((1, 4)), a)
    assert xyz.shape == (3, 1, 3)


def test_track_size_resamples_frames_K_and_anchor():
    frames, K, w2c, depth = make_scene(V=2, n=2, H=40, W=40)
    K[..., 0, :2] *= 0.75
    lifter = FakeLifter(depth, depth_div=2.0, plain_K_scaling=True)
    anchor = Anchor("gt_depth", depth=depth[:, 0], far_clip_m=80.0)
    xyz, vis, info = lifter.track(frames, K, w2c, np.array([[0, 0.1, 0.2, 0.3]]), anchor, track_size=(53, 40),
                                  return_info=True)
    ti = lifter.tracker_inputs
    assert ti["frames"].shape == (2, 2, 40, 53, 3) and ti["depths"].shape == (2, 2, 40, 53)
    assert info["grid"] == [53, 40] and info["resampled_from"] == [40, 40]
    np.testing.assert_allclose(ti["K"][..., 0, 0], K[..., 0, 0] * 53 / 40, rtol=1e-6)
    np.testing.assert_allclose(xyz[:, 0], [[0.1, 0.2, 0.3]] * 2, atol=1e-6)
    with pytest.warns(UserWarning, match="non-square"):
        lifter.track(frames, K, w2c, np.array([[0, 0.1, 0.2, 0.3]]), anchor)


def test_fov_override_reaches_the_estimator():
    frames, K, w2c, depth = make_scene(V=2, n=3)
    lifter = FakeLifter(depth)
    lifter.track(frames, K, w2c, np.zeros((1, 4)), Anchor("none"), fov_x=[70.0, 80.0])
    assert [list(c[2]) for c in lifter.depth_calls] == [[70.0] * 3, [80.0] * 3]
    lifter.depth_calls.clear()
    lifter.track(frames, K, w2c, np.zeros((1, 4)), Anchor("none"), fov_x=65.0)
    assert len(lifter.depth_calls) == 2 and all(np.all(c[2] == 65.0) for c in lifter.depth_calls)
    lifter.depth_calls.clear()
    lifter.prepare_anchor(Anchor("none"), frames, K, w2c, fov_x=[[70.0] * 3, [80.0] * 3])
    assert [list(c[2]) for c in lifter.depth_calls] == [[70.0], [80.0]]


def test_world_transform_round_trip():
    frames, K, w2c, depth = make_scene(V=2, n=3)
    rng = np.random.default_rng(7)
    q = np.concatenate([np.zeros((9, 1)), rng.uniform(-1, 1, (9, 3))], 1)
    anchor = Anchor("gt_depth", depth=depth[:, 0], far_clip_m=80.0)
    A = np.eye(4)
    A[:3, :3] = np.diag([1.0, -1.0, 1.0])
    A[:3, 3] = [0.3, -2.0, 0.5]
    lifter = FakeLifter(depth, depth_div=2.0)
    xyz, vis, info = lifter.track(frames, K, w2c, q, anchor, world_transform=A, return_info=True)
    np.testing.assert_allclose(xyz, np.broadcast_to(q[:, 1:], (3, 9, 3)), atol=1e-5)
    s, T = info["scene_norm"]["scale"], np.array(info["scene_norm"]["translation"])
    np.testing.assert_allclose(lifter.tracker_inputs["queries"][:, 1:], s * (q[:, 1:] @ A[:3, :3].T + A[:3, 3]) + T,
                               atol=1e-5)
    _, _, info0 = lifter.track(frames, K, w2c, q, anchor, return_info=True)
    assert info["scene_norm"]["scale"] == pytest.approx(info0["scene_norm"]["scale"], rel=1e-6)
    assert info["world_transform"] == A.tolist() and info0["world_transform"] is None
    with pytest.raises(ValueError):
        lifter.track(frames, K, w2c, q, anchor, world_transform=np.diag([2.0, 1, 1, 1]))


def test_source_world_transform():
    ts = {"A": [[100.0, 0, 0], [0, -100.0, 0], [0, 0, 100.0]], "b": [-14511.8067, 8590.8182, 102.6504],
          "source_units": "cm"}
    A = source_world_transform(ts)
    np.testing.assert_array_equal(A[:3, :3], np.diag([1.0, -1.0, 1.0]))
    np.testing.assert_array_equal(A[:3, 3], np.array(ts["b"]) / 100.0)
    np.testing.assert_array_equal(A[3], [0, 0, 0, 1])
    X = np.random.default_rng(9).uniform(-1, 1, (5, 3))
    np.testing.assert_allclose(X @ A[:3, :3].T + A[:3, 3], (X @ np.array(ts["A"]).T + ts["b"]) / 100.0, atol=1e-12)
    eye = {"A": np.eye(3).tolist(), "b": [0.0, 0.0, 0.0], "source_units": "m"}
    np.testing.assert_array_equal(source_world_transform(eye), np.eye(4))
    with pytest.raises(ValueError):
        source_world_transform({**eye, "A": (2 * np.eye(3)).tolist()})
    with pytest.raises(ValueError):
        source_world_transform({**eye, "source_units": "ft"})
    frames, K, w2c, depth = make_scene(V=2, n=2)
    lifter = FakeLifter(depth, depth_div=2.0)
    q = np.array([[0, 0.1, -0.2, 0.3], [1, 0.0, 0.5, 0.2]])
    xyz, _, info = lifter.track(frames, K, w2c, q, Anchor("gt_depth", depth=depth[:, 0]), world_transform=A,
                                return_info=True)
    np.testing.assert_allclose(xyz, np.broadcast_to(q[:, 1:], (2, 2, 3)), atol=1e-4)
    np.testing.assert_allclose(lifter.tracker_inputs["queries"][:, 1:],
                               info["scene_norm"]["scale"] * (q[:, 1:] @ A[:3, :3].T + A[:3, 3])
                               + np.array(info["scene_norm"]["translation"]), atol=1e-3)


def test_given_depth_and_input_validation():
    frames, K, w2c, depth = make_scene(V=2, n=2)
    lifter = FakeLifter(depth)
    d = depth.copy()
    d[0, 0, 0, 0] = 100.0
    xyz, vis, info = lifter.track(frames, K, w2c, np.zeros((2, 4)), Anchor("none", far_clip_m=80.0), depth=d,
                                  return_info=True)
    assert lifter.depth_calls == [] and info["depth_align"][0]["mode"] == "given"
    assert lifter.tracker_inputs["depths"][0, 0, 0, 0] == 0
    xyz, vis = lifter.track(frames, K, w2c, np.zeros((0, 4)), Anchor("none"))
    assert xyz.shape == (2, 0, 3) and vis.shape == (2, 0)
    for bad in (np.array([[0.5, 0, 0, 0]]), np.array([[2, 0, 0, 0]]), np.array([[0, np.nan, 0, 0]])):
        with pytest.raises(ValueError):
            lifter.track(frames, K, w2c, bad, Anchor("none"))
    with pytest.raises(ValueError):
        lifter.track(frames.astype(np.float32), K, w2c, np.zeros((1, 4)), Anchor("none"))
    with pytest.raises(ValueError):
        lifter.track(frames, K[:, :1], w2c, np.zeros((1, 4)), Anchor("none"))
    with pytest.raises(ValueError, match="norm_depth"):
        lifter.track(frames, K, w2c, np.zeros((1, 4)), Anchor("none", norm_depth=np.ones((2, 10, 10))))
    with pytest.raises(ValueError, match="points_vis"):
        lifter.track(frames, K, w2c, np.zeros((1, 4)),
                     Anchor("gt_points", points=np.zeros((3, 3)), points_vis=np.ones((1, 3), bool)))


def test_registry_and_lazy_construction():
    lifter = get_lifter("mvtracker", device="cpu", rgb_scale_quirk=True)
    assert isinstance(lifter, MVTrackerLifter) and isinstance(lifter, Lifter)
    assert lifter._tracker is None and lifter._moge is None
    assert lifter.rgb_scale_quirk and lifter.hub_repo == "ethz-vlg/mvtracker:ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
    assert lifter.moge_weights == "Ruicheng/moge-2-vitl-normal"
    with pytest.raises(KeyError):
        get_lifter("nope")
    register_lifter("fake", lambda **kw: FakeLifter(np.ones((1, 1, 2, 2), np.float32), **kw))
    assert isinstance(get_lifter("fake"), FakeLifter)
    with pytest.raises(ValueError):
        MVTrackerLifter(hub_source="ftp")


def test_mvtracker_rgb_scale_and_tensor_layout():
    torch = pytest.importorskip("torch")
    seen = {}

    class FakePredictor:
        def __call__(self, rgbs, depths, query_points_3d, intrs, extrs):
            seen.update(rgbs=rgbs, depths=depths, q=query_points_3d, intrs=intrs, extrs=extrs)
            n, N = rgbs.shape[2], query_points_3d.shape[1]
            return {"traj_e": query_points_3d[:, None, :, 1:].repeat(1, n, 1, 1),
                    "vis_e": torch.ones(1, n, N, dtype=torch.bool)}

    frames, K, w2c, depth = make_scene(V=2, n=3)
    frames[0, 1, 2, 3] = 255
    q = np.array([[1, 0.1, 0.2, 0.3], [0, 1, 2, 3]], np.float64)
    for quirk, top in ((False, 255.0), (True, 1.0)):
        lifter = MVTrackerLifter(device="cpu", rgb_scale_quirk=quirk)
        lifter._tracker = FakePredictor()
        tr, vs = lifter.run_tracker(frames, depth, K, w2c, q)
        assert seen["rgbs"].shape == (1, 2, 3, 3, 40, 48) and seen["rgbs"].dtype == torch.float32
        assert float(seen["rgbs"].max()) == pytest.approx(top)
        assert seen["depths"].shape == (1, 2, 3, 1, 40, 48) and seen["extrs"].shape == (1, 2, 3, 3, 4)
        assert seen["intrs"].shape == (1, 2, 3, 3, 3) and seen["q"].shape == (1, 2, 4)
        assert tr.shape == (3, 2, 3) and vs.dtype == bool and vs.all()


def test_mvtracker_moge_wrapper_batches_mask_and_fov():
    torch = pytest.importorskip("torch")
    calls = []

    class FakeMoGe:
        def infer(self, im, fov_x=None, resolution_level=9, use_fp16=True):
            fov = fov_x.tolist() if torch.is_tensor(fov_x) else fov_x
            calls.append(dict(shape=tuple(im.shape), fov=fov, level=resolution_level, fp16=use_fp16,
                              lo=float(im.min()), hi=float(im.max())))
            depth = im[:, 0] * 10.0 + 1.0
            mask = torch.ones(depth.shape, dtype=torch.bool)
            mask[:, 0, 0] = False
            return {"depth": torch.where(mask, depth, torch.inf), "mask": mask}

    frames = np.random.default_rng(8).integers(0, 256, (10, 6, 8, 3), dtype=np.uint8)
    lifter = MVTrackerLifter(device="cpu", moge_batch=4, moge_resolution_level=7)
    lifter._moge = FakeMoGe()
    d = lifter.estimate_depth(frames, np.full(10, 70.0))
    assert d.shape == (10, 6, 8) and d.dtype == np.float32
    assert np.isnan(d[:, 0, 0]).all() and np.isfinite(d[:, 1:]).all()
    np.testing.assert_allclose(d[:, 1:], frames[:, 1:, :, 0] / 255.0 * 10.0 + 1.0, rtol=1e-6)
    assert [c["shape"][0] for c in calls] == [4, 4, 2] and calls[0]["shape"][1:] == (3, 6, 8)
    assert all(c["fov"] == 70.0 and c["level"] == 7 and c["fp16"] is False for c in calls)
    assert min(c["lo"] for c in calls) >= 0.0 and max(c["hi"] for c in calls) <= 1.0
    calls.clear()
    lifter.estimate_depth(frames, np.linspace(60.0, 69.0, 10))
    assert [len(c["fov"]) for c in calls] == [4, 4, 2] and calls[2]["fov"] == pytest.approx([68.0, 69.0])


def test_mvtracker_weight_sources(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    seen = []

    class Dummy:
        def eval(self):
            return self

    def fake_hub_load(repo, model, source=None, pretrained=None, device=None, **kw):
        seen.append(dict(repo=repo, model=model, source=source, pretrained=pretrained, device=device, kw=kw))
        return Dummy()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    MVTrackerLifter(device="cpu")._get_tracker()
    assert seen[-1] == dict(repo="ethz-vlg/mvtracker:ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072", model="mvtracker",
                            source="github", pretrained=True, device="cpu",
                            kw={"trust_repo": True, "skip_validation": True})
    MVTrackerLifter(device="cpu", hub_repo=str(tmp_path))._get_tracker()
    assert seen[-1]["repo"] == str(tmp_path) and seen[-1]["source"] == "local" and seen[-1]["kw"] == {}
    MVTrackerLifter(device="cpu", hub_repo="ethz-vlg/mvtracker:main", hub_model="mvtracker_cleandepth",
                    hub_kwargs={"trust_repo": "check"})._get_tracker()
    assert seen[-1]["repo"] == "ethz-vlg/mvtracker:main" and seen[-1]["model"] == "mvtracker_cleandepth"
    assert seen[-1]["kw"] == {"trust_repo": "check"}
    lifter = MVTrackerLifter(device="cpu")
    first = lifter._get_tracker()
    assert lifter._get_tracker() is first and len(seen) == 4

    def broken_hub_load(*a, **k):
        raise ModuleNotFoundError("No module named 'kornia'")

    monkeypatch.setattr(torch.hub, "load", broken_hub_load)
    with pytest.raises(ImportError, match=r"loading MVTracker from .* failed \(No module named 'kornia'\); install"):
        MVTrackerLifter(device="cpu")._get_tracker()

    v2 = pytest.importorskip("moge.model.v2")
    got = []

    class FakeMoGeModel:
        def to(self, device):
            got.append(("to", device))
            return self

        def eval(self):
            return self

    monkeypatch.setattr(v2.MoGeModel, "from_pretrained",
                        classmethod(lambda cls, src: got.append(src) or FakeMoGeModel()))
    MVTrackerLifter(device="cpu")._get_moge()
    assert got[:2] == ["Ruicheng/moge-2-vitl-normal", ("to", "cpu")]
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    MVTrackerLifter(device="cpu", moge=str(weights))._get_moge()
    assert got[2] == str(weights)


def test_prepare_anchor_on_the_track_grid():
    frames, K, w2c, depth = make_scene(V=2, n=2, H=40, W=40)
    K[..., 0, :2] *= 0.75
    lifter = FakeLifter(depth, depth_div=2.0, plain_K_scaling=True)
    a = lifter.prepare_anchor(Anchor("none", far_clip_m=80.0), frames, K, w2c, track_size=(53, 40))
    assert a.norm_depth.shape == (2, 40, 53) and [c[1] for c in lifter.depth_calls] == [1, 1]
    np.testing.assert_allclose(lifter.depth_calls[0][2], fov_x_deg(K[0, :1] * [[53 / 40], [1], [1]], 53))
    _, _, info = lifter.track(frames, K, w2c, np.zeros((1, 4)), a, track_size=(53, 40), return_info=True)
    assert info["scene_norm"]["from"] == "norm_depth" and info["grid"] == [53, 40]
    with pytest.raises(ValueError, match="norm_depth"):
        lifter.track(frames, K, w2c, np.zeros((1, 4)), a)
