"""Tests for vpm_worldbench.tasks.segmentation.

A synthetic mini benchmark (written to tmp_path) covers every subset: static look-at cameras around the origin, a
moving target cluster, other clusters as non-target entities, depth archives rendered as point splats in front of a far
background plane, and a Real-General labels/human/masks.npz (np.packbits along x). Optional real-data checks run when
VPMWB_RELEASED_ROOT points to a VP-WorldBench data root.
"""
import json
import math
import os
import zipfile

import numpy as np
import pytest

from vpm_worldbench.constants import KIND_NAMES, SCHEMA_SAMPLE
from vpm_worldbench.data.benchmark import Benchmark
from vpm_worldbench.geometry import inside, project
from vpm_worldbench.metrics.segmentation import j_and_f, sample_mask
from vpm_worldbench.tasks import segmentation as seg
from vpm_worldbench.tasks.common import RunDirError

TP, HZ = 4, 12
W, H, T, V = 64, 48, 24, 2
KIND = {k: i for i, k in enumerate(KIND_NAMES)}


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


K0 = np.array([[40.0, 0, (W - 1) / 2], [0, 40.0, (H - 1) / 2], [0, 0, 1]])
CENTRES = np.array([(3.0, 0.2, 1.0), (0.3, 3.0, 1.2)])


def _cluster(rng, n, centre, radius):
    return np.asarray(centre, float) + rng.uniform(-radius, radius, (n, 3))


def _write(root, subset, sid, xyz, groups, group, kind, side, extra, attributes, captions=None, depth=False,
           labels=None, visible_mask=None):
    """Write one sample; visibility = in frustum (& visible_mask); depth = point splats (3x3, min z) over a plane."""
    import cv2
    from vpm_worldbench.io.video import write_video
    d = root / subset / sid
    (d / "videos").mkdir(parents=True)
    N = xyz.shape[1]
    w2c = np.stack([np.stack([_look_at(c)] * T) for c in CENTRES])
    valid = np.isfinite(xyz).all(-1)
    infr = np.zeros((T, N, V), bool)
    for v in range(V):
        for t in range(T):
            uv, z = project(xyz[t], K0, w2c[v, t])
            infr[t, :, v] = valid[t] & (z > 0.1) & inside(uv, (W, H))
    vis = infr.copy() if visible_mask is None else infr & visible_mask
    np.savez(d / "tracks3d.npz", xyz=xyz.astype(np.float32), valid=valid, visible=vis, in_frustum=infr,
             default_gt=valid, group=np.asarray(group, np.int16), group_names=np.array(groups),
             part=np.zeros(N, np.int16), part_names=np.array(["p"]), kind=np.asarray(kind, np.int8),
             kind_names=np.array(KIND_NAMES), side=np.asarray(side, np.int8), **extra)
    np.savez(d / "cameras.npz", view_names=np.array([f"view{v}" for v in range(V)]),
             source_view_names=np.array([f"cam{v}" for v in range(V)]), image_size=np.array([[W, H]] * V, np.int32),
             K=np.repeat(K0[None], V, 0), w2c=w2c, static=np.ones(V, bool))
    cap = {"caption": "a synthetic clip."}
    cap.update(captions or {})
    (d / "captions.json").write_text(json.dumps(cap))
    views = [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W, "height": H, "source_name": f"cam{v}"}
             for v in range(V)]
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid, "n_views": V,
            "n_frames": T, "fps": 10.0, "views": views,
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": None, "labels": {}},
            "tracks": {"n_points": N, "groups": list(groups), "default_report_group": "all"},
            "caption": cap["caption"], "attributes": attributes, "flags": [], "depth": None}
    if depth:
        (d / "depth").mkdir()
        with zipfile.ZipFile(d / "depth" / "depth.zip", "w") as zf:
            for v in range(V):
                for t in range(T):
                    D = np.full((H, W), 8.0)
                    uv, z = project(xyz[t], K0, w2c[v, t])
                    for (u, vv), zz, ok in zip(uv, z, valid[t]):
                        if not ok or zz <= 0:
                            continue
                        iu, iv = int(np.floor(u + 0.5)), int(np.floor(vv + 0.5))
                        y0, y1, x0, x1 = max(iv - 1, 0), min(iv + 2, H), max(iu - 1, 0), min(iu + 2, W)
                        if y0 < y1 and x0 < x1:
                            D[y0:y1, x0:x1] = np.minimum(D[y0:y1, x0:x1], zz)
                    img = np.round(D * 1000).astype(np.uint16)
                    zf.writestr(f"cam{v}/frame_{t:04d}.png", cv2.imencode(".png", img)[1].tobytes())
        meta["depth"] = {"file": "depth/depth.zip", "member": "{source_view}/frame_{t:04d}.png",
                         "view_member_names": [f"cam{v}" for v in range(V)], "dtype": "uint16", "unit_scale": 0.001,
                         "invalid_values": [0], "width": W, "height": H, "kind": "rendered (synthetic GT)"}
        meta["files"]["depth"] = "depth/depth.zip"
    for name, (rel, writer) in (labels or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        writer(p, w2c)
        meta["files"]["labels"][name] = rel
    (d / "sample.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(0)
    for v in range(V):
        write_video(d / "videos" / f"view{v}.mp4", rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8), fps=10.0)
    return d


def _moving(base, vel):
    return (base[None] + np.asarray(vel)[None, None] * np.arange(T)[:, None, None]).astype(np.float64)


def make_simul_robot(root, sid="sr0"):
    rng = np.random.default_rng(1)
    tgt = _moving(_cluster(rng, 12, (0, 0, 0), 0.06), (0.01, 0, 0))
    other = _moving(_cluster(rng, 10, (0, 0, 0.5), 0.06), (0, 0, 0))
    robot = _moving(_cluster(rng, 10, (0, 0, -0.5), 0.06), (0, 0, 0))
    xyz = np.concatenate([tgt, other, robot], 1)
    n1, n2, n3 = 12, 10, 10
    group = [0] * (n1 + n2) + [1] * n3
    kind = [KIND["object_moving"]] * n1 + [KIND["object_static"]] * n2 + [KIND["robot_hand"]] * n3
    side = [-1] * (n1 + n2) + [1] * n3
    extra = {"is_primary": np.array([True] * n1 + [False] * (n2 + n3)),
             "object_id": np.array(["small_yellow_cube"] * n1 + ["square_big_box"] * n2 + [""] * n3)}
    return _write(root, "Simul-Robot", sid, xyz, ("object", "robot"), group, kind, side, extra,
                  {"primary_object": "small_yellow_cube", "difficulty": "normal"}, depth=True)


def make_simul_general(root, sid="sg0"):
    rng = np.random.default_rng(2)
    xyz = _moving(_cluster(rng, 16, (0, 0, 0), 0.1), (0.01, 0.005, 0))
    N = xyz.shape[1]
    return _write(root, "Simul-General", sid, xyz, ("target",), [0] * N, [KIND["pedestrian"]] * N, [-1] * N, {},
                  {"category": "pedestrian", "difficulty": "normal"}, depth=True)


def make_real_robot(root, sid="rr0", hand="right"):
    rng = np.random.default_rng(3)
    left = _moving(_cluster(rng, 12, (0, 0, 0.4), 0.06), (0, 0, 0))
    right = _moving(_cluster(rng, 12, (0, 0, -0.4), 0.06), (0.01, 0, 0))
    xyz = np.concatenate([left, right], 1)
    group = ([0] * 6 + [1] * 6) * 2
    kind = ([KIND["robot_arm"]] * 6 + [KIND["robot_hand"]] * 6) * 2
    side = [0] * 12 + [1] * 12
    steps = [{"text": "do it", "hands": [hand if hand != "bimanual" else "right"]}]
    return _write(root, "Real-Robot", sid, xyz, ("arm", "hand"), group, kind, side, {},
                  {"hand": hand, "difficulty": "normal"}, captions={"structured": {"steps": steps}})


def _rg_mask_writer(xyz, person, n_frames=T, not_ok=((10, 1),), border_col=0):
    """actor mask = disks (radius 3 px) around the projected person points; mask_ok False at `not_ok` (t, v)."""
    def write(path, w2c):
        actor = np.zeros((n_frames, V, H, W), bool)
        yy, xx = np.mgrid[0:H, 0:W]
        for t in range(n_frames):
            for v in range(V):
                uv, z = project(xyz[t, person], K0, w2c[v, t])
                for u, vv in uv:
                    actor[t, v] |= (xx - u) ** 2 + (yy - vv) ** 2 <= 9.0
        ok = np.ones((n_frames, V), bool)
        for t, v in not_ok:
            ok[t, v] = False
        valid_px = np.ones((V, H, W), bool)
        valid_px[:, :, border_col] = False
        np.savez_compressed(path, actor=np.packbits(actor, axis=-1), other=np.packbits(np.zeros_like(actor), axis=-1),
                            ran=np.ones((n_frames, V), bool), mask_ok=ok,
                            hull_iou=np.ones((n_frames, V), np.float32),
                            area_px_src=actor.sum((2, 3)).astype(np.int32),
                            valid_px=np.packbits(valid_px, axis=-1),
                            view_names=np.array([f"view{v}" for v in range(V)]),
                            source_view_names=np.array([f"cam{v}" for v in range(V)]),
                            image_size=np.array([W, H], np.int32), info=np.array("{}"))
    return write


def make_real_general(root, sid="rg0", masks=True):
    rng = np.random.default_rng(4)
    body = _moving(_cluster(rng, 24, (0, 0, 0), 0.12), (0.008, 0, 0))
    near = body[:, :10] + rng.normal(0, 0.01, (1, 10, 3))
    far = _moving(_cluster(rng, 10, (0, 0, 0.8), 0.05), (0, 0, 0))
    xyz = np.concatenate([body, near, far], 1)
    n = xyz.shape[1]
    kind = [KIND["human_body"]] * 24 + [KIND["dynamic_surface"]] * 20
    group = [1] * 24 + [0] * 20
    extra = {"entity": np.array([-1] * 24 + [0] * 10 + [1] * 10, np.int16)}
    person = np.zeros(n, bool)
    person[:34] = True
    labels = {"masks": ("labels/human/masks.npz", _rg_mask_writer(xyz, person))} if masks else {}
    return _write(root, "Real-General", sid, xyz, ("dynamic_surface", "human_body"), group, kind, [-1] * n, extra,
                  {"difficulty": "normal"}, depth=True, labels=labels)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("vpmwb_seg")
    make_simul_robot(root)
    make_simul_general(root)
    make_real_robot(root)
    make_real_general(root)
    return Benchmark(root)


@pytest.fixture(scope="module")
def proto():
    return seg.SegProtocol(ref_frame=TP, horizon=HZ, bg_stride_px=8, max_pos=16, max_neg=16, sparse_frame_step=1,
                           min_prompt_area_px=4)


def _run(bench, tmp_path, proto, kind, **kw):
    pred = tmp_path / f"pred_{kind}"
    seg.run_baseline(bench, pred, kind, proto)
    return seg.evaluate(bench, pred, tmp_path / f"res_{kind}", proto, **kw)


def test_protocol_roundtrip_window_and_validation(bench, proto):
    p2 = seg.SegProtocol.from_dict(json.loads(json.dumps(proto.to_dict())))
    assert p2 == proto
    s = bench.sample("Simul-Robot/sr0")
    assert proto.window(s) == list(range(TP, TP + HZ + 1))
    assert seg.SegProtocol(ref_frame=TP, horizon=None).window(s)[-1] == T - 1
    assert seg.SegProtocol(ref_frame=TP, horizon=4, frame_step=3).window(s) == [4, 7, 10, 13, 16]
    assert not seg.SegProtocol(ref_frame=TP, horizon=40).evaluable(s)[0]
    with pytest.raises(ValueError):
        seg.SegProtocol(targets={"Real-Robot": "primary_object"})
    with pytest.raises(KeyError):
        seg.SegProtocol.from_dict({"bogus": 1})
    assert seg.SegProtocol(targets={"rr": "left_arm"}).target_for("Real-Robot") == "left_arm"
    p = seg.SegProtocol(targets={"rr": "left_arm"}, background={"real_general": False}, bg_margin_m={"SG": 0.5})
    assert p.targets == {"Real-Robot": "left_arm"} and p.background == {"Real-General": False}
    assert p.bg_margin_m == {"Simul-General": 0.5} and p.bg_margin_for("sg") == 0.5
    assert seg.SegProtocol.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert not p.background_for("Real-General") and p.background_for("Simul-Robot")
    with pytest.raises(KeyError):
        seg.SegProtocol(background={"Mars": True})
    with pytest.raises(ValueError):
        seg.SegProtocol(targets={"rr": "arms", "Real-Robot": "left_arm"})
    for bad in ((0, 48), (64,), "64x48"):
        with pytest.raises(ValueError, match="dense_size must be"):
            seg.SegProtocol(dense_size=bad)


def test_resize_mask_nearest_is_pixel_centre_exact():
    rng = np.random.default_rng(0)
    m = rng.random((6, 10)) > 0.5
    up = seg.resize_mask_nearest(m, (20, 12))
    assert np.array_equal(up, np.repeat(np.repeat(m, 2, 0), 2, 1))
    assert np.array_equal(seg.resize_mask_nearest(up, (10, 6)), m)
    assert seg.resize_mask_nearest(m, (10, 6)) is m
    stack = np.stack([m, ~m])
    assert seg.resize_mask_nearest(stack, (5, 3)).shape == (2, 3, 5)
    from vpm_worldbench.geometry import image_map, warp_image
    idx = np.arange(37 * 23).reshape(23, 37)
    for size in ((15, 9), (50, 31), (37, 23), (8, 40)):
        assert np.array_equal(seg.resize_mask_nearest(idx, size), warp_image(idx, image_map((37, 23), size), "nearest"))
    try:
        import cv2
        flag = getattr(cv2, "INTER_NEAREST_EXACT", None)
        if flag is not None:
            ref = cv2.resize(m.astype(np.uint8), (7, 4), interpolation=flag).astype(bool)
            assert np.array_equal(seg.resize_mask_nearest(m, (7, 4)), ref)
    except ImportError:
        pass


def test_fps_and_rng_are_deterministic():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
    assert seg._fps(pts, 3, 0) == [0, 2, 3]
    assert seg._fps(pts, 10, 1) == [1, 2, 3, 0]
    assert seg._fps(np.zeros((0, 2)), 3, 0) == []
    a = seg._rng(0, "x", 1).integers(0, 1 << 30, 5)
    b = seg._rng(0, "x", 1).integers(0, 1 << 30, 5)
    assert np.array_equal(a, b)


def test_actor_masks_unpack_orientation(bench):
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    tr = s.tracks
    person = np.zeros(tr.n_points, bool)
    person[:34] = True
    for t in (TP, 9):
        for v in range(V):
            uv, _ = project(tr.xyz[t, person], s.cameras.K[v], s.cameras.w2c[v, t])
            val, ins = sample_mask(m.get(t, v), uv)
            assert val[ins].all()
            uv2, _ = project(tr.xyz[t, ~person], s.cameras.K[v], s.cameras.w2c[v, t])
            val2, ins2 = sample_mask(m.get(t, v), uv2)
            assert not val2[ins2].any()
    assert not m.usable(10, 1) and m.usable(10, 0)
    vp = m.valid_px(0)
    assert vp.shape == (H, W) and not vp[:, 0].any() and vp[:, 1:].all()


def test_targets_per_subset(bench, proto):
    t = seg.resolve_target(bench.sample("Simul-Robot/sr0"), proto)
    assert t["name"] == "primary_object" and t["pos"].sum() == 12 and t["neg"].sum() == 20
    assert "small yellow cube" in t["text"] and t["background"]
    t = seg.resolve_target(bench.sample("Simul-Robot/sr0"), seg.SegProtocol(targets={"Simul-Robot": "robot"}))
    assert t["pos"].sum() == 10 and not t["background"]
    t = seg.resolve_target(bench.sample("Simul-General/sg0"), proto)
    assert t["pos"].all() and not t["neg"].any()
    t = seg.resolve_target(bench.sample("Real-Robot/rr0"), proto)
    assert t["info"]["side"] == "right" and t["pos"][12:].all() and not t["pos"][:12].any() and t["neg"][:12].all()
    assert not t["background"]
    t = seg.resolve_target(bench.sample("Real-Robot/rr0"), seg.SegProtocol(targets={"Real-Robot": "left_arm"}))
    assert t["pos"][:12].all()
    s = bench.sample("Real-General/rg0")
    lab = seg.rg_entity_labels(s)
    assert lab[0]["label"] == "person" and lab[1]["label"] == "other"
    assert lab[0]["in_actor"] > 0.85 and lab[1]["in_actor"] < 0.15
    t = seg.resolve_target(s, proto)
    assert t["pos"][:34].all() and not t["pos"][34:].any() and t["neg"][34:].all()


def test_sparse_labels(bench, proto):
    for uid in ("Simul-Robot/sr0", "Simul-General/sg0", "Real-Robot/rr0", "Real-General/rg0"):
        s = bench.sample(uid)
        tgt = seg.resolve_target(s, proto)
        lab = seg.sparse_labels(s, proto, tgt)
        lab2 = seg.sparse_labels(s, proto, tgt)
        tr = s.tracks
        for v, L in lab["views"].items():
            assert np.array_equal(L["uv"], lab2["views"][v]["uv"])
            assert set(np.unique(L["frame"]).tolist()) <= set(proto.window(s)[1:])
            for t in np.unique(L["frame"]):
                sel = L["frame"] == t
                assert (L["label"][sel]).sum() <= proto.max_pos and (~L["label"][sel]).sum() <= proto.max_neg
            assert np.array_equal(L["label"], L["source"] == seg.SRC_POS)
            uv_t = {}
            for t in np.unique(L["frame"]):
                uv_t[int(t)] = project(tr.xyz[t], s.cameras.K[v], s.cameras.w2c[v, t])[0]
            for t, uv, src in zip(L["frame"], L["uv"], L["source"]):
                if src == seg.SRC_BACKGROUND:
                    continue
                d = np.linalg.norm(uv_t[int(t)] - uv, axis=1)
                n = int(np.nanargmin(d))
                assert d[n] < 1e-9 and tr.visible[t, n, v]
                assert bool(tgt["pos"][n]) == (src == seg.SRC_POS)
            bg = L["source"] == seg.SRC_BACKGROUND
            if s.subset == "Real-Robot":
                assert not bg.any()
            else:
                assert bg.any()
                for t, (u, vv) in zip(L["frame"][bg], L["uv"][bg]):
                    D = s.depth(v, [int(t)])[0][int(vv), int(u)]
                    ray = np.linalg.inv(s.cameras.K[v]) @ np.array([u, vv, 1.0]) * D
                    c2w = np.linalg.inv(s.cameras.w2c[v, t])
                    X = c2w[:3, :3] @ ray + c2w[:3, 3]
                    dist = np.linalg.norm(tr.xyz[t, tgt["pos"]] - X, axis=1).min()
                    assert dist >= lab["margin_m"] - 1e-6


def test_prompts(bench, proto):
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    pr = seg.sample_prompts(s, proto, masks=m)
    for v, r in pr.items():
        assert r["ok"] and r["dense_ok"] and r["mask"] is not None
        assert np.array_equal(r["mask"], m.get(TP, v))
        val, ins = sample_mask(r["mask"], r["points"])
        assert val.all() and ins.all() and r["points_source"] == "track_points"
        vv, uu = np.nonzero(r["mask"])
        assert r["box"] == [uu.min(), vv.min(), uu.max(), vv.max()] and r["box_source"] == "gt_mask"
        assert len(r["points"]) == proto.n_points_export
    s = bench.sample("Simul-Robot/sr0")
    pr = seg.sample_prompts(s, proto)
    tr = s.tracks
    for v, r in pr.items():
        uv, _ = project(tr.xyz[TP], s.cameras.K[v], s.cameras.w2c[v, TP])
        for p in r["points"]:
            n = int(np.argmin(np.linalg.norm(uv - p, axis=1)))
            assert tr.extra["is_primary"][n] and tr.visible[TP, n, v]
        x0, y0, x1, y1 = r["box"]
        assert x0 <= x1 and y0 <= y1 and r["box_source"] == "track_points"
    p_bad = seg.SegProtocol(ref_frame=TP, horizon=HZ, min_prompt_points=1000)
    assert not any(r["ok"] for r in seg.sample_prompts(s, p_bad).values())


def test_oracle_and_label_raster_are_perfect(bench, proto, tmp_path):
    for kind in ("oracle", "label_raster"):
        summ = _run(bench, tmp_path, proto, kind)
        for sub, s in summ["subsets"].items():
            sp = s["sparse"]
            assert s["status"] == {"ok": 1}, (kind, sub, s["status"])
            assert sp["recall"]["mean"] == pytest.approx(100.0) and sp["precision"]["mean"] == pytest.approx(100.0)
            assert sp["iou"]["mean"] == pytest.approx(100.0)
            if kind == "oracle" and sub == "Real-General":
                assert s["dense"]["J&F"]["mean"] == pytest.approx(100.0)
                assert s["dense"]["J"]["mean"] == pytest.approx(100.0)
    rec = json.loads((tmp_path / "res_oracle" / "Real-General" / "rg0.json").read_text())
    assert 10 not in rec["views"]["view1"]["dense"]["frames"]
    assert 10 in rec["views"]["view0"]["dense"]["frames"]


def test_oracle_falls_back_to_label_raster_where_masks_are_unusable(bench, proto):
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    win = proto.window(s)
    o = seg.baseline_masks(s, 1, "oracle", proto)
    lr = seg.baseline_masks(s, 1, "label_raster", proto)
    j = win.index(10)
    assert not m.usable(10, 1) and m.get(10, 1).any()
    assert np.array_equal(o[j], lr[j]) and not np.array_equal(o[j], m.get(10, 1))
    for jj, t in enumerate(win):
        if m.usable(t, 1):
            assert np.array_equal(o[jj], m.get(t, 1))


def test_prediction_checks(bench, proto, tmp_path):
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    win = proto.window(s)
    pred = tmp_path / "chk"
    gt = [np.stack([m.get(t, v) for t in win]) for v in range(V)]
    seg.write_prediction(pred, s, 0, gt[0])
    seg.write_prediction(pred, s, 1, np.swapaxes(gt[1], 1, 2))
    with pytest.raises(ValueError, match="swapped"):
        seg.evaluate_sample(s, pred, proto)
    r = seg._eval_one((str(bench.root), s.subset, s.id, str(pred), proto.to_dict(), None))
    assert r["status"] == "invalid" and "swapped" in r["reason"]
    with pytest.warns(UserWarning, match="1 sample"):
        summ = seg.evaluate(bench, pred, tmp_path / "res_chk", proto, ids=["rg/rg0"])
    assert summ["subsets"]["Real-General"]["status"] == {"invalid": 1} and summ["evaluated"]["all_failed"]
    assert (summ["subsets"]["Real-General"]["n_invalid"], summ["subsets"]["Real-General"]["n_error"]) == (1, 0)
    assert summ["pred_root"] == str(pred.resolve())
    (pred / s.subset / s.id / "view1.npz").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="not a readable"):
        seg.evaluate_sample(s, pred, proto)
    seg.write_prediction(pred, s, 1, gt[1][:, :, : W // 2])
    with pytest.raises(ValueError, match="aspect"):
        seg.evaluate_sample(s, pred, proto)
    seg.write_prediction(pred, s, 1, np.concatenate([gt[1], gt[1][:1]]), frames=list(win) + [win[0]])
    with pytest.raises(ValueError, match="more than once"):
        seg.evaluate_sample(s, pred, proto)
    keep = [t for t in win if t % 2 == 0]
    seg.write_prediction(pred, s, 1, np.stack([m.get(t, 1) for t in keep] + [m.get(20, 1)]), frames=keep + [20])
    r = seg.evaluate_sample(s, pred, proto)
    v1 = r["views"]["view1"]
    assert r["status"] == "ok" and v1["n_extra_rows"] == 1
    assert v1["n_missing_frames"] == len([t for t in win[1:] if t % 2 == 1])
    assert r["views"]["view0"]["n_missing_frames"] == 0 and r["views"]["view0"]["n_extra_rows"] == 0


def test_empty_and_full(bench, proto, tmp_path):
    e = _run(bench, tmp_path, proto, "empty")
    f = _run(bench, tmp_path, proto, "full")
    for sub in e["subsets"]:
        assert e["subsets"][sub]["sparse"]["recall"]["mean"] == 0.0
        assert e["subsets"][sub]["sparse"]["precision"]["n"] == 0
        pooled = f["subsets"][sub]["sparse_pooled"]
        assert pooled["recall"] == 100.0
        assert pooled["precision"] == pytest.approx(100.0 * pooled["n_pos"] / (pooled["tp"] + pooled["fp"]))
        assert pooled["fp"] == pooled["n_neg_points"] + pooled["n_background"]
    assert e["subsets"]["Real-General"]["dense"]["J"]["mean"] == 0.0


def test_prompt_static_matches_direct_computation(bench, proto, tmp_path):
    summ = _run(bench, tmp_path, proto, "prompt_static")
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    vals = []
    for v in range(V):
        frames = [t for t in proto.window(s)[1:] if m.usable(t, v)]
        P = np.stack([m.get(TP, v)] * len(frames))
        G = np.stack([m.get(t, v) for t in frames])
        vals.append(j_and_f(P, G, void=~m.valid_px(v))["J&F"])
    assert summ["subsets"]["Real-General"]["dense"]["J&F"]["mean"] == pytest.approx(np.mean(vals))
    assert summ["subsets"]["Real-General"]["dense"]["J&F"]["mean"] < 100.0


def test_prediction_formats_resolution_and_missing(bench, proto, tmp_path):
    s = bench.sample("Real-General/rg0")
    m = seg.ActorMasks(s)
    win = proto.window(s)
    pred = tmp_path / "fmt"
    gt = [np.stack([m.get(t, v) for t in win]) for v in range(V)]
    perm = np.random.default_rng(0).permutation(len(win))
    d = pred / s.subset / s.id
    d.mkdir(parents=True)
    np.savez(d / "view0.npz", masks=gt[0][perm].astype(np.uint8), frames=np.asarray(win)[perm])
    half = seg.resize_mask_nearest(gt[1], (W // 2, H // 2))
    seg.write_prediction(pred, s, 1, half)
    loaded, fr = seg.load_prediction(seg.prediction_path(pred, s, 1))
    assert fr is None and np.array_equal(loaded, half)
    r = seg.evaluate_sample(s, pred, proto)
    assert r["status"] == "ok"
    assert r["views"]["view0"]["dense"]["J&F"] == pytest.approx(100.0)
    assert 80.0 < r["views"]["view1"]["dense"]["J&F"] < 100.0
    assert r["views"]["view1"]["sparse"]["recall"] > 90.0
    np.savez(d / "view0.npz", masks=gt[0].astype(np.float32) * 0.9)
    assert seg.evaluate_sample(s, pred, proto)["views"]["view0"]["dense"]["J&F"] == pytest.approx(100.0)
    np.savez(d / "view0.npz", masks=gt[0][:3])
    with pytest.raises(ValueError):
        seg.evaluate_sample(s, pred, proto)
    (d / "view0.npz").unlink()
    r = seg.evaluate_sample(s, pred, proto)
    assert r["n_views_missing_file"] == 1 and r["views"]["view0"]["sparse"]["recall"] == 0.0
    assert seg.evaluate_sample(s, tmp_path / "none", proto)["status"] == "missing"
    p_pts = seg.SegProtocol.from_dict({**proto.to_dict(), "min_prompt_points": 1000})
    pr = seg.sample_prompts(s, p_pts)
    assert all(r["ok"] and r["points_source"] == "mask_pixels" for r in pr.values())
    p_bad = seg.SegProtocol(ref_frame=TP, horizon=HZ, min_prompt_points=1000, min_prompt_area_px=10 ** 9)
    assert seg.evaluate_sample(s, pred, p_bad)["status"] == "skipped"
    p_long = seg.SegProtocol(ref_frame=TP, horizon=100)
    assert seg.evaluate_sample(s, pred, p_long)["status"] == "skipped"


def test_dense_size_option(bench, proto, tmp_path):
    p = seg.SegProtocol.from_dict({**proto.to_dict(), "dense_size": [32, 24]})
    summ = _run(bench, tmp_path, p, "oracle")
    assert summ["subsets"]["Real-General"]["dense"]["J&F"]["mean"] == pytest.approx(100.0)


def test_labels_cache_roundtrip(bench, proto, tmp_path):
    s = bench.sample("Simul-General/sg0")
    cache = tmp_path / "cache"
    a = seg.cached_sparse_labels(s, proto, cache)
    assert (cache / s.subset / f"{s.id}.npz").exists()
    b = seg.cached_sparse_labels(s, proto, cache)
    for v in a["views"]:
        for c in ("frame", "uv", "label", "source"):
            assert np.array_equal(a["views"][v][c], b["views"][v][c])
    p2 = seg.SegProtocol.from_dict({**proto.to_dict(), "max_pos": 4})
    c = seg.cached_sparse_labels(s, p2, cache)
    assert all(c["views"][v]["label"].sum() <= 4 * HZ for v in c["views"])


def test_export_inputs_has_no_gt_beyond_prompts(bench, proto, tmp_path):
    import cv2
    out = tmp_path / "inputs"
    res = seg.export_inputs(bench, out, proto, frames="png")
    assert res == {"exported": 4, "reused": 0, "skipped": 0}
    assert seg.export_inputs(bench, out, proto, frames="png")["reused"] == 4
    p2 = seg.SegProtocol.from_dict({**proto.to_dict(), "n_points": 1})
    with pytest.raises(ValueError):
        seg.export_inputs(bench, out, p2, frames="png")
    assert seg.export_inputs(bench, out, p2, frames="png", overwrite=True)["reused"] == 0
    assert json.loads((out / "Real-General" / "rg0" / "inputs.json").read_text())["protocol"]["n_points"] == 1
    man = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert {m["uid"] for m in man} == {"Simul-Robot/sr0", "Simul-General/sg0", "Real-Robot/rr0", "Real-General/rg0"}
    inp = json.loads((out / "Real-General" / "rg0" / "inputs.json").read_text())
    assert set(inp) == {"schema", "task", "uid", "subset", "id", "target", "prompt_frame", "frames", "fps", "tracks",
                        "views", "protocol"}
    assert inp["frames"] == list(range(TP, TP + HZ + 1)) and inp["tracks"] == ["dense", "sparse"]
    v0 = inp["views"]["view0"]
    assert set(v0) == {"video", "size", "status", "prompt"}
    png = cv2.imread(str(out / "Real-General" / "rg0" / v0["prompt"]["mask"]), cv2.IMREAD_UNCHANGED)
    s = bench.sample("Real-General/rg0")
    assert np.array_equal(png > 0, seg.ActorMasks(s).get(TP, 0))
    assert (out / "Real-General" / "rg0" / "frames" / "view1" / f"frame_{TP + HZ:04d}.png").exists()
    sr = json.loads((out / "Simul-Robot" / "sr0" / "inputs.json").read_text())
    assert sr["views"]["view0"]["prompt"]["mask"] is None and 1 <= len(sr["views"]["view0"]["prompt"]["points"]) <= 5


def test_summary_files_and_cli(bench, proto, tmp_path, capsys):
    pfile = tmp_path / "proto.json"
    pfile.write_text(json.dumps(proto.to_dict()))
    root = str(bench.root)
    assert seg.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--protocol", str(pfile)]) == 0
    assert seg.main(["baseline", "--root", root, "--pred", str(tmp_path / "pr"), "--kind", "oracle", "--protocol",
                     str(pfile), "--subsets", "rg,sr"]) == 0
    assert seg.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res"),
                     "--protocol", str(pfile), "--subsets", "rg,sr", "--jobs", "2"]) == 0
    out = capsys.readouterr().out
    assert "| Real-General | person |" in out and "| Real-General | 1 | 100.0" in out
    summ = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(summ["subsets"]) == {"Simul-Robot", "Real-General"}
    assert summ["pred_root"] == str((tmp_path / "pr").resolve())
    assert (summ["subsets"]["Simul-Robot"]["n_invalid"], summ["subsets"]["Simul-Robot"]["n_error"]) == (0, 0)
    rec = json.loads((tmp_path / "res" / "Simul-Robot" / "sr0.json").read_text())
    assert rec["status"] == "ok" and rec["task"] == "segmentation" and rec["sparse"]["iou"] == 100.0
    assert seg.main(["baseline", "--root", root, "--pred", str(tmp_path / "pe"), "--kind", "empty", "--protocol",
                     str(pfile), "--subsets", "rr", "Simul-General,", "--ids", "rr0,sg0"]) == 0
    other = ["evaluate", "--root", root, "--pred", str(tmp_path / "pe"), "--out", str(tmp_path / "res"),
             "--protocol", str(pfile)]
    with pytest.raises(RunDirError, match="results of the predictions in"):
        seg.main(other)
    from vpm_worldbench.cli import main as vpmwb
    capsys.readouterr()
    assert vpmwb(["segmentation", *other]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: ") and "--overwrite" in err and len(err.strip().splitlines()) == 1
    assert json.loads((tmp_path / "res" / "summary.json").read_text()) == summ
    with pytest.raises(RunDirError, match=r"inputs of other export options \(export.json: other frames\)"):
        seg.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--protocol", str(pfile), "--frames", "png"])
    for bad in ("12x", "0x0"):
        with pytest.raises(SystemExit) as e:
            seg.main(other + ["--out", str(tmp_path / "r_ds"), "--dense-size", bad])
        assert e.value.code == 2 and f"--dense-size must be WxH (e.g. 640x480), got '{bad}'" in \
            capsys.readouterr().err
    assert not (tmp_path / "r_ds").exists()
    with pytest.raises(SystemExit) as e:
        seg.main(["summarize", "--out", str(tmp_path / "r_ds")])
    assert e.value.code == 2 and "no such directory" in capsys.readouterr().err and not (tmp_path / "r_ds").exists()
    assert seg.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pe"), "--out", str(tmp_path / "res"),
                     "--protocol", str(pfile), "--subsets", "rr", "sg", "--overwrite"]) == 0
    summ = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(summ["subsets"]) == {"Real-Robot", "Simul-General"}
    assert summ["subsets"]["Real-Robot"]["sparse"]["recall"]["mean"] == 0.0
    with pytest.raises(SystemExit):
        seg.main(["baseline", "--root", root, "--pred", str(tmp_path / "pe"), "empty"])
    capsys.readouterr()
    for args, msg in ((["--subsets", "Mars"], "unknown subset 'Mars'"),
                      (["--protocol", "vpm-v1"], "neither the preset 'vpm-seg-v1'")):
        with pytest.raises(SystemExit) as e:
            seg.main(["export", "--root", root, "--out", str(tmp_path / "in3"), *args])
        assert e.value.code == 2 and msg in capsys.readouterr().err
    assert seg.main(["export", "--root", root, "--out", str(tmp_path / "in4"), "--protocol", str(pfile), "--ids", "rg0",
                     "--horizon", "0"]) == 0
    inp = json.loads((tmp_path / "in4" / "Real-General" / "rg0" / "inputs.json").read_text())
    assert inp["frames"][-1] == T - 1 and inp["protocol"]["horizon"] is None
    assert inp["protocol"]["name"] == "vpm-seg-v1+horizon=None"
    with pytest.warns(UserWarning, match="match no sample"):
        assert seg.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pe"), "--out", str(tmp_path / "r0"),
                         "--protocol", str(pfile), "--ids", "nope"]) == 1
    assert "no sample matches the selection" in capsys.readouterr().err
    assert vpmwb(["segmentation", "summarize", "--out", str(tmp_path / "res")]) == 0


def test_mean_se_and_jsonable(tmp_path):
    m = seg.mean_se([1.0, 2.0, float("nan"), 3.0])
    assert m["n"] == 3 and m["mean"] == 2.0 and m["se"] == pytest.approx(1.0 / math.sqrt(3))
    assert seg.mean_se([])["n"] == 0
    m = seg.mean_se([1.0, float("inf"), None, "inf"])
    assert (m["mean"], m["n"], m["n_inf"], m["n_undefined"]) == (1.0, 1, 2, 1)
    (tmp_path / "my_seg.yaml").write_text("horizon: 8\n")
    p = seg.get_protocol(tmp_path / "my_seg.yaml")
    assert p.name == "my_seg" and p.horizon == 8
    assert seg.jsonable({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(2), float("inf")]}) == \
        {"a": None, "b": 1.5, "c": [2, "inf"]}


def test_real_robot_arms_target_and_real_general_without_masks(tmp_path):
    root = tmp_path / "mini"
    make_real_robot(root, "rr1")
    make_real_general(root, "rg1", masks=False)
    b = Benchmark(root)
    p = seg.SegProtocol(ref_frame=TP, horizon=HZ, bg_stride_px=8, max_pos=16, max_neg=16, sparse_frame_step=1,
                        targets={"Real-Robot": "arms"})
    t = seg.resolve_target(b.sample("Real-Robot/rr1"), p)
    assert t["pos"].all() and not t["neg"].any() and t["text"] == "the robot's arms and hands"
    s = b.sample("Real-General/rg1")
    assert seg.load_actor_masks(s) is None
    lab = seg.rg_entity_labels(s)
    assert lab[0]["label"] == "person" and lab[1]["label"] == "other" and lab[0]["n_obs"] == 0
    assert seg.sample_inputs(s, p)["tracks"] == ["sparse"]
    seg.run_baseline(b, tmp_path / "pred", "label_raster", p)
    summ = seg.evaluate(b, tmp_path / "pred", tmp_path / "res", p)
    rg = summ["subsets"]["Real-General"]
    assert rg["dense"] is None and rg["sparse"]["iou"]["mean"] == pytest.approx(100.0)
    rr = summ["subsets"]["Real-Robot"]["sparse_pooled"]
    assert rr["recall"] == 100.0 and rr["n_neg_points"] == 0 and rr["n_background"] == 0


RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")
released = pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT to a VP-WorldBench data root")


@released
def test_released_masks_agree_with_tracks():
    b = Benchmark(RELEASED)
    for sid in b.ids("Real-General")[:2]:
        s = b.sample("Real-General", sid)
        m = seg.ActorMasks(s)
        tr = s.tracks
        hb = tr.kind == KIND["human_body"]
        n_in = n = 0
        for v in range(s.n_views):
            for t in range(16, 81, 8):
                if not m.usable(t, v):
                    continue
                sel = hb & tr.visible[t, :, v]
                uv, _ = project(tr.xyz[t, sel], s.cameras.K[v], s.cameras.w2c[v, t])
                val, ins = sample_mask(m.get(t, v), uv)
                n_in += int(val[ins].sum())
                n += int(ins.sum())
        assert n > 50 and n_in / n > 0.95, (sid, n_in, n)


@released
def test_released_oracle_one_sample_per_subset(tmp_path):
    b = Benchmark(RELEASED)
    p = seg.SegProtocol(horizon=16)
    ids = [b.ids(sub)[0] for sub in b.subsets()]
    seg.run_baseline(b, tmp_path / "pred", "oracle", p, ids=ids)
    summ = seg.evaluate(b, tmp_path / "pred", tmp_path / "res", p, ids=ids)
    for sub, s in summ["subsets"].items():
        assert s["n_ok"] == 1
        if sub == "Real-General":
            assert s["dense"]["J&F"]["mean"] == pytest.approx(100.0)
            assert s["sparse"]["recall"]["mean"] > 90.0
        else:
            assert s["sparse"]["iou"]["mean"] == pytest.approx(100.0)
