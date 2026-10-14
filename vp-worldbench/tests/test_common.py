"""Tests of vpm_worldbench.tasks.common (the utilities shared by the geometry tasks depth / recon4d / nvs).

This file also holds `make_bench`, a tiny synthetic benchmark written to tmp_path that test_depth.py, test_recon4d.py
and test_nvs.py load by file path: a textured ground plane z = 0 seen by static or moving cameras, analytic depth, GT
tracks on the plane (30 static and 10 moving points), real H.264 videos, uint16 depth zips (half resolution for
Real-General) and a narrow-raw-FOV distortion model (an invalid border) on Real-General."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench import Benchmark
from vpm_worldbench.constants import KIND_NAMES, SCHEMA_SAMPLE
from vpm_worldbench.geometry import image_map, project, warp_image
from vpm_worldbench.tasks import common as C
from vpm_worldbench.utils import UsageError


def look_at(centre, target):
    """World-to-camera 4x4 (OpenCV axes) of a camera at `centre` looking at `target` (world +z up)."""
    c, p = np.asarray(centre, float), np.asarray(target, float)
    z = (p - c) / np.linalg.norm(p - c)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c
    return T


def plane_depth(K, w2c, W, H):
    """Planar depth (H, W) of the ground plane z = 0 seen by camera (K, w2c) at the pixel centres."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    ray_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    R, t = w2c[:3, :3], w2c[:3, 3]
    Cc = -R.T @ t
    d_w = ray_c @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -Cc[2] / d_w[..., 2]
    return np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)


SPECS = {
    "Simul-Robot": (3, False, True, 1, False),
    "Simul-General": (3, True, True, 1, False),
    "Real-Robot": (2, False, False, 1, False),
    "Real-General": (3, False, True, 2, True),
}
CENTRES = [(-1.0, -2.0, 2.0), (1.5, -1.8, 2.2), (0.2, 2.1, 2.5)]


def make_sample(root: Path, subset: str, sid: str = "s000", T: int = 24, W: int = 64, H: int = 48, fps: float = 10.0,
                seed: int = 0, gappy: bool = False) -> Path:
    """One synthetic sample. gappy=True invalidates the moving GT tracks on every odd frame (validity gaps)."""
    from vpm_worldbench.io.video import write_video
    import cv2
    V, moving, has_depth, div, distort = SPECS[subset]
    rng = np.random.default_rng(seed)
    d = root / subset / sid
    (d / "videos").mkdir(parents=True, exist_ok=True)
    K = np.array([[50.0, 0.0, (W - 1) / 2], [0.0, 50.0, (H - 1) / 2], [0.0, 0.0, 1.0]])
    Ks = np.repeat(K[None], V, 0)
    w2c = np.zeros((V, T, 4, 4))
    for v in range(V):
        for t in range(T):
            shift = np.array([0.05 * t, 0.0, 0.0]) if moving else np.zeros(3)
            w2c[v, t] = look_at(np.asarray(CENTRES[v]) + shift, shift)
    N = 40
    xy0 = rng.uniform(-0.6, 0.6, (N, 2))
    xyz = np.zeros((T, N, 3), np.float32)
    for t in range(T):
        xyz[t, :, :2] = xy0
        xyz[t, 30:, 0] += 0.02 * t
    valid = np.ones((T, N), bool)
    if gappy:
        valid[1::2, 30:] = False
        xyz[~valid] = np.nan
    in_fr = np.zeros((T, N, V), bool)
    for v in range(V):
        for t in range(T):
            uv, z = project(np.nan_to_num(xyz[t]), K, w2c[v, t])
            in_fr[t, :, v] = (z > 0.05) & (uv[:, 0] >= -0.5) & (uv[:, 0] < W - 0.5) & (uv[:, 1] >= -0.5) & \
                (uv[:, 1] < H - 0.5) & valid[t]
    np.savez(d / "tracks3d.npz", xyz=xyz, valid=valid, visible=in_fr.copy(), in_frustum=in_fr, default_gt=valid.copy(),
             group=np.zeros(N, np.int16), group_names=np.array(["object"]), part=np.zeros(N, np.int16),
             part_names=np.array(["plane"]), kind=np.zeros(N, np.int8), kind_names=np.array(KIND_NAMES),
             side=np.full(N, -1, np.int8))
    cams = dict(view_names=np.array([f"view{v}" for v in range(V)]),
                source_view_names=np.array([f"cam{v}" for v in range(V)]),
                image_size=np.tile(np.array([W, H], np.int32), (V, 1)), K=Ks, w2c=w2c,
                static=np.array([not moving] * V))
    if distort:
        Kr = K.copy() * 2.0
        Kr[0, 0] = Kr[1, 1] = 2.0 * 50.0 * 1.25
        Kr[0, 2], Kr[1, 2], Kr[2, 2] = W - 0.5, H - 0.5, 1.0
        cams.update(K_raw=np.repeat(Kr[None], V, 0), dist=np.zeros((V, 5)), raw_size=np.array([2 * W, 2 * H], np.int32))
    np.savez(d / "cameras.npz", **cams)
    u, vv = np.meshgrid(np.arange(W), np.arange(H))
    for v in range(V):
        fr = np.zeros((T, H, W, 3), np.uint8)
        for t in range(T):
            fr[t, ..., 0] = (u * 3 + 40 * v + 2 * t) % 256
            fr[t, ..., 1] = (vv * 4 + 70 * v) % 256
            fr[t, ..., 2] = (u + vv + 90 * v + 5 * t) % 256
        write_video(d / "videos" / f"view{v}.mp4", fr, fps=fps)
    depth_spec = None
    if has_depth:
        dw, dh = W // div, H // div
        (d / "depth").mkdir(exist_ok=True)
        with zipfile.ZipFile(d / "depth" / "depth.zip", "w") as z:
            for v in range(V):
                Kd = K.copy()
                Kd[0, 0] /= div
                Kd[1, 1] /= div
                Kd[0, 2] = (K[0, 2] + 0.5) / div - 0.5
                Kd[1, 2] = (K[1, 2] + 0.5) / div - 0.5
                for t in range(T):
                    dep = plane_depth(Kd, w2c[v, t], dw, dh)
                    ok, buf = cv2.imencode(".png", np.round(dep * 1000).astype(np.uint16))
                    z.writestr(f"cam{v}/frame_{t:04d}.png", buf.tobytes())
        depth_spec = {"file": "depth/depth.zip", "member": "{source_view}/frame_{t:04d}.png",
                      "view_member_names": [f"cam{v}" for v in range(V)], "dtype": "uint16", "unit_scale": 0.001,
                      "invalid_values": [0], "width": dw, "height": dh, "planar_z": True,
                      "kind": "rendered (synthetic GT)"}
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid,
            "source": {"dataset": "synthetic"}, "n_views": V, "n_frames": T, "fps": fps, "duration_s": T / fps,
            "views": [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W, "height": H,
                       "source_name": f"cam{v}", "role": None, "static": not moving} for v in range(V)],
            "world": {"units": "m", "handedness": "right", "up": "+z"},
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": "depth/depth.zip" if has_depth else None, "labels": {}},
            "tracks": {"n_points": N, "groups": ["object"], "default_report_group": "all"},
            "depth": depth_spec, "caption": "a textured floor", "attributes": {"difficulty": "normal"}, "flags": []}
    (d / "sample.json").write_text(json.dumps(meta))
    (d / "captions.json").write_text(json.dumps({"caption": "a textured floor"}))
    return d


def make_bench(root: Path, subsets=tuple(SPECS), **kw) -> Benchmark:
    for s in subsets:
        make_sample(Path(root), s, **kw)
    return Benchmark(root)


SMALL = dict(ref_frame=2, horizon=8, frame_step=2)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return make_bench(tmp_path_factory.mktemp("synthbench_common"))


def test_nearest_index_matches_core_warp():
    """The exact pixel-centre rule, identical to geometry.warp_image(interp='nearest') (integer arithmetic)."""
    for src, dst in ((1280, 480), (1024, 640), (512, 1024), (12, 5), (7, 7), (5, 12), (3, 2), (848, 320)):
        got = C.nearest_index(src, dst)
        centre = (np.arange(dst) + 0.5) * src / dst - 0.5
        assert np.abs(got - centre).max() <= 0.5 + 1e-9 and got.min() >= 0 and got.max() < src
        img = np.arange(src, dtype=np.int32)[None].repeat(3, 0)
        ref = warp_image(img, image_map((src, 3), (dst, 3)), interp="nearest")[0]
        assert np.array_equal(got, ref), (src, dst)


def test_resize_nearest_axes():
    x = np.arange(4 * 6 * 2).reshape(4, 6, 2)
    y = C.resize_nearest(x, (3, 2), axes=(-3, -2))
    assert y.shape == (2, 3, 2) and np.array_equal(y, x[[1, 3]][:, [1, 3, 5]])
    assert C.resize_nearest(x[..., 0], (6, 4)) is not None


def test_resize_map_bilinear_keeps_hole_footprint():
    """Missing values never mix into valid ones; a hole keeps its nearest-neighbour footprint (no dilation, no
    fill-in) and valid pixels are exact bilinear averages of the valid support."""
    x = np.full((1, 8, 8), 2.0, np.float32)
    x[0, 3, 3] = np.nan
    up = C.resize_map(x, (16, 16), "bilinear")
    nn_hole = np.isnan(C.resize_nearest(x, (16, 16)))
    assert np.array_equal(np.isnan(up), nn_hole) and nn_hole.sum() == 4
    assert np.allclose(up[np.isfinite(up)], 2.0)
    ramp = np.tile(np.arange(8, dtype=np.float32), (8, 1))[None]
    ramp[0, :, 0] = np.nan
    down = C.resize_map(ramp, (4, 4), "bilinear")
    assert np.array_equal(np.isnan(down), np.isnan(C.resize_nearest(ramp, (4, 4))))
    assert np.isfinite(down[0, :, 0]).all() and np.allclose(down[0, 0], [1.0, 2.5, 4.5, 6.5])
    full = np.random.default_rng(0).random((2, 6, 10)).astype(np.float32)
    import cv2
    assert np.allclose(C.resize_map(full, (5, 3))[1], cv2.resize(full[1], (5, 3), interpolation=cv2.INTER_LINEAR))
    nn = C.resize_map(x, (4, 4), "nearest")
    assert nn.shape == (1, 4, 4) and np.isnan(nn).sum() <= 1
    with pytest.raises(ValueError):
        C.resize_map(x, (4, 4), "cubic")


def test_sample_map_nearest_and_bilinear():
    img = np.arange(12, dtype=float).reshape(3, 4)
    uv = np.array([[1.0, 1.0], [1.4, 0.6], [1.5, 1.5], [-0.6, 0.0], [3.6, 0.0]])
    nn = C.sample_map(img, uv, "nearest")
    assert nn[:3].tolist() == [5.0, 5.0, 10.0] and np.isnan(nn[3:]).all()
    bl = C.sample_map(img, uv[:3], "bilinear")
    assert np.allclose(bl, 4 * uv[:3, 1] + uv[:3, 0])
    img[1, 2] = np.nan
    assert np.isnan(C.sample_map(img, np.array([[1.5, 1.5]]), "bilinear")[0])
    assert C.sample_map(img, np.array([[1.0, 1.0]]), "bilinear")[0] == 5.0


def test_json_safe_and_mean_se():
    s = C.json_safe({"a": np.float32(1.5), "b": float("nan"), "c": [np.inf, -np.inf], "d": np.arange(2)})
    assert s == {"a": 1.5, "b": None, "c": ["inf", "-inf"], "d": [0, 1]}
    json.dumps(s, allow_nan=False)
    m = C.mean_se([1.0, 2.0, 3.0, None, "inf"])
    assert m["mean"] == pytest.approx(2.0) and m["se"] == pytest.approx(1.0 / np.sqrt(3))
    assert (m["n"], m["n_undefined"], m["n_inf"]) == (3, 1, 1)


def test_protocol_subset_aliases_and_unknown_subsets(tmp_path):
    from vpm_worldbench.tasks.depth import DepthProtocol
    p = DepthProtocol(eval_size={"sr": (32, 32), "real_general": [16, 12]}, max_depth={"RG": 5.0})
    assert p.eval_size == {"Simul-Robot": (32, 32), "Real-General": [16, 12]}
    assert p.eval_size_for("Simul-Robot") == (32, 32) and p.eval_size_for("rg") == (16, 12)
    assert p.max_depth_for("Real-General") == 5.0 and p.max_depth_for("Simul-Robot") == 80.0
    p.save(tmp_path / "p.json")
    assert DepthProtocol.load(tmp_path / "p.json").to_dict() == p.to_dict()
    (tmp_path / "short_window.yaml").write_text("horizon: 4\nframe_step: 2\n")
    q = DepthProtocol.load(tmp_path / "short_window.yaml")
    assert q.name == "short_window" and q.horizon == 4 and q.frame_step == 2
    (tmp_path / "empty.json").write_text("{}")
    assert DepthProtocol.load(tmp_path / "empty.json").name == "empty"
    (tmp_path / "null_name.yaml").write_text("name:\nhorizon: 4\n")
    assert DepthProtocol.load(tmp_path / "null_name.yaml").name == "null_name"
    (tmp_path / "list.yaml").write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        DepthProtocol.load(tmp_path / "list.yaml")
    with pytest.raises(ValueError, match="unknown subset"):
        DepthProtocol(eval_size={"Simul-Robots": (8, 8)})
    with pytest.raises(ValueError, match="twice"):
        DepthProtocol(max_depth={"sr": 1.0, "Simul-Robot": 2.0})
    with pytest.raises(KeyError):
        DepthProtocol.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        C.TaskProtocol(frame_step=0)


def test_fov_mask(bench):
    assert C.fov_mask(bench.sample("Simul-Robot/s000"), 0) is None
    s = bench.sample("Real-General/s000")
    m = C.fov_mask(s, 0)
    assert m.shape == (48, 64) and m[24, 32] and not m[0, 0] and not m.all()
    u = np.arange(64) - 31.5
    assert np.array_equal(m[24], np.abs(u) * 125 / 50 <= 63.5 + 1e-9)
    assert C.fov_mask(s, 0, (32, 24)).shape == (24, 32)
    lb = C.fov_mask(s, 0, (64, 64), "letterbox")
    assert not lb[:8].any() and not lb[-8:].any() and lb[32, 32]


def test_track_observations(bench):
    s = bench.sample("Simul-Robot/s000")
    obs = C.track_observations(s, 1, [2, 5])
    uv, z = project(s.tracks.xyz[5], s.cameras.K[1], s.cameras.w2c[1, 5])
    m = obs.mask[1]
    assert m.sum() > 10 and np.allclose(obs.uv[1, m], uv[m]) and np.allclose(obs.z[1, m], z[m])
    assert np.isnan(obs.uv[~obs.mask]).all() and np.array_equal(obs.mask[1], s.tracks.visible[5, :, 1])
    with pytest.raises(ValueError):
        C.track_observations(s, 1, [2], visibility="maybe")


def test_gt_depth_grid_and_consistency_filter(bench):
    s = bench.sample("Real-General/s000")
    g = C.gt_depth(s, 0, [3], (64, 48))
    assert g.shape == (1, 48, 64) and (g > 0).mean() > 0.9
    assert np.array_equal(g, s.depth(0, [3]))
    assert np.array_equal(C.gt_depth(s, 0, [3, 4], (40, 30)), s.depth(0, [3, 4], size=(40, 30)))
    obs = C.track_observations(s, 0, [3, 4])
    _, keep = C.read_gt_depth(s, 0, [3, 4], None, obs, (0.05, 0.05))
    assert keep.sum() == obs.mask.sum()
    obs.z[0, np.flatnonzero(obs.mask[0])[0]] += 1.0
    _, keep = C.read_gt_depth(s, 0, [3, 4], None, obs, (0.05, 0.05))
    assert keep.sum() == obs.mask.sum() - 1
    assert C.read_gt_depth(bench.sample("Real-Robot/s000"), 0, [3], None)[0] is None


def test_track_path_length_bridges_gaps():
    xyz = np.zeros((6, 4, 3))
    xyz[:, 0, 0] = np.arange(6) * 0.1
    xyz[:, 1, 0] = np.arange(6) * 0.1
    xyz[:, 3, 0] = np.arange(6) * 0.1
    valid = np.ones((6, 4), bool)
    valid[1::2, 1] = False
    valid[:, 2] = False
    valid[2, 2] = True
    valid[:, 3] = [True, False, False, False, False, False]
    xyz[~valid] = np.nan
    path, nv = C.track_path_length(xyz, valid)
    assert np.allclose(path, [0.5, 0.4, 0.0, 0.0]) and nv.tolist() == [6, 3, 1, 1]
    xyz2 = xyz.copy()
    xyz2[5, 0] = np.nan
    assert C.track_path_length(xyz2, np.ones((6, 4), bool))[0][0] == pytest.approx(0.4)


def _dummy_eval(sample, pred_root, protocol):
    p = Path(pred_root) / sample.subset / sample.id / "x.txt"
    if not p.exists():
        raise C.MissingPrediction(str(p))
    return {"status": "ok", "metrics": {"value": float(p.read_text()), "fail": float("inf")}}


def _dummy_summary(out_dir):
    return C.summarize_records(out_dir, "dummy", "dummy", [("value", "value", 2), ("fail", "fail", 2)])


def _write(pred, uid, value):
    p = Path(pred) / uid / "x.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def _bad_eval(sample, pred_root, protocol):
    raise C.InvalidPrediction(f"{sample.uid}: broken file")


def test_status_vocabulary_and_records_of_earlier_versions(bench, tmp_path):
    """Statuses ok / skipped / missing / invalid / error; records written with the earlier names not_evaluable /
    missing_prediction are reused and summarised under the new names."""
    ev = dict(progress=False, summarize_fn=_dummy_summary)
    proto = C.TaskProtocol(**SMALL)
    _write(tmp_path / "a", "Simul-Robot/s000", 1.0)
    s = C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res",
                        ids=["Simul-Robot/s000", "Real-Robot/s000"], **ev)
    assert s["subsets"]["Simul-Robot"]["status"] == {"ok": 1} and s["subsets"]["Real-Robot"]["status"] == {"missing": 1}
    assert (s["subsets"]["Real-Robot"]["n_invalid"], s["subsets"]["Real-Robot"]["n_error"]) == (0, 0)
    assert s["evaluated"] == {"n_samples": 2, "status": {"missing": 1, "ok": 1}, "n_failed": 0, "all_failed": False,
                              "unmatched_ids": []}
    assert "evaluated" not in json.loads((tmp_path / "res" / "summary.json").read_text())
    long = C.TaskProtocol(ref_frame=2, horizon=30)
    s = C.evaluate_task("dummy", _dummy_eval, long, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res2",
                        ids=["Simul-Robot/s000"], **ev)
    assert s["subsets"]["Simul-Robot"]["status"] == {"skipped": 1}
    p = tmp_path / "res2" / "Simul-Robot" / "s000.json"
    r = json.loads(p.read_text())
    p.write_text(json.dumps(dict(r, status="not_evaluable")))
    s = C.evaluate_task("dummy", _dummy_eval, long, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res2",
                        ids=["Simul-Robot/s000"], **ev)
    assert s["subsets"]["Simul-Robot"]["status"] == {"skipped": 1} and s["evaluated"]["status"] == {"skipped": 1}
    assert json.loads(p.read_text())["status"] == "not_evaluable"
    p.write_text(json.dumps(dict(r, status="missing_prediction")))
    assert _dummy_summary(tmp_path / "res2")["subsets"]["Simul-Robot"]["status"] == {"missing": 1}
    assert C.canonical_status("not_evaluable") == "skipped" and C.canonical_status("partial") == "partial"
    with pytest.warns(UserWarning, match="2 sample.*broken file"):
        s = C.evaluate_task("dummy", _bad_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res3",
                            ids=["Simul-Robot/s000", "Simul-General/s000"], **ev)
    assert s["subsets"]["Simul-Robot"]["status"] == {"invalid": 1} and s["evaluated"]["all_failed"]
    assert (s["subsets"]["Simul-Robot"]["n_invalid"], s["subsets"]["Simul-Robot"]["n_error"]) == (1, 0)
    with pytest.raises(C.InvalidPrediction):
        C.evaluate_task("dummy", _bad_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res3",
                        ids=["Simul-Robot/s000"], strict=True, overwrite=True, **ev)
    rs = C.run_status(["ok", "error", "skipped"], ["nope"])
    assert rs["n_failed"] == 1 and not rs["all_failed"] and rs["unmatched_ids"] == ["nope"]
    assert C.run_status(["missing_prediction", "error"])["all_failed"] and C.run_status(["missing"])["all_failed"]
    assert C.run_status(["missing", "skipped"])["status"] == {"missing": 1, "skipped": 1}
    assert not C.run_status(["skipped", "not_evaluable"])["all_failed"] and not C.run_status([])["all_failed"]
    assert not C.run_status(["partial", "missing", "error"])["all_failed"]
    with pytest.warns(UserWarning, match="match no sample"):
        s = C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res",
                            ids=["Simul-Robot/s000", "Real-Robot/s000", "rr/nope"], **ev)
    assert s["evaluated"]["status"] == {"missing": 1, "ok": 1} and not s["evaluated"]["all_failed"]
    assert s["evaluated"]["unmatched_ids"] == ["rr/nope"]


def test_ids_with_subset_aliases_and_unmatched_ids(bench):
    assert C.canonical_ids(None) is None
    assert C.canonical_ids(["sr/s000", "s001", " rg/x ", "Mars/s0"]) == ["Simul-Robot/s000", "s001", "Real-General/x",
                                                                          "Mars/s0"]
    assert [s.uid for s in C.iter_samples(bench, ["sr", "rr"], ["sr/s000"])] == ["Simul-Robot/s000"]
    with pytest.warns(UserWarning, match=r"2 requested id\(s\) match no sample") as rec:
        assert C.select_samples(bench, ["rr"], ["sr/s000", "nope"]) == []
    assert len([w for w in rec if "match no sample" in str(w.message)]) == 1
    assert len(C.select_samples(bench, None, ["s000"])) == 4
    assert [s.uid for s in C.select_samples(bench, "rg", ["s000"])] == ["Real-General/s000"]


def test_transposed_hint_and_read_npz(tmp_path):
    assert "swapped" in C.transposed_hint((48, 64), (64, 48))
    assert C.transposed_hint((64, 48), (64, 48)) is None and C.transposed_hint((128, 96), (64, 48)) is None
    assert C.transposed_hint((32, 32), (64, 48)) is None
    assert C.transposed_hint((48, 48), (48, 48)) is None
    np.savez(tmp_path / "ok.npz", a=np.arange(3), b=np.ones(2))
    names, z = C.read_npz(tmp_path / "ok.npz", ("a",))
    assert names == ["a", "b"] and list(z) == ["a"] and z["a"].tolist() == [0, 1, 2]
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    np.save(tmp_path / "plain.npy", np.ones(3))
    np.savez(tmp_path / "obj.npz", a=np.array([{"x": 1}], dtype=object))
    for f in ("bad.npz", "plain.npy", "obj.npz"):
        with pytest.raises(C.InvalidPrediction):
            C.read_npz(tmp_path / f)
    assert C.frame_index_array(np.array([2.0, 4.0]), "x").tolist() == [2, 4]
    for bad in (np.array([2.5]), np.array([[1, 2]]), np.array([True])):
        with pytest.raises(C.InvalidPrediction):
            C.frame_index_array(bad, "x")


def test_override_protocol_names_the_change():
    p = C.TaskProtocol(**SMALL)
    assert C.override_protocol(p, {"horizon": 8}) is p
    q = C.override_protocol(p, {"horizon": 4, "frame_step": 2, "views": [0, 1]})
    assert q.name == "task+horizon=4,views=[0,1]" and q.horizon == 4 and q.views == [0, 1]


def test_evaluate_task_ties_results_to_one_run(bench, tmp_path):
    """Results are reused only for the same run (protocol + prediction root) and unchanged prediction files; another
    run in the same folder raises unless overwrite, which drops the stale per-sample results."""
    proto = C.TaskProtocol(**SMALL)
    ids = ["Simul-Robot/s000", "Real-Robot/s000"]
    ev = dict(subsets=None, ids=ids, progress=False, summarize_fn=_dummy_summary)
    for uid in ids:
        _write(tmp_path / "a", uid, 1.0)
        _write(tmp_path / "b", uid, 2.0)
    s = C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res", **ev)
    assert s["subsets"]["Simul-Robot"]["metrics"]["value"]["mean"] == 1.0
    assert s["subsets"]["Simul-Robot"]["metrics"]["fail"]["n_inf"] == 1
    assert "(1 inf)" in (tmp_path / "res" / "summary.md").read_text()
    run = json.loads((tmp_path / "res" / "run.json").read_text())
    rec = json.loads((tmp_path / "res" / "Simul-Robot" / "s000.json").read_text())
    assert rec["run_id"] == run["run_id"] and rec["pred_stamp"] and run["pred_root"] == str((tmp_path / "a").resolve())
    assert s["pred_root"] == run["pred_root"] == C.run_pred_root(tmp_path / "res")
    with pytest.raises(C.RunDirError, match="predictions in") as e:
        C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "b", tmp_path / "res", **ev)
    assert isinstance(e.value, ValueError) and isinstance(e.value, UsageError) and "--overwrite" in str(e.value)
    exp = tmp_path / "exp"
    proto.save(exp / "protocol.json")
    with pytest.raises(C.RunDirError, match="holds a protocol.json without run.json"):
        C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "a", exp, **ev)
    with pytest.raises(C.RunDirError, match="another protocol"):
        C.evaluate_task("dummy", _dummy_eval, C.TaskProtocol(**dict(SMALL, horizon=4)), C.TaskProtocol, bench,
                        tmp_path / "a", tmp_path / "res", **ev)
    import os
    import time
    _write(tmp_path / "a", "Simul-Robot/s000", 3.0)
    t = time.time() + 5
    os.utime(tmp_path / "a" / "Simul-Robot" / "s000" / "x.txt", (t, t))
    s = C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "a", tmp_path / "res", **ev)
    assert s["subsets"]["Simul-Robot"]["metrics"]["value"]["mean"] == 3.0
    s = C.evaluate_task("dummy", _dummy_eval, proto, C.TaskProtocol, bench, tmp_path / "b", tmp_path / "res",
                        overwrite=True, **dict(ev, ids=["Simul-Robot/s000"]))
    assert s["subsets"]["Simul-Robot"]["metrics"]["value"]["mean"] == 2.0 and "Real-Robot" not in s["subsets"]
    assert not (tmp_path / "res" / "Real-Robot" / "s000.json").exists()


def test_task_cli_options(bench, tmp_path, capsys):
    """The uniform CLI: --kind, comma / space separated --subsets (aliases) and --ids, window overrides."""
    from vpm_worldbench.tasks import depth as D
    root = str(bench.root)
    pf = tmp_path / "p.json"
    D.DepthProtocol(eval_size={s: (32, 24) for s in SPECS}, align=("none",), **SMALL).save(pf)
    assert D.main(["baseline", "--kind", "constant", "--root", root, "--pred", str(tmp_path / "pred"),
                   "--protocol", str(pf), "--subsets", "sr,rr"]) == 0
    assert sorted(p.name for p in (tmp_path / "pred").iterdir()) == ["Real-Robot", "Simul-Robot"]
    assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res"),
                   "--protocol", str(pf), "--ids", "Simul-Robot/s000", "Real-Robot/s000", "--subsets", "sr", "rr"]) == 0
    sm = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(sm["subsets"]) == {"Simul-Robot", "Real-Robot"}
    assert D.main(["summarize", "--out", str(tmp_path / "res")]) == 0
    assert D.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--frames", "none", "--subsets", "sg",
                   "--horizon", "4", "--frame-step", "3", "--ref-frame", "1"]) == 0
    rec = json.loads((tmp_path / "in" / "Simul-General" / "s000" / "inputs.json").read_text())
    assert rec["frame_indices"] == [1, 4, 7, 10, 13]
    pr = json.loads((tmp_path / "in" / "protocol.json").read_text())
    assert pr["name"] == "depth-v1+ref_frame=1,horizon=4,frame_step=3"
    with pytest.raises(SystemExit) as e:
        D.main(["baseline", "oracle", "--root", root, "--pred", str(tmp_path / "p2")])
    assert e.value.code == 2
    capsys.readouterr()
    for args, msg in ((["--subsets", "nope"], "unknown subset 'nope'"),
                      (["--protocol", "legacy-prerelease"], "neither the preset 'depth-v1'"),
                      (["--horizon", "0"], "must be >= 1"),
                      (["--frame-step", "0"], "frame_step >= 1")):
        with pytest.raises(SystemExit) as e:
            D.main(["export", "--root", root, "--out", str(tmp_path / "x"), *args])
        assert e.value.code == 2 and msg in capsys.readouterr().err, args
    with pytest.warns(UserWarning, match="match no sample"):
        assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pred"), "--out", str(tmp_path / "res0"),
                       "--protocol", str(pf), "--ids", "Real-Robot/nope"]) == 1
    assert "no sample matches the selection" in capsys.readouterr().err
    bad = tmp_path / "bad" / "Simul-Robot" / "s000"
    bad.mkdir(parents=True)
    for k in range(3):
        np.savez(bad / f"view{k}.npz", depth=np.ones((3, 48, 64), np.float32))
    with pytest.warns(UserWarning, match="1 sample"):
        assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "bad"), "--out", str(tmp_path / "res1"),
                       "--protocol", str(pf), "--subsets", "sr", "rr"]) == 1
    err = capsys.readouterr().err
    assert "every evaluated sample failed (invalid 1, missing 1)" in err
    assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "nowhere"), "--out", str(tmp_path / "res2"),
                   "--protocol", str(pf), "--subsets", "sr", "rr"]) == 1
    assert "every evaluated sample failed (missing 2)" in capsys.readouterr().err
    assert D.main(["evaluate", "--root", root, "--pred", str(tmp_path / "nowhere"), "--out", str(tmp_path / "res3"),
                   "--protocol", str(pf), "--subsets", "sr", "--ref-frame", "20"]) == 0
    assert json.loads((tmp_path / "res3" / "summary.json").read_text())["subsets"]["Simul-Robot"]["status"] == \
        {"skipped": 1}
    p = D.DepthProtocol.load(pf)
    assert D.run_baseline(bench, tmp_path / "f1", kind="constant", protocol=p, subsets=["sr"], progress=False,
                          value=2.0) == {"written": 1, "skipped": 0}
    assert D.run_baseline("constant", bench, tmp_path / "f2", p, subsets=["sr"], progress=False, value=2.0) == \
        {"written": 1, "skipped": 0}
    a, b = (np.load(tmp_path / f / "Simul-Robot" / "s000" / "view0.npz")["depth"] for f in ("f1", "f2"))
    assert np.array_equal(a, b) and a.max() == 2.0
    with pytest.raises(TypeError, match="needs a vpm_worldbench.Benchmark"):
        D.run_baseline(root, tmp_path / "f3", kind="constant", protocol=p)
    from vpm_worldbench import cli
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli.main(["depth", "evaluate", "--help"])
    assert e.value.code == 0 and "usage: vpmwb depth evaluate" in capsys.readouterr().out


def test_check_out_stamp_names_what_differs(tmp_path):
    """The stamp check of segmentation / understanding (run.json, export.json): RunDirError (a ValueError and a
    utils.UsageError) naming what differs and --overwrite; overwrite returns True (the caller then clears the dir)."""
    run = {"task": "t", "protocol": {"a": 1}, "pred_root": "/x"}
    assert C.check_out_stamp(tmp_path, "run.json", run, False, "t") is False
    C.write_json(tmp_path / "run.json", run)
    assert C.check_out_stamp(tmp_path, "run.json", dict(run), False, "t") is False
    with pytest.raises(C.RunDirError, match=r"holds t results of the predictions in /x \(run.json\); .*--overwrite"):
        C.check_out_stamp(tmp_path, "run.json", dict(run, pred_root="/y"), False, "t")
    with pytest.raises(C.RunDirError, match="results of another protocol"):
        C.check_out_stamp(tmp_path, "run.json", dict(run, protocol={"a": 2}), False, "t")
    with pytest.raises(C.RunDirError, match=r"results of another run \(run.json: other answer_key\)"):
        C.check_out_stamp(tmp_path, "run.json", dict(run, answer_key="k"), False, "t")
    assert C.check_out_stamp(tmp_path, "run.json", dict(run, pred_root="/y"), True, "t") is True
    assert C.run_pred_root(tmp_path) == "/x" and C.run_pred_root(tmp_path / "nope") is None
    C.write_json(tmp_path / "export.json", {"task": "t", "frames": "none", "protocol": {"a": 1}})
    with pytest.raises(C.RunDirError, match=r"inputs of other export options \(export.json: other frames\)"):
        C.check_out_stamp(tmp_path, "export.json", {"task": "t", "frames": "png", "protocol": {"a": 1}}, False, "t")
    (tmp_path / "export.json").write_text("{not json")
    with pytest.raises(C.RunDirError, match="unreadable export.json"):
        C.check_out_stamp(tmp_path, "export.json", {"task": "t"}, False, "t")


def test_task_cli_setup_errors_are_one_line(bench, tmp_path, capsys, monkeypatch):
    """An out dir of another run (`RunDirError`) ends `vpmwb` with one line `vpmwb: error: ...` and exit status 2
    (VPMWB_TRACEBACK=1: the traceback), also behind `python -m vpm_worldbench.tasks.<task>`, and nothing is deleted; a
    malformed --size is a usage error (exit status 2) before anything is written."""
    import os
    import subprocess
    import sys

    from vpm_worldbench import cli
    from vpm_worldbench.tasks import depth as D
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    pf = tmp_path / "p.json"
    D.DepthProtocol(eval_size={s: (32, 24) for s in SPECS}, align=("none",), **SMALL).save(pf)
    common = ["--root", str(bench.root), "--protocol", str(pf), "--subsets", "sr"]
    for kind in ("oracle", "constant"):
        assert D.main(["baseline", "--kind", kind, "--pred", str(tmp_path / kind), *common]) == 0
    out = tmp_path / "res"
    assert D.main(["evaluate", "--pred", str(tmp_path / "oracle"), "--out", str(out), *common]) == 0
    before = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*.json")}
    other = ["evaluate", "--pred", str(tmp_path / "constant"), "--out", str(out), *common]
    with pytest.raises(C.RunDirError, match="--overwrite"):
        D.main(other)
    capsys.readouterr()
    assert cli.main(["depth", *other]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"vpmwb: error: {out} holds depth results of the predictions in {tmp_path / 'oracle'}")
    assert "--overwrite" in err and len(err.strip().splitlines()) == 1
    monkeypatch.setenv("VPMWB_TRACEBACK", "1")
    with pytest.raises(C.RunDirError):
        cli.main(["depth", *other])
    monkeypatch.setenv("VPMWB_TRACEBACK", "0")
    assert cli.main(["depth", *other]) == 2
    monkeypatch.delenv("VPMWB_TRACEBACK")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(C.__file__).resolve().parents[2])] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    r = subprocess.run([sys.executable, "-m", "vpm_worldbench.tasks.depth", *other], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 2 and r.stderr.startswith("vpmwb: error: ") and "--overwrite" in r.stderr, r.stderr
    assert "Traceback" not in r.stderr and len(r.stderr.strip().splitlines()) == 1
    assert {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*.json")} == before
    for size in ("12x", "abc", "0x48", "64x48x2"):
        with pytest.raises(SystemExit) as e:
            D.main(["export", "--out", str(tmp_path / "in"), "--size", size, *common])
        assert e.value.code == 2 and f"--size must be WxH (e.g. 640x480), got '{size}'" in \
            capsys.readouterr().err, size
    assert not (tmp_path / "in").exists()
    with pytest.raises(SystemExit) as e:
        D.main(["summarize", "--out", str(tmp_path / "in")])
    assert e.value.code == 2 and "no such directory" in capsys.readouterr().err and not (tmp_path / "in").exists()
    assert D.main(["export", "--out", str(tmp_path / "in"), "--size", "32x24", "--frames", "none", *common]) == 0
    rec = json.loads((tmp_path / "in" / "Simul-Robot" / "s000" / "inputs.json").read_text())
    assert rec["frames"]["view0"]["size"] == [32, 24]
