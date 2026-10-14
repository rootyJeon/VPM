"""Core tests: geometry, video IO, protocol, predictions, model inputs, the Sample API, the validator, the index and the
usage / setup error contract of `vpmwb` (one line `vpmwb: error: ...`, exit status 2, VPMWB_TRACEBACK).

Synthetic data only (the mini benchmark of test_common.make_sample, written to tmp_path). With VPMWB_RELEASED_ROOT set
to a VP-WorldBench data root, `test_released_*` also validate a few released samples (read-only)."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_common import look_at, make_sample
from vpm_worldbench import Benchmark, Protocol, get_protocol
from vpm_worldbench.geometry import (camera_center, image_map, inside, project, rotation_angle_deg, umeyama,
                                     unproject_depth, warp_image)
from vpm_worldbench.index import build_index
from vpm_worldbench.inputs import camera_phrase, export_inputs, prompts, window_caption
from vpm_worldbench.io.video import frame_to_rgb, iter_frames, read_frames, video_info, write_video
from vpm_worldbench.predictions import load_prediction, load_reference
from vpm_worldbench.protocol import legacy_option
from vpm_worldbench.validate import validate, validate_sample

RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")


@pytest.fixture()
def mini(tmp_path):
    """A one-sample benchmark (Simul-Robot layout: 3 static views, dense depth), 24 frames of 64x48."""
    make_sample(tmp_path, "Simul-Robot", "s000")
    build_index(tmp_path)
    return Benchmark(tmp_path)


SMALL = Protocol(ref_frame=4, horizon=8, frame_step=2)


@pytest.mark.parametrize("mode", ["resize", "center_crop", "letterbox"])
def test_image_map_keeps_projection(mode):
    """Projecting with the mapped K equals mapping the projected pixel (pixel-centre convention)."""
    K = np.array([[300.0, 0, 319.5], [0, 310.0, 239.5], [0, 0, 1]])
    w2c = look_at([0.3, -2.0, 1.0], [0, 0, 0.5])
    X = np.random.default_rng(0).uniform(-0.5, 0.5, (50, 3)) + [0, 0, 0.5]
    uv, z = project(X, K, w2c)
    for dst in [(320, 240), (480, 480), (848, 480), (101, 77)]:
        m = image_map((640, 480), dst, mode)
        uv2, _ = project(X, m.apply_K(K), w2c)
        np.testing.assert_allclose(m.apply_uv(uv), uv2, atol=1e-9)


def test_image_map_pixel_centres():
    m = image_map((640, 480), (320, 240), "resize")
    np.testing.assert_allclose(m.apply_uv(np.array([[-0.5, -0.5], [639.5, 479.5], [0.0, 0.0]])),
                               [[-0.5, -0.5], [319.5, 239.5], [-0.25, -0.25]])
    img = np.zeros((480, 640, 3), np.uint8)
    for mode, shape in [("resize", (480, 480, 3)), ("center_crop", (480, 480, 3)), ("letterbox", (480, 480, 3))]:
        assert warp_image(img, image_map((640, 480), (480, 480), mode)).shape == shape


def test_project_unproject_roundtrip():
    K = np.array([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]])
    w2c = look_at([1.0, -2.0, 1.5], [0, 0, 0])
    depth = np.full((48, 64), 2.5)
    X = unproject_depth(depth, K, w2c).reshape(-1, 3)
    uv, z = project(X, K, w2c)
    u, v = np.meshgrid(np.arange(64.0), np.arange(48.0))
    np.testing.assert_allclose(uv, np.stack([u.ravel(), v.ravel()], -1), atol=1e-9)
    np.testing.assert_allclose(z, 2.5, atol=1e-9)
    assert inside(np.array([[-0.5, -0.5], [63.49, 47.49], [63.5, 0.0], [-0.51, 0.0]]), (64, 48)).tolist() == \
        [True, True, False, False]


def test_umeyama_and_camera_center():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(100, 3))
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    dst = 2.5 * src @ R.T + [1.0, -2.0, 0.5]
    s, R2, t = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9 and rotation_angle_deg(R, R2) < 1e-4
    np.testing.assert_allclose(t, [1.0, -2.0, 0.5], atol=1e-9)
    np.testing.assert_allclose(camera_center(look_at([1, 2, 3], [0, 0, 0])), [1, 2, 3], atol=1e-12)


def test_video_lossless_roundtrip(tmp_path):
    fr = np.random.default_rng(0).integers(0, 256, (5, 24, 32, 3), dtype=np.uint8)
    p = write_video(tmp_path / "a.mp4", fr, fps=15, lossless=True)
    assert video_info(p) == {"width": 32, "height": 24, "fps": 15.0, "n_frames": 5}
    np.testing.assert_array_equal(read_frames(p), fr)
    np.testing.assert_array_equal(read_frames(p, [4, 0, 4]), fr[[4, 0, 4]])
    with pytest.raises(IndexError):
        read_frames(p, [5])
    assert sum(1 for _ in iter_frames(p)) == 5


def test_yuv_conversion_formula(tmp_path):
    """yuv420p -> RGB: chroma replicated over 2x2 blocks, BT.601 limited range, round half up (module docstring)."""
    import av
    fr = np.random.default_rng(2).integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    p = write_video(tmp_path / "b.mp4", fr, fps=10)
    with av.open(str(p)) as c:
        f = next(c.decode(video=0))
        assert f.format.name == "yuv420p"
        y, u, v = [np.frombuffer(pl, np.uint8).reshape(pl.height, pl.line_size)[:, :pl.width].astype(float)
                   for pl in f.planes]
        rgb = frame_to_rgb(f)
    u = np.repeat(np.repeat(u, 2, 0), 2, 1) - 128
    v = np.repeat(np.repeat(v, 2, 0), 2, 1) - 128
    y = 1.164383562 * (y - 16)
    ref = np.stack([y + 1.596026786 * v, y - 0.391762290 * u - 0.812967647 * v, y + 2.017232143 * u], -1)
    np.testing.assert_array_equal(rgb, np.clip(np.floor(ref + 0.5), 0, 255).astype(np.uint8))


def test_protocol_defaults_and_variants():
    p = Protocol()
    assert p.context_indices == [16] and p.eval_indices == list(range(17, 81)) and p.window_indices[0] == 16
    assert p.met3r_positions() == list(range(4, 65, 4))
    assert p.eval_size_for("Real-Robot") == (848, 480) and p.eval_size_for("Simul-Robot") == (480, 480)
    v2v = Protocol(context_frames=9)
    assert v2v.context_indices == list(range(8, 17)) and v2v.eval_indices == p.eval_indices
    st = Protocol(ref_frame=32, horizon=32, frame_step=2)
    assert st.eval_indices[:2] == [34, 36] and st.last_frame == 96
    assert st.met3r_positions() == list(range(2, 33, 2))
    with pytest.raises(ValueError):
        Protocol(context_frames=0)
    with pytest.raises(ValueError):
        Protocol(text="bogus")


def test_protocol_io(tmp_path):
    p = Protocol(ref_frame=8, context_frames=3, eval_size={"Real-General": (512, 384)}, clip_filter={"difficulty": "hard"})
    p.save(tmp_path / "p.json")
    q = Protocol.load(tmp_path / "p.json")
    assert q.to_dict() == p.to_dict() and q.eval_size_for("Real-General") == (512, 384)
    (tmp_path / "p.yaml").write_text("ref_frame: 0\nhorizon: 80\nviews: [0, 1]\n")
    y = get_protocol(str(tmp_path / "p.yaml"))
    assert y.window_indices == list(range(0, 81)) and y.views == [0, 1]
    with pytest.raises(KeyError):
        Protocol.from_dict({"ref_frame": 3, "typo": 1})
    leg = get_protocol("legacy-prerelease")
    assert leg.rgb_scale_quirk and legacy_option(leg, "Real-General", "fvd_windows") == 2
    assert legacy_option(Protocol(), "Real-General", "fvd_windows", "x") == "x"


def test_evaluable(mini):
    s = mini.sample("Simul-Robot/s000")
    assert SMALL.evaluable(s)[0]
    assert not Protocol().evaluable(s)[0]
    assert not Protocol(ref_frame=4, context_frames=6, horizon=2).evaluable(s)[0]
    assert not Protocol(ref_frame=4, horizon=2, views=[3]).evaluable(s)[0]


def test_sample_api(mini):
    s = mini.sample("Simul-Robot", "s000")
    assert (s.n_views, s.n_frames, s.fps, s.image_size(0)) == (3, 24, 10.0, (64, 48))
    fr = s.frames(1, [0, 5, 5], size=(32, 24))
    assert fr.shape == (3, 24, 32, 3) and np.array_equal(fr[1], fr[2])
    uv, vis, inf = s.tracks2d(0, size=(32, 24))
    K = s.K(0, (32, 24))
    uv_ref, _ = project(s.tracks.xyz[7], K, s.w2c(0, [7])[0])
    np.testing.assert_allclose(uv[7], uv_ref, atol=1e-5)
    assert vis.shape == (24, s.tracks.n_points) and (vis <= inf).all()
    d = s.depth(0, [3], size=(32, 24))
    assert d.shape == (1, 24, 32) and (d > 0).any()
    assert s.tracks.group_mask("all").all() and s.tracks.group_mask("object").all()


def _write_pred(root, sample, frames_per_view, meta=None):
    d = Path(root) / sample.subset / sample.id
    d.mkdir(parents=True, exist_ok=True)
    for k, fr in enumerate(frames_per_view):
        write_video(d / f"view{k}.mp4", fr, fps=sample.fps, lossless=True)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))


def test_prediction_grid_and_meta(mini, tmp_path):
    s = mini.sample("Simul-Robot/s000")
    p = Protocol(ref_frame=4, horizon=8, frame_step=2, eval_size={"Simul-Robot": (64, 48)})
    real = [load_reference(s, k, p) for k in range(3)]
    assert real[0].shape == (9, 48, 64, 3)
    _write_pred(tmp_path / "a", s, real)
    np.testing.assert_array_equal(load_prediction(tmp_path / "a", s, 0, p), real[0])
    _write_pred(tmp_path / "b", s, [r[1:] for r in real], {"first_frame": "future"})
    np.testing.assert_array_equal(load_prediction(tmp_path / "b", s, 1, p, ref_frame=real[1][0]), real[1])
    fine = np.repeat(s.frames(2, range(4, 24), size=(64, 48)), 2, axis=0)
    _write_pred(tmp_path / "c", s, [fine] * 3, {"fps": 20.0})
    np.testing.assert_array_equal(load_prediction(tmp_path / "c", s, 2, p), real[2])
    _write_pred(tmp_path / "d", s, [np.repeat(np.repeat(r, 2, 1), 2, 2) for r in real])
    assert load_prediction(tmp_path / "d", s, 0, p).shape == (9, 48, 64, 3)
    _write_pred(tmp_path / "e", s, [r[:5] for r in real])
    with pytest.raises(ValueError):
        load_prediction(tmp_path / "e", s, 0, p)


def test_camera_phrase():
    a = look_at([0, -3, 1], [0, 0, 1])
    assert camera_phrase(a, a) == "The camera is static."
    left = look_at([0, -3, 1], [-1.5, 0, 1])
    right = look_at([0, -3, 1], [1.5, 0, 1])
    assert "panning left" in camera_phrase(a, left) and "panning right" in camera_phrase(a, right)
    assert camera_phrase(a, look_at([0, -1, 1], [0, 2, 1])).startswith("The camera travels about 2.0 m")


def test_export_inputs(mini, tmp_path):
    p = Protocol(ref_frame=4, context_frames=3, horizon=8, frame_step=2, text="caption+camera", camera_conditioning=True)
    rep = export_inputs(mini, tmp_path / "in", p)
    assert rep == {"exported": 1, "skipped": 0}
    rec = json.loads((tmp_path / "in" / "Simul-Robot" / "s000" / "inputs.json").read_text())
    assert rec["context_indices"] == [2, 3, 4] and rec["n_frames_to_generate"] == 8
    assert len(rec["cameras"]["view0"]["w2c_window"]) == 9
    assert sorted(os.listdir(tmp_path / "in" / rec["context_dir"] / "view2")) == \
        ["frame_0002.png", "frame_0003.png", "frame_0004.png"]
    assert rec["prompts"]["view0"].startswith("a textured floor")
    rep = export_inputs(mini, tmp_path / "in2", Protocol())
    assert rep == {"exported": 0, "skipped": 1}
    s = mini.sample("Simul-Robot/s000")
    assert window_caption(s, p) == "a textured floor"
    assert prompts(s, Protocol(text="none"))["view1"] == ""


def test_validate_ok_and_index(mini):
    r = validate_sample(mini.root, "Simul-Robot", "s000", deep=True)
    assert r["ok"], r["errors"]
    rep = validate(mini.root, jobs=1)
    assert rep["n_ok"] == rep["n_samples"] == 1
    idx = json.loads((mini.root / "index.json").read_text())
    rec = idx["samples"][0]
    assert rec["uid"] == "Simul-Robot/s000" and rec["n_frames"] == 24 and len(rec["queries_per_frame"]) == 24
    assert [r["id"] for r in mini.records("Simul-Robot", difficulty="normal")] == ["s000"]
    assert mini.records("Simul-Robot", difficulty="hard") == []


@pytest.mark.parametrize("breakage", ["visible_outside_frustum", "nan_valid", "missing_video", "wrong_frame_count",
                                      "improper_rotation", "missing_depth_member"])
def test_validate_detects(tmp_path, breakage):
    import zipfile
    d = make_sample(tmp_path, "Simul-Robot", "bad")
    if breakage in ("visible_outside_frustum", "nan_valid"):
        z = dict(np.load(d / "tracks3d.npz"))
        if breakage == "visible_outside_frustum":
            t, n, v = np.argwhere(~z["in_frustum"])[0] if (~z["in_frustum"]).any() else (0, 0, 0)
            z["in_frustum"][t, n, v] = False
            z["visible"][t, n, v] = True
        else:
            z["xyz"][3, 0] = np.nan
        np.savez(d / "tracks3d.npz", **z)
    elif breakage == "missing_video":
        (d / "videos" / "view1.mp4").unlink()
    elif breakage == "wrong_frame_count":
        write_video(d / "videos" / "view0.mp4", np.zeros((23, 48, 64, 3), np.uint8), fps=10.0)
    elif breakage == "improper_rotation":
        z = dict(np.load(d / "cameras.npz"))
        z["w2c"][1, 5, :3, 0] *= -1
        np.savez(d / "cameras.npz", **z)
    else:
        src = d / "depth" / "depth.zip"
        tmp = d / "depth" / "x.zip"
        with zipfile.ZipFile(src) as a, zipfile.ZipFile(tmp, "w") as b:
            for n in a.namelist()[1:]:
                b.writestr(n, a.read(n))
        shutil.move(tmp, src)
    r = validate_sample(tmp_path, "Simul-Robot", "bad")
    assert not r["ok"], breakage


@pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT to a VP-WorldBench data root")
def test_released_samples_validate():
    bench = Benchmark(RELEASED)
    for sub in bench.subsets():
        sid = bench.ids(sub)[0]
        r = validate_sample(RELEASED, sub, sid)
        assert r["ok"], (sub, sid, r["errors"])
        s = bench.sample(sub, sid)
        assert Protocol().evaluable(s)[0]


def test_resample_ties_round_half_up():
    """A 30 fps prediction on a 20 fps grid: window step = 1.5 video frames; ties (x.5) take the later frame."""
    from vpm_worldbench.predictions import _resample
    p = Protocol(ref_frame=0, horizon=6)
    frames = np.arange(12, dtype=np.uint8)[:, None, None, None] * np.ones((1, 2, 2, 3), np.uint8)
    out = _resample(frames, p, fps_gt=20.0, meta={"fps": 30.0}, ref_frame=None)
    assert out[:, 0, 0, 0].tolist() == [0, 2, 3, 5, 6, 8, 9]


def test_protocol_subset_alias_keys():
    p = Protocol(eval_size={"rr": (424, 240)}, delta_ladder_m={"real-general": (0.1, 0.2)})
    assert p.eval_size == {"Real-Robot": (424, 240)} and p.eval_size_for("Real-Robot") == (424, 240)
    assert p.get("delta_ladder_m", "Real-General") == (0.1, 0.2)
    with pytest.raises(KeyError):
        Protocol(eval_size={"bogus": (1, 1)})


def test_export_fps_window(mini, tmp_path):
    p = Protocol(ref_frame=4, horizon=8, frame_step=2)
    export_inputs(mini, tmp_path / "in", p)
    rec = json.loads((tmp_path / "in" / "Simul-Robot" / "s000" / "inputs.json").read_text())
    assert rec["fps"] == 10.0 and rec["fps_window"] == 5.0


@pytest.mark.parametrize("fps_gt,fps", [(20.0, 30.0), (20.0, 25.0), (20.0, 50.0), (24.0, 60.0), (30000 / 1001, 60.0)])
def test_resample_exact_ties(fps_gt, fps):
    """vpm-v1 window (ref 16, horizon 64): window position j sits at x = j * fps / fps_gt, computed exactly, and a tie
    takes the later frame (float arithmetic gave 41 * 30 / 20 = 61.49999...)."""
    from fractions import Fraction
    from math import floor

    from vpm_worldbench.predictions import _resample, window_positions
    p = Protocol()
    n = int(64 * fps / fps_gt) + 2
    frames = np.arange(n, dtype=np.int32)[:, None, None, None] * np.ones((1, 1, 1, 3), np.int32)
    out = _resample(frames, p, fps_gt=fps_gt, meta={"fps": fps}, ref_frame=None)[:, 0, 0, 0]
    r = Fraction(fps).limit_denominator(10 ** 6) / Fraction(fps_gt).limit_denominator(10 ** 6)
    assert out.tolist() == [floor(j * r + Fraction(1, 2)) for j in range(65)]
    assert window_positions(p, fps_gt, fps)[41] == 41 * r
    if (fps_gt, fps) == (20.0, 30.0):
        assert out[41] == 62


def test_resample_blend_and_meta_checks(mini, tmp_path):
    from vpm_worldbench.predictions import _resample, check_meta
    frames = (np.arange(8, dtype=np.uint8) * 30)[:, None, None, None] * np.ones((1, 2, 2, 3), np.uint8)
    out = _resample(frames, Protocol(ref_frame=0, horizon=4), fps_gt=10.0, meta={"fps": 5.0}, ref_frame=None)
    assert out[:, 0, 0, 0].tolist() == [0, 15, 30, 45, 60]
    for bad in ({"fps": 0}, {"fps": -5}, {"fps": float("nan")}, {"fps": float("inf")}, {"fps": "abc"},
                {"fps": "24"}, {"fps": "15.0"}, {"fps": True}, {"fps": False}, {"fps": [24]}, {"fps": 10 ** 400},
                {"first_frame": "Future"}, {"first_frame": None}, [1, 2]):
        with pytest.raises(ValueError):
            check_meta(bad)
        with pytest.raises(ValueError):
            _resample(frames, Protocol(ref_frame=0, horizon=4), 10.0, bad, None)
    for good in ({"fps": 24, "first_frame": "future"}, {"fps": 29.97}, {"fps": np.float64(5.0)}, {"fps": np.int64(5)},
                 {"fps": None}, {}):
        assert check_meta(good) is good
    s = mini.sample("Simul-Robot/s000")
    p = Protocol(ref_frame=4, horizon=8, frame_step=2, eval_size={"Simul-Robot": (64, 48)})
    real = [load_reference(s, k, p) for k in range(3)]
    for i, meta in enumerate(['{"fps": 0}', '{"first_frame": "Future"}', '{"fps": -5}', "not json", "[1]",
                              '{"fps": "10"}', '{"fps": true}']):
        _write_pred(tmp_path / f"m{i}", s, real)
        (tmp_path / f"m{i}" / "Simul-Robot" / "s000" / "meta.json").write_text(meta)
        with pytest.raises(ValueError, match="meta.json"):
            load_prediction(tmp_path / f"m{i}", s, 0, p)


def test_export_follows_protocol_resize_mode(mini, tmp_path):
    """--size is reached with the protocol's resize mode of the subset, so the exported conditioning frame is the
    reference frame on the eval grid; an explicit other mode or an incompatible size gives a warning."""
    import warnings

    import cv2
    s = mini.sample("Simul-Robot/s000")
    p = Protocol(ref_frame=4, horizon=8, frame_step=2, eval_size={"sr": (48, 48)}, resize_mode={"sr": "center_crop"})
    ref = load_reference(s, 0, p)[0]

    def exported(d):
        img = cv2.imread(str(d / "Simul-Robot" / "s000" / "context" / "view0" / "frame_0004.png"))
        rec = json.loads((d / "Simul-Robot" / "s000" / "inputs.json").read_text())
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), rec

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert export_inputs(mini, tmp_path / "a", p, size=(48, 48)) == {"exported": 1, "skipped": 0}
        export_inputs(mini, tmp_path / "n", p)
        export_inputs(mini, tmp_path / "x", p, size=(96, 96))
    img, rec = exported(tmp_path / "a")
    np.testing.assert_array_equal(img, ref)
    np.testing.assert_allclose(rec["cameras"]["view0"]["K"], s.K(0, (48, 48), "center_crop"))
    assert rec["cameras"]["view0"]["resize_mode"] == "center_crop"
    assert exported(tmp_path / "n")[1]["cameras"]["view0"]["resize_mode"] is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        export_inputs(mini, tmp_path / "b", p, size=(48, 48), mode="resize")
    msgs = [str(x.message) for x in w]
    assert len(msgs) == 2 and "differs from the protocol's resize_mode center_crop" in msgs[0]
    assert "frames of 64x48 exported at 48x48 with resize are framed differently" in msgs[1]
    assert not np.array_equal(exported(tmp_path / "b")[0], ref)
    with pytest.warns(UserWarning, match="framed differently"):
        export_inputs(mini, tmp_path / "c", p, size=(64, 32))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        export_inputs(mini, tmp_path / "d", SMALL, size=(80, 32))
    from vpm_worldbench import inputs
    p.save(tmp_path / "crop.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert inputs.main(["--root", str(mini.root), "--out", str(tmp_path / "e"), "--protocol",
                            str(tmp_path / "crop.json"), "--size", "48x48", "--subsets", "sr"]) == 0
    np.testing.assert_array_equal(exported(tmp_path / "e")[0], ref)
    with pytest.raises(SystemExit):
        inputs.main(["--root", str(mini.root), "--out", str(tmp_path / "f"), "--subsets", "nope"])
    assert inputs.main(["--root", str(mini.root), "--out", str(tmp_path / "g"), "--subsets", "rr"]) == 1


def test_benchmark_ids_aliases_and_unmatched(mini, tmp_path):
    import warnings

    from vpm_worldbench.data.benchmark import DataRootError
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [s.uid for s in mini.samples(ids=["sr/s000"])] == ["Simul-Robot/s000"]
        assert [s.uid for s in mini.samples("simul_robot", ids=["SR/s000"])] == ["Simul-Robot/s000"]
        assert [s.uid for s in mini.samples(["sr", "rr"], ids=["s000"])] == ["Simul-Robot/s000"]
        assert list(mini.samples("Real-Robot", ids=["s000"])) == []
        assert list(mini.samples("sr", ids=["sr/s999"])) == []
        assert list(mini.samples(ids=[])) == []
    with pytest.warns(UserWarning, match=r"1 requested id\(s\) match no sample of Simul-Robot: \['nope'\]"):
        assert [s.id for s in mini.samples(ids=["s000", "nope"])] == ["s000"]
    with pytest.warns(UserWarning, match=r"2 requested id\(s\) .*: \['Simul-Robot/s999', 'other'\]"):
        assert list(mini.samples(["sr"], ids=["sr/s999", "other", "SR/s999"])) == []
    with pytest.warns(UserWarning, match="after the attribute filters"):
        assert list(mini.samples(ids=["s000"], difficulty="hard")) == []
    assert mini.unmatched_ids(["s000", "sr/s000", "rr/s000", "nope", "qq/s000"]) == ["rr/s000", "nope", "qq/s000"]
    assert mini.unmatched_ids(["s000"], ["rr"]) == ["s000"]
    assert mini.unmatched_ids(["sr/s999"], "sr") == ["sr/s999"]
    with pytest.raises(KeyError):
        list(mini.samples("qq"))
    with pytest.raises(DataRootError, match="does not exist"):
        Benchmark(tmp_path / "nowhere")
    assert issubclass(DataRootError, FileNotFoundError)


def test_validate_cli_selection(mini, capsys):
    from vpm_worldbench import validate as V
    root = str(mini.root)
    assert V.main(["--root", root, "--subsets", "sr", "--jobs", "1"]) == 0
    assert "1/1 samples valid" in capsys.readouterr().out
    assert V.main(["--root", root, "--subsets", "sr", "Simul-Robot,simul_robot", "--jobs", "1"]) == 0
    assert "1/1 samples valid" in capsys.readouterr().out
    assert V.main(["--root", root, "--ids", "sr/s000", "Simul-Robot/s000", "--jobs", "1"]) == 0
    assert V.main(["--root", root, "--subsets", "Simul-Robot,real_robot", "--jobs", "1"]) == 2
    assert "error: no subset folder Real-Robot" in capsys.readouterr().err
    assert V.main(["--root", root, "--ids", "s000", "nope", "--jobs", "1"]) == 2
    out, err = capsys.readouterr()
    assert "1/1 samples valid" in out and "error: 1 requested id(s) match no sample: ['nope']" in err
    assert V.main(["--root", root, "--ids", "nope", "--jobs", "1"]) == 2
    assert "error: no sample selected" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        V.main(["--root", root, "--subsets", "xx"])
    assert e.value.code == 2
    rep = validate(mini.root, subsets=["sr"], jobs=1, ids=["sr/s000"])
    assert rep["n_ok"] == 1 and rep["unmatched_ids"] == [] and rep["missing_subsets"] == []


def test_protocol_file_names_and_unknown_protocol(tmp_path):
    from vpm_worldbench.protocol import ProtocolNotFoundError
    (tmp_path / "noname.yaml").write_text("horizon: 32\n")
    assert get_protocol(str(tmp_path / "noname.yaml")).name == "noname"
    (tmp_path / "legacy-prerelease.yaml").write_text("horizon: 32\n")
    q = Protocol.load(tmp_path / "legacy-prerelease.yaml")
    assert q.name == "custom-legacy-prerelease" and legacy_option(q, "Real-General", "fvd_windows", "x") == "x"
    (tmp_path / "named.json").write_text(json.dumps({"name": "mine", "horizon": 8}))
    assert Protocol.load(tmp_path / "named.json").name == "mine"
    (tmp_path / "empty.yaml").write_text("# only a comment\n")
    assert Protocol.load(tmp_path / "empty.yaml").to_dict() == Protocol(name="empty").to_dict()
    (tmp_path / "list.yaml").write_text("- 1\n")
    with pytest.raises(ValueError):
        Protocol.load(tmp_path / "list.yaml")
    with pytest.raises(ProtocolNotFoundError, match="neither a preset"):
        get_protocol("vpm-v2")
    with pytest.raises(FileNotFoundError):
        get_protocol(str(tmp_path / "missing.yaml"))
    from vpm_worldbench.protocol import ProtocolFileError
    (tmp_path / "typo.yaml").write_text("horizn: 32\n")
    (tmp_path / "broken.yaml").write_text("horizon: [1\n")
    (tmp_path / "zero.json").write_text('{"horizon": 0}')
    (tmp_path / "subset.json").write_text('{"eval_size": {"Mars": [1, 1]}}')
    for f, cause in [("typo.yaml", r"KeyError: unknown protocol fields \['horizn'\]"), ("broken.yaml", r"\w+Error"),
                     ("zero.json", "ValueError: horizon and frame_step must be >= 1"),
                     ("subset.json", "KeyError: unknown subset 'Mars'"), ("list.yaml", "ValueError")]:
        with pytest.raises(ProtocolFileError, match=f"protocol file '.*{f}': {cause}"):
            get_protocol(str(tmp_path / f))
    with pytest.raises(KeyError):
        Protocol.load(tmp_path / "typo.yaml")
    assert issubclass(ProtocolFileError, ValueError)


def test_cli_one_line_errors(mini, monkeypatch, capsys):
    from vpm_worldbench import cli
    from vpm_worldbench.tasks import generation as G
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    assert cli.main(["protocol", "--protocol", "nonexistent"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: protocol 'nonexistent' is neither a preset") and err.count("\n") == 1
    (mini.root / "_typo.yaml").write_text("horizn: 32\n")
    for argv in (["protocol"], ["generation", "export", "--root", str(mini.root), "--out", str(mini.root / "_x")]):
        assert cli.main(argv + ["--protocol", str(mini.root / "_typo.yaml")]) == 2
        err = capsys.readouterr().err
        assert err.startswith("vpmwb: error: protocol file ") and "unknown protocol fields ['horizn']" in err
        assert err.count("\n") == 1
    for argv in (["info"], ["generation", "export", "--out", str(mini.root / "_x")]):
        assert cli.main(argv + ["--root", str(mini.root / "nowhere")]) == 2
        err = capsys.readouterr().err
        assert err.startswith("vpmwb: error: data root ") and "does not exist" in err and err.count("\n") == 1

    def boom(exc):
        def f(argv):
            raise exc
        return f

    monkeypatch.setattr(cli, "_info", boom(ImportError("LPIPS needs torch and lpips:\n pip install 'x[metrics]'")))
    assert cli.main(["info", "--root", str(mini.root)]) == 2
    assert capsys.readouterr().err == "vpmwb: error: LPIPS needs torch and lpips: pip install 'x[metrics]'\n"
    monkeypatch.setattr(cli, "_info", boom(G.ModelLoadError("loading the metric models failed: RuntimeError: no GPU")))
    assert cli.main(["info", "--root", str(mini.root)]) == 2
    assert "loading the metric models failed" in capsys.readouterr().err
    monkeypatch.setattr(cli, "_info", boom(ModuleNotFoundError("No module named 'torch'")))
    assert cli.main(["info", "--root", str(mini.root)]) == 2
    assert "see the extras in the README" in capsys.readouterr().err
    from vpm_worldbench.utils import UnknownSubsetError, parse_subsets
    monkeypatch.setattr(cli, "_info", lambda argv: parse_subsets(["sr", "nope"]))
    assert cli.main(["info", "--root", str(mini.root)]) == 2
    assert capsys.readouterr().err.startswith("vpmwb: error: unknown subset 'nope'; expected one of")
    with pytest.raises(KeyError):
        parse_subsets("sr,nope")
    assert issubclass(UnknownSubsetError, KeyError)
    monkeypatch.setattr(cli, "_info", boom(RuntimeError("a bug")))
    with pytest.raises(RuntimeError):
        cli.main(["info", "--root", str(mini.root)])
    monkeypatch.setenv("VPMWB_TRACEBACK", "1")
    monkeypatch.setattr(cli, "_info", boom(ImportError("x")))
    with pytest.raises(ImportError):
        cli.main(["info", "--root", str(mini.root)])


def test_docstring_examples(mini):
    """The usage lines of the data API docstrings."""
    s = mini.sample("Simul-Robot/s000")
    uv, vis, in_frame = s.tracks2d(0, size=(32, 24))
    assert uv.shape[:2] == vis.shape == in_frame.shape
    assert [x.uid for x in mini.samples("Simul-Robot", difficulty="normal")] == ["Simul-Robot/s000"]
    with pytest.warns(UserWarning, match="Real-Robot/episode_000"):
        assert [x.uid for x in mini.samples(["sr", "rr"], ids=["rr/episode_000", "s000"])] == ["Simul-Robot/s000"]


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG_KEYS = {"camera_control": "d52ab6f6607c", "half_rate": "30897e8ef043", "hard_only": "d52ab6f6607c",
               "i2v_first_frame": "80a057a3e0b4", "long_horizon": "ef6d86a7525e", "square_center_crop": "8d97387d20f3",
               "v2v_17": "d52ab6f6607c", "v2v_9": "d52ab6f6607c", "vpm-v1": "d52ab6f6607c"}


def _env_without_traceback():
    """Environment of a `python -m vpm_worldbench...` subprocess: this checkout first, VPMWB_TRACEBACK unset."""
    import vpm_worldbench
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(vpm_worldbench.__file__).resolve().parents[1])] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    env.pop("VPMWB_TRACEBACK", None)
    return env


def test_parse_size_errors():
    from vpm_worldbench.utils import SizeError, UsageError, parse_size
    assert parse_size("640x480") == (640, 480) and parse_size("1280X720") == (1280, 720)
    assert parse_size(None) is None and parse_size("") is None and parse_size("native") is None
    for bad in ("12x", "abc", "x", "640x480x3", "0x480", "-640x480", "640*480", "1.5x2"):
        with pytest.raises(SizeError, match=r"^--size must be WxH \(e\.g\. 640x480\), got "):
            parse_size(bad)
    with pytest.raises(ValueError, match="^--dense-size must be WxH"):
        parse_size("12x", flag="--dense-size")
    assert issubclass(SizeError, UsageError) and issubclass(SizeError, ValueError)


BAD_PROTOCOL_FILES = {
    "eval_size: {sr: 480}\n": "eval_size of Simul-Robot", "eval_size: {rr: [848, 0]}\n": "eval_size of Real-Robot",
    "eval_size: {rr: [848.0, 480]}\n": "eval_size of Real-Robot",
    "resize_mode: {rr: crop}\n": "resize_mode of Real-Robot",
    "clip_filter: [1]\n": "clip_filter", "views: 5\n": "views", "views: [0, 0]\n": "views", "views: []\n": "views",
    "views: [-1]\n": "views", "met3r_aspect: bogus\n": "met3r_aspect", "fvd_cov: bogus\n": "fvd_cov",
    "lifter: nope\n": "lifter 'nope'", "depth_anchor: {rr: bogus}\n": "depth_anchor of Real-Robot",
    "depth_anchor: {sr: gt-depth}\n": "depth_anchor of Simul-Robot", 'horizon: "64"\n': "horizon",
    "horizon: true\n": "horizon", "ref_frame: -1\n": "ref_frame", "met3r_step: 0\n": "met3r_step",
    "fvd_windows: 0\n": "fvd_windows", "camera_conditioning: 'yes'\n": "camera_conditioning",
    "reference_frames: raw\n": "reference_frames", "delta_ladder_m: {rr: [0.01, -1]}\n": "delta_ladder_m of Real-Robot",
    "delta_survival_m: {rg: .nan}\n": "delta_survival_m of Real-General", "delta_group: {sr: 3}\n": "delta_group",
    "name: 7\n": "name",
}


def test_protocol_values_are_checked(tmp_path, monkeypatch, capsys):
    """Every field is checked when the protocol is built: an invalid file is one ProtocolFileError (vpmwb: one line,
    exit status 2), before anything is exported or scored."""
    import re

    from vpm_worldbench import cli
    from vpm_worldbench.protocol import ProtocolFileError
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    for i, (text, what) in enumerate(BAD_PROTOCOL_FILES.items()):
        f = tmp_path / f"bad{i}.yaml"
        f.write_text(text)
        with pytest.raises(ProtocolFileError, match=re.escape(what)):
            get_protocol(str(f))
        assert cli.main(["protocol", "--protocol", str(f)]) == 2, text
        err = capsys.readouterr().err
        assert err.startswith(f"vpmwb: error: protocol file '{f}': ") and what in err and err.count("\n") == 1, err
    for bad in ("my package:X", "pkg:", ":X", "a:b:c", 3):
        with pytest.raises(ValueError, match="lifter"):
            Protocol(lifter=bad)
    assert Protocol(lifter="my_package.lifters:MyLifter").lifter == "my_package.lifters:MyLifter"
    p = Protocol(views=(0, 2), eval_size={"sr": (np.int64(64), 48)}, depth_anchor={"rr": "none"}, fvd_windows=2,
                 delta_ladder_m={"rg": [1, 2.5]}, delta_group={"sr": None}, clip_filter={"difficulty": ["hard"]})
    assert p.to_dict()["views"] == [0, 2] and p.eval_size_for("sr") == (64, 48)


def test_presets_and_configs_keep_their_names_and_keys():
    """The checks and the preset-name rule leave the presets and every shipped config loadable without a warning, under
    their own names and with the reference keys they had (vpm-v1 d52ab6f6607c, legacy-prerelease 2395018aa26d)."""
    import warnings

    import yaml

    from vpm_worldbench.tasks.generation import protocol_hash
    assert protocol_hash(get_protocol("vpm-v1")) == "d52ab6f6607c"
    assert protocol_hash(get_protocol("legacy-prerelease")) == "2395018aa26d"
    files = sorted(CONFIGS.glob("*.yaml"))
    if not files:
        pytest.skip("configs/ ships with the repository, not the wheel")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = {f.stem: get_protocol(str(f)) for f in files}
    for f in files:
        assert loaded[f.stem].name == yaml.safe_load(f.read_text())["name"]
    keys = {k: protocol_hash(p) for k, p in loaded.items()}
    for k in ("vpm-v1", "v2v_9", "v2v_17", "camera_control", "hard_only"):
        assert keys.get(k, "d52ab6f6607c") == "d52ab6f6607c", k
    assert {k: v for k, v in CONFIG_KEYS.items() if k in keys} == {k: keys[k] for k in CONFIG_KEYS if k in keys}


def test_preset_names_in_files(tmp_path):
    """A file named legacy-prerelease is that preset only when it evaluates like it (model inputs and clip_filter aside;
    a saved preset loads back as the preset); otherwise it is custom-legacy-prerelease with a warning, so the name alone
    never switches the legacy recipes on. 'vpm-v1', the default name of every Protocol, switches nothing on and is kept,
    so save / load round trips keep the name."""
    import warnings
    from dataclasses import replace

    from vpm_worldbench.protocol import legacy_prerelease
    from vpm_worldbench.tasks.generation import protocol_hash
    legacy_prerelease().save(tmp_path / "saved_legacy.json")
    replace(legacy_prerelease(), text="legacy", camera_conditioning=True, context_frames=9,
            clip_filter={"difficulty": "hard"}).save(tmp_path / "legacy_inputs.json")
    (tmp_path / "legacy_spelled_out.yaml").write_text(
        "name: legacy-prerelease\nrgb_scale_quirk: true\nplain_K_scaling: true\nreference_frames: _reproduce\n"
        "eval_size: {rg: [480, 480], sr: [480, 480]}\nresize_mode: {rr: resize}\nfvd_windows: null\n")
    Protocol(horizon=32).save(tmp_path / "v1_32.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = get_protocol(str(tmp_path / "saved_legacy.json"))
        assert get_protocol(str(tmp_path / "legacy_inputs.json")).name == "legacy-prerelease"
        assert get_protocol(str(tmp_path / "legacy_spelled_out.yaml")).name == "legacy-prerelease"
        v1 = get_protocol(str(tmp_path / "v1_32.json"))
    assert q.name == "legacy-prerelease" and protocol_hash(q) == "2395018aa26d"
    assert legacy_option(q, "Real-General", "fvd_windows") == 2
    assert v1.name == "vpm-v1" and v1.to_dict() == Protocol(horizon=32).to_dict()
    (tmp_path / "only_name.yaml").write_text("name: legacy-prerelease\n")
    with pytest.warns(UserWarning, match=r"evaluates differently \(eval_size, rgb_scale_quirk, plain_K_scaling, "
                                         r"reference_frames\); loaded as 'custom-legacy-prerelease'"):
        q = get_protocol(str(tmp_path / "only_name.yaml"))
    assert q.name == "custom-legacy-prerelease" and legacy_option(q, "Real-General", "fvd_windows", "x") == "x"
    assert protocol_hash(q) == "d52ab6f6607c"
    replace(legacy_prerelease(), horizon=32).save(tmp_path / "legacy_32.json")
    with pytest.warns(UserWarning, match=r"evaluates differently \(horizon\)"):
        assert get_protocol(str(tmp_path / "legacy_32.json")).name == "custom-legacy-prerelease"
    assert Protocol(name="legacy-prerelease").name == "legacy-prerelease"


def test_index_and_validate_roots(tmp_path, monkeypatch, capsys):
    """A missing --root (and a root without samples or without index.json) is a one-line usage error, exit status 2;
    `vpmwb index` writes no empty index.json."""
    import subprocess
    import sys

    from vpm_worldbench import cli
    from vpm_worldbench.data.benchmark import DataRootError
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    nowhere, empty = tmp_path / "nowhere", tmp_path / "empty"
    (empty / "Simul-Robot").mkdir(parents=True)
    with pytest.raises(DataRootError, match="does not exist"):
        build_index(nowhere)
    with pytest.raises(DataRootError, match="holds no sample"):
        build_index(empty)
    assert not (empty / "index.json").exists()
    with pytest.raises(DataRootError, match="does not exist"):
        validate(nowhere)
    for argv, what in ((["index", "--root", str(nowhere)], "does not exist"),
                       (["index", "--root", str(empty)], "holds no sample"),
                       (["validate", "--root", str(nowhere)], "does not exist"),
                       (["info", "--root", str(empty)], "has no index.json: run `vpmwb index --root")):
        assert cli.main(argv) == 2, argv
        out, err = capsys.readouterr()
        assert err.startswith("vpmwb: error: data root ") and what in err and err.count("\n") == 1, err
        assert out == ""
    for mod in ("index", "validate"):
        r = subprocess.run([sys.executable, "-m", f"vpm_worldbench.{mod}", "--root", str(nowhere)],
                           capture_output=True, text=True, env=_env_without_traceback())
        assert r.returncode == 2 and r.stderr.startswith("vpmwb: error: data root ") and "Traceback" not in r.stderr
        assert len(r.stderr.strip().splitlines()) == 1, r.stderr


def test_cli_errors_are_uniform(tmp_path, monkeypatch, capsys):
    """Every usage / setup error of vpmwb ends with one `vpmwb: error: ...` line and exit status 2 (an unknown command,
    an invalid --size, a command exiting with a message, a run directory of another run); VPMWB_TRACEBACK set to
    anything but '' / '0' shows the traceback instead."""
    from vpm_worldbench import cli
    from vpm_worldbench.tasks import generation as G
    from vpm_worldbench.utils import SizeError, error_line, parse_size, traceback_requested
    monkeypatch.delenv("VPMWB_TRACEBACK", raising=False)
    assert cli.main(["nope"]) == 2
    out, err = capsys.readouterr()
    assert err.startswith("vpmwb: error: unknown command 'nope'; commands: info, validate,") and out == ""
    assert err.count("\n") == 1

    def boom(exc):
        def f(argv):
            raise exc
        return f

    cases = [(lambda argv: parse_size("12x"), "vpmwb: error: --size must be WxH (e.g. 640x480), got '12x'"),
             (boom(SystemExit("--run: expected NAME=DIR,\n got 'x'")),
              "vpmwb: error: --run: expected NAME=DIR, got 'x'"),
             (boom(G.RunDirError("runs/a holds the results of the predictions in X")),
              "vpmwb: error: runs/a holds the results of the predictions in X")]
    for fn, line in cases:
        monkeypatch.setattr(cli, "_info", fn)
        assert cli.main(["info"]) == 2
        assert capsys.readouterr().err == line + "\n"
    monkeypatch.setattr(cli, "_info", boom(SystemExit(3)))
    with pytest.raises(SystemExit) as e:
        cli.main(["info"])
    assert e.value.code == 3
    for value, shown in (("1", True), ("yes", True), ("0", False), ("", False)):
        monkeypatch.setenv("VPMWB_TRACEBACK", value)
        assert traceback_requested() is shown
        monkeypatch.setattr(cli, "_info", lambda argv: parse_size("12x"))
        if shown:
            with pytest.raises(SizeError):
                cli.main(["info"])
            monkeypatch.setattr(cli, "_info", boom(SystemExit("bad --pred")))
            with pytest.raises(SystemExit):
                cli.main(["info"])
        else:
            assert cli.main(["info"]) == 2 and capsys.readouterr().err.startswith("vpmwb: error: --size")
    assert error_line(KeyError("unknown subset 'x'")) == "vpmwb: error: unknown subset 'x'"
    assert error_line("a\n  b") == "vpmwb: error: a b" and error_line(ValueError()) == "vpmwb: error: ValueError"
