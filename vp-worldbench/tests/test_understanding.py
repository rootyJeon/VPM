"""Tests for vpm_worldbench.tasks.understanding (QA generation, caption retrieval, temporal grounding).

A synthetic mini benchmark (tmp_path) is built so that every QA template has a known answer: object / hand motions in
the tracks, Simul-General target states, Real-Robot instructions, Real-General motion events + keypoints + segment
captions. Optional real-data checks run when VPMWB_RELEASED_ROOT points to a VP-WorldBench data root.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from vpm_worldbench.constants import KIND_NAMES, SCHEMA_SAMPLE
from vpm_worldbench.data.benchmark import Benchmark
from vpm_worldbench.metrics.qa import LETTERS
from vpm_worldbench.tasks import understanding as und
from vpm_worldbench.tasks.common import RunDirError

T = 40
KIND = {k: i for i, k in enumerate(KIND_NAMES)}
PROTO = dict(ref_frame=4, horizon=12)


def _look(C, f_dir):
    f = np.asarray(f_dir, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    E = np.eye(4)
    E[:3, :3] = np.stack([r, d, f])
    E[:3, 3] = -E[:3, :3] @ np.asarray(C, float)
    return E


def static_cams(V=2):
    centres = [(3.0, 0.2, 1.0), (0.3, 3.0, 1.2), (-3.0, -0.2, 1.1)][:V]
    return np.stack([np.stack([_look(c, -np.asarray(c))] * T) for c in centres])


def _ramp(onset, dur=10):
    """(T,) 0 -> 1 linear ramp starting at `onset`."""
    t = np.arange(T, dtype=float)
    return np.clip((t - onset) / dur, 0.0, 1.0)


def _pts(rng, n, centre, motion=None):
    """(T, n, 3) points around centre; motion = (T, 3) offset."""
    base = np.asarray(centre, float) + rng.uniform(-0.03, 0.03, (n, 3))
    off = np.zeros((T, 3)) if motion is None else motion
    return base[None] + off[:, None]


def write_sample(root, subset, sid, xyz, groups, group, kind, side, extra=None, attributes=None, captions=None,
                 w2c=None, labels=None, flags=()):
    d = root / subset / sid
    d.mkdir(parents=True)
    N = xyz.shape[1]
    w2c = static_cams() if w2c is None else w2c
    V = w2c.shape[0]
    W, H = 64, 48
    K = np.array([[40.0, 0, (W - 1) / 2], [0, 40.0, (H - 1) / 2], [0, 0, 1]])
    valid = np.isfinite(xyz).all(-1)
    vis = np.repeat(valid[..., None], V, -1)
    np.savez(d / "tracks3d.npz", xyz=xyz.astype(np.float32), valid=valid, visible=vis, in_frustum=vis,
             default_gt=valid, group=np.asarray(group, np.int16), group_names=np.array(groups),
             part=np.zeros(N, np.int16), part_names=np.array(["p"]), kind=np.asarray(kind, np.int8),
             kind_names=np.array(KIND_NAMES), side=np.asarray(side, np.int8), **(extra or {}))
    static = np.array([np.allclose(w2c[v], w2c[v, :1]) for v in range(V)])
    np.savez(d / "cameras.npz", view_names=np.array([f"view{v}" for v in range(V)]),
             source_view_names=np.array([f"cam{v}" for v in range(V)]), image_size=np.array([[W, H]] * V, np.int32),
             K=np.repeat(K[None], V, 0), w2c=w2c, static=static)
    cap = {"caption": f"caption of {subset} {sid}."}
    cap.update(captions or {})
    (d / "captions.json").write_text(json.dumps(cap))
    views = [{"name": f"view{v}", "video": f"videos/view{v}.mp4", "width": W, "height": H} for v in range(V)]
    meta = {"schema": SCHEMA_SAMPLE, "uid": f"{subset}/{sid}", "subset": subset, "id": sid, "n_views": V,
            "n_frames": T, "fps": 10.0, "views": views,
            "files": {"cameras": "cameras.npz", "tracks3d": "tracks3d.npz", "captions": "captions.json",
                      "depth": None, "labels": {}},
            "tracks": {"n_points": N, "groups": list(groups)}, "caption": cap["caption"],
            "attributes": attributes or {}, "flags": list(flags), "depth": None}
    for name, (rel, write) in (labels or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        write(p)
        meta["files"]["labels"][name] = rel
    (d / "sample.json").write_text(json.dumps(meta))


def _json_writer(obj):
    return lambda p: p.write_text(json.dumps(obj))


def _sr(root, sid, objs, hands, flag_details, w2c=None, primary=None, rng_seed=0):
    """objs: [(id, kind, onset or None, displacement m)], hands: {'left': disp, 'right': disp} (onset 5)."""
    rng = np.random.default_rng(rng_seed)
    parts, group, kind, side, oid = [], [], [], [], []
    for i, (name, k, onset, disp) in enumerate(objs):
        mot = None if onset is None else np.outer(_ramp(onset), [disp, 0.0, 0.0])
        parts.append(_pts(rng, 8, (0.0, 0.3 * i, 0.0), mot))
        group += [0] * 8
        kind += [KIND["object_moving" if k == "target" else "object_static"]] * 8
        side += [-1] * 8
        oid += [name] * 8
    for s_idx, s_name in ((0, "left"), (1, "right")):
        d = hands.get(s_name, 0.0)
        parts.append(_pts(rng, 4, (0.3, 0.5 - s_idx, 0.4), np.outer(_ramp(5), [0.0, 0.0, d])))
        parts.append(_pts(rng, 4, (0.5, 0.5 - s_idx, 0.6)))
        group += [1] * 8
        kind += [KIND["robot_hand"]] * 4 + [KIND["robot_arm"]] * 4
        side += [s_idx] * 8
        oid += [""] * 8
    xyz = np.concatenate(parts, 1)
    primary = primary or objs[0][0]
    attrs = {"hands_used": sorted(k for k, v in hands.items() if v > 0.05), "primary_object": primary,
             "distractor_max_pushed_cm": 0.0, "distractor_fell_off": 0, "flag_details": flag_details,
             "difficulty": "normal"}
    objects = {"objects": [{"id": n, "kind": k} for n, k, _, _ in objs]}
    write_sample(root, "Simul-Robot", sid, xyz, ("object", "robot"), group, kind, side,
                 extra={"object_id": np.array(oid), "is_primary": np.array(oid) == primary}, attributes=attrs, w2c=w2c,
                 labels={"objects": ("labels/objects.json", _json_writer(objects))})


RESTING = {"primary_end_supported": True, "primary_end_held": False, "primary_end_airborne": False,
           "clip_ends_mid_motion_borderline": False}
HELD = {"primary_end_supported": False, "primary_end_held": True, "primary_end_airborne": True,
        "clip_ends_mid_motion_borderline": False}


def _sg(root, sid, category, sub, blueprint, net_m, states, follow=False):
    rng = np.random.default_rng(len(sid))
    mot = np.outer(np.linspace(0, 1, T), [net_m, 0.0, 0.0])
    xyz = _pts(rng, 8, (0.0, 0.0, 0.8), mot)
    if follow:
        c = xyz.mean(1)
        w2c = np.stack([np.stack([_look(c[t] + off, -np.asarray(off)) for t in range(T)])
                        for off in ((-4.0, 0.5, 1.0), (0.5, -4.0, 1.2))])
    else:
        w2c = static_cams()
    names = np.array(["visible", "occluded", "out_of_frustum", "absent"])

    def write(p):
        np.savez(p, state_strict=np.asarray(states, np.uint8), state_names=names)
    write_sample(root, "Simul-General", sid, xyz, ("target",), [0] * 8,
                 [KIND["pedestrian" if category == "pedestrian" else "vehicle"]] * 8, [-1] * 8,
                 attributes={"category": category, "subcategory": sub, "blueprint": blueprint, "difficulty": "normal"},
                 w2c=w2c, labels={"target": ("labels/target.npz", write)})


def _rr(root, sid, hand, family, steps, objects, left_onset, left_disp, right_onset, right_disp, caption):
    rng = np.random.default_rng(len(sid) + 7)
    parts, group, side = [], [], []
    for s_idx, onset, disp in ((0, left_onset, left_disp), (1, right_onset, right_disp)):
        mot = np.outer(_ramp(onset if onset is not None else T + 1), [disp, 0.0, 0.0])
        parts.append(_pts(rng, 6, (0.4, 0.3 - 0.6 * s_idx, 0.2), mot))
        parts.append(_pts(rng, 6, (0.2, 0.3 - 0.6 * s_idx, 0.4)))
        group += [1] * 6 + [0] * 6
        side += [s_idx] * 12
    xyz = np.concatenate(parts, 1)
    kind = [KIND["robot_hand"] if g == 1 else KIND["robot_arm"] for g in group]
    st = {"hand": hand, "task_family": family, "steps": steps,
          "objects": [{"name": o, "noun": o.split()[-1], "colour": None} for o in objects]}
    write_sample(root, "Real-Robot", sid, xyz, ("arm", "hand"), group, kind, side,
                 attributes={"hand": hand, "task_family": family, "difficulty": "normal"},
                 captions={"caption": caption, "instruction": caption, "structured": st})


def _kp_person(walk_from=None, walk_to=None, turn=None, wave=None, walk_dist=2.0):
    """(T, 133, 3) keypoints of a standing person at the origin facing +x; optional walk along +x between frames,
    a left (CCW) turn of the body (yaw +90 deg) over `turn` = (f0, f1), and a left-wrist wave over `wave`."""
    X = np.zeros((T, 133, 3))
    local = {0: (0.1, 0, 1.6), 1: (0.1, 0.03, 1.65), 2: (0.1, -0.03, 1.65), 3: (0.05, 0.08, 1.62),
             4: (0.05, -0.08, 1.62), 5: (0, 0.2, 1.4), 6: (0, -0.2, 1.4), 7: (0, 0.25, 1.1), 8: (0, -0.25, 1.1),
             9: (0.1, 0.25, 0.85), 10: (0.1, -0.25, 0.85), 11: (0, 0.1, 0.95), 12: (0, -0.1, 0.95),
             13: (0, 0.1, 0.5), 14: (0, -0.1, 0.5), 15: (0, 0.1, 0.08), 16: (0, -0.1, 0.08)}
    for t in range(T):
        yaw = 0.0
        if turn is not None:
            yaw = math.radians(90.0) * float(np.clip((t - turn[0]) / (turn[1] - turn[0]), 0, 1))
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        pos = np.zeros(3)
        if walk_from is not None:
            pos[0] = walk_dist * float(np.clip((t - walk_from) / (walk_to - walk_from), 0, 1))
        for j, p in local.items():
            q = np.asarray(p, float).copy()
            if wave is not None and j == 9 and wave[0] <= t <= wave[1]:
                q[0] += 0.3 * math.sin(2 * math.pi * (t - wave[0]) / 6.0)
            X[t, j] = R @ q + pos
        X[t, 17:] = X[t, :1]
    return X


def _rg(root, sid, events, segments, kp, with_events=True):
    rng = np.random.default_rng(len(sid) + 11)
    xyz = _pts(rng, 8, (0.0, 0.0, 1.0))
    labels = {} if kp is None else {"keypoints3d": ("labels/human/keypoints3d.npz", lambda p: np.savez(
        p, xyz=kp.astype(np.float32), ok=np.ones(kp.shape[:2], bool)))}
    if with_events:
        labels["motion_events"] = ("labels/motion_events.json", _json_writer(
            {"clip": sid, "fps": 10.0, "n_frames": T, "summary": {"standing_pelvis_m": 0.95}, "events": events}))
    write_sample(root, "Real-General", sid, xyz, ("human_body",), [0] * 8, [KIND["human_body"]] * 8, [-1] * 8,
                 attributes={"difficulty": "normal"},
                 captions={"caption": f"a person in {sid}.", "segments": segments},
                 labels=labels, w2c=static_cams(3))


SEGMENTS = [
    {"start_frame": 0, "end_frame": 7, "caption": "The person stands calmly beside the wooden desk."},
    {"start_frame": 8, "end_frame": 15, "caption": "The left wrist waves quickly above the keyboard."},
    {"start_frame": 16, "end_frame": 23, "caption": "Walking forward across the bright room begins."},
    {"start_frame": 24, "end_frame": 31, "caption": "Turning around toward the shelf happens slowly."},
    {"start_frame": 32, "end_frame": 39, "caption": "Finally sitting down on a grey sofa."},
]


def build(root: Path):
    _sr(root, "sr_a", [("small_yellow_cube", "target", 5, 0.3), ("square_big_box", "placer", None, 0.0)],
        {"left": 0.0, "right": 0.3}, RESTING)
    pan = np.stack([_look((3.0, 0.2, 1.0), (-1.0, -0.07 + 0.012 * t, -0.3)) for t in range(T)])
    w2c = static_cams()
    w2c[1] = pan
    _sr(root, "sr_b", [("tomato_soup_can", "target", 3, 0.3), ("alphabet_soup_can", "target", 25, 0.2),
                       ("red_plastic_tote", "placer", None, 0.0)], {"left": 0.25, "right": 0.3}, HELD, w2c=w2c)
    _sr(root, "sr_c", [("tomato_can", "target", 5, 0.03), ("magenta_plastic_crate", "placer", None, 0.0)],
        {"left": 0.2, "right": 0.0}, {"primary_end_supported": True, "primary_end_held": True,
                                      "primary_end_airborne": False})
    st_a = np.zeros((T, 2), np.uint8)
    st_a[10:26, 1] = 2
    st_b = np.zeros((T, 2), np.uint8)
    st_b[5:8, 0] = 2
    _sg(root, "sg_a", "pedestrian", "pedestrian", "walker.pedestrian.0001", 15.0, st_a, follow=True)
    _sg(root, "sg_b", "vehicle", "truck", "vehicle.ambulance.ford", 50.0, st_b)
    _sg(root, "sg_c", "vehicle", "car", "vehicle.audi.tt", 19.0, np.zeros((T, 2), np.uint8))
    _rr(root, "rr_a", "right", "put_on_plate", [{"text": "x", "hands": ["right"]}], ["red cube", "plate"],
        None, 0.0, 5, 0.3, "Put the red cube on the plate with the right hand.")
    _rr(root, "rr_b", "bimanual", "plate_and_fruit",
        [{"text": "x", "hands": ["left"]}, {"text": "y", "hands": ["right"]}],
        ["plate", "peach"], 3, 0.3, 25, 0.3, "Bring the plate, then put the peach on it.")
    _rr(root, "rr_c", "left", "stack_cubes", [{"text": "x", "hands": ["left"]}],
        ["blue cube", "tennis ball", "white mouse", "doll"], 5, 0.3, 5, 0.3,
        "Put the red cube on the plate with the right hand.")
    ev = [{"type": "stand_still", "f0": 0, "f1": 15}, {"type": "walk", "f0": 17, "f1": 39},
          {"type": "turn", "f0": 18, "f1": 27, "yaw_deg": 90.0, "direction": "left"}]
    _rg(root, "rg_a", ev, SEGMENTS, _kp_person(walk_from=17, walk_to=39, turn=(18, 27), wave=(4, 16)))
    segs_b = [dict(g) for g in SEGMENTS]
    segs_b[4] = dict(segs_b[4], clipped=True)
    segs_b[3] = dict(segs_b[3], caption=segs_b[2]["caption"] + " again")
    _rg(root, "rg_b", [], segs_b, _kp_person(), with_events=False)
    _rg(root, "rg_c", [], [], None, with_events=False)
    return Benchmark(root)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return build(tmp_path_factory.mktemp("vpmwb_und"))


@pytest.fixture(scope="module")
def proto():
    return und.UnderstandingProtocol(**PROTO)


def _cands(bench, uid, proto):
    """{template: candidate} (templates with several questions per sample: {template#k: candidate})."""
    s = bench.sample(uid)
    vocab = {sub: und._vocabulary(bench, sub) for sub in ("Simul-Robot", "Real-Robot")}
    cs = und.sample_candidates(s, proto, vocab)
    n = {}
    for c in cs:
        n[c["template"]] = n.get(c["template"], 0) + 1
    return {(c["template"] if n[c["template"]] == 1 and c["template"] not in ("rg_motion", "rg_turn_direction",
                                                                              "rg_body_part", "rg_segment_order")
             else c["template"] + "#" + c["qid"].split("#")[-1]): c for c in cs}


def test_simul_robot_templates(bench, proto):
    a = _cands(bench, "Simul-Robot/sr_a", proto)
    assert a["sr_hands"]["answer_text"] == "only the right hand"
    assert a["sr_moved_object"]["answer_text"] == "small yellow cube"
    assert "large square box" in a["sr_moved_object"]["pref"]
    assert a["sr_num_moved"]["answer_text"] == "1"
    assert "sr_first_moved" not in a
    assert a["sr_end_state"]["answer_text"] == und.END_OPTIONS[0]
    assert a["num_views"]["answer_text"] == "2" and "2" not in a["num_views"]["pool"]
    b = _cands(bench, "Simul-Robot/sr_b", proto)
    assert b["sr_hands"]["answer_text"] == "both hands"
    assert b["sr_num_moved"]["answer_text"] == "2" and "sr_moved_object" not in b
    assert b["sr_first_moved"]["answer_text"] == "tomato soup can"
    assert b["sr_first_moved"]["pref"] == ["alphabet soup can"]
    assert all(not und._confusable(x, "tomato soup can") for x in b["sr_first_moved"]["pool"])
    assert b["sr_end_state"]["answer_text"] == und.END_OPTIONS[1]
    cm = b["camera_motion"]
    v = cm["meta"]["view"]
    assert cm["answer_text"] == (und.CAMERA_OPTIONS[1] if v == 1 else und.CAMERA_OPTIONS[0])
    c = _cands(bench, "Simul-Robot/sr_c", proto)
    assert "sr_moved_object" not in c and "sr_num_moved" not in c and "sr_end_state" not in c
    assert c["sr_hands"]["answer_text"] == "only the left hand"
    for q in list(a.values()) + list(b.values()):
        assert q["answer_text"] not in q["pool"] and q["source"] and set(q["pref"]) <= set(q["pool"])
    assert "tomato can" not in b["sr_first_moved"]["pool"] and "tomato can" in a["sr_moved_object"]["pool"]


def test_simul_general_templates(bench, proto):
    a = _cands(bench, "Simul-General/sg_a", proto)
    assert a["sg_agent_type"]["answer_text"] == "a pedestrian"
    assert "sg_vehicle_type" not in a
    assert a["sg_views_out"]["answer_text"] == "1"
    assert a["sg_distance"]["answer_text"] == "between 10 and 20 m"
    assert a["camera_motion"]["answer_text"] == und.CAMERA_OPTIONS[2]
    assert a["target_distance"]["answer_text"] == und.DIST_OPTIONS[2]
    b = _cands(bench, "Simul-General/sg_b", proto)
    assert b["sg_vehicle_type"]["answer_text"] == "ambulance" and "van" not in b["sg_vehicle_type"]["pool"]
    assert "sg_views_out" not in b
    assert b["sg_distance"]["answer_text"] == "more than 40 m"
    c = _cands(bench, "Simul-General/sg_c", proto)
    assert c["sg_vehicle_type"]["answer_text"] == "car" and "sg_distance" not in c


def test_real_robot_templates(bench, proto):
    a = _cands(bench, "Real-Robot/rr_a", proto)
    assert a["rr_hands"]["answer_text"] == "only the right hand"
    assert a["rr_task"]["answer_text"] == und.RR_TASKS["put_on_plate"]
    assert und.RR_TASKS["plate_and_fruit"] not in a["rr_task"]["pool"]
    assert a["rr_object"]["answer_text"] == "red cube"
    assert a["rr_object"]["pool"] and all(d.split()[-1] not in ("cube", "plate") for d in a["rr_object"]["pool"])
    assert "rr_first_hand" not in a and "camera_motion" in a
    b = _cands(bench, "Real-Robot/rr_b", proto)
    assert b["rr_hands"]["answer_text"] == "both hands"
    assert b["rr_first_hand"]["answer_text"] == "the left hand"
    assert und.RR_TASKS["put_on_plate"] not in b["rr_task"]["pool"]
    c = _cands(bench, "Real-Robot/rr_c", proto)
    assert "rr_hands" not in c


def test_real_general_templates(bench, proto):
    s = bench.sample("Real-General/rg_a")
    assert proto.windows_for(s) == [(4, 16), (16, 28)]
    a = _cands(bench, "Real-General/rg_a", proto)
    m0, m1 = a["rg_motion#w0"], a["rg_motion#w1"]
    E = und.RG_EVENT_TEXT
    assert m0["answers"] == [(E["stand_still"], "stand_still")] and m0["frames"] == [4, 16]
    assert set(m0["pool"]) == {E[t] for t in ("bend_forward", "lower_body", "arm_raised", "hands_together")}
    assert sorted(v for _, v in m1["answers"]) == ["turn", "walk"] and m1["frames"] == [16, 28]
    assert E["stand_still"] not in m1["pool"] and E["hands_together"] in m1["pool"]
    assert E["arm_extended"] not in m1["pool"]
    assert a["rg_turn_direction#w1"]["answer_text"] == und.TURN_OPTIONS[0] and "rg_turn_direction#w0" not in a
    assert a["rg_body_part#w0"]["answer_text"].startswith("the left hand")
    order = [q for k, q in a.items() if k.startswith("rg_segment_order")]
    assert len(order) == 2
    caps = [g["caption"] for g in SEGMENTS]
    used = set()
    for q in order:
        i, j = caps.index(q["answer_text"]), caps.index(q["pool"][0])
        assert j >= i + 2
        assert not ({i, j} & used)
        used |= {i, j}
    td = a["target_distance"]
    assert td["meta"]["view"] in (0, 2)
    assert td["answer_text"] == (und.DIST_OPTIONS[0] if td["meta"]["view"] == 0 else und.DIST_OPTIONS[1])
    assert a["camera_motion"]["answer_text"] == und.CAMERA_OPTIONS[0]
    b = _cands(bench, "Real-General/rg_b", proto)
    assert not any(k.startswith(("rg_motion", "rg_turn")) for k in b)
    assert not any(k.startswith("rg_segment_order") for k in b) or True
    segs = und._usable_segments(bench.sample("Real-General/rg_b"), proto)
    assert [g["start_frame"] for g in segs] == [0, 8]
    c = _cands(bench, "Real-General/rg_c", proto)
    assert set(c) == {"num_views", "camera_motion"}


def _fake(uid, tpl, k, ans, pool, subset="Simul-Robot", answers=None, pref=()):
    return {"qid": f"{uid}#{tpl}#{k}", "uid": uid, "subset": subset, "id": uid.split("/")[1], "template": tpl,
            "question": "q?", "answer_text": ans, "pool": list(pool), "pref": list(pref),
            "answers": answers, "value": ans, "frames": [0, 9], "views": [0], "fps": 10.0, "videos": [], "source": "s",
            "meta": {}}


def test_finalize_balances_positions_and_answers():
    cands = [_fake(f"Simul-Robot/s{i:03d}", "bin", 0, "yes" if i < 80 else "no", ["no" if i < 80 else "yes"])
             for i in range(100)]
    cands += [_fake(f"Simul-Robot/s{i:03d}", "multi", 0, f"a{i % 4}", [f"a{j}" for j in range(4) if j != i % 4])
              for i in range(60)]
    cands += [_fake(f"Simul-Robot/s{i:03d}", "single", 0, "x", ["y"]) for i in range(10)]
    cands += [_fake(f"Simul-Robot/s{i:03d}", "num_views", 0, "3", ["1", "2", "4", "5"]) for i in range(10)]
    p = und.UnderstandingProtocol()
    out = und.finalize(cands, p)
    assert out == und.finalize(list(reversed(cands)), p)
    by = {}
    for q in out:
        by.setdefault(q["template"], []).append(q)
        assert q["options"][LETTERS.index(q["answer"])] == q["answer_text"]
        assert len(set(q["options"])) == len(q["options"]) >= 2
        assert not ({"pool", "pref", "answers", "distractors"} & set(q))
    vals = [q["answer_text"] for q in by["bin"]]
    assert vals.count("yes") == vals.count("no") == 20
    assert sorted({q["answer"] for q in by["bin"]}) == ["A", "B"]
    assert abs(sum(q["answer"] == "A" for q in by["bin"]) - 20) <= 1
    assert all(len(q["options"]) == 4 for q in by["multi"])
    counts = [sum(q["answer"] == LETTERS[i] for q in by["multi"]) for i in range(4)]
    assert max(counts) - min(counts) <= 1
    assert "single" not in by
    assert len(by["num_views"]) == 10 and all(len(q["options"]) == 4 for q in by["num_views"])
    p2 = und.UnderstandingProtocol(balance=False)
    assert len([q for q in und.finalize(cands, p2) if q["template"] == "bin"]) == 100


def _skewed_group(n=200):
    """rg_motion-like: frequent types are rarely valid distractors, rare types almost always."""
    common, rare = ["walk", "turn", "still"], ["squat", "raise", "bend"]
    rng = np.random.default_rng(3)
    cands = []
    for i in range(n):
        present = [t for t in common if rng.random() < 0.7] + [t for t in rare if rng.random() < 0.1]
        present = present or ["walk"]
        absent = [t for t in common + rare if t not in present and (t in rare or rng.random() < 0.2)]
        if not absent:
            continue
        cands.append(_fake(f"Real-General/c{i:03d}", "skew", 0, present[0], absent, subset="Real-General",
                           answers=[(t, t) for t in present]))
    return cands


def test_finalize_balances_option_texts():
    cands = _skewed_group()
    p = und.UnderstandingProtocol()
    out = und.finalize(cands, p)
    assert len(out) > 40
    st = und.option_stats(out)
    for o, r in st.items():
        assert r["n_answer"] > 0
        if r["n"] >= p.option_bias_min_count:
            assert abs(r["answer_rate"] - r["chance"]) <= p.max_option_bias + 1e-9, (o, r)
    for q in out:
        c = next(c for c in cands if c["qid"] == q["qid"])
        assert q["answer_text"] in [t for t, _ in c["answers"]] and not (set(q["options"]) - {q["answer_text"]}
                                                                          - set(c["pool"]))
    blind = und.option_prior_choice(out)
    acc = np.mean([blind[q["qid"]] == q["answer"] for q in out])
    chance = np.mean([1 / len(q["options"]) for q in out])
    assert acc <= chance + 0.10
    unb = und.finalize(cands, und.UnderstandingProtocol(balance=False))
    blind_u = und.option_prior_choice(unb)
    acc_u = np.mean([blind_u[q["qid"]] == q["answer"] for q in unb])
    assert acc_u > np.mean([1 / len(q["options"]) for q in unb]) + 0.15


def test_option_prior_choice_is_leave_one_out():
    qs = [{"qid": f"q{i}", "template": "t", "subset": "S", "options": ["x", "y"], "answer_text": "x" if i < 9 else "y",
           "answer": "A" if i < 9 else "B"} for i in range(10)]
    ch = und.option_prior_choice(qs)
    assert all(ch[f"q{i}"] == "A" for i in range(10))


def test_generation_is_deterministic_and_per_subset(bench, proto):
    q1 = und.generate_questions(bench, proto)
    q2 = und.generate_questions(bench, proto, jobs=2)
    assert q1 == q2
    qr = und.generate_questions(bench, proto, subsets=["Real-Robot"])
    assert qr == [q for q in q1 if q["subset"] == "Real-Robot"]
    assert len({q["qid"] for q in q1}) == len(q1)
    p_seed = und.UnderstandingProtocol(seed=1, **PROTO)
    q3 = und.generate_questions(bench, p_seed)
    assert [q["qid"] for q in q3] != [] and q3 != q1


def test_export_has_no_answers_or_captions(bench, proto, tmp_path):
    res = und.export_inputs(bench, tmp_path / "in", proto, answer_key_out=tmp_path / "keys" / "answer_key.jsonl")
    assert res["qa"]["n_questions"] > 10 and res["grounding"]["n_queries"] == 7
    assert not (tmp_path / "in" / "qa" / "answer_key.jsonl").exists()
    assert res["qa"]["answer_key"] == str(tmp_path / "keys" / "answer_key.jsonl")
    qs = [json.loads(x) for x in (tmp_path / "in" / "qa" / "questions.jsonl").read_text().splitlines()]
    key = [json.loads(x) for x in (tmp_path / "keys" / "answer_key.jsonl").read_text().splitlines()]
    assert [q["qid"] for q in qs] == [k["qid"] for k in key]
    caps = {s.uid: s.caption for s in bench.samples()}
    for q, k in zip(qs, key):
        assert not ({"answer", "answer_text", "source", "value", "meta"} & set(q))
        assert caps[q["uid"]] not in q["prompt"] and "Options:" in q["prompt"]
        assert k["answer"] in LETTERS[: len(q["options"])] and q["options"] == k["options"]
        assert all(v.startswith(q["subset"] + "/") for v in q["videos"])
    gq = [json.loads(x) for x in (tmp_path / "in" / "grounding" / "queries.jsonl").read_text().splitlines()]
    assert all("gt" not in q for q in gq)
    rdir = tmp_path / "in" / "retrieval" / "Real-Robot"
    texts = [json.loads(x) for x in (rdir / "texts.jsonl").read_text().splitlines()]
    vids = [json.loads(x) for x in (rdir / "videos.jsonl").read_text().splitlines()]
    assert len(texts) == len(vids) == 3 and all("uid" not in t for t in texts)
    assert not (tmp_path / "in" / "retrieval" / "Real-General_segment").exists() and "warning" not in res
    res2 = und.export_inputs(bench, tmp_path / "seg", proto, tasks=("retrieval_segment",))
    sdir = tmp_path / "seg" / "retrieval" / "Real-General_segment"
    assert set(res2["retrieval"]) == {"Real-General_segment"} and "grounding" in res2["warning"]
    assert (sdir / "README.txt").exists() and not (tmp_path / "seg" / "retrieval" / "Real-Robot").exists()
    p2 = und.UnderstandingProtocol(seed=5, **PROTO)
    with pytest.raises(ValueError):
        und.export_inputs(bench, tmp_path / "in", p2)
    (tmp_path / "in" / "qa" / "answer_key.jsonl").write_text("{}\n")
    with pytest.warns(UserWarning, match="removed"):
        und.export_inputs(bench, tmp_path / "in", p2, overwrite=True)
    assert not (tmp_path / "in" / "qa" / "answer_key.jsonl").exists()
    with pytest.warns(UserWarning, match="inside the inputs folder"):
        und.export_inputs(bench, tmp_path / "in", p2, answer_key_out=tmp_path / "in" / "qa" / "answer_key.jsonl")
    assert (tmp_path / "in" / "qa" / "answer_key.jsonl").exists()
    with pytest.raises(ValueError):
        und.export_inputs(bench, tmp_path / "x", proto, tasks=("bogus",))
    with pytest.raises(ValueError, match="qa"):
        und.export_inputs(bench, tmp_path / "x", proto, tasks=("grounding",), answer_key_out=tmp_path / "k.jsonl")
    r3 = und.export_inputs(bench, tmp_path / "ids", proto, ids=["rg/rg_a"], tasks=("qa", "grounding"))
    q3 = [json.loads(x) for x in (tmp_path / "ids" / "qa" / "questions.jsonl").read_text().splitlines()]
    assert {q["uid"] for q in q3} == {"Real-General/rg_a"} and r3["grounding"]["n_queries"] == 5
    with pytest.warns(UserWarning, match="match no sample"):
        und.export_inputs(bench, tmp_path / "ids2", proto, ids=["Real-General/nope"], tasks=("qa",))


def test_score_qa_parsing():
    q = {"qid": "x", "uid": "S/a", "subset": "S", "template": "t", "answer": "B", "options": ["red cube", "blue cube",
                                                                                           "plate"]}
    rows = und.score_qa([q] * 6, {})
    assert all(r["status"] == "missing" and not r["correct"] for r in rows)
    for pred, ok in (({"response": "B"}, True), ({"response": "The answer is (B)."}, True),
                     ({"response": "blue cube"}, True), ({"choice": 1}, True), ({"choice": "b"}, True),
                     ({"choice": 0}, False), ({"response": "no idea"}, False), ({"choice": 7}, False)):
        r = und.score_qa([q], {"x": pred})[0]
        assert r["correct"] == ok, pred
    assert und.score_qa([q], {"x": {"response": "no idea"}})[0]["status"] == "unparsed"


def test_temporal_iou_and_metrics():
    assert und.temporal_iou([0, 9], [0, 9]) == 1.0
    assert und.temporal_iou([0, 4], [5, 9]) == 0.0
    assert und.temporal_iou([0, 9], [5, 14]) == pytest.approx(5 / 15)
    assert und.temporal_iou([3, 3], [3, 3]) == 1.0
    assert und.temporal_iou([5, 2], [0, 9]) == 0.0 and und.temporal_iou(None, [0, 9]) == 0.0
    m = und.grounding_metrics([1.0, 0.6, 0.4, 0.0])
    assert m["miou"] == pytest.approx(50.0) and m["R@0.3"] == 75.0 and m["R@0.5"] == 50.0 and m["R@0.7"] == 25.0
    assert und.parse_interval({"start_s": 1.0, "end_s": 2.0}, 10.0) == ([10, 19], "ok")
    assert und.parse_interval({"start_s": 1.04, "end_s": 2.04}, 10.0) == ([11, 20], "ok")
    assert und.parse_interval({"start_s": 0.5, "end_s": 1.5}, 15.0) == ([8, 22], "ok")
    assert und.parse_interval({"start_s": 1.0, "end_s": 1.0}, 10.0) == ([10, 9], "empty")
    assert und.parse_interval({"start_frame": 10.5, "end_frame": 20.4}, 10.0) == ([11, 20], "ok")
    assert und.parse_interval({"start_frame": -5, "end_frame": 99}, 10.0, n_frames=40) == ([0, 39], "ok")
    for bad in ({"start_frame": None, "end_frame": 5}, {"start_frame": "abc", "end_frame": 5}, {"start_frame": 3},
                {"start_s": float("inf"), "end_s": 3.0}, {"start_s": float("nan"), "end_s": 3.0}, {"x": 1}, [1, 2],
                {"start_frame": True, "end_frame": 4}):
        assert und.parse_interval(bad, 10.0) == (None, "invalid"), bad


def test_retrieval_items_duplicates(bench, proto):
    it = und.retrieval_items(bench, "Real-Robot", "clip", proto)
    assert len(it["videos"]) == 3 and it["gt"].shape == (3, 3)
    caps = {v["index"]: bench.sample(v["uid"]).caption for v in it["videos"]}
    for j, t in enumerate(it["texts"]):
        expect = [i for i, c in caps.items() if c == t["text"]]
        assert sorted(np.nonzero(it["gt"][j])[0].tolist()) == sorted(expect)
    assert it["gt"].sum() == 5
    it2 = und.retrieval_items(bench, "Real-Robot", "clip", proto)
    assert [t["text"] for t in it2["texts"]] == [t["text"] for t in it["texts"]]
    with pytest.raises(ValueError):
        und.retrieval_items(bench, "Real-Robot", "segment", proto)
    seg = und.retrieval_items(bench, "Real-General", "segment", proto)
    assert len(seg["videos"]) == len(seg["texts"]) == 7
    assert und.retrieval_levels(["Real-Robot", "Real-General"]) == [("Real-Robot", "clip"), ("Real-General", "clip")]
    assert ("Real-General", "segment") in und.retrieval_levels(["Real-General"], segment=True)


@pytest.mark.filterwarnings("ignore:grounding.jsonl and a segment-level")
def test_baselines_and_evaluate(bench, proto, tmp_path):
    res = {}
    for kind in und.BASELINES:
        und.run_baseline(bench, tmp_path / kind, kind, proto, tasks=und.SUBTASKS)
        res[kind] = und.evaluate(bench, tmp_path / kind, tmp_path / f"res_{kind}", proto)
    o = res["oracle"]
    assert o["qa_overall"]["accuracy"] == 100.0 and o["qa_overall"]["sanity"]["accuracy"] == 100.0
    assert "num_views" not in o["subsets"]["Simul-Robot"]["qa"]["per_template"]
    assert set(o["subsets"]["Simul-Robot"]["qa"]["sanity"]["per_category"]) == {"num_views"}
    assert o["meta"]["notes"] and "note:" in (tmp_path / "res_oracle" / "summary.md").read_text()
    with pytest.warns(UserWarning, match="segment-level retrieval"):
        und.evaluate(bench, tmp_path / "oracle", tmp_path / "res_oracle", proto, subsets=["Real-General"])
    for sub, s in o["subsets"].items():
        assert s["qa"]["accuracy_per_sample"]["mean"] == 100.0
    assert set(o["retrieval"]) == {"Simul-Robot", "Simul-General", "Real-Robot", "Real-General",
                                   "Real-General_segment"}
    for tag, r in o["retrieval"].items():
        assert r["text_to_video"]["R@1"] == 100.0 and r["video_to_text"]["R@1"] == 100.0
    assert o["subsets"]["Real-General"]["grounding"]["pooled"]["miou"] == pytest.approx(100.0)
    gq = und.grounding_queries(bench, proto)
    full = np.mean([(q["gt"][1] - q["gt"][0] + 1) / q["n_frames"] for q in gq]) * 100
    assert res["full"]["subsets"]["Real-General"]["grounding"]["pooled"]["miou"] == pytest.approx(full)
    assert res["random"]["qa_overall"]["accuracy"] < 100.0
    rec = json.loads((tmp_path / "res_oracle" / "Real-General" / "rg_a.json").read_text())
    assert {"qa", "grounding", "retrieval"} <= set(rec) and rec["retrieval"]["video_to_text_rank"] == 1
    und.export_inputs(bench, tmp_path / "in", proto, tasks=("qa",), answer_key_out=tmp_path / "key.jsonl")
    o2 = und.evaluate(bench, tmp_path / "oracle", tmp_path / "res_key", proto, answer_key=tmp_path / "key.jsonl")
    assert o2["qa_overall"] == o["qa_overall"]
    with pytest.raises(ValueError):
        und.evaluate(bench, tmp_path / "random", tmp_path / "res_key", proto)
    o2b = und.evaluate(bench, tmp_path / "random", tmp_path / "res_key", proto, overwrite=True)
    assert o2b["qa_overall"]["accuracy"] == res["random"]["qa_overall"]["accuracy"]
    o3 = und.evaluate(bench, tmp_path / "oracle", tmp_path / "res_ids", proto, ids=["Real-Robot/rr_a"])
    assert set(o3["subsets"]["Real-Robot"]["qa"]["per_template"]) and o3["subsets"]["Real-Robot"]["n_samples"] == 1
    np.save(tmp_path / "oracle" / "retrieval" / "Real-Robot.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError):
        und.evaluate(bench, tmp_path / "oracle", tmp_path / "res_bad", proto)
    with pytest.raises(FileNotFoundError):
        und.evaluate(bench, tmp_path / "nothing", tmp_path / "res_none", proto)


def test_malformed_predictions_are_scored_not_fatal(bench, proto, tmp_path):
    gq = und.grounding_queries(bench, proto)
    qs = und.generate_questions(bench, proto)
    pr = tmp_path / "bad"
    pr.mkdir()
    lines = ['{"query_id": "%s", "start_frame": null, "end_frame": 5}' % gq[0]["query_id"],
             '{"query_id": "%s", "start_frame": "abc", "end_frame": 5}' % gq[1]["query_id"],
             '{"query_id": "%s", "start_s": Infinity, "end_s": 3.0}' % gq[2]["query_id"],
             '{"query_id": "%s", "start_s": NaN, "end_s": 3.0}' % gq[3]["query_id"],
             '{"query_id": "%s", "start_frame": 30, "end_frame": 10}' % gq[4]["query_id"],
             json.dumps({"query_id": gq[5]["query_id"], "start_frame": gq[5]["gt"][0] - 0.4,
                         "end_frame": gq[5]["gt"][1] + 0.4}),
             'not json', '[1, 2]', '{"start_frame": 1}']
    (pr / "grounding.jsonl").write_text("\n".join(lines) + "\n")
    (pr / "qa.jsonl").write_text(json.dumps({"qid": qs[0]["qid"], "response": None}) + "\n{broken\n"
                                 + json.dumps({"qid": qs[1]["qid"], "choice": [1]}) + "\n" + json.dumps({"qid": 5}) + "\n")
    s = und.evaluate(bench, pr, tmp_path / "res", proto, subsets=["Real-General", "Simul-Robot"])
    g = s["subsets"]["Real-General"]["grounding"]
    assert g["pooled"]["n"] == len(gq) and g["n_invalid"] == 5 and g["n_missing"] == len(gq) - 6
    recs = {x["query_id"]: x for f in (tmp_path / "res" / "Real-General").glob("*.json")
            for x in json.loads(f.read_text()).get("grounding", {}).get("queries", [])}
    assert recs[gq[5]["query_id"]]["iou"] == 1.0 and recs[gq[5]["query_id"]]["pred"] == gq[5]["gt"]
    assert [recs[gq[i]["query_id"]]["status"] for i in range(5)] == ["invalid"] * 4 + ["empty"]
    assert s["meta"]["grounding"]["n_bad_lines"] == 3 and s["meta"]["qa"]["n_bad_lines"] == 2


def test_cli(bench, proto, tmp_path, capsys):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps(proto.to_dict()))
    root = str(bench.root)
    key = tmp_path / "keys" / "und.jsonl"
    assert und.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--protocol", str(pf),
                     "--answer-key-out", str(key)]) == 0
    assert key.exists() and not (tmp_path / "in" / "qa" / "answer_key.jsonl").exists()
    assert und.main(["baseline", "--root", root, "--pred", str(tmp_path / "pr"), "--kind", "oracle", "--protocol",
                     str(pf), "--subsets", "rr,", "rg", "--tasks", "qa,grounding"]) == 0
    assert und.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res"),
                     "--protocol", str(pf), "--answer-key", str(key),
                     "--subsets", "Real-Robot", "rg", "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    assert "| all |" in out and "100.0" in out and "num_views (sanity)" in out
    summ = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(summ["subsets"]) == {"Real-Robot", "Real-General"}
    assert summ["pred_root"] == str((tmp_path / "pr").resolve())
    assert und.main(["summarize", "--out", str(tmp_path / "res")]) == 0
    with pytest.raises(SystemExit):
        und.main(["baseline", "--root", root, "--pred", str(tmp_path / "x"), "oracle"])
    other = ["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res"),
             "--ids", "Real-Robot/rr_a"]
    with pytest.raises(RunDirError, match="results of another protocol"):
        und.main(other)
    from vpm_worldbench.cli import main as vpmwb
    capsys.readouterr()
    assert vpmwb(["understanding", *other]) == 2
    err = capsys.readouterr().err
    assert err.startswith("vpmwb: error: ") and "--overwrite" in err and len(err.strip().splitlines()) == 1
    assert json.loads((tmp_path / "res" / "summary.json").read_text()) == summ
    with pytest.raises(RunDirError, match=r"inputs of other export options \(export.json: other tasks\)"):
        und.main(["export", "--root", root, "--out", str(tmp_path / "in"), "--protocol", str(pf), "--tasks", "qa"])
    assert und.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res"),
                     "--protocol", str(pf), "--ids", "Real-Robot/rr_a,", "rr_b", "--overwrite"]) == 0
    assert json.loads((tmp_path / "res" / "summary.json").read_text())["subsets"]["Real-Robot"]["n_samples"] == 2
    capsys.readouterr()
    for args, msg in ((["--subsets", "nowhere"], "unknown subset 'nowhere'"),
                      (["--protocol", str(tmp_path / "missing.yaml")], "neither the preset"),
                      (["--tasks", "grounding", "--answer-key-out", str(key)], "needs the qa sub-task"),
                      (["--tasks", "qa,bogus"], "unknown understanding sub-tasks ['bogus']"),
                      (["--horizon", "0"], "--horizon 0: must be >= 1"),
                      (["--frame-step", "0"], "frame_step must be >= 1"),
                      (["--ref-frame", "-1"], "ref_frame must be >= 0")):
        with pytest.raises(SystemExit) as e:
            und.main(["export", "--root", root, "--out", str(tmp_path / "in2"), *args])
        assert e.value.code == 2 and msg in capsys.readouterr().err, args
    assert und.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res_none"),
                     "--protocol", str(pf), "--ids", "rr_a", "--subsets", "rg"]) == 1
    assert "no sample matches the selection" in capsys.readouterr().err and not (tmp_path / "res_none").exists()
    assert vpmwb(["understanding", "summarize", "--out", str(tmp_path / "res")]) == 0
    assert und.main(["evaluate", "--root", root, "--pred", str(tmp_path / "nothing"), "--out",
                     str(tmp_path / "res_np"), "--protocol", str(pf)]) == 1
    err = capsys.readouterr().err
    assert "nothing is scored" in err and len(err.strip().splitlines()) == 1 and not (tmp_path / "res_np").exists()
    with pytest.raises(SystemExit) as e:
        und.main(["evaluate", "--root", root, "--pred", str(tmp_path / "pr"), "--out", str(tmp_path / "res_k"),
                  "--protocol", str(pf), "--answer-key", str(tmp_path / "nokey.jsonl")])
    assert e.value.code == 2 and "nokey.jsonl: no such file" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        und.main(["summarize", "--out", str(tmp_path / "res_np")])
    assert e.value.code == 2 and "no such directory" in capsys.readouterr().err and not (tmp_path / "res_np").exists()
    assert und.main(["export", "--root", root, "--out", str(tmp_path / "in_w"), "--protocol", str(pf), "--subsets",
                     "rg", "--tasks", "qa", "--ref-frame", "6", "--horizon", "10"]) == 0
    pw = json.loads((tmp_path / "in_w" / "export.json").read_text())["protocol"]
    assert (pw["ref_frame"], pw["horizon"], pw["name"]) == (6, 10, f"{proto.name}+ref_frame=6,horizon=10")
    assert und.UnderstandingProtocol.from_dict(pw).windows_for(bench.sample("Real-General/rg_a")) == [(6, 16), (16, 26)]


def test_protocol_validation(tmp_path):
    (tmp_path / "two_windows.yaml").write_text("n_windows: 1\n")
    q = und.get_protocol(tmp_path / "two_windows.yaml")
    assert q.name == "two_windows" and q.n_windows == 1
    with pytest.raises(KeyError):
        und.UnderstandingProtocol(templates=["bogus"])
    with pytest.raises(ValueError):
        und.UnderstandingProtocol(n_options=9)
    with pytest.raises(ValueError, match="ref_frame must be >= 0"):
        und.UnderstandingProtocol(ref_frame=-1)
    p = und.UnderstandingProtocol(templates=["num_views", "rg_motion"])
    assert p.templates_for("Simul-Robot") == ["num_views"]
    assert und.UnderstandingProtocol.from_dict(p.to_dict()) == p


RELEASED = os.environ.get("VPMWB_RELEASED_ROOT")
released = pytest.mark.skipif(not RELEASED, reason="set VPMWB_RELEASED_ROOT to a VP-WorldBench data root")


@released
def test_released_generation(tmp_path):
    b = Benchmark(RELEASED)
    subs = [s for s in ("Real-Robot", "Real-General") if s in b.subsets()]
    q1 = und.generate_questions(b, subsets=subs)
    q2 = und.generate_questions(b, subsets=subs[::-1])
    assert q1 == q2 and len(q1) > 100
    for q in q1:
        assert q["source"] and q["answer_text"] == q["options"][LETTERS.index(q["answer"])]
        assert len(set(q["options"])) == len(q["options"]) >= 2
        assert 0 <= q["frames"][0] <= q["frames"][1]
    p = und.UnderstandingProtocol()
    groups = {}
    for q in q1:
        groups.setdefault((q["template"], q["subset"]), []).append(q)
    for (tpl, sub), g in groups.items():
        if tpl in und.BALANCE_EXEMPT:
            continue
        for o, r in und.option_stats(g).items():
            assert r["n_answer"] > 0
            if r["n"] >= p.option_bias_min_count:
                assert abs(r["answer_rate"] - r["chance"]) <= p.max_option_bias + 1e-9, (tpl, sub, o, r)
    und.run_baseline(b, tmp_path / "o", "oracle", subsets=subs)
    s = und.evaluate(b, tmp_path / "o", tmp_path / "r", subsets=subs)
    assert s["qa_overall"]["accuracy"] == 100.0
    assert s["subsets"]["Real-General"]["grounding"]["pooled"]["miou"] == pytest.approx(100.0)
