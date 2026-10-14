# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-view world-model samples with cross-view 3D point tracks and per-view camera conditioning.

Dataset layout (``vision_path`` of each jsonl row is the anchor view of a scene, absolute or relative to the jsonl)::

    <root>/videos/{scene}__{view}.mp4
    <root>/camera/{scene}_cameras.json      {view: {measured_intrinsics, optimized_extrinsics}}
    <root>/tracks/{scene}__{view}_3d.npz    points_3d [T,N,3] in the view's camera frame, valid_3d [T,N]
    <root>/tracks/{scene}__{view}_2d.npz    visibility [T,N]

``measured_intrinsics`` is a pixel 3x3 K at the jsonl ``(width, height)``. ``optimized_extrinsics`` is a
world-to-camera 4x4 (static camera) or [T,4,4] (moving camera, indexed by video frame), in metres.

A sample holds V views of the same temporal window: the anchor plus up to ``num_views - 1`` sibling views.

* ``video``: list of V float tensors [3,T,H,W] in [-1, 1].
* ``action``: 9D framewise camera action per view (domain ``camera_pose``), [V*(T-1), max_action_dim].
* ``camera_c2w``, ``camera_intrinsics``, ``camera_distortion``, ``camera_model``, ``camera_image_size``:
  per-view camera bundle for ray conditioning (see ``camera_bundle``).
* ``point_tracks`` [V,T,N,2]: normalised (x, y); point n is the same physical 3D point in every view.
* ``track_visibility``, ``track_valid``, ``track_vis_valid`` [V,T,N]; ``track_depth`` [V,T,N] metric depth in
  each view's camera frame; ``track_src`` [N] source view of each point.
* ``track_intrinsics`` [V,3,3] image-normalised; ``track_extrinsics`` [V,T,4,4] world-to-camera.
* ``sequence_plan``: clean past latents followed by generated future latents.
"""

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.local_datasets.camera_bundle import (
    build_camera_bundle,
    build_framewise_camera_action,
)
from cosmos_framework.data.vfm.local_datasets.helper import download_from_s3, ffmpeg_decode_video
from cosmos_framework.data.vfm.local_datasets.sft_dataset import SFTDataset, get_sft_dataset
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils import log

CAMERA_DOMAIN = "camera_pose"
CAMERA_RAW_DIM = 9


def _decode_video_window(
    s3_client,
    vision_path: str,
    start_frame: int,
    end_frame: int,
    temporal_interval: int,
    resize_hw: tuple[int, int],
    crop_yx: tuple[int, int],
    target_hw: tuple[int, int],
) -> torch.Tensor | None:
    """Decode frames ``start_frame..end_frame`` (step ``temporal_interval``) of one video with the same resize and
    centre crop as ``SFTDataset.process_one_sample``. Returns uint8 [3,T,H,W], or None if nothing was decoded."""
    resize_h, resize_w = resize_hw
    crop_y, crop_x = crop_yx
    target_h, target_w = target_hw
    video_bytes = download_from_s3(s3_client, vision_path)
    if video_bytes is None:
        return None
    chunk = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        for idx, frame in enumerate(ffmpeg_decode_video(tmp.name, scale_hw=(resize_h, resize_w), num_threads=2)):
            if idx < start_frame:
                continue
            if idx <= end_frame:
                if (idx - start_frame) % temporal_interval == 0:
                    chunk.append(frame)
            else:
                break
    if not chunk:
        return None
    arr = np.stack(chunk, axis=0)
    arr = arr[:, crop_y : crop_y + target_h, crop_x : crop_x + target_w]
    arr = np.transpose(arr, (3, 0, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.uint8)


def _camera_views(cams_json: dict) -> list[str]:
    return [s for s in cams_json if s != "optimization_summary"]


class MultiViewTrackDataset(SFTDataset):
    """SFTDataset producing multi-view world-model samples with cross-view point tracks (see module docstring).
    Built via ``get_multiview_track_dataset``, which sets the ``_mvt_*`` attributes."""

    def _mvt_get(self, name, default):
        return getattr(self, name, default)

    def _mvt_rng(self) -> random.Random:
        rng = getattr(self, "_mvt_rng_obj", None)
        if rng is None:
            rng = random.Random()
            self._mvt_rng_obj = rng
        return rng

    def _mvt_load_json(self, path: Path) -> dict:
        cache = getattr(self, "_mvt_json_cache", None)
        if cache is None:
            cache = {}
            self._mvt_json_cache = cache
        key = str(path)
        if key not in cache:
            with open(path) as f:
                cache[key] = json.load(f)
            if len(cache) > 4096:
                cache.clear()
        return cache[key]

    @staticmethod
    def _mvt_K(K_raw) -> np.ndarray:
        """Pixel intrinsics at the jsonl ``(width, height)``, used as stored."""
        K = np.array(K_raw, np.float64).copy()
        assert K.shape == (3, 3), f"[mvt] expected a 3x3 K, got {K.shape}"
        return K

    def _mvt_camera_condition(self, view_sers, c2w, K, geom: dict):
        """9D framewise camera action (one block per view) and the per-view camera bundle."""
        ts = self._mvt_get("_mvt_translation_scale", 1.0)
        max_dim = self._mvt_get("_mvt_max_action_dim", 64)
        action = torch.cat(
            [build_framewise_camera_action(c2w[s], translation_scale=ts, max_action_dim=max_dim) for s in view_sers],
            dim=0,
        )
        camera_bundle = build_camera_bundle(
            c2w_views=[c2w[s] for s in view_sers],
            intrinsics_views=[K[s] for s in view_sers],
            native_sizes=[(geom["input_height"], geom["input_width"])] * len(view_sers),
            unit_to_meters=1.0,
            **geom,
        )
        return action, camera_bundle

    def _mvt_cross_view_tracks(self, root: Path, scene: str, view_sers, E, K, frame_idx, input_w, input_h):
        """Union of every view's 3D tracks, projected into all views: ``x_tgt = K_tgt E_tgt E_src^-1 X_src``.

        Visibility in the source view is the annotated ``valid_3d & visibility``. In another view it is
        estimated in three tiers: (1) a mutual nearest neighbour (< 5 cm in world) among the target view's own
        tracks inherits that track's visibility; (2) a nearer target-view point (>= 10 cm closer, < 25 px away)
        marks the point occluded; (3) otherwise in-bounds and positive depth, with the label marked untrusted
        (``track_vis_valid`` = 0). Returns pixel coords [V,T,N,2], visibility, vis trust, depth [V,T,N] and the
        source view index [N], or None when no view has tracks."""
        sel_cache = {}

        def _npz_window(ser: str):
            d3 = np.load(root / "tracks" / f"{scene}__{ser}_3d.npz")
            d2 = np.load(root / "tracks" / f"{scene}__{ser}_2d.npz")
            T_full = d3["points_3d"].shape[0]
            if T_full not in sel_cache:
                sel_cache[T_full] = frame_idx(T_full)
            idx = sel_cache[T_full]
            P = d3["points_3d"][idx].astype(np.float64)
            val = d3["valid_3d"][idx] & d2["visibility"][idx]
            return P, val

        per_view = {}
        for src in view_sers:
            P, val = _npz_window(src)
            Ph = np.concatenate([P, np.ones_like(P[..., :1])], axis=-1)
            per_view[src] = dict(P=P, val=val, Xw=np.einsum("tij,tnj->tni", np.linalg.inv(E[src]), Ph))

        all_xy, all_vis, all_visok, all_z, all_src = [], [], [], [], []
        for src_i, src in enumerate(view_sers):
            sv = per_view[src]
            N_src = sv["P"].shape[1]
            if N_src == 0:
                continue
            xy_views, vis_views, ok_views, z_views = [], [], [], []
            for tgt in view_sers:
                Xc = np.einsum("tij,tnj->tni", E[tgt], sv["Xw"])[..., :3]
                z = Xc[..., 2]
                uv = Xc @ K[tgt].T
                uv = uv[..., :2] / np.clip(uv[..., 2:3], 1e-6, None)
                inb = (
                    (uv[..., 0] >= 0) & (uv[..., 0] < input_w) & (uv[..., 1] >= 0) & (uv[..., 1] < input_h) & (z > 1e-6)
                )
                if tgt == src:
                    vis, vis_ok = sv["val"], np.ones_like(sv["val"])
                else:
                    vis = sv["val"] & inb
                    vis_ok = np.zeros_like(sv["val"])
                    tv = per_view[tgt]
                    if tv["P"].shape[1] > 0:
                        Tw = vis.shape[0]
                        match = np.full(N_src, -1, np.int64)
                        for t in range(0, Tw, max(1, Tw // 8)):
                            am, tm = sv["val"][t], tv["val"][t]
                            if not (am.any() and tm.any()):
                                continue
                            A = sv["Xw"][t][am][:, :3]
                            Bm = tv["Xw"][t][tm][:, :3]
                            dw = np.linalg.norm(A[:, None] - Bm[None], axis=-1)
                            ia, ib = dw.argmin(1), dw.argmin(0)
                            a_idx, t_idx = np.nonzero(am)[0], np.nonzero(tm)[0]
                            for r in range(dw.shape[0]):
                                c = ia[r]
                                if ib[c] == r and dw[r, c] < 0.05 and match[a_idx[r]] < 0:
                                    match[a_idx[r]] = t_idx[c]
                        mi = match >= 0
                        if mi.any():
                            vis[:, mi] = tv["val"][:, match[mi]] & inb[:, mi]
                            vis_ok[:, mi] = True
                        t_uv = tv["P"] @ K[tgt].T
                        t_uv = t_uv[..., :2] / np.clip(tv["P"][..., 2:3], 1e-6, None)
                        for t in range(Tw):
                            tm = tv["val"][t]
                            if not tm.any():
                                continue
                            tu = t_uv[t][tm]
                            zn = tv["P"][t][tm][:, 2]
                            du = np.linalg.norm(uv[t][:, None] - tu[None], axis=-1)
                            occl = ((du < 25.0) & (zn[None] < z[t][:, None] - 0.10)).any(1) & ~mi
                            vis[t, occl] = False
                            vis_ok[t, occl] = True
                xy_views.append(uv)
                vis_views.append(vis)
                ok_views.append(vis_ok)
                z_views.append(z)
            all_xy.append(np.stack(xy_views, axis=0))
            all_vis.append(np.stack(vis_views, axis=0))
            all_visok.append(np.stack(ok_views, axis=0))
            all_z.append(np.stack(z_views, axis=0))
            all_src.append(np.full(N_src, src_i, np.int64))
        if not all_xy:
            return None
        return (
            np.concatenate(all_xy, axis=2),
            np.concatenate(all_vis, axis=2),
            np.concatenate(all_visok, axis=2),
            np.concatenate(all_z, axis=2),
            np.concatenate(all_src),
        )

    def _mvt_draw_num_future(self, latent_t: int) -> int:
        """Number of generated latents: ``variable_past_frac`` (clean past = random fraction of the clip), else a
        draw from ``variable_past_latents``, else the fixed ``num_future_latents``."""
        vp = self._mvt_get("_mvt_variable_past_latents", None)
        vpf = self._mvt_get("_mvt_variable_past_frac", None)
        if vpf:
            lo, hi = float(vpf[0]), float(vpf[-1])
            p_min = max(1, round(lo * latent_t))
            p_max = min(latent_t - 1, max(p_min, round(hi * latent_t)))
            p_lat = int(self._mvt_rng().randint(p_min, p_max))
            num_future = latent_t - p_lat
        elif vp:
            p_lat = int(self._mvt_rng().choice(list(vp)))
            assert 1 <= p_lat < latent_t, f"[mvt] variable past {p_lat} not in [1,{latent_t - 1}]"
            num_future = latent_t - p_lat
        else:
            num_future = int(self._mvt_get("_mvt_num_future_latents", 4))
        assert 1 <= num_future < latent_t, (
            f"[mvt] num_future_latents={num_future} must be in [1, latent_t-1={latent_t - 1}]"
        )
        return num_future

    def process_one_sample(self, metadata: dict):
        try:
            ret = super().process_one_sample(metadata)
        except Exception as e:
            log.warning(f"[mvt] skipping unreadable anchor {metadata.get('uuid')}: {e}")
            return None
        if ret is None:
            return None

        try:
            anchor = Path(os.path.abspath(metadata["vision_path"]))
            root = anchor.parents[1]
            scene, cam0 = anchor.stem.rsplit("__", 1)
            cams_json = self._mvt_load_json(root / "camera" / f"{scene}_cameras.json")

            def _have(ser: str) -> bool:
                return (
                    (root / "videos" / f"{scene}__{ser}.mp4").exists()
                    and (root / "tracks" / f"{scene}__{ser}_2d.npz").exists()
                    and (root / "tracks" / f"{scene}__{ser}_3d.npz").exists()
                )

            fs, fe, nm = ret["frame_start"], ret["frame_end"], ret["num_multiplier"]
            T = ret["num_frames"]

            def _frame_idx(T_full: int) -> np.ndarray:
                return np.minimum(fs + np.arange(T) * nm, T_full - 1)

            input_w, input_h = metadata["width"], metadata["height"]
            target_w, target_h = self.output_sizes[metadata["aspect_ratio"]]
            resize_ratio = max(target_w / input_w, target_h / input_h)
            resize_h, resize_w = round(input_h * resize_ratio), round(input_w * resize_ratio)
            crop_y, crop_x = round((resize_h - target_h) / 2), round((resize_w - target_w) / 2)
            geom = dict(
                input_height=input_h,
                input_width=input_w,
                resize_height=resize_h,
                resize_width=resize_w,
                crop_y=crop_y,
                crop_x=crop_x,
                output_height=target_h,
                output_width=target_w,
            )

            def _decode(ser: str):
                v = _decode_video_window(
                    self.s3_client,
                    str(root / "videos" / f"{scene}__{ser}.mp4"),
                    fs,
                    fe,
                    nm,
                    (resize_h, resize_w),
                    (crop_y, crop_x),
                    (target_h, target_w),
                )
                if v is None or v.shape[1] < T:
                    log.warning(f"[mvt] view {scene}__{ser} short/unreadable; skipping it")
                    return None
                return v[:, :T]

            num_views = int(self._mvt_get("_mvt_num_views", 2))
            view_sers = [cam0]
            view_videos = [ret["video"]]
            if num_views < 2 and bool(self._mvt_get("_mvt_random_anchor", False)):
                cands = [s for s in _camera_views(cams_json) if _have(s)]
                pick = self._mvt_rng().choice(cands) if cands else cam0
                if pick != cam0:
                    v0 = _decode(pick)
                    if v0 is not None:
                        view_sers = [pick]
                        view_videos = [v0]
            if num_views >= 2:
                others = [s for s in _camera_views(cams_json) if s != cam0 and _have(s)]
                n_sib = num_views - 1
                picks = self._mvt_rng().sample(others, n_sib) if len(others) > n_sib else list(others)
                for ser in picks:
                    v1 = _decode(ser)
                    if v1 is not None:
                        view_sers.append(ser)
                        view_videos.append(v1)
            V = len(view_sers)

            E = {}
            for ser in view_sers:
                e = np.array(
                    cams_json[ser].get("optimized_extrinsics", cams_json[ser].get("vggt_extrinsics")), np.float64
                )
                E[ser] = np.repeat(e[None], T, axis=0) if e.ndim == 2 else e[_frame_idx(e.shape[0])]
            c2w = {ser: np.linalg.inv(E[ser]) for ser in view_sers}
            K = {ser: self._mvt_K(cams_json[ser]["measured_intrinsics"]) for ser in view_sers}

            action, camera_bundle = self._mvt_camera_condition(view_sers, c2w, K, geom)

            tracks = self._mvt_cross_view_tracks(root, scene, view_sers, E, K, _frame_idx, input_w, input_h)
            if tracks is None:
                log.warning(f"[mvt] no 3D tracks for {scene}; skip")
                return None
            g, gv, gok, gz, src_all = tracks

            x = g[..., 0] * resize_ratio - crop_x
            y = g[..., 1] * resize_ratio - crop_y
            inb = (x >= 0) & (x < target_w) & (y >= 0) & (y < target_h)
            valid = (gv & inb).astype(np.float32)
            visf = gv.astype(np.float32)
            xn, yn = (x / target_w).astype(np.float32), (y / target_h).astype(np.float32)

            cand = np.nonzero(valid.any(axis=(0, 1)))[0]
            if len(cand) == 0:
                log.warning(f"[mvt] no valid track points in window for {scene}; skip")
                return None
            Ntgt = int(self._mvt_get("_mvt_num_track_points", 128))
            rng = self._mvt_rng()
            if len(cand) >= Ntgt:
                sel_idx = np.array(rng.sample(cand.tolist(), Ntgt))
            else:
                extra = np.array([rng.choice(cand.tolist()) for _ in range(Ntgt - len(cand))])
                sel_idx = np.concatenate([cand, extra])

            track_xy = np.stack([xn[:, :, sel_idx], yn[:, :, sel_idx]], axis=-1)
            ret["point_tracks"] = torch.from_numpy(np.ascontiguousarray(track_xy)).float()
            ret["track_visibility"] = torch.from_numpy(np.ascontiguousarray(visf[:, :, sel_idx])).float()
            ret["track_valid"] = torch.from_numpy(np.ascontiguousarray(valid[:, :, sel_idx])).float()
            ret["track_vis_valid"] = torch.from_numpy(np.ascontiguousarray(gok[:, :, sel_idx].astype(np.float32)))
            ret["track_depth"] = torch.from_numpy(np.ascontiguousarray(gz[:, :, sel_idx].astype(np.float32)))
            ret["track_src"] = torch.from_numpy(np.ascontiguousarray(src_all[sel_idx])).long()

            Kn = np.zeros((V, 3, 3), np.float32)
            En = np.zeros((V, T, 4, 4), np.float32)
            for i, ser in enumerate(view_sers):
                Kv = K[ser]
                Kn[i, 0, 0] = Kv[0, 0] * resize_ratio / target_w
                Kn[i, 1, 1] = Kv[1, 1] * resize_ratio / target_h
                Kn[i, 0, 2] = (Kv[0, 2] * resize_ratio - crop_x) / target_w
                Kn[i, 1, 2] = (Kv[1, 2] * resize_ratio - crop_y) / target_h
                Kn[i, 2, 2] = 1.0
                En[i] = E[ser].astype(np.float32)
            ret["track_intrinsics"] = torch.from_numpy(Kn)
            ret["track_extrinsics"] = torch.from_numpy(En)
        except Exception as e:
            log.warning(f"[mvt] no sample for {metadata.get('uuid')}: {e}")
            return None

        assert action.shape[0] == V * (T - 1), f"action {action.shape[0]} != {V}*(T-1) {V * (T - 1)}"

        tcf = int(self.temporal_compression_factor)
        latent_t = T // tcf + 1
        num_future = self._mvt_draw_num_future(latent_t)
        past_cond_idx = list(range(latent_t - num_future))

        if self._mvt_get("_mvt_future_camera", "full") == "hold":
            p_px = tcf * (latent_t - num_future - 1) + 1
            if p_px < T:
                c2w_held = {}
                for ser in view_sers:
                    h = c2w[ser].copy()
                    h[p_px:] = h[p_px - 1]
                    c2w_held[ser] = h
                action, camera_bundle = self._mvt_camera_condition(view_sers, c2w_held, K, geom)

        image_size = ret.get("image_size")
        ret["video"] = [v.float().div_(127.5).sub_(1.0) for v in view_videos]
        ret["image_size"] = [image_size] * V
        ret["num_frames"] = T
        ret["action"] = action
        ret["domain_id"] = torch.tensor(get_domain_id(CAMERA_DOMAIN), dtype=torch.long)
        ret["raw_action_dim"] = torch.tensor(CAMERA_RAW_DIM, dtype=torch.long)
        ret.update(camera_bundle)
        ret["mode"] = "forward_dynamics"
        ret["__key__"] = (
            f"{ret.get('__key__', metadata.get('uuid'))}__{V}views_fut{num_future}__{'_'.join(view_sers)}"
        )
        ret["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=past_cond_idx,
            num_source_views=0,
            has_action=True,
            condition_frame_indexes_action=list(range(action.shape[0])),
            action_start_frame_offset=1,
        )
        return ret


def get_multiview_track_dataset(
    *,
    num_views: "int | list[int]" = 2,
    num_future_latents: int = 4,
    variable_past_latents: "list[int] | None" = None,
    variable_past_frac: "list[float] | None" = None,
    future_camera: str = "full",
    translation_scale: float = 1.0,
    ref_translation_scale: float = 1.0,
    max_action_dim: int = 64,
    num_track_points: int = 128,
    random_anchor: bool = False,
    **kwargs,
) -> MultiViewTrackDataset:
    """Build an SFT dataset (``kwargs`` go to ``get_sft_dataset``) and promote it to ``MultiViewTrackDataset``.

    ``num_views``: maximum views per sample (a list means its max). V is data-driven: the anchor plus up to
    ``num_views - 1`` siblings sampled without replacement among the views that cover the window.
    ``random_anchor``: with ``num_views=1``, view 0 is drawn uniformly among the scene's views per sample.
    ``num_future_latents`` / ``variable_past_latents`` / ``variable_past_frac``: past/future latent split
    (``variable_past_frac`` takes precedence, then ``variable_past_latents``).
    ``future_camera``: ``"full"`` gives every frame's pose; ``"hold"`` freezes each view's pose at the last clean
    frame for the generated frames (action and camera bundle).
    ``translation_scale``: scale of the action translation (extrinsics are metric).
    ``ref_translation_scale``: accepted for config compatibility; unused.
    ``max_action_dim``: action zero-padding width. ``num_track_points``: points per sample (N)."""
    ds = get_sft_dataset(**kwargs)
    ds.__class__ = MultiViewTrackDataset
    ds._mvt_num_future_latents = int(num_future_latents)
    ds._mvt_translation_scale = float(translation_scale)
    ds._mvt_ref_translation_scale = float(ref_translation_scale)
    ds._mvt_max_action_dim = int(max_action_dim)
    ds._mvt_num_track_points = int(num_track_points)
    ds._mvt_num_views = num_views if isinstance(num_views, int) else max(num_views)
    ds._mvt_future_camera = str(future_camera)
    ds._mvt_variable_past_frac = [float(v) for v in variable_past_frac] if variable_past_frac else None
    ds._mvt_variable_past_latents = [int(v) for v in variable_past_latents] if variable_past_latents else None
    ds._mvt_random_anchor = bool(random_anchor)
    ds._mvt_json_cache = {}
    ds._mvt_rng_obj = random.Random()
    log.info(
        f"[mvt] MultiViewTrackDataset: num_views={ds._mvt_num_views} num_future_latents={num_future_latents} "
        f"variable_past_latents={ds._mvt_variable_past_latents} variable_past_frac={ds._mvt_variable_past_frac} "
        f"future_camera={future_camera} num_track_points={num_track_points} translation_scale={translation_scale}"
    )
    return ds
