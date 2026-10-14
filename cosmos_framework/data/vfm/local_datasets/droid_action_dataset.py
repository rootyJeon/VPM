# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DROID samples (multi-view video, cross-view 3D tracks, camera bundle) with a robot action stream.

The sample matches ``MultiViewTrackDataset`` except that the per-view 9D camera action is replaced by a single
8D joint-space action stream (domain ``oxe/droid``): video flow matching, track losses and action flow matching
train in one forward pass.

Action labels: ``<root>/actions/{scene}.npz`` with ``joint_states`` and ``joint_actions`` [n_frames, 8], indexed
by video frame (7 joint positions in radians, gripper 0 open .. 1 closed). With ``gripper_flip`` the model sees
``1 - gripper_position`` (1 = open, 0 = closed).

Window: ``p_lat`` clean past latents are drawn from ``variable_past_latents``; the window is ``p_px + action_chunk``
frames with ``p_px = tcf * (p_lat - 1) + 1``, starting uniformly in the episode, so every sample generates exactly
``action_chunk`` future frames. Action rows: row 0 = joint state at the last clean frame ``t0`` (clean), rows
1..chunk = ``joint_actions[t0 .. t0+chunk-1]`` (generated); ``action_start_frame_offset = p_px - 1``.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.transforms import pad_action_to_max_dim
from cosmos_framework.data.vfm.local_datasets.multiview_track_dataset import (
    MultiViewTrackDataset,
    get_multiview_track_dataset,
)
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.utils import log

ROBOT_DOMAIN = "oxe/droid"
ROBOT_RAW_ACTION_DIM = 8


class DroidActionDataset(MultiViewTrackDataset):
    """``MultiViewTrackDataset`` with a single-stream robot action (see module docstring).
    Built via ``get_droid_action_dataset``."""

    def process_one_sample(self, metadata: dict):
        try:
            anchor = Path(os.path.abspath(metadata["vision_path"]))
            scene = anchor.stem.rsplit("__", 1)[0]
            act = np.load(anchor.parents[1] / "actions" / f"{scene}.npz")
            js, ja = act["joint_states"], act["joint_actions"]
        except Exception as e:
            log.warning(f"[droid-act] no action labels for {metadata.get('uuid')}: {e}")
            return None
        n = int(js.shape[0])
        chunk = int(self._mvt_get("_mvt_action_chunk", 32))
        tcf = int(self.temporal_compression_factor)
        vp = list(self._mvt_get("_mvt_variable_past_latents", None) or [1])
        p_lat = int(random.choice(vp))
        while p_lat > 1 and tcf * (p_lat - 1) + 1 + chunk > n:
            p_lat -= 1
        p_px = tcf * (p_lat - 1) + 1
        T = p_px + chunk
        if T > n:
            return None
        if metadata.get("window_start") is not None:
            start = int(metadata["window_start"])
            if not 0 <= start <= n - T:
                return None
        else:
            start = random.randint(0, n - T)

        windows = metadata.get("t2w_windows") or [{}]
        win = dict(windows[0])
        win.update(start_frame=start, end_frame=start + T - 1, temporal_interval=1)
        md = dict(metadata)
        md["t2w_windows"] = [win]
        saved = self._mvt_get("_mvt_variable_past_latents", None)
        self._mvt_variable_past_latents = [p_lat]
        try:
            ret = super().process_one_sample(md)
        finally:
            self._mvt_variable_past_latents = saved
        if ret is None:
            return None
        if int(ret["num_frames"]) != T or int(ret["frame_start"]) != start:
            log.warning(
                f"[droid-act] window mismatch for {metadata.get('uuid')}: got T={ret['num_frames']} "
                f"start={ret['frame_start']}, want T={T} start={start} (video shorter than labels?)"
            )
            return None

        t0 = start + p_px - 1
        rows = np.concatenate([js[t0 : t0 + 1], ja[t0 : t0 + chunk]], axis=0).astype(np.float32)
        assert rows.shape == (1 + chunk, ROBOT_RAW_ACTION_DIM), rows.shape
        if self._mvt_get("_mvt_gripper_flip", True):
            rows[:, -1] = 1.0 - rows[:, -1]
        max_dim = int(self._mvt_get("_mvt_max_action_dim", 64))
        ret["action"] = pad_action_to_max_dim(torch.from_numpy(rows), max_dim)
        ret["domain_id"] = torch.tensor(get_domain_id(ROBOT_DOMAIN), dtype=torch.long)
        ret["raw_action_dim"] = torch.tensor(ROBOT_RAW_ACTION_DIM, dtype=torch.long)
        ret["mode"] = "policy"
        ret["__key__"] = f"{ret['__key__']}__act{chunk}_p{p_lat}"
        old = ret["sequence_plan"]
        ret["sequence_plan"] = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=list(old.condition_frame_indexes_vision),
            num_source_views=0,
            has_action=True,
            condition_frame_indexes_action=[0],
            action_start_frame_offset=p_px - 1,
            action_single_stream=True,
            action_view_index=0,
        )
        assert len(old.condition_frame_indexes_vision) == p_lat, (old.condition_frame_indexes_vision, p_lat)
        return ret


def get_droid_action_dataset(action_chunk: int = 32, gripper_flip: bool = True, **kwargs) -> DroidActionDataset:
    """``get_multiview_track_dataset`` plus the robot action stream (other kwargs as there).

    ``action_chunk``: generated frames = generated action rows per sample (default 32); must be a positive multiple
    of the VAE temporal compression factor. ``gripper_flip``: the model sees ``1 - gripper_position``
    (1 = open, 0 = closed)."""
    ds = get_multiview_track_dataset(**kwargs)
    tcf = int(ds.temporal_compression_factor)
    assert int(action_chunk) > 0 and int(action_chunk) % tcf == 0, (
        f"action_chunk={action_chunk} must be a positive multiple of {tcf}: the window p_px + chunk must stay on "
        "the 4n+1 latent grid, else every sample is truncated and rejected"
    )
    ds.__class__ = DroidActionDataset
    ds._mvt_action_chunk = int(action_chunk)
    ds._mvt_gripper_flip = bool(gripper_flip)
    log.info(
        f"[droid-act] DroidActionDataset: robot action stream (domain {ROBOT_DOMAIN}, 8D joint), "
        f"chunk={action_chunk}, gripper_flip={gripper_flip}, past latents={ds._mvt_variable_past_latents}"
    )
    return ds
