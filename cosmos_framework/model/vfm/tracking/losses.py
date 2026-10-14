# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-view point-tracking losses (adapted from MV-TAP ``model_utils.py``).

L_track = w_seq * L_seq + w_conf * L_conf + w_vis * L_vis, with MV-TAP's defaults
(w_seq=0.05, w_conf=w_vis=1.0). L_seq is a visibility-weighted Huber on 2D coords, with the
per-iteration gamma discount; L_conf calibrates predicted confidence against the realized
regression error; L_vis is BCE on visibility logits. We process the whole (short) latent clip
as a SINGLE refinement window, so unlike MV-TAP there is no sliding-window reduction here —
predictions come as per-iteration lists and the GT is the full ``[G, T, N, *]`` clip
(G = flattened sample*view groups). The outer MVTrack4Gen weight (lambda_track=0.01) is
applied by the model, not here.
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


def reduce_masked_mean(x, mask):
    """Mean of x over a (broadcastable) mask (MV-TAP model_utils.reduce_masked_mean)."""
    mask = mask.expand_as(x)
    return torch.sum(x * mask) / (EPS + torch.sum(mask))


def huber_loss(x, y, delta=1.0):
    """Element-wise Huber loss (MV-TAP model_utils.huber_loss)."""
    diff = x - y
    a = diff.abs()
    flag = (a <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (a - 0.5 * delta)


def multiview_track_loss(
    coord_preds, vis_preds, conf_preds, traj_gt, vis_gt, valid_gt,
    gamma=0.8, huber_delta=1.5, dist_thresh=3.0,
    w_seq=0.05, w_conf=1.0, w_vis=1.0, point_weight=None,
    w_vel=0.0, vel_huber_delta=0.75, vis_mask=None,
    seq_cap=0.0, frame_weight=None, past_frames=None,
    depth_preds=None, depth_gt_log=None, depth_valid=None, w_depth=0.0, depth_huber_delta=0.2,
):
    """MV-TAP tracking loss over per-iteration predictions of one window.

    UNITS: our coords/GT are in FEATURE-GRID cells (0..W0 on the ~8x8 latent grid), NOT pixels.
    MV-TAP applies its Huber ``delta=6`` and confidence ``dist_thresh=12`` in PIXELS at stride 4
    (feature grid ~96), i.e. 1.5 and 3.0 FEATURE CELLS. So the cell-equivalent defaults are 1.5/3.0.
    (Using the raw pixel 6/12 here silently broke L_conf: on an 8x8 grid max err^2 ~= 128 < 12^2=144,
    so ``valid_c`` was always 1 and confidence collapsed to always-high.)

    Args:
        coord_preds: list[M] of [G, T, N, 2] predicted coords (feature-grid cell units).
        vis_preds:   list[M] of [G, T, N] visibility LOGITS.
        conf_preds:  list[M] of [G, T, N] confidence LOGITS.
        traj_gt:  [G, T, N, 2] GT coords (same units as coord_preds).
        vis_gt:   [G, T, N] GT visibility in {0,1} (float).
        valid_gt: [G, T, N] validity mask in {0,1} (float) — point exists / in-bounds.
        point_weight: optional [N] per-point weight (e.g. motion weighting, mean ~1).
        w_vel: weight of the VELOCITY (anti-jitter) term — Huber on the difference between predicted
            and GT per-frame displacement, over co-visible consecutive frames. Penalizes predicted
            motion where the GT point is still. 0 disables.
        vis_mask: optional [G, T, N] in {0,1} — cells whose ``vis_gt`` label is TRUSTWORTHY. Cells
            with 0 (e.g. cross-projected points with unknown occlusion) are excluded from the
            visibility BCE only; coord/conf supervision is unaffected.
        seq_cap: ROBUST-SEQ truncation — per-point-frame Huber values of the seq AND vel terms
            are clamped at this value (0 = off). The Huber tail is LINEAR and predicted coords
            are unbounded, so a diverged refinement (e.g. on heavily-noised FUTURE frames whose
            features carry no content) otherwise yields arbitrarily large losses (the observed
            0.xx -> 1000 spikes). Beyond the cap the gradient is CUT: a wildly-wrong mode is not
            worth chasing — conf/vis carry that signal instead.
        frame_weight: optional [G, T] in [0,1] — per-frame weight on the seq/vel terms only
            (conf/vis keep full weight). Used for SIGMA-AWARE supervision of future prediction:
            weight ~ (1 - sigma) so clean past frames get full coordinate supervision while
            frames near pure noise (where the future point position is fundamentally
            under-determined) are not forced to commit to exact coordinates.
        past_frames: optional [G, T] in {0,1} — 1 on the CLEAN conditioning (past) frames. Only
            adds the diagnostics ``track_loss/seq_past`` / ``track_loss/seq_future``: the final
            iteration's capped coordinate Huber averaged over past vs future frames, UNWEIGHTED by
            frame_weight — i.e. how well the head localizes on clean vs noised frames regardless of
            how the objective weights them (the sigma-weighting A/B reads these two curves).
        depth_preds: optional list[M] of [G, T, N] predicted LOG-depth along the pixel ray
            (the head's ``out["depth"]``). Enables the 3D term.
        depth_gt_log: [G, T, N] GT log(metric depth) in each view's own camera frame.
        depth_valid: [G, T, N] in {0,1} — GT depth is finite and positive. Combined with the same
            visibility/validity/frame weighting as the seq term.
        w_depth: weight of the depth term (0 disables even when depth_preds is given).
        depth_huber_delta: Huber delta in LOG-depth (0.2 ~ 22% relative error). The depth term is
            NEVER truncated — no seq_cap analogue — per the no-truncation training setup; log-space
            errors are bounded by the workspace scale anyway.
    Returns:
        (loss scalar, dict of component scalars for logging).
    """
    M = len(coord_preds)
    seq = traj_gt.new_zeros(())
    conf = traj_gt.new_zeros(())
    vis = traj_gt.new_zeros(())
    vel = traj_gt.new_zeros(())
    pw = 1.0 if point_weight is None else point_weight.view(1, 1, -1)
    vis_weight = valid_gt * vis_gt * pw
    valid_w = valid_gt * pw
    vis_w = valid_w if vis_mask is None else valid_w * vis_mask
    covis_pair = vis_weight[:, 1:] * vis_weight[:, :-1]
    seq_w = vis_weight if frame_weight is None else vis_weight * frame_weight[..., None]
    vel_w = covis_pair if frame_weight is None else covis_pair * frame_weight[:, 1:, None]
    gt_disp = traj_gt[:, 1:] - traj_gt[:, :-1]
    for i in range(M):
        w = gamma ** (M - i - 1)
        l = huber_loss(coord_preds[i], traj_gt, delta=huber_delta).mean(dim=-1)
        if seq_cap > 0:
            l = l.clamp(max=seq_cap)
        seq = seq + w * reduce_masked_mean(l, seq_w)

        err = ((coord_preds[i].detach() - traj_gt) ** 2).sum(-1)
        valid_c = (err <= dist_thresh**2).float()
        lp = F.binary_cross_entropy_with_logits(conf_preds[i], valid_c, reduction="none") * vis_gt
        conf = conf + reduce_masked_mean(lp, valid_w)

        lv = F.binary_cross_entropy_with_logits(vis_preds[i], vis_gt, reduction="none")
        vis = vis + reduce_masked_mean(lv, vis_w)

        if w_vel > 0:
            pred_disp = coord_preds[i][:, 1:] - coord_preds[i][:, :-1]
            dl = huber_loss(pred_disp, gt_disp, delta=vel_huber_delta).mean(dim=-1)
            if seq_cap > 0:
                dl = dl.clamp(max=seq_cap)
            vel = vel + w * reduce_masked_mean(dl, vel_w)

    dep = traj_gt.new_zeros(())
    _dep_on = w_depth > 0 and depth_preds is not None and depth_gt_log is not None
    if _dep_on:
        dep_w = seq_w if depth_valid is None else seq_w * depth_valid
        for i, dp in enumerate(depth_preds):
            w = gamma ** (M - i - 1)
            dl = huber_loss(dp, depth_gt_log, delta=depth_huber_delta)
            dep = dep + w * reduce_masked_mean(dl, dep_w)

    seq, conf, vis, vel, dep = seq / M, conf / M, vis / M, vel / M, dep / M
    loss = w_seq * seq + w_conf * conf + w_vis * vis + w_vel * vel
    if _dep_on:
        loss = loss + w_depth * dep
    logs = {
        "track_loss/seq": seq.detach(),
        "track_loss/conf": conf.detach(),
        "track_loss/vis": vis.detach(),
        "track_loss/total": loss.detach(),
    }
    if w_vel > 0:
        logs["track_loss/vel"] = vel.detach()
    if _dep_on:
        logs["track_loss/depth"] = dep.detach()
    if past_frames is not None:
        l_last = huber_loss(coord_preds[-1].detach(), traj_gt, delta=huber_delta).mean(dim=-1)
        if seq_cap > 0:
            l_last = l_last.clamp(max=seq_cap)
        pf = past_frames.to(l_last.dtype)[..., None]
        logs["track_loss/seq_past"] = reduce_masked_mean(l_last, vis_weight * pf)
        logs["track_loss/seq_future"] = reduce_masked_mean(l_last, vis_weight * (1.0 - pf))
    return loss, logs
