# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Multi-view point-tracking head over DiT attention features (MVTrack4Gen + MV-TAP).

Given the post-RoPE query/key features of ONE intermediate 3D-attention layer, reshaped into
per-view latent-grid feature maps, this head builds MV-TAP's multi-scale local 4D correlation
volumes (query-side from Q, search-side from K — exactly MVTrack4Gen Eq.4) and runs the
``Refiner`` for ``iters`` updates to predict per-frame, per-view 2D tracks + visibility +
confidence. The Refiner's factorized *view* attention exchanges cross-view correspondence
across the V symmetric views of a single multi-view sample.

Differences vs MV-TAP's ``MVTAP`` module:
  * No image encoder (``BasicEncoder``): features ARE the DiT's attention Q/K, projected to
    ``latent_dim`` by a small trainable linear (separate for Q and K) and L2-normalized.
  * Q != K: ``get_track_feat`` samples Q at the query points; ``get_correlation_feat`` samples
    K around the running estimate. (MV-TAP self-correlates one feature map.)
  * Single refinement window covering the whole (short) latent clip — no sliding window — and
    one multi-view group per call so view attention stays within the group.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.model.vfm.tracking.blocks import Mlp, Refiner
from cosmos_framework.model.vfm.tracking.cam import plucker_from_tracks
from cosmos_framework.model.vfm.tracking.ops import (
    get_1d_sincos_pos_embed_from_grid,
    get_support_points,
    posenc,
    sample_features5d,
    bilinear_sampler,
)


class MultiViewTrackingHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 128,
        corr_radius: int = 3,
        corr_levels: int = 4,
        num_virtual_tracks: int = 64,
        hidden_size: int = 384,
        corr_hidden: int = 256,
        iters: int = 4,
        max_frames: int = 32,
        view_att: bool = True,
        use_checkpoint: bool = True,
        use_camera: bool = False,
        upsample: int = 1,
        predict_depth: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.iters = iters
        self.use_camera = use_camera
        self.upsample = int(upsample)
        self.q_proj = nn.Linear(in_dim, latent_dim * self.upsample ** 2, bias=True)
        self.k_proj = nn.Linear(in_dim, latent_dim * self.upsample ** 2, bias=True)
        r = 2 * corr_radius + 1
        self.corr_mlp = Mlp(in_features=r * r * r * r, hidden_features=384, out_features=corr_hidden)
        self.former_input_dim = corr_levels * corr_hidden + 86
        if use_camera:
            self.cam_proj = nn.Linear(6, self.former_input_dim)
            self.cam_norm = nn.LayerNorm(self.former_input_dim)
        self.predict_depth = bool(predict_depth)
        self.depth_init = -0.6931471805599453
        self.updateformer = Refiner(
            space_depth=3, time_depth=3, view_depth=3,
            input_dim=self.former_input_dim, hidden_size=hidden_size,
            output_dim=5 if self.predict_depth else 4,
            mlp_ratio=4.0, num_virtual_tracks=num_virtual_tracks,
            use_checkpoint=use_checkpoint, view_att=view_att,
        )

    def _project(self, feat: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """[V,T,Cin,H,W] -> [V,T,latent,H*u,W*u] FLOAT32, L2-normalized over channel (MV-TAP fmaps).

        ``u = self.upsample`` pixel-shuffles the projected ``latent*u^2`` channels into a u-times finer
        spatial grid (learned unpatchify). The Linear runs in the head's param dtype (bf16 under mixed
        precision); the returned maps are float32 so grid_sample / correlation geometry stays stable."""
        V, T, _, H, W = feat.shape
        u = self.upsample
        x = feat.permute(0, 1, 3, 4, 2)
        x = proj(x.to(proj.weight.dtype))
        if u > 1:
            x = x.reshape(V, T, H, W, u, u, self.latent_dim)
            x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(V, T, H * u, W * u, self.latent_dim)
        x = F.normalize(x.float(), dim=-1, eps=1e-12, p=2)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    @staticmethod
    def _pyramid(fmaps: torch.Tensor, levels: int) -> list:
        """4-level feature pyramid via avg_pool2d (MV-TAP). fmaps [V,T,C,H,W]."""
        V, T, C, H, W = fmaps.shape
        pyr = [fmaps]
        cur = fmaps
        for _ in range(levels - 1):
            x = cur.reshape(V * T, C, cur.shape[-2], cur.shape[-1])
            x = F.avg_pool2d(x, 2, stride=2)
            cur = x.reshape(V, T, C, x.shape[-2], x.shape[-1])
            pyr.append(cur)
        return pyr

    def _get_track_feat(self, fmaps, q_frames, q_coords):
        """Query-side support features at query points. fmaps [V,T,C,H,W]; returns [V,(2r+1)^2,N,C]."""
        sample_frames = q_frames[:, None, :, None]
        sample_coords = torch.cat([sample_frames, q_coords[:, None]], dim=-1)
        support = get_support_points(sample_coords, self.corr_radius)
        return sample_features5d(fmaps, support)

    def _get_correlation_feat(self, fmaps, coords):
        """Search-side features around `coords`. fmaps [V,T,C,H,W], coords [V*T,N,2] -> [V,T,N,2r+1,2r+1,C]."""
        V, T, D, H_, W_ = fmaps.shape
        N = coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat([torch.zeros_like(coords[..., :1]), coords], dim=-1)[:, None]
        support = get_support_points(sample_coords, r, reshape_back=False)
        feat = bilinear_sampler(fmaps.reshape(V * T, D, 1, H_, W_), support)
        return feat.reshape(V, T, D, N, 2 * r + 1, 2 * r + 1).permute(0, 1, 3, 4, 5, 2)

    def reset_parameters(self):
        """(Re)initialize ALL head params on the real device (it is built on `meta` inside `net`,
        so every Linear/LayerNorm/Parameter must be explicitly materialized here)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.updateformer.flow_head.weight, std=0.001)
        nn.init.trunc_normal_(self.updateformer.vis_conf_head.weight, std=0.001)
        with torch.no_grad():
            self.updateformer.virual_tracks.normal_(0, 1)

    def _time_embed(self, T: int, device, dtype) -> torch.Tensor:
        """Sinusoidal [1, T, former_input_dim] time embedding, computed at the needed length."""
        pos = torch.linspace(0, T - 1, T)
        return get_1d_sincos_pos_embed_from_grid(self.former_input_dim, pos).to(device=device, dtype=dtype)

    def _cam_embed(self, coords, intrinsics, extrinsics, model_hw, out_dtype):
        """Plücker camera embedding for the current track estimate, shaped [V, N, T, former_input_dim]
        to ADD onto the Refiner input. ``coords`` [V,T,N,2] in feature-grid units; intrinsics/extrinsics
        already image-normalized / windowed to T. Returns None if camera conditioning is off/unavailable."""
        if not (self.use_camera and intrinsics is not None and extrinsics is not None):
            return None
        V, T, N, _ = coords.shape
        H0, W0 = model_hw
        intr = intrinsics.to(coords.device).float()
        if intr.dim() == 3:
            intr = intr[:, None].expand(V, T, 3, 3)
        extr = extrinsics.to(coords.device).float()
        coords01 = torch.stack([coords[..., 0] / W0, coords[..., 1] / H0], dim=-1)
        plk = plucker_from_tracks(coords01, intr, extr)
        cam = self.cam_proj(plk.to(self.cam_proj.weight.dtype))
        cam = F.normalize(cam, dim=-1)
        cam = self.cam_norm(cam)
        return cam.permute(0, 2, 1, 3).to(out_dtype)

    def _refine(self, q_pyr, k_pyr, q_support_pyr, coords, vis, conf, model_hw,
                return_features=False, intrinsics=None, extrinsics=None):
        r = 2 * self.corr_radius + 1
        V, T, N, _ = coords.shape
        H0, W0 = model_hw
        scale = torch.tensor([W0, H0], device=coords.device, dtype=torch.float32)
        pdt = self.corr_mlp.fc1.weight.dtype
        coord_preds, vis_preds, conf_preds, dep_preds = [], [], [], []
        dep = coords.new_full((*coords.shape[:3], 1), self.depth_init) if self.predict_depth else None
        feats = None
        for _it in range(self.iters):
            coords = coords.detach()
            if dep is not None:
                dep = dep.detach()
            coords_flat = coords.reshape(V * T, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self._get_correlation_feat(k_pyr[i], coords_flat / 2**i)
                tfs = q_support_pyr[i].reshape(V, 1, r, r, N, self.latent_dim).squeeze(1).permute(0, 3, 1, 2, 4)
                corr_volume = torch.einsum("btnhwc,bnijc->btnhwij", corr_feat, tfs)
                corr_embs.append(self.corr_mlp(corr_volume.reshape(V * T * N, r * r * r * r).to(pdt)))
            corr_embs = torch.cat(corr_embs, dim=-1).reshape(V, T, N, -1)

            rel_fwd = F.pad(coords[:, :-1] - coords[:, 1:], (0, 0, 0, 0, 0, 1)) / scale
            rel_bwd = F.pad(coords[:, 1:] - coords[:, :-1], (0, 0, 0, 0, 1, 0)) / scale
            rel = posenc(torch.cat([rel_fwd, rel_bwd], dim=-1), min_deg=0, max_deg=10)

            x = torch.cat([vis.to(pdt), conf.to(pdt), corr_embs, rel.to(pdt)], dim=-1)
            x = x.permute(0, 2, 1, 3).reshape(V * N, T, -1)
            x = x + self._time_embed(T, x.device, pdt)
            x = x.reshape(V, N, T, -1)
            cam = self._cam_embed(coords, intrinsics, extrinsics, model_hw, pdt)
            if cam is not None:
                x = x + cam
            if return_features and _it == self.iters - 1:
                delta, feats = self.updateformer(x, return_tokens=True)
                delta = delta.float()
            else:
                delta = self.updateformer(x).float()
            coords = coords + delta[..., :2].permute(0, 2, 1, 3)
            _o = 2
            if dep is not None:
                dep = dep + delta[..., 2:3].permute(0, 2, 1, 3)
                dep_preds.append(dep[..., 0])
                _o = 3
            vis = vis + delta[..., _o:_o + 1].permute(0, 2, 1, 3)
            conf = conf + delta[..., _o + 1:].permute(0, 2, 1, 3)
            coord_preds.append(coords[..., :2])
            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        if return_features:
            return coord_preds, vis_preds, conf_preds, dep_preds, feats
        return coord_preds, vis_preds, conf_preds, dep_preds

    def forward(self, q_feat: torch.Tensor, k_feat: torch.Tensor, queries: torch.Tensor,
                intrinsics: torch.Tensor = None, extrinsics: torch.Tensor = None):
        """Track query points across the V-view clip.

        Args:
            q_feat / k_feat: [V, T, Cin, H, W] DiT query/key feature maps at one layer.
            queries: [V, N, 3] = (frame_idx, x, y) in feature-grid coords.
            intrinsics: optional [V,3,3] or [V,T,3,3] image-normalized pinhole (Plücker conditioning).
            extrinsics: optional [V,T,4,4] world-to-camera; T = latent frames.
        Returns dict with per-iteration lists (training) + final tensors:
            coords: list[iters] of [V, T, N, 2]; vis/conf: list[iters] of [V, T, N] (logits);
            coords_final/vis_final/conf_final: [V, T, N, *] (conf/vis sigmoid-activated).
        """
        V, T = q_feat.shape[0], q_feat.shape[1]
        H0, W0 = q_feat.shape[-2] * self.upsample, q_feat.shape[-1] * self.upsample
        N = queries.shape[1]
        qmaps = self._project(q_feat, self.q_proj)
        kmaps = self._project(k_feat, self.k_proj)
        q_pyr = self._pyramid(qmaps, self.corr_levels)
        k_pyr = self._pyramid(kmaps, self.corr_levels)

        q_frames = queries[..., 0].long()
        q_coords = queries[..., 1:3].float()
        q_support_pyr = []
        for i in range(self.corr_levels):
            tfs = self._get_track_feat(q_pyr[i], q_frames, q_coords / 2**i)
            q_support_pyr.append(tfs.unsqueeze(1))

        coords0 = q_coords.unsqueeze(1).expand(-1, T, -1, -1).contiguous()
        vis0 = torch.zeros(V, T, N, 1, dtype=torch.float32, device=q_feat.device)
        conf0 = torch.zeros(V, T, N, 1, dtype=torch.float32, device=q_feat.device)
        coord_preds, vis_preds, conf_preds, dep_preds = self._refine(
            q_pyr, k_pyr, q_support_pyr, coords0, vis0, conf0, (H0, W0),
            intrinsics=intrinsics, extrinsics=extrinsics,
        )
        out = {
            "coords": coord_preds,
            "vis": vis_preds,
            "conf": conf_preds,
            "coords_final": coord_preds[-1],
            "vis_final": torch.sigmoid(vis_preds[-1]),
            "conf_final": torch.sigmoid(conf_preds[-1]),
        }
        if self.predict_depth:
            out["depth"] = dep_preds
            out["depth_final"] = torch.exp(dep_preds[-1])
        return out

    @staticmethod
    def unproject(coords: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor,
                  grid_hw: "tuple[int, int]") -> torch.Tensor:
        """2D track + ray depth -> metric 3D in each view's camera frame.

        Args:
            coords: [V,T,N,2] track positions in FEATURE-GRID cells (the head's output units).
            depth: [V,T,N] metric depth along the pixel ray (e.g. ``depth_final``).
            intrinsics: [V,3,3] image-NORMALIZED K (the dataset's ``track_intrinsics``: fx' = fx/W etc.).
            grid_hw: (H0, W0) of the track grid, to map cells -> normalized image coords.
        Returns:
            [V,T,N,3] points, OpenCV convention (+x right, +y down, +z forward), meters.
        """
        H0, W0 = grid_hw
        uv = torch.stack([coords[..., 0] / W0, coords[..., 1] / H0], dim=-1)
        ones = torch.ones_like(uv[..., :1])
        pix = torch.cat([uv, ones], dim=-1)
        Kinv = torch.linalg.inv(intrinsics.float().cpu()).to(pix.device, pix.dtype)
        rays = torch.einsum("vij,vtnj->vtni", Kinv, pix)
        return rays * depth[..., None]

    @property
    def feature_dim(self) -> int:
        """Hidden dim of the refined track tokens returned by ``feedback_features``."""
        return self.updateformer.hidden_size

    def feedback_features(self, q_feat: torch.Tensor, k_feat: torch.Tensor,
                          intrinsics: torch.Tensor = None, extrinsics: torch.Tensor = None) -> torch.Tensor:
        """Dense track-feature feedback: query EVERY latent-grid cell (frame 0) and return the Refiner's
        refined per-cell token, reshaped to ``[V, T, h, w, hidden]`` — a per-token, tracking-aware
        feature for injecting back into the DiT. q_feat/k_feat: [V, T, Cin, h, w]. intrinsics/extrinsics
        optional Plücker camera conditioning (same convention as ``forward``)."""
        V, T = q_feat.shape[0], q_feat.shape[1]
        h0, w0 = q_feat.shape[-2], q_feat.shape[-1]
        H0, W0 = h0 * self.upsample, w0 * self.upsample
        dev = q_feat.device
        q_pyr = self._pyramid(self._project(q_feat, self.q_proj), self.corr_levels)
        k_pyr = self._pyramid(self._project(k_feat, self.k_proj), self.corr_levels)
        gy, gx = torch.meshgrid(torch.arange(H0, device=dev), torch.arange(W0, device=dev), indexing="ij")
        xy = torch.stack([gx.reshape(-1).float(), gy.reshape(-1).float()], dim=-1)
        N = xy.shape[0]
        q_frames = torch.zeros(V, N, dtype=torch.long, device=dev)
        q_coords = xy[None].expand(V, N, 2).contiguous()
        q_support_pyr = [self._get_track_feat(q_pyr[i], q_frames, q_coords / 2**i).unsqueeze(1)
                         for i in range(self.corr_levels)]
        coords0 = q_coords.unsqueeze(1).expand(-1, T, -1, -1).contiguous()
        vis0 = torch.zeros(V, T, N, 1, dtype=torch.float32, device=dev)
        conf0 = torch.zeros(V, T, N, 1, dtype=torch.float32, device=dev)
        *_, feats = self._refine(q_pyr, k_pyr, q_support_pyr, coords0, vis0, conf0, (H0, W0),
                                 return_features=True, intrinsics=intrinsics, extrinsics=extrinsics)
        out = feats.reshape(V, H0, W0, T, feats.shape[-1]).permute(0, 3, 1, 2, 4).contiguous()
        if self.upsample > 1:
            C = out.shape[-1]
            out = out.permute(0, 1, 4, 2, 3).reshape(V * T, C, H0, W0)
            out = F.avg_pool2d(out, self.upsample, stride=self.upsample)
            out = out.reshape(V, T, C, h0, w0).permute(0, 1, 3, 4, 2).contiguous()
        return out
