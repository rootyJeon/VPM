# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Sampling / correlation primitives for the tracking head (vendored from MV-TAP).

Faithful copies of MV-TAP's bilinear sampler, 5D feature sampler, support-grid builder,
sinusoidal time embedding, and the relative-position ``posenc``. These are the geometry
ops the local 4D correlation volume is built on; kept identical so the correlation matches
the reference tracker's behaviour.
"""

import torch
import torch.nn.functional as F


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    """grid_sample with a (t,x,y) coordinate convention (MV-TAP models/utils.py)."""
    sizes = input.shape[2:]
    assert len(sizes) in [2, 3]
    if len(sizes) == 3:
        coords = coords[..., [1, 2, 0]]
    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device)
    else:
        coords = coords * torch.tensor([2 / size for size in reversed(sizes)], device=coords.device)
    coords = coords - 1
    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features5d(input, coords, mode="border"):
    """Sample spatio-temporal features. input [B,T,C,H,W], coords [B,R1,R2,3]->[B,R1,R2,C]."""
    B, T, _, _, _ = input.shape
    input = input.permute(0, 2, 1, 3, 4)
    coords = coords.unsqueeze(3)
    feats = bilinear_sampler(input, coords, padding_mode=mode if mode in ("zeros", "border") else "border")
    return feats.permute(0, 2, 3, 1, 4).view(B, feats.shape[2], feats.shape[3], feats.shape[1])


def get_support_points(coords, r, reshape_back=True):
    """(2r+1)x(2r+1) neighbourhood of (t,x,y) coords (MV-TAP MVTAP.get_support_points).

    coords: [B, S, N, 3]. Returns [B, S*N? ...] -- see reshape_back."""
    B, _, N, _ = coords.shape
    device = coords.device
    centroid_lvl = coords.reshape(B, N, 1, 1, 3)
    dx = torch.linspace(-r, r, 2 * r + 1, device=device)
    dy = torch.linspace(-r, r, 2 * r + 1, device=device)
    xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
    zgrid = torch.zeros_like(xgrid)
    delta = torch.stack([zgrid, xgrid, ygrid], dim=-1).reshape(1, 1, 2 * r + 1, 2 * r + 1, 3)
    coords_lvl = centroid_lvl + delta
    if reshape_back:
        return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
    return coords_lvl


def posenc(x, min_deg, max_deg):
    """Concatenate x with its sinusoidal positional encoding (MV-TAP models/mvtap.py)."""
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device)
    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x, four_feat], dim=-1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sin/cos positional embedding (MV-TAP models/embeddings.py). Returns [1, M, embed_dim]."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb[None].float()
