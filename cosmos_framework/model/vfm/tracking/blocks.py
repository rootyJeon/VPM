# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transformer blocks for the multi-view tracking head (vendored from MV-TAP).

Faithful copies of MV-TAP's ``models/blocks.py`` (Mlp / Attention / AttnBlock) and the
``CrossAttnBlock`` + ``Refiner`` from ``models/transformer.py``. The image-feature
encoder (BasicEncoder / ResidualBlock) is intentionally NOT vendored: this head consumes
the DiT's own attention features, not raw pixels. Kept byte-for-byte where possible so the
factorized time -> view -> space refinement matches the reference tracker.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class Mlp(nn.Module):
    """MLP as used in ViT / MLP-Mixer (MV-TAP models/blocks.py)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """Multi-head (cross-)attention with fused SDPA (MV-TAP models/blocks.py)."""

    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=48,
                 qkv_bias=False, fused_attn=True, attn_drop=0.0):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.fused_attn = fused_attn
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        if self.fused_attn and attn_bias is None:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            sim = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                sim = sim + attn_bias
            x = sim.softmax(dim=-1) @ v
        x = x.transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)


class AttnBlock(nn.Module):
    """Pre-norm self-attention + MLP residual block (MV-TAP models/blocks.py)."""

    def __init__(self, hidden_size, num_heads, attn_class=Attention, mlp_ratio=4.0, **kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, fused_attn=True, **kw)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x, mask=None):
        attn_bias = None
        if mask is not None:
            m = (mask[:, None] * mask[:, :, None]).unsqueeze(1).expand(-1, self.attn.heads, -1, -1)
            attn_bias = (~m) * (-torch.finfo(x.dtype).max)
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + MLP residual block (MV-TAP models/transformer.py)."""

    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(hidden_size, context_dim=context_dim, num_heads=num_heads,
                                    qkv_bias=True, fused_attn=True, **kw)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio),
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

    def forward(self, x, context, mask=None):
        attn_bias = None
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(-1, self.cross_attn.heads, -1, context.shape[1])
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)
            attn_bias = (~mask) * (-torch.finfo(x.dtype).max)
        x = x + self.cross_attn(self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class Refiner(nn.Module):
    """Track-update transformer with factorized time -> view -> space attention.

    Vendored from MV-TAP ``models/transformer.py``. Input ``[B, N, T, input_dim]`` (B = the
    multi-view group folded as the view-attention batch, N = tracked points, T = frames);
    returns per-iteration residuals ``[B, N, T, 4]`` = (dx, dy, dvis, dconf). With
    ``view_att=True`` the view blocks attend across the B (view) axis, which is how a single
    multi-view group exchanges cross-view correspondence.
    """

    def __init__(self, space_depth=3, time_depth=3, input_dim=320, hidden_size=384,
                 num_heads=8, output_dim=4, mlp_ratio=4.0, num_virtual_tracks=64,
                 view_depth=3, use_checkpoint=True, view_att=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.view_att = view_att
        self.num_virtual_tracks = num_virtual_tracks
        self.use_checkpoint = use_checkpoint

        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim - 2, bias=True)
        self.vis_conf_head = nn.Linear(hidden_size, 2, bias=True)
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))

        self.time_blocks = nn.ModuleList(
            [AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention)
             for _ in range(time_depth)]
        )
        if self.view_att:
            self.view_blocks = nn.ModuleList(
                [AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention)
                 for _ in range(view_depth)]
            )
        self.space_virtual_blocks = nn.ModuleList(
            [AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_class=Attention)
             for _ in range(space_depth)]
        )
        self.space_point2virtual_blocks = nn.ModuleList(
            [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
             for _ in range(space_depth)]
        )
        self.space_virtual2point_blocks = nn.ModuleList(
            [CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
             for _ in range(space_depth)]
        )
        assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            nn.init.trunc_normal_(self.flow_head.weight, std=0.001)
            nn.init.trunc_normal_(self.vis_conf_head.weight, std=0.001)
        self.apply(_basic)

    def forward(self, input_tensor, mask=None, view_att_mask=None, return_tokens=False):
        use_ckpt = self.training and self.use_checkpoint

        def _cp(module, *args, _use_ckpt=False, **kwargs):
            if not _use_ckpt:
                return module(*args, **kwargs)
            return cp.checkpoint(lambda *t: module(*t, **kwargs), *args, use_reentrant=False)

        tokens = self.input_transform(input_tensor)
        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)
        _, N, _, _ = tokens.shape

        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)
            time_tokens = _cp(self.time_blocks[i], time_tokens, _use_ckpt=use_ckpt)
            tokens = time_tokens.view(B, N, T, -1)

            if self.view_att:
                view_tokens = tokens.permute(1, 2, 0, 3).contiguous().view(N * T, B, -1)
                view_tokens = _cp(self.view_blocks[i], view_tokens, view_att_mask, _use_ckpt=use_ckpt)
                tokens = view_tokens.view(N, T, B, -1).permute(2, 0, 1, 3)

            space_tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
            point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
            virtual_tokens = space_tokens[:, N - self.num_virtual_tracks:]

            virtual_tokens = _cp(self.space_virtual2point_blocks[i], virtual_tokens, point_tokens,
                                 _use_ckpt=use_ckpt, mask=mask)
            virtual_tokens = _cp(self.space_virtual_blocks[i], virtual_tokens, _use_ckpt=use_ckpt)
            point_tokens = _cp(self.space_point2virtual_blocks[i], point_tokens, virtual_tokens,
                               _use_ckpt=use_ckpt, mask=mask)

            space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
            tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)

        tokens = tokens[:, : N - self.num_virtual_tracks]
        flow = self.flow_head(tokens)
        vis_conf = self.vis_conf_head(tokens)
        out = torch.cat([flow, vis_conf], dim=-1)
        if return_tokens:
            return out, tokens
        return out
