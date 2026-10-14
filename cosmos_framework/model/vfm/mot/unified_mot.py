# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cosmos_framework.model.attention import attention as imaginaire_attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.utils import log
from cosmos_framework.data.vfm.sequence_packing import (
    FactoredSequencePack,
    from_joint,
    from_und_gen_splits,
    get_device_and_dtype,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)
from cosmos_framework.model.vfm.mot.attention import (
    AttentionMaskType,
    dispatch_attention,
)
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryState, MemoryValue

from cosmos_framework.model.vfm.vlm.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.vfm.vlm.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLPreTrainedModel,
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
)

from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl import (
    apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb,
)
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import (
    prepare_multimodal_reasoner_inputs,
)

from cosmos_framework.model.vfm.vlm.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from cosmos_framework.model.vfm.vlm.qwen3_vl_moe.qwen3_vl_moe import (
    LBLMetadata,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionModel,
)

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096


class LayerTypes:
    """Architecture-family dispatch table for the shared MoT layers.

    A single ``LayerTypes(variant)`` instance bundles the four sub-layer
    classes / functions that the MoT decoder layers swap in based on the
    text-model family: the MLP block, the RMSNorm class, the rotary
    embedding module, and the ``apply_rotary_pos_emb`` function.  Passed
    through ``_impl_init`` to ``MoTDecoderLayer`` so a single decoder
    implementation works across all three families.

    Supported variants: ``"qwen3_vl_dense"``, ``"qwen3_vl_moe"``,
    ``"nemotron_dense"``.
    """

    def __init__(self, variant: str):
        self.variant = variant
        if variant == "qwen3_vl_dense":
            self.mlp = Qwen3VLTextMLP
            self.rms_norm = Qwen3VLTextRMSNorm
            self.rotary_embedding = Qwen3VLTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        elif variant == "qwen3_vl_moe":
            self.mlp = Qwen3VLMoeTextMLP
            self.rms_norm = Qwen3VLMoeTextRMSNorm
            self.rotary_embedding = Qwen3VLMoeTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        elif variant == "nemotron_dense":
            self.mlp = Nemotron3DenseVLMLP
            self.rms_norm = Nemotron3DenseVLRMSNorm
            self.rotary_embedding = MultiModalRotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
        else:
            raise ValueError(f"Unknown LayerTypes variant: {variant!r}")

    @property
    def is_moe(self) -> bool:
        return self.variant == "qwen3_vl_moe"


class _MoTConfigBase(object):
    """Shared MoT wrapper logic for all three architecture families.

    Concrete subclasses: :class:`Qwen3VLMoTConfig` (dense Qwen3-VL),
    :class:`Qwen3VLMoeMoTConfig` (Qwen3-VL MoE), and
    :class:`Nemotron3DenseVLMoTConfig`.

    Stores the **JSON config dict** (as loaded from the per-family
    config JSON) plus a small fixed set of MoT-specific fields, and
    materializes fresh HF configs on demand via :pyattr:`full_config`,
    :pyattr:`text_config`, and :pyattr:`vision_config`.  Three concerns
    are kept separate:

    1. The HF *text* config that ``*TextModel`` / decoder layers
       actually consume — built from the nested ``text_config``
       sub-section of the JSON (or the flat dict for LLM-only configs),
       optionally massaged by :meth:`_transform_text_dict`, then splatted
       into :pyattr:`_text_config_cls`.  MoT-specific fields are *not*
       merged in here; see point 3.
    2. The HF *vision* sub-section (``config_dict["vision_config"]``)
       that the visual tower needs but the text model must never see —
       surfaced via :pyattr:`vision_config`.
    3. MoT-specific knobs (``qk_norm_for_text`` / ``qk_norm_for_diffusion``,
       ``include_visual``) that are read directly off the wrapper:
       ``qk_norm_for_*`` are passed as constructor kwargs to the
       ``*TextModel`` (and from there to the decoder layers / packed
       attention), and ``include_visual`` gates whether the ViT is built
       at all by the ``*TextForCausalLM`` __init__.

    Constructor surface:

    - ``config_dict``: full JSON dict (nested-VLM with top-level
      ``text_config`` / ``vision_config`` / ``image_token_id``, or flat
      LLM-only).  Stored verbatim under ``self.config_dict`` and never
      mutated; the materialization properties build fresh dicts.
    - MoT kwargs (``qk_norm_for_text``, ``qk_norm_for_diffusion``,
      ``include_visual``): stored as plain attributes.  These do *not*
      flow into :pyattr:`text_config` — the ``*TextForCausalLM.__init__``
      flow reads them directly off the wrapper.
    - ``text_config_overrides``: optional ``{field: value}`` mapping
      that the :pyattr:`text_config` property merges into the text
      dict *after* :meth:`_transform_text_dict` (so SMOKE-style
      shrinks like ``num_hidden_layers=2`` win over both the JSON
      defaults and the Nemotron ``56 -> 28`` fold).  Stored as a
      plain attribute so the ``create_vlm_config`` ``setattr`` flow
      can drop a fresh mapping in post-construction; the next
      ``text_config`` access picks it up.

    Post-construction overrides via plain ``setattr`` (the
    ``create_vlm_config`` flow in ``cosmos_framework/configs/base/defaults/vlm.py``)
    just update the same plain attributes, so the next property access
    picks up the latest values.  No cache, no ``__setattr__``
    interception, no override bucket — the property rebuild is cheap
    relative to model init and consumers (``*TextForCausalLM.__init__``)
    read each property once and store the result locally.

    Subclasses customize four things:

    - :pyattr:`_full_config_cls`: HF full-config class (e.g.
      ``Qwen3VLConfig``); used by :pyattr:`full_config` to build the
      base ``PreTrainedModel.config`` for the ``*ForCausalLM`` wrapper.
    - :pyattr:`_text_config_cls`: HF text-config class (e.g.
      ``Qwen3VLTextConfig``); used by :pyattr:`text_config`.
    - :pyattr:`_vision_config_cls`: HF vision-config class for the
      visual tower (only relevant when ``include_visual=True``).
    - :meth:`_transform_text_dict`: optional hook for architecture-
      specific massaging of the text-config dict (e.g. the Nemotron 3
      Dense VL ``56 -> 28`` layer fold).  Default is identity.
    """

    _full_config_cls: type = type(None)
    _text_config_cls: type = type(None)
    _vision_config_cls: type = type(None)

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        include_visual: bool = False,
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        self.config_dict = dict(config_dict)
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.include_visual = include_visual
        self.text_config_overrides: dict[str, Any] = dict(text_config_overrides) if text_config_overrides else {}

    @property
    def full_config(self) -> Any:
        """Materialize the full HF LLM/VLM config from the stored dict.

        Splats :pyattr:`config_dict` directly into :pyattr:`_full_config_cls`.
        Used by ``*TextForCausalLM.__init__`` to build the
        ``PreTrainedModel.config`` that the HF base class stores under
        ``self.config``.
        """
        if self._full_config_cls is type(None):
            raise ValueError(f"No _full_config_cls defined for {self.__class__.__name__}")
        return self._full_config_cls(**self.config_dict)

    @property
    def text_config(self) -> Any:
        """Materialize a fresh HF text config from the stored dict.

        Built from scratch on every access — cheap relative to model
        init and naturally correct under the ``create_vlm_config``
        post-init ``setattr`` flow.  Construction steps:

        1. Pick the source dict: the ``text_config`` sub-section of
           :pyattr:`config_dict` when nested, or the dict itself for
           flat / LLM-only configs.
        2. Run :meth:`_transform_text_dict` on it (default identity;
           the Nemotron variant uses it for the ``56 -> 28`` layer
           fold).
        3. Merge :pyattr:`text_config_overrides` on top (last write
           wins) so caller-supplied per-field overrides (SMOKE
           shrinks like ``num_hidden_layers=2``) beat both the JSON
           defaults and the Nemotron transform.  Applied as a
           post-transform overlay rather than a pre-transform merge
           so transform-aware overrides (final layer counts) don't
           get re-halved by the Nemotron fold.
        4. Splat into :pyattr:`_text_config_cls`.

        MoT-specific fields (``qk_norm_for_*``, ``include_visual``) are
        deliberately *not* injected into the returned HF text config —
        they're read directly off the wrapper by
        ``*TextForCausalLM.__init__`` and forwarded as ``*TextModel``
        constructor kwargs.  Likewise ``tie_word_embeddings`` is only
        consumed via :pyattr:`full_config` (it lives at the JSON
        top-level for nested VLM configs and is not duplicated into the
        text sub-section).
        """
        nested = self.config_dict.get("text_config")
        text_dict = nested if isinstance(nested, dict) else self.config_dict
        text_dict = self._transform_text_dict(text_dict)
        overrides = getattr(self, "text_config_overrides", None) or {}
        if overrides:
            text_dict = {**text_dict, **overrides}

        if self._text_config_cls is type(None):
            raise ValueError(f"No _text_config_cls defined for {self.__class__.__name__}")
        return self._text_config_cls(**text_dict)

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Subclass hook: transform the text-config dict before MoT fields are merged.

        Default implementation is identity.  Override in subclasses to
        apply architecture-specific massaging (e.g. the Nemotron 3
        Dense VL ``56 -> 28`` layer fold).  The hook receives the dict
        already split out of the nested ``text_config`` sub-section
        (or the flat dict for LLM-only configs) and must return a
        dict-like object suitable for splatting into
        :pyattr:`_text_config_cls`.
        """
        return text_dict

    @property
    def vision_config(self) -> Any | None:
        """Materialize the HF vision config for the visual tower, or None.

        Mirrors :pyattr:`text_config`'s "build on demand" pattern:
        gates on :pyattr:`include_visual`, splats the JSON's
        ``vision_config`` sub-section into :pyattr:`_vision_config_cls`,
        and returns the built HF config (caller plugs it into
        ``Qwen3VL{,Moe}VisionModel._from_config``).

        Returns ``None`` when ``include_visual=False``.  When
        ``include_visual=True`` but the JSON has no ``vision_config``
        sub-section (e.g. an LLM-only checkpoint mistakenly used as a
        VLM), raises ``ValueError`` eagerly rather than silently
        building an empty ViT.
        """
        if not self.include_visual:
            return None
        vision_dict = self.config_dict.get("vision_config")
        if vision_dict is None:
            raise ValueError(
                "include_visual=True requires a vision_config sub-section in the language-model JSON config."
            )
        if self._vision_config_cls is type(None):
            raise ValueError(f"No _vision_config_cls defined for {self.__class__.__name__}")
        return self._vision_config_cls(**vision_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "_MoTConfigBase":
        """Load the JSON file verbatim and wrap it.

        The full dict is stored under ``self.config_dict``; sibling
        ``vision_config`` / ``image_token_id`` / ``vision_*_token_id``
        fields (when present) are surfaced lazily via
        :pyattr:`vision_config` and by HF downstream consumers reading
        the dict directly.
        """
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls(config_dict=config_dict)


class Qwen3MoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Qwen3 family."""

    _full_config_cls = Qwen3VLTextConfig
    _text_config_cls = Qwen3VLTextConfig


class Qwen3VLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the dense Qwen3-VL family."""

    _full_config_cls = Qwen3VLConfig
    _text_config_cls = Qwen3VLTextConfig
    _vision_config_cls = Qwen3VLVisionConfig


class Qwen3VLMoeMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Qwen3-VL MoE family."""

    _full_config_cls = Qwen3VLMoeConfig
    _text_config_cls = Qwen3VLMoeTextConfig
    _vision_config_cls = Qwen3VLMoeVisionConfig


class Nemotron3DenseVLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Nemotron 3 Dense VL family.

    Inherits the shared ``config_dict``-storing wrapper, MoT field
    handling, nested-vs-flat ``text_config`` resolution, and
    ``from_json_file`` loader from :class:`_MoTConfigBase`.  Only two
    things are Nemotron-specific:

    1. :pyattr:`_text_config_cls` — the materialized HF text config
       class is :class:`Nemotron3DenseVLTextConfig`.
    2. :meth:`_transform_text_dict` — folds the upstream VLM's
       alternating ``(attn, MLP)`` blocks (``num_hidden_layers == 56``)
       into MoT's fused decoder layers (28 effective).

    Nemotron 3 Dense VL has no nested ``vision_config`` sub-section in
    its JSON, so the inherited ``vision_config`` property is unused —
    callers leave the inherited ``include_visual`` at its default of
    ``False`` and never access the property.
    """

    _full_config_cls = Nemotron3DenseVLTextConfig
    _text_config_cls = Nemotron3DenseVLTextConfig

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        if text_dict.get("num_hidden_layers") == 56:
            return {**text_dict, "num_hidden_layers": 28}
        return text_dict


class PackedAttentionMoT(nn.Module):
    """
    Dual-pathway packed attention for MoT architectures.
    Implements understanding and generation pathways with separate projections.

    Used for Qwen3VL (Dense), Qwen3VL-MoE, and Nemotron 3 Dense VL variants.
    QK normalisation and RoPE function are selected via ``layer_types`` and the
    ``qk_norm_for_text`` config attribute (the generation pathway always installs
    QK norm).
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        eps = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        if qk_norm_for_text:
            self.q_norm = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        if qk_norm_for_diffusion:
            self.q_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._apply_rotary_pos_emb = layer_types.apply_rotary_pos_emb
        self.dispatch_attention_fn = dispatch_attention

        self._track_capture = False
        self._cap_q: torch.Tensor | None = None
        self._cap_k: torch.Tensor | None = None

        self.camera_roce_mode = "none"
        self.camera_roce_qk: nn.Sequential | None = None
        self.camera_roce_vo: nn.Sequential | None = None
        self._camera_roce_spatial_slots: tuple[int, ...] = ()

    def enable_camera_roce(self, camera_code_dim: int, mode: str) -> None:
        """Attach ReDirector-style per-layer camera phase heads.

        Qwen's half-rotary layout interleaves T/H/W frequency slots.  Only H
        and W slots receive a camera phase; temporal slots remain exactly zero.
        One phase is predicted per KV head and repeated over its GQA query
        group instead of learning inconsistent Q- and KV-head phases.
        """

        if mode not in {"none", "qk", "qk_vo"}:
            raise ValueError(f"Unsupported camera RoCE mode: {mode!r}")
        self.camera_roce_mode = mode
        if mode == "none":
            return

        rope_scaling = getattr(self.config, "rope_scaling", None) or {}
        sections = rope_scaling.get("mrope_section", getattr(self.config, "mrope_section", [24, 20, 20]))
        half_dim = self.head_dim // 2
        if len(sections) != 3 or sum(sections) != half_dim:
            raise ValueError(
                "Camera RoCE requires Qwen interleaved mRoPE sections summing to head_dim/2; "
                f"got sections={sections}, head_dim={self.head_dim}."
            )
        h_slots = tuple(range(1, 3 * int(sections[1]), 3))
        w_slots = tuple(range(2, 3 * int(sections[2]), 3))
        self._camera_roce_spatial_slots = h_slots + w_slots
        phase_dim = self.num_key_value_heads * len(self._camera_roce_spatial_slots)
        self.camera_roce_qk = nn.Sequential(nn.SiLU(), nn.Linear(camera_code_dim, phase_dim, bias=False))
        if mode == "qk_vo":
            self.camera_roce_vo = nn.Sequential(nn.SiLU(), nn.Linear(camera_code_dim, phase_dim, bias=False))
        self.reset_camera_roce_parameters()

    def reset_camera_roce_parameters(self) -> None:
        """Zero phase outputs so a pretrained checkpoint starts as an exact no-op."""

        for head in (self.camera_roce_qk, self.camera_roce_vo):
            if head is not None:
                nn.init.zeros_(head[-1].weight)

    @staticmethod
    def _apply_camera_rotation(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Apply a per-token/per-head split-half rotary angle."""

        input_dtype = x.dtype
        x_float = x.float()
        angle = angle.float()
        x1, x2 = x_float.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x_float * angle.cos() + rotated * angle.sin()).to(input_dtype)

    def _camera_angle(self, code: torch.Tensor, head: nn.Sequential) -> torch.Tensor:
        """Predict KV-head angles and scatter them into Qwen's H/W slots."""

        spatial = head(code).view(
            code.shape[0],
            self.num_key_value_heads,
            len(self._camera_roce_spatial_slots),
        )
        half = spatial.new_zeros(code.shape[0], self.num_key_value_heads, self.head_dim // 2)
        half[..., list(self._camera_roce_spatial_slots)] = spatial
        return torch.cat((half, half), dim=-1)

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask: AttentionMaskType,
        packed_position_embeddings: tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        camera_code: FactoredSequencePack | None = None,
    ) -> tuple[FactoredSequencePack, KVToStore | None]:
        """Forward pass with optional memory-augmented attention.

        When ``memory_value`` is provided, ``dispatch_attention_fn`` routes to
        the appropriate attention kernel (e.g. three-way KV-cache attention
        for training, or AR inference concat + dense attention).

        ``kv_to_store`` is produced when ``memory_value`` is present:
        ``(gen_k, gen_v, und_k, und_v)`` for the caller to write back via
        ``MemoryState.write_for_layer()``.  The tensors are passed with
        gradients attached; each ``MemoryState`` decides whether to detach
        (e.g. for truncated BPTT) or keep gradients (e.g. teacher forcing).

        Args:
            pack: Packed sequence with und/gen tokens
            attention_mask: Attention mask (BlockMask or SplitInfo)
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Optional read-only tensor container for memory-augmented attention.
        """

        q_und_in = self.q_proj(get_und_seq(pack))
        q_gen_in = self.q_proj_moe_gen(get_gen_seq(pack))

        k_und_in = self.k_proj(get_und_seq(pack))
        k_gen_in = self.k_proj_moe_gen(get_gen_seq(pack))

        v_und_in = self.v_proj(get_und_seq(pack))
        v_gen_in = self.v_proj_moe_gen(get_gen_seq(pack))

        q_und = q_und_in.view(-1, self.num_attention_heads, self.head_dim)
        k_und = k_und_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_und = v_und_in.view(-1, self.num_key_value_heads, self.head_dim)

        q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)
        k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)

        q_und = self.q_norm(q_und)
        k_und = self.k_norm(k_und)

        q_gen = self.q_norm_moe_gen(q_gen)
        k_gen = self.k_norm_moe_gen(k_gen)

        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]

        q_und_, k_und_ = self._apply_rotary_pos_emb(
            q_und,
            k_und,
            get_und_seq(packed_cos),
            get_und_seq(packed_sin),
            unsqueeze_dim=1,
        )
        q_gen_, k_gen_ = self._apply_rotary_pos_emb(
            q_gen,
            k_gen,
            get_gen_seq(packed_cos),
            get_gen_seq(packed_sin),
            unsqueeze_dim=1,
        )

        camera_vo_angle_und: torch.Tensor | None = None
        camera_vo_angle_gen: torch.Tensor | None = None
        if camera_code is not None and self.camera_roce_qk is not None:
            qk_und = self._camera_angle(get_und_seq(camera_code), self.camera_roce_qk)
            qk_gen = self._camera_angle(get_gen_seq(camera_code), self.camera_roce_qk)
            q_und_ = self._apply_camera_rotation(q_und_, qk_und.repeat_interleave(self.num_key_value_groups, dim=1))
            k_und_ = self._apply_camera_rotation(k_und_, qk_und)
            q_gen_ = self._apply_camera_rotation(q_gen_, qk_gen.repeat_interleave(self.num_key_value_groups, dim=1))
            k_gen_ = self._apply_camera_rotation(k_gen_, qk_gen)

            if self.camera_roce_vo is not None:
                camera_vo_angle_und = self._camera_angle(get_und_seq(camera_code), self.camera_roce_vo)
                camera_vo_angle_gen = self._camera_angle(get_gen_seq(camera_code), self.camera_roce_vo)
                v_und = self._apply_camera_rotation(v_und, -camera_vo_angle_und)
                v_gen = self._apply_camera_rotation(v_gen, -camera_vo_angle_gen)

        packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)
        packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)
        packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)

        if self._track_capture:
            self._cap_q = packed_query_states_
            self._cap_k = packed_key_states_

        packed_attn_output, kv_to_store = self.dispatch_attention_fn(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
        )

        if memory_value is not None and kv_to_store is None:
            und_len = pack["_num_causal_tokens"]
            gen_len = pack["_num_full_tokens"]
            kv_to_store = (
                k_gen_[:gen_len].unsqueeze(0),
                v_gen[:gen_len].unsqueeze(0),
                k_und_[:und_len].unsqueeze(0),
                v_und[:und_len].unsqueeze(0),
            )

        und_seq = self.o_proj(get_und_seq(packed_attn_output))
        gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))
        if camera_vo_angle_und is not None and camera_vo_angle_gen is not None:
            und_angle_q = camera_vo_angle_und.repeat_interleave(self.num_key_value_groups, dim=1)
            gen_angle_q = camera_vo_angle_gen.repeat_interleave(self.num_key_value_groups, dim=1)
            und_seq = self._apply_camera_rotation(
                und_seq.view(und_seq.shape[0], self.num_attention_heads, self.head_dim),
                und_angle_q,
            ).reshape(und_seq.shape[0], self.hidden_size)
            gen_seq = self._apply_camera_rotation(
                gen_seq.view(gen_seq.shape[0], self.num_attention_heads, self.head_dim),
                gen_angle_q,
            ).reshape(gen_seq.shape[0], self.hidden_size)
        return from_und_gen_splits(und_seq, gen_seq, pack), kv_to_store

    def reasoner_forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "ReasonerKVCache | None",
        layer_idx: int,
    ) -> torch.Tensor:
        """Run the reasoner (und) projections with a per-layer KV cache.

        Operates in standard ``[B, T, hidden_size] -> [B, T, hidden_size]``
        layout (rather than the factored sequence-packed layout used during
        training in :meth:`forward`) so it can drive a per-layer KV cache
        in a clean AR loop.

        All attention compute is dispatched through
        ``imaginaire.attention.attention`` (per repo policy) which expects the
        heads-last contiguous layout ``[B, S, H, D]`` and natively handles GQA
        (``H_KV != H``) — no manual head expansion is needed.

        Causal masking semantics:
            - Prefill (``T>1`` with empty cache): ``is_causal=True`` with
              ``CausalType.DontCare`` (``seq_q == seq_kv``).
            - Decode (``T==1`` with non-empty cache): ``is_causal=False``;
              the single query attends to all cached keys (it is always the
              rightmost position, so no masking is needed).

        Multi-token incremental prefill on top of an existing cache is not
        supported here — ``_impl_generate_reasoner_text`` never triggers it.
        """
        B, T, _ = hidden_states.shape
        H = self.num_attention_heads
        H_kv = self.num_key_value_heads
        D = self.head_dim

        q = self.q_proj(hidden_states).view(B, T, H, D)
        k = self.k_proj(hidden_states).view(B, T, H_kv, D)
        v = self.v_proj(hidden_states).view(B, T, H_kv, D)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        if cache is not None:
            k_full, v_full = cache.update(layer_idx, k, v)
        else:
            k_full, v_full = k, v

        is_causal = T > 1 and k_full.shape[1] == T

        out = imaginaire_attention(
            query=q,
            key=k_full,
            value=v_full,
            is_causal=is_causal,
            causal_type=CausalType.DontCare if is_causal else None,
            scale=self.scaling,
        )

        return self.o_proj(out.reshape(B, T, H * D))


def _impl_init(
    self,
    config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
    *,
    layer_types: LayerTypes,
    qk_norm_for_text: bool,
    qk_norm_for_diffusion: bool,
):
    """Shared ``__init__`` body for the three MoT text-model variants.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.  Sub-layer classes (MLP, RMSNorm,
    rotary embedding) are dispatched through ``layer_types``.
    """
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    self.layers = nn.ModuleList()
    for layer_idx in range(config.num_hidden_layers):
        self.layers.append(
            MoTDecoderLayer(
                config=config,
                layer_types=layer_types,
                layer_idx=layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
            )
        )

    self.norm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    self.norm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    self.rotary_emb = layer_types.rotary_embedding(config)

    self.post_init()


def _impl_init_taylorseer(self, cache_dic=None, current=None):
    """Initialize TaylorSeer acceleration attributes.

    Shared implementation for ``init_taylorseer`` on
    ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.
    """
    self.cache_dic = cache_dic or {}
    self.current = current or {
        "step": 0,
        "type": "full",
        "stream": "layers_stream",
        "layer": 0,
        "module": "total",
        "activated_steps": [0],
    }
    self.enable_taylorseer = True


def _impl_forward(
    self,
    pack: FactoredSequencePack,
    attention_mask,
    position_ids: torch.Tensor,
    natten_metadata_list: list | None = None,
    memory: MemoryState | None = None,
    camera_code: FactoredSequencePack | None = None,
) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
    """Shared training forward pass for the three MoT text models.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.

    Args:
        pack: Packed sequence with und/gen tokens.
        attention_mask: Attention mask (BlockMask or SplitInfo).
        position_ids: Position IDs (1D ``[N]`` for standard RoPE or 2D
            ``[3, N]`` for mrope).
        natten_metadata_list: Optional per-layer NATTEN metadata.
        memory: Optional ``MemoryState`` for persistent memory across
            forward passes.
        camera_code: Optional vision-only camera sidecar with the same
            factored packing as ``pack``. Non-vision rows must be zero.
    """

    device, dtype = get_device_and_dtype(pack)
    _meta_tensor = torch.tensor([], dtype=dtype, device=device)
    cos, sin = self.rotary_emb(
        _meta_tensor, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    position_embeddings = (
        from_joint(cos, pack),
        from_joint(sin, pack),
    )

    lbl_metadata_all = dict(und=[], gen=[])

    hidden_states = pack

    if memory is not None:
        memory.init(hidden_states, device)

    memory_gen_only = memory.is_gen_only() if memory is not None else False

    for i, decoder_layer in enumerate(self.layers):
        if getattr(self, "_track_feedback_fn", None) is not None and i == getattr(
            self, "_track_feedback_inject_layer", -1
        ):
            hidden_states = self._track_feedback_fn(hidden_states)

        memory_value = memory.read_for_layer(i) if memory is not None else None

        hidden_states, lbl_metadata_dict, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            position_embeddings,
            natten_metadata=None if natten_metadata_list is None else natten_metadata_list[i],
            memory_value=memory_value,
            gen_only=memory_gen_only,
            camera_code=camera_code,
        )

        if kv_to_store is not None and memory is not None:
            memory.write_for_layer(i, kv_to_store)

        for pathway, lbl_metadata in lbl_metadata_dict.items():
            lbl_metadata_all[pathway].append(lbl_metadata)

    final_lbl_metadata: dict[str, LBLMetadata] = dict()
    for pathway, lbl_metadata_list in lbl_metadata_all.items():
        if len(lbl_metadata_list) > 0:
            num_tokens_per_expert = torch.stack(
                [lbl_metadata.num_tokens_per_expert for lbl_metadata in lbl_metadata_list]
            )
            num_tokens = torch.stack([lbl_metadata.num_tokens for lbl_metadata in lbl_metadata_list])
            mean_router_prob_per_expert = torch.stack(
                [lbl_metadata.mean_router_prob_per_expert for lbl_metadata in lbl_metadata_list]
            )
            final_lbl_metadata[pathway] = LBLMetadata(
                num_tokens_per_expert=num_tokens_per_expert,
                num_tokens=num_tokens,
                mean_router_prob_per_expert=mean_router_prob_per_expert,
            )

    hidden_states_out = zeros_like(hidden_states)
    set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))
    set_gen_seq(hidden_states_out, self.norm_moe_gen(get_gen_seq(hidden_states)))

    return hidden_states_out, final_lbl_metadata


def _run_mlp(
    mlp: torch.nn.Module,
    input: torch.Tensor,
) -> tuple[torch.Tensor, LBLMetadata | None]:
    """Run an MLP block and normalize the return shape across dense / MoE.

    ``Qwen3VLMoeTextSparseMoeBlock`` returns ``(Tensor, LBLMetadata)`` so the
    caller can aggregate the load-balancing-loss metadata across layers; every
    other MLP block (dense ``*MLP``) returns just the output tensor.  This
    helper unifies both into a single ``(output, lbl_metadata_or_None)`` shape
    so the decoder layers and the reasoner-tower forward don't need to branch
    on the MLP type.
    """
    if isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
        (
            output_tensor,
            lbl_metadata,
        ) = mlp(input)
    else:
        output_tensor = mlp(input)
        lbl_metadata = None
    return output_tensor, lbl_metadata


class MoTDecoderLayer(nn.Module):
    """
    Unified MoT (Mixture of Transformers) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionMoT(
            config,
            layer_types=layer_types,
            layer_idx=layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

        if (
            hasattr(config, "mlp_only_layers")
            and (layer_idx not in config.mlp_only_layers)
            and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
            self.mlp_moe_gen = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = layer_types.mlp(config)
            self.mlp_moe_gen = layer_types.mlp(config)

        self.input_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input: FactoredSequencePack,
        attention_mask,
        packed_position_embeddings: tuple[FactoredSequencePack, FactoredSequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        gen_only: bool = False,
        camera_code: FactoredSequencePack | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata], KVToStore | None]:
        """Forward pass with MoT routing and optional memory-augmented attention.

        Returns a 3-tuple: ``(hidden_states, lbl_metadata_dict, kv_to_store)``.
        ``kv_to_store`` is non-None when ``memory_value`` is provided,
        containing ``(gen_k, gen_v, und_k, und_v)`` to be written back by
        ``MemoryState.write_for_layer()`` outside the ``torch.compile``
        boundary.

        Args:
            input: Packed sequence with und/gen tokens
            attention_mask: Attention mask
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Read-only tensor container from MemoryState.read_for_layer().
            gen_only: When True, skip the understanding pathway (und K/V come from cache).
            camera_code: Optional packed per-token camera code.
        """
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(input)),
            self.input_layernorm_moe_gen(get_gen_seq(input)),
            input,
        )

        kv_to_store: KVToStore | None = None
        if gen_only:
            assert natten_metadata is None
            _gen_norm = get_gen_seq(pack_norm_out)
            gen_pack = from_und_gen_splits(
                _gen_norm.new_empty(0, _gen_norm.shape[-1]),
                _gen_norm,
                pack_norm_out,
            )

            _cos, _sin = packed_position_embeddings
            _empty_cos_und = get_und_seq(_cos)[:0]
            _empty_sin_und = get_und_seq(_sin)[:0]
            gen_position_embeddings = (
                from_und_gen_splits(_empty_cos_und, get_gen_seq(_cos), _cos),
                from_und_gen_splits(_empty_sin_und, get_gen_seq(_sin), _sin),
            )
            gen_camera_code = None
            if camera_code is not None:
                _camera_gen = get_gen_seq(camera_code)
                gen_camera_code = from_und_gen_splits(
                    _camera_gen.new_empty(0, _camera_gen.shape[-1]),
                    _camera_gen,
                    camera_code,
                )

            pack_attn_out, kv_to_store = self.self_attn(
                gen_pack,
                attention_mask,
                gen_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
                camera_code=gen_camera_code,
            )
            gen_attn_out = get_gen_seq(pack_attn_out)
            residual_gen = get_gen_seq(input) + gen_attn_out
        else:
            pack_attn_out, kv_to_store = self.self_attn(
                pack_norm_out,
                attention_mask,
                packed_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
                camera_code=camera_code,
            )
            residual_und = get_und_seq(input) + get_und_seq(pack_attn_out)
            residual_gen = get_gen_seq(input) + get_gen_seq(pack_attn_out)

        lbl_metadata_dict: dict[str, LBLMetadata] = dict()

        if gen_only:
            ln_out_und = residual_gen.new_empty(0, residual_gen.shape[-1])
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)

            gen_len = pack_attn_out["_num_full_tokens"]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]

            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)

            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)

            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            mlp_out_und_seq = residual_gen.new_empty(0, residual_gen.shape[-1])
            mlp_out_gen_seq = residual_gen + mlp_out_gen
        else:
            ln_out_und = self.post_attention_layernorm(residual_und)
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)


            gen_len = pack_attn_out["_num_full_tokens"]
            und_len = pack_attn_out["_num_causal_tokens"]
            ln_out_und_unpadded = ln_out_und[:und_len]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]

            mlp_out_und_unpadded, lbl_metadata_und = _run_mlp(self.mlp, ln_out_und_unpadded)
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)

            mlp_out_und = torch.cat([mlp_out_und_unpadded, ln_out_und[und_len:]], dim=0)
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)

            if lbl_metadata_und is not None:
                lbl_metadata_dict["und"] = lbl_metadata_und
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            mlp_out_und_seq = residual_und + mlp_out_und
            mlp_out_gen_seq = residual_gen + mlp_out_gen

        return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict, kv_to_store

    def reasoner_forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: "ReasonerKVCache | None",
        layer_idx: int,
    ) -> torch.Tensor:
        """Run this decoder layer through the reasoner (und) pathway only.

        Skips the ``*_moe_gen`` sub-modules — the dual-pathway training
        :meth:`forward` runs both und and gen towers, but autoregressive
        text decoding only needs und.  Called by
        :func:`_impl_reasoner_forward` (which iterates over
        ``model.layers`` in the AR loop).

        FSDP2 note: under ``data_parallel_shard_degree > 1`` each decoder
        layer is its own FSDP unit (see
        ``parallelize_unified_mot.apply_fsdp``) and its parameters live
        as ``DTensor`` shards until the layer's pre-forward hook
        materializes them.  This method is registered as a
        forward-equivalent via ``register_fsdp_forward_method`` in
        :func:`parallelize_unified_mot.apply_fsdp` so the layer's
        pre-forward unshard / post-forward reshard hooks fire here
        exactly as they would for :meth:`forward`, even though we call
        sub-modules (``input_layernorm`` / ``self_attn`` / ``mlp``)
        directly — without that registration,
        ``layer.input_layernorm.weight`` et al. stay sharded and
        ``self.weight * hidden_states`` raises ``RuntimeError:
        aten.mul.Tensor: got mixed torch.Tensor and DTensor``.  In
        single-rank settings the registration is a no-op and this
        method just runs the und pathway directly.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out = self.self_attn.reasoner_forward(h, cos, sin, cache, layer_idx)
        hidden = residual + attn_out

        residual = hidden
        h = self.post_attention_layernorm(hidden)

        B, T, H = h.shape
        mlp_out, _ = _run_mlp(self.mlp, h.reshape(B * T, H))
        mlp_out = mlp_out.view(B, T, H)
        return residual + mlp_out


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text model for MoT with dense MLPs.
    This is a wrapper around the _impl_forward defined above,
    specialized for dense models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


class Qwen3VLMoeTextModel(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text model for MoT with MoE MLPs.
    This is a wrapper around the _impl_* helpers defined above,
    specialized for MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_moe"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


class Nemotron3DenseVLTextModel(Nemotron3DenseVLPreTrainedModel):
    """
    Nemotron 3 Dense VL text model adapted for MoT training.
    This is a wrapper around the _impl_* helpers defined above,
    specialized for Nemotron 3 Dense VL models.
    """

    def __init__(
        self,
        config: Nemotron3DenseVLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("nemotron_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
        )

    def init_taylorseer(self, cache_dic=None, current=None) -> None:
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


@dataclass
class ReasonerKVCache:
    """Per-layer KV cache for the reasoner-tower autoregressive loop.

    Tensors are stored in the heads-last BSHD layout that
    ``imaginaire.attention.attention`` expects::

        keys[layer_idx]:   [B, T, num_kv_heads, head_dim]
        values[layer_idx]: [B, T, num_kv_heads, head_dim]

    where ``T`` grows monotonically as new tokens are appended each
    decode step.  ``None`` entries indicate an empty cache for that layer
    (set by ``empty()``); the first ``update`` populates them.
    """

    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> "ReasonerKVCache":
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return int(self.keys[0].shape[1])

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V along the seq dim and return the full cached K/V."""
        cached_k = self.keys[layer_idx]
        cached_v = self.values[layer_idx]
        if cached_k is None or cached_v is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = torch.cat([cached_k, k], dim=1)
            self.values[layer_idx] = torch.cat([cached_v, v], dim=1)
        return self.keys[layer_idx], self.values[layer_idx]

    def reset(self) -> None:
        for i in range(self.num_layers):
            self.keys[i] = None
            self.values[i] = None


def _impl_reasoner_forward(
    self,
    input_ids: torch.Tensor | None,
    cache: ReasonerKVCache | None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Shared reasoner-tower forward used by all three MoT text-model variants.

    Returns the final post-norm hidden states ``[B, T, hidden_size]``.

    Each per-layer call goes through ``MoTDecoderLayer.reasoner_forward``,
    which is registered as an FSDP2 forward-equivalent by
    ``parallelize_unified_mot.apply_fsdp`` so per-layer FSDP collectives
    fire correctly under ``data_parallel_shard_degree > 1``.  The
    outer ``self.embed_tokens`` / ``self.norm`` / ``self.rotary_emb``
    parameters live on the top-level FSDP unit and are materialized by
    the corresponding ``register_fsdp_forward_method(top, "generate_reasoner_text")``
    in ``parallelize_vfm_network`` — together those two registrations
    cover every FSDP-wrapped weight touched on the AR path.

    Args:
        input_ids: ``[B, T]`` integer token ids for this step.
        cache: Per-layer KV cache (mutated in place when not ``None``).
        position_ids: ``[B, T]`` absolute positions for these tokens.  When
            ``None``, positions are inferred from the cache so that prompts
            are indexed ``[0..T-1]`` and decode steps continue from the
            cache length.
        inputs_embeds: Pre-embedded inputs ``[B, T, hidden_size]``.
            Mutually exclusive with ``input_ids``.
        visual_pos_masks: ``[B, T]`` bool mask of visual token positions.
            Required (and must be on the same device as ``inputs_embeds``)
            when ``deepstack_visual_embeds`` is provided; ignored otherwise.
        deepstack_visual_embeds: Optional per-layer deepstack visual
            embeddings for the I2V path.  When provided, layer ``i``'s
            output is additively updated at the visual positions.  Every
            element must already match ``inputs_embeds.device`` and
            ``inputs_embeds.dtype``; the canonical producer
            ``prepare_multimodal_reasoner_inputs`` aligns both.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        assert input_ids is not None
        B, T = input_ids.shape
        device = input_ids.device
        h = self.embed_tokens(input_ids)
    else:
        B, T, _ = inputs_embeds.shape
        device = inputs_embeds.device
        h = inputs_embeds

    if position_ids is None:
        past_len = 0 if cache is None else cache.seq_len
        position_ids = torch.arange(past_len, past_len + T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

    cos, sin = self.rotary_emb(h, position_ids=position_ids)

    if deepstack_visual_embeds is not None and visual_pos_masks is None:
        raise ValueError("visual_pos_masks is required when deepstack_visual_embeds is provided.")

    for layer_idx, decoder_layer in enumerate(self.layers):
        h = decoder_layer.reasoner_forward(h, cos, sin, cache, layer_idx)
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            h = h.clone()
            h[visual_pos_masks, :] = h[visual_pos_masks, :] + deepstack_visual_embeds[layer_idx]

    return self.norm(h)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    seen_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Greedy / multinomial sampling with optional top-k, top-p, and presence/repetition penalties.

    Pipeline (order matters):
        1. Repetition penalty (CTRL/HF formula) — multiplicative rescale
           of every logit at a position seen in history (``seen_mask``).
           ``>1.0`` discourages repetition, ``<1.0`` encourages it,
           ``1.0`` is identity.
        2. Presence penalty (OpenAI semantics) — additive shift of every
           logit at a position seen in history.  ``>0`` discourages,
           ``<0`` encourages, ``0`` is identity.  Applied once per token
           regardless of how often it appeared (presence, not frequency).
        3. ``do_sample=False`` short-circuits to argmax.  The two
           penalties above are applied *before* this branch so they
           legitimately shift the greedy argmax — they're logit
           transformations, not sampling-only tricks.
        4. ``do_sample=True``: temperature → top-k → top-p → multinomial.

    ``seen_mask`` is the canonical "has this vocab id appeared in this
    sample's history" matrix maintained by
    :func:`_impl_generate_reasoner_text`.  Both penalties default to
    identity, and the fast path (both off) skips all penalty work and
    leaves the existing greedy/sampling logic bit-identical.

    ``generator`` is the only RNG-consuming primitive in this module:
    when provided, it is threaded into ``torch.multinomial`` so the
    sampling branch becomes reproducible across calls that share the
    same seeded generator (see :func:`_impl_generate_reasoner_text`'s
    ``seed`` argument).  Passing ``None`` (default) preserves the
    pre-seed behavior of consuming the device's default RNG and is
    bit-identical to the previous call signature.
    """
    if seen_mask is not None and (repetition_penalty != 1.0 or presence_penalty != 0.0):
        if repetition_penalty != 1.0:
            penalty_factor = torch.where(
                logits > 0,
                torch.full_like(logits, 1.0 / repetition_penalty),
                torch.full_like(logits, repetition_penalty),
            )
            logits = torch.where(seen_mask, logits * penalty_factor, logits)
        if presence_penalty != 0.0:
            logits = torch.where(
                seen_mask,
                logits - presence_penalty,
                logits,
            )

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth_values = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = (cumprobs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _all_ranks_finished(finished: torch.Tensor) -> bool:
    """Reduce ``finished.all()`` across every rank in the default process group.

    Single-process / non-distributed runs return the local result.  Under
    multi-rank inference each rank may see different prompts (e.g. test
    fixtures that build prompts from per-rank ``torch.randint`` without
    seed sync, or batched serving where each rank handles a different
    request slice) and therefore reach the EOS-driven early-exit at
    different decode steps.      Letting any one rank ``break`` while the
    others keep running strands the still-decoding ranks on the next
    FSDP all-gather collective inside ``*TextModel.reasoner_forward``
    (the AR loop is sharded by ``parallelize_unified_mot`` whenever
    ``parallel_dims.dp_enabled`` is True), causing an NCCL hang.

    Reducing the per-rank ``finished.all()`` flag with a logical AND —
    implemented as ``ReduceOp.MIN`` on a 0/1 ``uint8`` tensor, which is
    natively supported by NCCL — guarantees every rank breaks at the
    same iteration: only when *every* rank's every sample has emitted
    EOS.  Ranks that finished earlier keep iterating in lockstep with
    the slowest rank; their per-sample pad bookkeeping
    (``torch.where(finished, pad_token_id, next_token)``) already pins
    finished samples to ``pad_token_id``, so the extra steps just
    extend the rank-local KV cache with pad-fill — no semantic change
    to the returned tokens.

    Returns the boolean global flag (True iff every rank reports every
    sample finished).
    """
    local = finished.all()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        flag = local.to(dtype=torch.uint8)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())
    return bool(local)


@torch.no_grad()
def _impl_generate_reasoner_text(
    causal_lm: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | list[int] | None = None,
    pad_token_id: int | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    return_only_new_tokens: bool = False,
) -> torch.Tensor:
    """Run a reasoner-tower autoregressive decode loop with a per-layer KV cache.

    The loop has two phases:
        1. Prefill — process the prompt in one forward pass. For I2V, this
           first reuses Qwen3-VL visual helpers to scatter image embeddings.
        2. Decode — repeatedly feed the just-sampled token (length 1) back
           through the model, attending to the cached K/V.

    Each sample maintains its own ``finished`` flag.  Once a sample emits an
    ``eos_token_id``, subsequent positions are filled with ``pad_token_id``;
    the loop exits early once every sample has finished.

    **Distributed semantics.**  Under multi-rank execution
    (``torch.distributed`` initialized, the model wrapped by FSDP via
    ``parallelize_unified_mot``), the early-exit decision is reduced
    across ranks via :func:`_all_ranks_finished` so every rank breaks
    out of the decode loop at the *same* iteration.  This guarantees
    the AR loop's FSDP all-gather collectives stay in lockstep even
    when ranks see different prompts (and therefore reach EOS at
    different decode steps).  Ranks that finish earlier keep iterating
    in lockstep with the slowest rank; their per-sample pad bookkeeping
    is unchanged, so already-finished samples on early-finishing ranks
    simply continue emitting ``pad_token_id`` until the slowest rank
    catches up.  The returned ``T_new`` therefore matches the
    *globally slowest* rank's decode length.

    Args:
        causal_lm: A ``*ForCausalLM`` instance providing ``embed_tokens``
            (via ``causal_lm.model``) and ``lm_head``.
        input_ids: ``[B, T_prompt]`` integer token ids of the prompt.
            For image conditioning, the prompt must contain
            ``model.config.image_token_id`` placeholder tokens (one per
            patch *after* spatial merging) where the image features get
            scattered in.
        pixel_values: Optional image inputs in the native Qwen3-VL image
            processor format: a single ``[N_patches, C, H, W]`` float
            tensor (``model.visual.dtype``) holding *all* image patches
            from *all* images in *all* samples in the batch, concatenated
            along the leading patch dimension.  ``N_patches`` equals
            ``sum_i(t_i * h_i * w_i)`` over the rows of
            ``image_grid_thw`` below.  This is the same layout that
            ``Qwen3VLProcessor`` emits — pass it through unchanged.
            Moved to the prompt's device internally.  ``None`` (default)
            means text-only prompt; in that case the multimodal prefill
            path is skipped entirely.  Videos are *not* supported here —
            this function has no ``pixel_values_videos`` / ``video_grid_thw``
            parameters; for I2V conditioning, frames must be passed as
            images.
        image_grid_thw: Optional ``[num_images, 3]`` long tensor giving
            ``(t, h, w)`` — the temporal / height / width feature-grid
            size per image as produced by ``Qwen3VLProcessor`` (``t`` is
            typically 1 for still images).  ``num_images`` is the *total*
            image count across the entire batch, not per-sample.  Moved
            to the prompt's device internally.  Must be supplied together
            with ``pixel_values``; passing exactly one of the two raises
            ``ValueError``.  The number of image placeholder tokens in
            ``input_ids`` must equal
            ``sum_i(t_i * h_i * w_i) // model.visual.spatial_merge_size ** 2``;
            :func:`prepare_multimodal_reasoner_inputs` (which wraps
            :func:`get_placeholder_mask`) raises if this invariant is
            violated.  ``causal_lm.visual`` must exist when this is
            provided (a combined / language-only checkpoint without a
            vision tower will raise ``ValueError``).
        max_new_tokens: Maximum number of new tokens to generate per sample.
        eos_token_id: Token id (or list of ids) that terminates a sample.
            ``None`` disables early stopping.
        pad_token_id: Token id used to fill positions of finished samples
            so generated sequences can be returned as a single padded
            tensor.  Defaults to ``eos_token_id`` (or ``0`` if neither is
            provided).
        do_sample, temperature, top_k, top_p: Sampling controls.
        repetition_penalty: CTRL/HF-style multiplicative penalty
            applied to every logit at a vocab position seen in
            history (prompt + everything generated so far for this
            sample).  ``>1.0`` discourages repetition, ``<1.0``
            encourages it, ``1.0`` (default) is identity and skips
            all penalty bookkeeping.
        presence_penalty: OpenAI-style additive penalty applied once
            to every logit at a vocab position seen in history.
            ``>0`` discourages reuse, ``<0`` encourages it, ``0``
            (default) is identity.  Presence (binary), not frequency
            — appearing twice costs the same as appearing once.
            Both penalties are applied *before* the ``do_sample``
            argmax/multinomial branch, so they shift the greedy
            argmax too.  When both are at identity, the per-sample
            ``seen_mask`` is never allocated and the loop is
            bit-identical to the un-penalized fast path.
        seed: Optional integer seed for the sampling RNG.  When provided
            (and ``do_sample=True``), a fresh ``torch.Generator`` is
            allocated on ``input_ids.device`` and seeded once with
            ``manual_seed(seed)``; this generator is threaded into every
            ``torch.multinomial`` call inside :func:`_sample_next_token`,
            making the decoded sequence reproducible across calls that
            share the same seed.  ``None`` (default) consumes the
            device's default RNG, preserving the pre-seed call surface
            and behavior.  Has no effect when ``do_sample=False``
            (greedy / argmax doesn't consume any RNG); the generator
            is still allocated in that case but is never read.  Under
            multi-rank execution, callers that need cross-rank agreement
            on the sampled tokens must pass the *same* ``seed`` on
            every rank — the generator is rank-local, so distinct seeds
            (or ``None``) produce per-rank divergent samples even
            though the prompts and logits agree.
        return_only_new_tokens: If ``True``, return only the generated
            suffix ``[B, T_new]``; otherwise return prompt + generated
            tokens ``[B, T_prompt + T_new]``.

    Returns:
        Token ids ``[B, T_prompt + T_new]`` (or ``[B, T_new]`` when
        ``return_only_new_tokens=True``).  ``T_new <= max_new_tokens``;
        early termination only occurs when every sample emits EOS.
    """
    if input_ids.dim() != 2 or input_ids.shape[1] < 1:
        raise ValueError(f"input_ids must have shape [B, T_prompt>=1], got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    eos_ids: list[int] = []
    if isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    elif eos_token_id is not None:
        eos_ids = [int(t) for t in eos_token_id]
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else 0

    B = input_ids.shape[0]
    device = input_ids.device

    if max_new_tokens == 0:
        empty = input_ids.new_zeros((B, 0))
        return empty if return_only_new_tokens else input_ids.clone()

    model = causal_lm.model
    cache = ReasonerKVCache.empty(num_layers=len(model.layers))

    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError("pixel_values and image_grid_thw must be provided together.")

    _prefill_start = time.time()

    mrope_position_deltas: torch.Tensor | None = None
    if pixel_values is None:
        hidden = model.reasoner_forward(input_ids, cache=cache)
    else:
        if not hasattr(causal_lm, "visual"):
            raise ValueError("Combined checkpoint does not include a visual module on the reasoner language model.")
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            causal_lm,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        hidden = model.reasoner_forward(
            input_ids=None,
            cache=cache,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
    logits = causal_lm.lm_head(hidden[:, -1, :])

    apply_penalties = repetition_penalty != 1.0 or presence_penalty != 0.0
    seen_mask: torch.Tensor | None = None
    if apply_penalties:
        seen_mask = torch.zeros(B, logits.size(-1), dtype=torch.bool, device=device)
        seen_mask.scatter_(1, input_ids, True)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    next_token = _sample_next_token(
        logits,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        seen_mask=seen_mask,
        generator=generator,
    )
    if seen_mask is not None:
        seen_mask.scatter_(1, next_token.unsqueeze(1), True)

    pad_tensor = torch.full((B,), pad_token_id, dtype=next_token.dtype, device=device)
    eos_tensor: torch.Tensor | None = None
    if eos_ids:
        eos_tensor = torch.as_tensor(eos_ids, dtype=next_token.dtype, device=device)
    base_mrope_position_ids: torch.Tensor | None = None
    if mrope_position_deltas is not None:
        base_mrope_position_ids = mrope_position_deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    if eos_tensor is not None:
        finished |= (next_token.unsqueeze(-1) == eos_tensor).any(dim=-1)

    torch.cuda.synchronize()
    _prefill_time = time.time() - _prefill_start
    log.info(f"[generate_reasoner_text] prefill time: {_prefill_time:.2f} sec")

    _decode_start = time.time()
    output_buf = torch.empty(B, max_new_tokens, dtype=next_token.dtype, device=device)
    output_buf[:, 0] = next_token
    num_generated = 1

    for _ in range(max_new_tokens - 1):
        if eos_tensor is not None and _all_ranks_finished(finished):
            break
        step_input = next_token.unsqueeze(1)
        position_ids = None
        if base_mrope_position_ids is not None:
            position_ids = base_mrope_position_ids + cache.seq_len
        hidden = model.reasoner_forward(step_input, cache=cache, position_ids=position_ids)
        logits = causal_lm.lm_head(hidden[:, -1, :])
        next_token = _sample_next_token(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seen_mask=seen_mask,
            generator=generator,
        )
        next_token = torch.where(finished, pad_tensor, next_token)
        if seen_mask is not None:
            seen_mask.scatter_(1, next_token.unsqueeze(1), True)
        if eos_tensor is not None:
            finished |= (next_token.unsqueeze(-1) == eos_tensor).any(dim=-1)
        output_buf[:, num_generated] = next_token
        num_generated += 1

    new_tokens_tensor = output_buf[:, :num_generated]

    torch.cuda.synchronize()
    _decode_time = time.time() - _decode_start
    log.info(f"[generate_reasoner_text] decode time: {_decode_time:.2f} sec, number of tokens: {num_generated}")
    log.info(f"[generate_reasoner_text] average decode time per token: {_decode_time * 1e3 / num_generated:.2f} ms")

    if return_only_new_tokens:
        return new_tokens_tensor
    return torch.cat([input_ids, new_tokens_tensor], dim=1)


class Qwen3VLTextForCausalLM(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text causal language model for MoT.
    This variant is used for dense-only MLP models.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoTConfig):

        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Qwen3VLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        vision_config = config.vision_config
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel._from_config(vision_config)

        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters.

        Iterates over every parameter whose name contains ``moe_gen`` (i.e.
        every gen-tower weight) and copies its und-tower counterpart in.
        ``q_norm`` / ``k_norm`` get a special-case skip: when
        ``qk_norm_for_text=False`` the und tower's q_norm/k_norm are
        ``nn.Identity()`` (no parameters), but the gen tower may still have
        real ``q_norm_moe_gen`` / ``k_norm_moe_gen`` RMSNorm modules — those
        keep their default ``ones`` init and we just skip the copy rather
        than raising.
        """
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name or "lora_" in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
        camera_code: FactoredSequencePack | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
        """Training forward pass — delegates to the dense text model."""
        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
            camera_code=camera_code,
        )
        return outputs

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        return_only_new_tokens: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate text tokens using only the reasoner tower.

        Handles both text-only and image-conditioned (I2V) prompts through a
        single entry point.  Pass ``pixel_values`` + ``image_grid_thw`` (and
        optionally ``attention_mask``) to drive image-conditioned prefill via
        the Qwen3-VL visual encoder; omit them for text-only prefill.  The
        two arguments are mutually required: passing exactly one raises
        ``ValueError`` inside :func:`_impl_generate_reasoner_text`.

        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus the model-level ``embed_tokens`` / ``norm`` / ``lm_head``, and —
        for the I2V path — the visual encoder under ``self.visual``.  The
        generation pathway and all VFM-level multimodal embedders / heads
        (``vae2llm``, ``llm2vae``, etc.) are bypassed.  See
        :func:`_impl_generate_reasoner_text` for full argument docs.
        """
        return _impl_generate_reasoner_text(
            self,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            return_only_new_tokens=return_only_new_tokens,
        )


class Qwen3VLMoeTextForCausalLM(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text causal language model for MoT with MoE on the generation pathway.
    This variant is used for MoE MLP models.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoeMoTConfig):
        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Qwen3VLMoeTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        vision_config = config.vision_config
        if vision_config is not None:
            self.visual = Qwen3VLMoeVisionModel._from_config(vision_config)

        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters.

        See :meth:`Qwen3VLTextForCausalLM.init_moe` for the q_norm/k_norm
        Identity-tower handling shared with the dense variant.
        """
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name or "lora_" in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
        camera_code: FactoredSequencePack | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
        """Training forward pass — delegates to the MoE text model."""

        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
            camera_code=camera_code,
        )

        return outputs

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        return_only_new_tokens: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate text tokens using only the reasoner tower.

        Handles both text-only and image-conditioned (I2V) prompts through a
        single entry point.  Pass ``pixel_values`` + ``image_grid_thw`` (and
        optionally ``attention_mask``) to drive image-conditioned prefill via
        the Qwen3-VL visual encoder; omit them for text-only prefill.  The
        two arguments are mutually required: passing exactly one raises
        ``ValueError`` inside :func:`_impl_generate_reasoner_text`.

        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus the model-level ``embed_tokens`` / ``norm`` / ``lm_head``, and —
        for the I2V path — the visual encoder under ``self.visual``.  The
        MoE generation pathway is bypassed entirely; the reasoner MLP can
        itself be a dense MLP or an MoE block — both are handled by
        :func:`_run_mlp` inside the loop.  See
        :func:`_impl_generate_reasoner_text` for full argument docs.
        """
        return _impl_generate_reasoner_text(
            self,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            return_only_new_tokens=return_only_new_tokens,
        )


class Nemotron3DenseVLTextForCausalLM(Nemotron3DenseVLPreTrainedModel):
    """Causal LM head on top of the Nemotron 3 Dense VL MoT text model."""

    _tied_weights_keys: list[str] = []

    def __init__(self, config: Nemotron3DenseVLMoTConfig) -> None:
        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Nemotron3DenseVLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        assert config.vision_config is None, "Nemotron 3 Dense VL has no vision config"

        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name or "lora_" in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            elif any(norm_key in original_name for norm_key in ("q_norm", "k_norm")):
                pass
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
        camera_code: FactoredSequencePack | None = None,
    ) -> tuple[FactoredSequencePack, dict[str, LBLMetadata]]:
        return self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
            camera_code=camera_code,
        )

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        return_only_new_tokens: bool = False,
    ) -> torch.Tensor:

        raise NotImplementedError("This method is not implemented for Nemotron 3 Dense VL.")
