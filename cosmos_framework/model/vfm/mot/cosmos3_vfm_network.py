# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from cosmos_framework.data.vfm.sequence_packing import (
    ModalityData,
    PackedSequence,
    from_joint,
    verify_natten_parameter_list,
)
from cosmos_framework.model.vfm.mot.attention import build_packed_sequence
from cosmos_framework.model.vfm.mot.camera_conditioning import CameraConditioner
from cosmos_framework.model.vfm.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
    get_context_parallel_sharded_sequence,
)
from cosmos_framework.model.vfm.mot.domain_aware_linear import DomainAwareLinear
from cosmos_framework.model.vfm.mot.modeling_utils import (
    FlattenedSinCosPositionEmbedding,
    TimestepEmbedder,
    VideoRopePosition3DEmb,
)
from cosmos_framework.model.vfm.utils.memory import MemoryState


class Cosmos3VFMNetworkConfig(PretrainedConfig):
    def __init__(
        self,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        vlm_config=None,
        latent_patch_size=2,
        latent_downsample_factor=8,
        latent_channel_size=16,
        position_embedding_type="3d_rope",
        max_latent_h=32,
        max_latent_w=32,
        max_latent_t=32,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        enable_fps_modulation=False,
        base_fps=24,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        timestep_scale=0.001,
        predict_text_tokens=False,
        joint_attn_implementation="two_way",
        action_dim=32,
        num_embodiment_domains=32,
        temporal_compression_factor_vision=4,
        temporal_compression_factor_action=1,
        natten_parameter_list=None,
        video_temporal_causal=False,
        camera_conditioning_enabled=False,
        camera_code_dim=512,
        camera_ray_grid="latent16",
        camera_roce_mode="none",
        camera_roce_output_position="post_o",
        sound_dim: int | None = None,
        temporal_compression_factor_sound=1,
        sound_latent_fps: int = 25,
        **kwargs,
    ):
        self.vision_gen = vision_gen
        self.sound_gen = sound_gen
        self.vlm_config = vlm_config
        self.latent_patch_size = latent_patch_size
        self.latent_downsample_factor = latent_downsample_factor
        self.latent_channel_size = latent_channel_size
        self.position_embedding_type = position_embedding_type
        self.max_latent_h = max_latent_h
        self.max_latent_w = max_latent_w
        self.max_latent_t = max_latent_t
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.enable_fps_modulation = enable_fps_modulation
        self.base_fps = base_fps
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.timestep_scale = timestep_scale
        self.predict_text_tokens = predict_text_tokens
        self.joint_attn_implementation = joint_attn_implementation
        self.temporal_compression_factor_vision = temporal_compression_factor_vision
        self.natten_parameter_list = natten_parameter_list
        self.video_temporal_causal = video_temporal_causal
        self.camera_conditioning_enabled = camera_conditioning_enabled
        self.camera_code_dim = camera_code_dim
        self.camera_ray_grid = camera_ray_grid
        self.camera_roce_mode = camera_roce_mode
        self.camera_roce_output_position = camera_roce_output_position
        if self.camera_conditioning_enabled and self.camera_code_dim <= 0:
            raise ValueError(f"camera_code_dim must be positive, got {self.camera_code_dim}.")
        if self.camera_conditioning_enabled and not self.vision_gen:
            raise ValueError("Camera conditioning requires vision_gen=True.")
        if self.camera_ray_grid != "latent16":
            raise ValueError(f"Unsupported camera ray grid: {self.camera_ray_grid!r}")
        if self.camera_roce_mode not in {"none", "qk", "qk_vo"}:
            raise ValueError(f"Unsupported camera RoCE mode: {self.camera_roce_mode!r}")
        if self.camera_roce_output_position != "post_o":
            raise ValueError(
                "Only ReDirector-compatible camera_roce_output_position='post_o' is supported, "
                f"got {self.camera_roce_output_position!r}."
            )
        if self.camera_conditioning_enabled and self.latent_patch_size != 2:
            raise ValueError(
                "The latent16 camera adapter requires latent_patch_size=2 so its learned "
                f"2x2 aggregation matches vision patchify; got {self.latent_patch_size}."
            )
        if self.camera_conditioning_enabled and self.latent_downsample_factor != 16:
            raise ValueError(
                "camera_ray_grid='latent16' requires latent_downsample_factor=16; "
                f"got {self.latent_downsample_factor}."
            )
        if self.camera_conditioning_enabled and self.position_embedding_type != "unified_3d_mrope":
            raise ValueError(
                "Camera RoCE requires position_embedding_type='unified_3d_mrope'; "
                f"got {self.position_embedding_type!r}."
            )
        if self.camera_conditioning_enabled and self.temporal_compression_factor_vision != 4:
            raise ValueError(
                "The 24-channel camera adapter requires temporal_compression_factor_vision=4; "
                f"got {self.temporal_compression_factor_vision}."
            )

        self.action_gen = action_gen
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        self.temporal_compression_factor_action = temporal_compression_factor_action
        if self.action_gen:
            assert self.vision_gen, (
                "Action generation requires visual generation! We do NOT support action only training!"
            )

        self.sound_dim = sound_dim
        self.temporal_compression_factor_sound = temporal_compression_factor_sound
        self.sound_latent_fps = sound_latent_fps
        if self.sound_gen:
            assert self.vision_gen, (
                "Sound generation requires visual generation! We do NOT support sound only training!"
            )

        super().__init__(**kwargs)


class Cosmos3VFMNetwork(PreTrainedModel):
    config_class = Cosmos3VFMNetworkConfig
    base_model_prefix = "cosmos3"

    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(config)
        self.language_model = language_model

        text_config = config.vlm_config.text_config if hasattr(config.vlm_config, "text_config") else config.vlm_config
        self.hidden_size = text_config.hidden_size
        self.num_heads = text_config.num_attention_heads
        self.head_dim = text_config.head_dim
        self.num_hidden_layers = text_config.num_hidden_layers
        self._track_capture_layer: int | None = None
        self._track_feedback_layer: int | None = None
        self.predict_text_tokens = config.predict_text_tokens

        if config.natten_parameter_list is not None and config.joint_attn_implementation != "three_way":
            raise NotImplementedError(
                f"Sparsity is only supported with 'three_way' attention, but got {config.joint_attn_implementation=}, "
                "and 'natten_parameter_list' was not None."
            )
        self.natten_parameter_list = verify_natten_parameter_list(
            config.natten_parameter_list, num_layers=self.num_hidden_layers
        )

        if config.video_temporal_causal and config.joint_attn_implementation != "three_way":
            raise ValueError(
                f"video_temporal_causal=True requires joint_attn_implementation='three_way', "
                f"but got {config.joint_attn_implementation!r}."
            )
        self.video_temporal_causal = config.video_temporal_causal
        self.pad_for_cuda_graphs = False

        self.camera_conditioner: CameraConditioner | None = None
        if config.camera_conditioning_enabled:
            self.camera_conditioner = CameraConditioner(config.camera_code_dim, self.hidden_size)
            for layer in self._track_decoder_layers():
                layer.self_attn.enable_camera_roce(config.camera_code_dim, config.camera_roce_mode)

        if config.vision_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.timestep_scale = config.timestep_scale
            self.latent_downsample = config.latent_downsample_factor * config.latent_patch_size
            self.max_latent_h = config.max_latent_h
            self.max_latent_w = config.max_latent_w
            self.max_latent_t = config.max_latent_t
            self.latent_channel = config.latent_channel_size
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

            assert config.position_embedding_type in ["3d_rope", "flattened_sin_cos", "unified_3d_mrope"]
            if config.position_embedding_type == "3d_rope":
                self.latent_pos_embed = VideoRopePosition3DEmb(
                    head_dim=self.hidden_size,
                    len_h=self.max_latent_h,
                    len_w=self.max_latent_w,
                    len_t=self.max_latent_t,
                    h_extrapolation_ratio=config.rope_h_extrapolation_ratio,
                    w_extrapolation_ratio=config.rope_w_extrapolation_ratio,
                    t_extrapolation_ratio=config.rope_t_extrapolation_ratio,
                    enable_fps_modulation=config.enable_fps_modulation,
                    base_fps=config.base_fps,
                    base_temporal_compression_factor=config.temporal_compression_factor_vision,
                    temporal_compression_factor=config.temporal_compression_factor_vision,
                )
            elif config.position_embedding_type == "flattened_sin_cos":
                self.latent_pos_embed = FlattenedSinCosPositionEmbedding(
                    max_latent_h=self.max_latent_h, max_latent_w=self.max_latent_w, hidden_size=self.hidden_size
                )
            elif config.position_embedding_type == "unified_3d_mrope":
                self.latent_pos_embed = None
            else:
                raise ValueError(f"Unknown position_embedding_type: {config.position_embedding_type!r}")

        if config.action_gen:
            self.action_dim = config.action_dim
            self.num_embodiment_domains = config.num_embodiment_domains
            self.action2llm = DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.llm2action = DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)

            if config.position_embedding_type == "3d_rope":
                self.action_pos_embed = VideoRopePosition3DEmb(
                    head_dim=self.hidden_size,
                    len_h=1,
                    len_w=1,
                    len_t=self.max_latent_t * config.temporal_compression_factor_vision,
                    h_extrapolation_ratio=config.rope_h_extrapolation_ratio,
                    w_extrapolation_ratio=config.rope_w_extrapolation_ratio,
                    t_extrapolation_ratio=config.rope_t_extrapolation_ratio,
                    enable_fps_modulation=config.enable_fps_modulation,
                    base_fps=config.base_fps,
                    base_temporal_compression_factor=config.temporal_compression_factor_vision,
                    temporal_compression_factor=config.temporal_compression_factor_action,
                )
            elif config.position_embedding_type == "unified_3d_mrope":
                self.action_pos_embed = None
            else:
                raise ValueError(f"Unknown position_embedding_type: {config.position_embedding_type!r}")

            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if config.sound_gen:
            self.sound_dim = config.sound_dim
            self.sound2llm = nn.Linear(config.sound_dim, self.hidden_size)
            self.llm2sound = nn.Linear(self.hidden_size, config.sound_dim)
            self.sound_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        self.config = config
        self.parallel_dims = None

    def init_weights(self, buffer_device: torch.device | None):
        if self.config.vision_gen or self.config.action_gen or self.config.sound_gen:
            self.time_embedder._init_weights()

        if self.config.vision_gen:
            std = 1.0 / math.sqrt(self.patch_latent_dim)
            torch.nn.init.trunc_normal_(self.vae2llm.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.vae2llm.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2vae.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2vae.bias)

            if self.latent_pos_embed is not None:
                self.latent_pos_embed._init_weights()

        if self.config.action_gen:
            std = 1.0 / math.sqrt(self.action_dim)
            torch.nn.init.trunc_normal_(self.action2llm.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.action2llm.bias.weight)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2action.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2action.bias.weight)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.action_modality_embed, std=std, a=-3 * std, b=3 * std)

            if self.action_pos_embed is not None:
                self.action_pos_embed._init_weights()

        if self.config.sound_gen:
            std = 1.0 / math.sqrt(self.sound_dim)
            torch.nn.init.trunc_normal_(self.sound2llm.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.sound2llm.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2sound.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2sound.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.sound_modality_embed, std=std, a=-3 * std, b=3 * std)

        self.language_model.init_weights(buffer_device=buffer_device)
        if self.camera_conditioner is not None:
            self.camera_conditioner.reset_parameters()
            for layer in self._track_decoder_layers():
                layer.self_attn.reset_camera_roce_parameters()

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

        Thin pass-through to ``self.language_model.generate_reasoner_text``
        (see ``unified_mot._impl_generate_reasoner_text`` for full argument
        documentation).  Handles both text-only and image-conditioned (I2V)
        prompts through this single entry point: pass
        ``pixel_values`` + ``image_grid_thw`` (and optionally
        ``attention_mask``) for image-conditioned prefill via the Qwen3-VL
        visual encoder, or omit them for text-only prefill.  Uses the
        und-pathway weights (those WITHOUT the ``_moe_gen`` suffix) plus
        ``embed_tokens`` / ``norm`` / ``lm_head``; the generation pathway
        and all VFM-level multimodal embedders / heads (``vae2llm``,
        ``llm2vae``, ``sound2llm``, etc.) are bypassed.

        ``repetition_penalty`` / ``presence_penalty`` are pass-through
        sampling controls applied inside
        :func:`unified_mot._impl_generate_reasoner_text` as logit
        transformations *before* the ``do_sample`` argmax / multinomial
        branch (so they shift the greedy argmax too).  Identity defaults
        (``1.0`` / ``0.0``) keep the un-penalized fast path
        bit-identical.

        ``seed`` is a pass-through sampling-RNG knob: when provided,
        :func:`unified_mot._impl_generate_reasoner_text` allocates a
        device-local ``torch.Generator``, seeds it once with
        ``manual_seed(seed)``, and threads it into every
        ``torch.multinomial`` draw — making the decoded sequence a
        deterministic function of the seed, the prompt, and the
        penalty masks.  ``None`` (default) consumes the device's
        default RNG and is bit-identical to the pre-seed call surface.
        Has no effect under greedy decoding (the argmax branch never
        reads the generator).
        """

        return self.language_model.generate_reasoner_text(
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

    def patchify_and_pack_latents(
        self, tokens_vision: torch.Tensor, token_shapes_vision: List[Tuple[int, int, int]]
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.latent_patch_size
        packed_latent = []
        original_latent_shapes = []

        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)

            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))

            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p

            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (self.latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded

            h_patches = h_padded // p
            w_patches = w_padded // p

            latent = latent.reshape(
                self.latent_channel, t_actual, h_patches, p, w_patches, p
            )
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(
                -1, p * p * self.latent_channel
            )
            packed_latent.append(latent)

        packed_latent = torch.cat(packed_latent, dim=0)
        return packed_latent, original_latent_shapes

    def _track_decoder_layers(self):
        """The list of MoT decoder layers (handles bare text-model vs CausalLM-wrapped)."""
        lm = self.language_model
        return lm.layers if hasattr(lm, "layers") else lm.model.layers

    def enable_track_capture(self, layer_idx: int, per_head: bool = False) -> None:
        """Tap layer ``layer_idx``'s post-RoPE Q/K for the multi-view tracking head (MVTrack4Gen).

        ``per_head=True`` additionally emits the un-pooled per-head Q/K (track_q_heads / track_k_heads)
        needed for the multi-view correspondence loss L_corr (the attention map is per-head)."""
        layers = self._track_decoder_layers()
        assert 0 <= layer_idx < len(layers), f"tracking_layer {layer_idx} out of range [0,{len(layers)})"
        self._track_capture_layer = int(layer_idx)
        self._track_capture_per_head = bool(per_head)
        layers[layer_idx].self_attn._track_capture = True

    def run_tracking_head(self, *args, **kwargs):
        """FSDP2 entry point for the tracking head (registered via ``register_fsdp_forward_method``).

        The head is called OUTSIDE ``forward`` (after the denoise pass, from
        ``_compute_tracking_loss``/``_track_forward``). When the whole generation branch is frozen
        (curriculum stage 2), ``forward``'s outputs carry no grad, so FSDP2 never plants its
        post-backward on them — without this registered entry the root param group would neither
        reduce-scatter the head grads (optimizer no-op: sharded params keep grad=None) nor reshard
        after backward (net stays unsharded -> EMA update shape mismatch vs the sharded net_ema).
        Registering the head call as a forward-equivalent gives it the standard
        unshard/post-backward/reshard lifecycle regardless of whether ``forward`` carried grad."""
        return self.tracking_head(*args, **kwargs)

    def _track_model_owner(self):
        """The module that runs the decoder layer loop (_impl_forward)."""
        lm = self.language_model
        return lm if hasattr(lm, "layers") else lm.model

    @staticmethod
    def _track_full_positions(fpack, seq_idx):
        """Map packed-sequence indices ``seq_idx`` -> rows of the gen (full_only) stream, or identity
        for a joint pack (returns (positions, is_joint))."""
        if "packed_sequence" in fpack:
            return seq_idx, True
        fi = fpack["_full_indices"].long()
        nf = int(fpack["_num_full_tokens"])
        n_total = int(fpack["_num_causal_tokens"]) + nf
        inv = torch.full((n_total,), -1, dtype=torch.long, device=fi.device)
        inv[fi[:nf]] = torch.arange(nf, dtype=torch.long, device=fi.device)
        return inv[seq_idx], False

    @classmethod
    def _gather_track_tokens(cls, fpack, seq_idx):
        """Gather the vision tokens (post-RoPE Q/K or hidden states) from a sequence pack by packed
        index. Returns [N_vis, ...] or None."""
        if "packed_sequence" in fpack:
            return fpack["packed_sequence"][seq_idx]
        pos, _ = cls._track_full_positions(fpack, seq_idx)
        return fpack["full_only_seq"][pos] if bool((pos >= 0).all()) else None

    @staticmethod
    def _scatter_qk_grids(vq, vk, shapes, num_items):
        """[N_vis, C] vision Q/K -> list[B] of (qf, kf) each [V, T, C, h, w] (item-major t,h,w)."""
        C = vq.shape[-1]
        iq, ik, off = [], [], 0
        for (t, h, w) in shapes:
            n = int(t) * int(h) * int(w)
            iq.append(vq[off:off + n].reshape(int(t), int(h), int(w), C))
            ik.append(vk[off:off + n].reshape(int(t), int(h), int(w), C))
            off += n
        num_items = num_items or [len(shapes)]
        out, i0 = [], 0
        for V in num_items:
            qf = torch.stack(iq[i0:i0 + V], 0).permute(0, 1, 4, 2, 3).contiguous()
            kf = torch.stack(ik[i0:i0 + V], 0).permute(0, 1, 4, 2, 3).contiguous()
            out.append((qf, kf))
            i0 += V
        return out

    def enable_track_feedback(self, inject_layer: int) -> None:
        """Inject the tracking head's refined per-cell features (zero-init projected) into the vision
        tokens entering layer ``inject_layer`` (track-feature feedback). Requires enable_track_capture."""
        layers = self._track_decoder_layers()
        assert self._track_capture_layer is not None, "call enable_track_capture before enable_track_feedback"
        assert self._track_capture_layer < inject_layer < len(layers), (
            f"feedback inject layer {inject_layer} must be in ({self._track_capture_layer}, {len(layers)})"
        )
        self._track_feedback_layer = int(inject_layer)
        owner = self._track_model_owner()
        owner._track_feedback_fn = self._apply_track_feedback
        owner._track_feedback_inject_layer = int(inject_layer)

    def _apply_track_feedback(self, pack):
        """Compute track-feature feedback from layer ``_track_capture_layer``'s Q/K and ADD it to the
        vision (gen) tokens of ``pack``. Called inside the text-model layer loop right before the inject
        layer. Returns the (new) pack; a no-op until the zero-init projection learns."""
        if not (hasattr(self, "tracking_head") and hasattr(self, "track_feedback_proj")):
            return pack
        seq_idx = getattr(self, "_track_fb_seq_idx", None)
        shapes = getattr(self, "_track_fb_shapes", None)
        if seq_idx is None or shapes is None:
            return pack
        attn = self._track_decoder_layers()[self._track_capture_layer].self_attn
        cap_q, cap_k = attn._cap_q, attn._cap_k
        if cap_q is None or cap_k is None:
            return pack
        vq = self._gather_track_tokens(cap_q, seq_idx)
        vk = self._gather_track_tokens(cap_k, seq_idx)
        if vq is None or vk is None:
            return pack
        per_sample = self._scatter_qk_grids(vq.mean(1), vk.mean(1), shapes,
                                            getattr(self, "_track_fb_num_items", None))
        intr_all = getattr(self, "_track_fb_intr", None)
        extr_all = getattr(self, "_track_fb_extr", None)
        tcf = int(getattr(self, "_track_fb_tcf", 1) or 1)
        feats = []
        for b, (qf, kf) in enumerate(per_sample):
            intr = extr = None
            if intr_all is not None and extr_all is not None and b < len(intr_all):
                ii, ee = intr_all[b], extr_all[b]
                while isinstance(ii, (list, tuple)): ii = ii[0]
                while isinstance(ee, (list, tuple)): ee = ee[0]
                if torch.is_tensor(ii) and torch.is_tensor(ee) and ii.shape[0] == qf.shape[0]:
                    Tlat, Tpix = qf.shape[1], ee.shape[1]
                    wp = torch.tensor([min(tcf * i, Tpix - 1) for i in range(Tlat)], device=qf.device)
                    intr, extr = ii.to(qf.device).float(), ee.to(qf.device).float()[:, wp]
            f = self.tracking_head.feedback_features(qf, kf, intrinsics=intr, extrinsics=extr)
            feats.append(f.reshape(-1, f.shape[-1]))
        feat = torch.cat(feats, dim=0)
        feat = self.track_feedback_proj(feat.to(self.track_feedback_proj.weight.dtype))
        pos, _ = self._track_full_positions(pack, seq_idx)
        if not bool((pos >= 0).all()):
            return pack
        key = "packed_sequence" if "packed_sequence" in pack else "full_only_seq"
        out = dict(pack)
        out[key] = pack[key].index_add(0, pos, feat.to(pack[key].dtype))
        return out

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        unpatchified_latents = []

        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            if original_latent_shapes is not None:
                t_orig, h_orig, w_orig = original_latent_shapes[i]
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                t_orig, h_orig, w_orig = t_c, h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)

            output_tensor = torch.zeros(
                (self.latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(
                    t_n, h_patches, w_patches, p, p, self.latent_channel
                )
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(
                    self.latent_channel, t_n, h_patches * p, w_patches * p
                )

                latent = latent[:, :, :h_orig, :w_orig]

                output_tensor[:, noisy_frame_indexes] = latent

                start_idx = end_idx

            unpatchified_latents.append(output_tensor.unsqueeze(0))

        return unpatchified_latents

    def pack_action(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, ...]],
        domain_id_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length action tokens into a 1D sequence for transformer input.

        Args:
            tokens_action: List of action tensors, each [T_i, action_dim] (T_i may vary).
            token_shapes_action: List of (T_i,) tuples per sample.
            domain_id_action: List of domain ID tensors, each of shape [1].

        Returns:
            Tuple of (packed_tokens, per_token_domain_id):
                packed_tokens: [total_action_tokens, action_dim]
                per_token_domain_id: [total_action_tokens]
        """
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for tokens, shape, d_id in zip(tokens_action, token_shapes_action, domain_id_action):
            T = shape[0]
            packed.append(tokens[:T])
            domain_ids.append(d_id.expand(T))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def unpack_action(
        self,
        packed_action_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, ...]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack action predictions back into per-sample action tensors.

        Args:
            packed_action_preds: Packed action predictions of shape (total_noisy_tokens, action_dim)
            token_shapes_action: Per-sample token shapes, each (T_i,) tuple.
            noisy_frame_indexes_action: List of tensors, each with shape (Tn_i,), where the values
                are the noisy frame indices for sample i.

        Returns:
            List of per-sample tensors, each of shape (T_i, action_dim), with predictions
            placed at noisy positions. Clean positions are left as zeros.
        """
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros(
                (T, self.action_dim),
                device=packed_action_preds.device,
                dtype=packed_action_preds.dtype,
            )
            t_n = len(noisy_frame_indexes)
            if t_n > 0:
                end_idx = start_idx + t_n
                output[noisy_frame_indexes] = packed_action_preds[start_idx:end_idx]
                start_idx = end_idx
            unpacked.append(output)
        return unpacked

    def pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Pack sound latents into a 1D sequence for transformer input.

        Args:
            tokens_sound: List of sound latent tensors, each [C, T]
            token_shapes_sound: List of (T, 1, 1) tuples per sample

        Returns:
            Packed tensor of shape [total_sound_tokens, C]
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            sound_tokens = sound[:, :T].permute(1, 0)
            packed.append(sound_tokens)
        return torch.cat(packed, dim=0)

    def unpack_sound_latents(
        self,
        packed_sound_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack sound predictions back into per-sample sound latents.

        Args:
            packed_sound_preds: Packed sound predictions of shape (total_noisy_tokens, sound_dim)
            token_shapes_sound: List of (T, 1, 1) tuples per sample
            noisy_frame_indexes_action: List of tensors, each with shape (T_i,), where the values
                are the noisy frame indices. T_i <= max_T.

        Returns:
            List of per-sample tensors, each [C, T], with predictions placed at noisy positions.
            Clean positions are left as zeros.
        """
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros(
                (self.sound_dim, T),
                device=packed_sound_preds.device,
                dtype=packed_sound_preds.dtype,
            )

            t_n = len(noisy_frame_indexes)

            if t_n > 0:
                end_idx = start_idx + t_n
                output[:, noisy_frame_indexes] = packed_sound_preds[
                    start_idx:end_idx
                ].T
                start_idx = end_idx

            unpacked.append(output)
        return unpacked

    def _encode_text(
        self,
        packed_seq: PackedSequence,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens and initialize packed_sequence.

        Args:
            packed_seq: PackedSequence containing text_ids and text_indexes.

        Returns:
            tuple of (packed_sequence, target_dtype) where packed_sequence has text embeddings filled in.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            size=(packed_seq.sequence_length, self.hidden_size)
        )
        packed_sequence[packed_seq.text_indexes] = (
            packed_text_embedding
        )
        return packed_sequence, packed_text_embedding.dtype

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps: Optional[torch.Tensor] = None,
    ) -> tuple[List[Tuple[int, int, int]] | None, torch.Tensor | None]:
        """Project vision tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing vision tokens and metadata.
            packed_sequence: The packed sequence tensor to fill vision embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
            fps: Optional FPS tensor for RoPE modulation.

        Returns:
            Original latent shapes before padding (for unpadding during decode), or None if no vision tokens.
        """
        if packed_seq.vision is None or packed_seq.vision.tokens is None:
            return None, None

        vision = packed_seq.vision
        assert vision.tokens is not None
        assert vision.token_shapes is not None
        assert isinstance(vision.sequence_indexes, torch.Tensor)
        assert isinstance(vision.timesteps, torch.Tensor)
        assert isinstance(vision.mse_loss_indexes, torch.Tensor)

        packed_tokens_vision, original_latent_shapes = self.patchify_and_pack_latents(
            vision.tokens, vision.token_shapes
        )

        packed_tokens_vision = self.vae2llm(packed_tokens_vision)

        camera_code: torch.Tensor | None = None
        if self.camera_conditioner is not None:
            if packed_seq.camera_conditioning is None:
                raise ValueError(
                    "camera_conditioning_enabled=True requires calibrated camera metadata in PackedSequence."
                )
            camera_code, camera_additive = self.camera_conditioner(
                packed_seq.camera_conditioning,
                vision.tokens,
                vision.token_shapes,
            )
            if camera_code.shape[0] != packed_tokens_vision.shape[0]:
                raise RuntimeError(
                    f"Camera code has {camera_code.shape[0]} tokens, but vision has "
                    f"{packed_tokens_vision.shape[0]} tokens."
                )
            packed_tokens_vision = packed_tokens_vision + camera_additive.to(packed_tokens_vision.dtype)

        if self.latent_pos_embed is not None:
            latent_token_pos_emb = self.latent_pos_embed(vision.token_shapes, fps=fps).to(
                target_dtype
            )
            packed_tokens_vision = packed_tokens_vision + latent_token_pos_emb

        has_noisy_vision = vision.mse_loss_indexes.numel() > 0

        if has_noisy_vision:
            timesteps_vision = vision.timesteps * self.timestep_scale

            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_vision = self.time_embedder(
                    timesteps_vision
                )
            packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(
                target_dtype
            )

            packed_tokens_vision = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_vision,
                packed_timestep_embeds=packed_timestep_embeds_vision,
                noisy_frame_indexes=vision.noisy_frame_indexes,
                token_shapes=vision.token_shapes,
            )

        packed_sequence[vision.sequence_indexes] = (
            packed_tokens_vision
        )
        return original_latent_shapes, camera_code

    def _decode_vision(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Decode vision tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing mse_loss_indexes_vision and token_shapes_vision.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with mse_preds (modified in-place).
            original_latent_shapes: Original latent shapes before padding (for unpadding).
        """
        vision = packed_seq.vision
        has_noisy_vision = (
            vision is not None
            and vision.tokens is not None
            and isinstance(vision.mse_loss_indexes, torch.Tensor)
            and vision.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_vision:
            preds_vision = torch.zeros(
                [1, self.patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            preds_vision = self.vae2llm(preds_vision)
            preds_vision = self.llm2vae(preds_vision)
            if vision is not None and vision.tokens is not None:
                preds_vision_list = [torch.zeros_like(tok) for tok in vision.tokens]
                preds_vision_list[0] = preds_vision_list[0] + 0.0 * preds_vision.sum()
            else:
                preds_vision_list = [preds_vision]
            output_dict.update(preds_vision=preds_vision_list)
        else:
            assert vision is not None
            assert isinstance(vision.mse_loss_indexes, torch.Tensor)
            assert vision.noisy_frame_indexes is not None
            preds_vision = self.llm2vae(
                last_hidden_state[vision.mse_loss_indexes]
            )
            preds_vision = self.unpatchify_and_unpack_latents(
                preds_vision,
                token_shapes_vision=vision.token_shapes,
                noisy_frame_indexes_vision=vision.noisy_frame_indexes,
                original_latent_shapes=original_latent_shapes,
            )
            output_dict.update(preds_vision=preds_vision)

    def _encode_action(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps_action: Optional[torch.Tensor] = None,
    ) -> None:
        """Encode action tokens and fill into packed_sequence."""
        if packed_seq.action is None or packed_seq.action.tokens is None:
            return

        action: ModalityData = packed_seq.action
        assert action.token_shapes is not None
        assert isinstance(action.sequence_indexes, torch.Tensor)
        assert isinstance(action.timesteps, torch.Tensor)
        assert isinstance(action.mse_loss_indexes, torch.Tensor)

        packed_tokens_action, per_token_domain_id = self.pack_action(
            action.tokens, action.token_shapes, action.domain_id
        )
        packed_tokens_action = self.action2llm(packed_tokens_action, per_token_domain_id)

        if self.action_pos_embed is not None:
            action_shapes_3d = [(ts[0], 1, 1) for ts in action.token_shapes]
            action_token_pos_emb = self.action_pos_embed(
                action_shapes_3d,
                fps=fps_action,
                start_frame_offset=1,
            ).to(target_dtype)
            packed_tokens_action = packed_tokens_action + action_token_pos_emb

        packed_tokens_action = packed_tokens_action + self.action_modality_embed.view(
            1, -1
        )

        has_noisy_actions = action.mse_loss_indexes.numel() > 0
        if has_noisy_actions:
            timesteps_action = action.timesteps * self.timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_action = self.time_embedder(
                    timesteps_action
                )
            packed_timestep_embeds_action = packed_timestep_embeds_action.to(
                target_dtype
            )

            packed_tokens_action = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_action,
                packed_timestep_embeds=packed_timestep_embeds_action,
                noisy_frame_indexes=action.noisy_frame_indexes,
                token_shapes=action.token_shapes,
            )

        packed_sequence[action.sequence_indexes] = (
            packed_tokens_action
        )

    def _decode_action(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode action tokens from hidden states and update output_dict."""
        action = packed_seq.action
        has_noisy_action = (
            action is not None
            and action.tokens is not None
            and isinstance(action.mse_loss_indexes, torch.Tensor)
            and action.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_action:
            preds_action = torch.zeros(
                [1, self.action_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            dummy_domain_id = torch.zeros([1], device=last_hidden_state.device, dtype=torch.long)
            preds_action = self.action2llm(preds_action, dummy_domain_id) + self.action_modality_embed.view(
                1, -1
            )
            preds_action = self.llm2action(preds_action, dummy_domain_id)
            if action is not None and action.tokens is not None:
                preds_action_list = [torch.zeros_like(tok) for tok in action.tokens]
                preds_action_list[0] = preds_action_list[0] + 0.0 * preds_action.sum()
            else:
                preds_action_list = [preds_action]
            output_dict.update(preds_action=preds_action_list)
        else:
            assert action is not None
            assert isinstance(action.mse_loss_indexes, torch.Tensor)
            assert action.condition_mask is not None
            assert len(action.domain_id) > 0

            action_hidden_states = last_hidden_state[action.mse_loss_indexes]

            domain_ids: list[torch.Tensor] = []
            for nfi, d_id in zip(action.noisy_frame_indexes, action.domain_id):
                domain_ids.append(d_id.expand(len(nfi)))
            per_token_domain_id = torch.cat(domain_ids, dim=0)

            preds_action = self.llm2action(
                action_hidden_states, per_token_domain_id
            )
            preds_action = self.unpack_action(preds_action, action.token_shapes, action.noisy_frame_indexes)
            output_dict.update(preds_action=preds_action)

    def _encode_sound(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
        fps_sound: Optional[torch.Tensor] = None,
    ) -> None:
        """Encode sound tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing sound tokens and metadata.
            packed_sequence: The packed sequence tensor to fill sound embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
            fps_sound: FPS tensor for RoPE modulation. Should be the sound latent rate (e.g., 25 Hz).
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        packed_tokens_sound = self.pack_sound_latents(
            sound.tokens, sound.token_shapes
        )
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)


        packed_tokens_sound = (
            self.sound2llm(packed_tokens_sound) + self.sound_modality_embed
        )

        has_noisy_sound = sound.mse_loss_indexes.numel() > 0
        if has_noisy_sound:
            timesteps_sound = sound.timesteps * self.timestep_scale
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                packed_timestep_embeds_sound = self.time_embedder(timesteps_sound)
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(
                target_dtype
            )

            packed_tokens_sound = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )

        packed_sequence[sound.sequence_indexes] = (
            packed_tokens_sound
        )

    def _decode_sound(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode sound tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing sound modality data.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with preds_sound (modified in-place).
        """
        sound = packed_seq.sound
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_sound:
            preds_sound = torch.zeros(
                [1, self.sound_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )
            preds_sound = self.sound2llm(preds_sound) + self.sound_modality_embed
            preds_sound = self.llm2sound(preds_sound)
            if sound is not None and sound.tokens is not None:
                preds_sound_list = [torch.zeros_like(tok) for tok in sound.tokens]
                preds_sound_list[0] = preds_sound_list[0] + 0.0 * preds_sound.sum()
            else:
                preds_sound_list = [preds_sound]
            output_dict.update(preds_sound=preds_sound_list)
        else:
            assert sound is not None
            assert isinstance(sound.mse_loss_indexes, torch.Tensor)
            assert sound.condition_mask is not None
            preds_sound = self.llm2sound(
                last_hidden_state[sound.mse_loss_indexes]
            )
            preds_sound = self.unpack_sound_latents(
                preds_sound, sound.token_shapes, sound.noisy_frame_indexes
            )
            output_dict.update(preds_sound=preds_sound)

    def forward(
        self,
        packed_seq: PackedSequence,
        fps_vision: Optional[torch.Tensor] = None,
        fps_action: Optional[torch.Tensor] = None,
        fps_sound: Optional[torch.Tensor] = None,
        memory: MemoryState | None = None,
    ) -> dict:
        """
        Forward pass for Cosmos3VFMNetwork.

        Args:
            packed_seq: PackedSequence containing all packed tensors and metadata.
                See PackedSequence dataclass for field details.
            fps_vision: Optional FPS tensor for vision RoPE modulation.
            fps_action: Optional FPS tensor for action RoPE modulation.
            fps_sound: Optional FPS tensor for sound RoPE modulation (e.g., sound_latent_fps=25).
            memory: Optional MemoryState for persistent KV-cache memory
                (AR inference or rolling-KV-cache training).  Built by
                ``OmniMoTModel.build_memory_state()``.

        Returns:
            dict with keys:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity predictions for action tokens (if action_gen).
                - "preds_sound": Velocity predictions for sound tokens (if sound_gen).
                - "last_hidden_state": Last hidden state from the transformer.
                - "lbl_metadata_*": Load balancing metadata.
                - "ce_preds": Cross-entropy predictions (if predict_text_tokens is True).
        """

        packed_sequence, target_dtype = self._encode_text(packed_seq)

        original_latent_shapes: List[Tuple[int, int, int]] | None = None
        camera_code_vision: torch.Tensor | None = None
        if self.config.vision_gen:
            original_latent_shapes, camera_code_vision = self._encode_vision(
                packed_seq, packed_sequence, target_dtype, fps_vision
            )

        if self.config.action_gen:
            self._encode_action(packed_seq, packed_sequence, target_dtype, fps_action)

        if self.config.sound_gen:
            self._encode_sound(packed_seq, packed_sequence, target_dtype, fps_sound)

        assert packed_seq.attn_modes is not None
        assert packed_seq.split_lens is not None

        all_gen_indexes = []
        if packed_seq.vision is not None:
            assert packed_seq.vision.token_shapes is not None
            assert isinstance(packed_seq.vision.sequence_indexes, torch.Tensor)
            all_gen_indexes.append(packed_seq.vision.sequence_indexes)
        if packed_seq.action is not None and isinstance(packed_seq.action.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.action.sequence_indexes)
        if packed_seq.sound is not None and isinstance(packed_seq.sound.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.sound.sequence_indexes)
        vision_sequence_indexes = torch.cat(all_gen_indexes, dim=0) if all_gen_indexes else None

        if self.video_temporal_causal:
            assert packed_seq.sound is None, "Sound generation is not supported with video_temporal_causal=True."
            if vision_sequence_indexes is not None:
                vision_sequence_indexes = vision_sequence_indexes.sort().values

        vision_token_shapes = packed_seq.vision.token_shapes if packed_seq.vision else None

        num_action_tokens_per_supertoken = packed_seq.num_action_tokens_per_supertoken

        input_pack, attention_meta, natten_metadata_list = build_packed_sequence(
            self.config.joint_attn_implementation,
            packed_sequence=packed_sequence,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
            packed_und_token_indexes=packed_seq.text_indexes,
            packed_gen_token_indexes=vision_sequence_indexes,
            num_heads=self.num_heads,
            is_image_batch=packed_seq.is_image_batch,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            token_shapes=packed_seq.vision.token_shapes,
            natten_parameter_list=self.natten_parameter_list,
            cp_world_size=self.parallel_dims.cp_size if self.parallel_dims else 1,
            video_temporal_causal=self.video_temporal_causal,
            use_rolling_kv_cache=memory is not None and memory.uses_rolling_kv_cache,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=packed_seq.action.token_shapes if packed_seq.action else None,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            pad_for_cuda_graphs=self.pad_for_cuda_graphs,
        )

        camera_code_pack = None
        if camera_code_vision is not None:
            cp_size = self.parallel_dims.cp_size if self.parallel_dims else 1
            if cp_size != 1:
                raise NotImplementedError(
                    "Camera conditioning currently requires context_parallel_shard_degree=1."
                )
            assert packed_seq.vision is not None
            assert isinstance(packed_seq.vision.sequence_indexes, torch.Tensor)
            camera_code_full = camera_code_vision.new_zeros(
                packed_seq.sequence_length, camera_code_vision.shape[-1]
            )
            camera_code_full[packed_seq.vision.sequence_indexes] = camera_code_vision
            camera_code_pack = from_joint(camera_code_full, input_pack)

        input_pack, packed_position_ids = get_context_parallel_sharded_sequence(
            attn_implementation=self.config.joint_attn_implementation,
            input_pack=input_pack,
            position_ids=packed_seq.position_ids,
            parallel_dims=self.parallel_dims,
        )

        if self._track_feedback_layer is not None and packed_seq.vision is not None \
                and (self.parallel_dims.cp_size if self.parallel_dims else 1) == 1 \
                and isinstance(packed_seq.vision.sequence_indexes, torch.Tensor):
            self._track_fb_seq_idx = packed_seq.vision.sequence_indexes.long()
            self._track_fb_shapes = list(packed_seq.vision.token_shapes)
        else:
            self._track_fb_seq_idx = None
            self._track_fb_shapes = None

        packed_outputs, lbl_metadata = self.language_model(
            input_pack,
            attention_mask=attention_meta,
            position_ids=packed_position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
            camera_code=camera_code_pack,
        )
        last_hidden_state = get_context_parallel_last_hidden_state(
            packed_outputs=packed_outputs,
            parallel_dims=self.parallel_dims,
        )
        output_dict = dict()

        if self._track_capture_layer is not None and packed_seq.vision is not None:
            cp_size = self.parallel_dims.cp_size if self.parallel_dims else 1
            attn = self._track_decoder_layers()[self._track_capture_layer].self_attn
            cap_q, cap_k = attn._cap_q, attn._cap_k
            attn._cap_q = attn._cap_k = None
            if cp_size == 1 and cap_q is not None and isinstance(packed_seq.vision.sequence_indexes, torch.Tensor):
                def _gather_vision(fpack, seq_idx):
                    if "packed_sequence" in fpack:
                        return fpack["packed_sequence"][seq_idx]
                    fi = fpack["_full_indices"].long()
                    nf = int(fpack["_num_full_tokens"])
                    n_total = int(fpack["_num_causal_tokens"]) + nf
                    inv = torch.full((n_total,), -1, dtype=torch.long, device=fi.device)
                    inv[fi[:nf]] = torch.arange(nf, dtype=torch.long, device=fi.device)
                    gpos = inv[seq_idx]
                    return fpack["full_only_seq"][gpos] if bool((gpos >= 0).all()) else None

                seq_idx = packed_seq.vision.sequence_indexes.long()
                vq = _gather_vision(cap_q, seq_idx)
                vk = _gather_vision(cap_k, seq_idx)
                if vq is not None and vk is not None:
                    output_dict["track_q"] = vq.mean(dim=1)
                    output_dict["track_k"] = vk.mean(dim=1)
                    output_dict["track_token_shapes"] = list(packed_seq.vision.token_shapes)
                    if getattr(self, "_track_capture_per_head", False):
                        output_dict["track_q_heads"] = vq
                        output_dict["track_k_heads"] = vk

        if self.config.vision_gen:
            self._decode_vision(packed_seq, last_hidden_state, output_dict, original_latent_shapes)

        if self.config.action_gen:
            self._decode_action(packed_seq, last_hidden_state, output_dict)

        if self.config.sound_gen:
            self._decode_sound(packed_seq, last_hidden_state, output_dict)

        output_dict.update(last_hidden_state=last_hidden_state)
        for lbl_metadata_key, lbl_metadata_value in lbl_metadata.items():
            output_dict.update({f"lbl_metadata_{lbl_metadata_key}": lbl_metadata_value})
        if self.predict_text_tokens:
            packed_ce_preds = self.language_model.lm_head(
                last_hidden_state[packed_seq.ce_loss_indexes]
            )
            output_dict["ce_preds"] = packed_ce_preds

        return output_dict


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Apply timestep embeddings to noisy tokens.
    Tn is the number of noisy frames for a given sample.
    Tc is the number of clean frames for a given sample.
    T is the total number of frames for a given sample.
    T = Tn + Tc

    Args:
        packed_tokens: The packed tokens to apply timestep embeddings to.
        packed_timestep_embeds: The packed timestep embeddings to apply.
        noisy_frame_indexes: The frame indices to apply timestep embeddings to
            (list of tensors, each with shape (Tn,)).
        token_shapes: The token shapes for each sample. Each entry is a tuple
            shaped like ``(T, ...)`` where trailing dimensions represent the spatial grid.

    Returns:
        The packed tokens with timestep embeddings applied to the noisy tokens.
    """

    start_noisy_index = 0
    flattened_noisy_frame_indexes = []

    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
        assert noisy_indexes_i.numel() <= token_shape_i[0]
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
        noisy_indexes_i = (
            (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
        )
        noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())
        start_noisy_index += math.prod(token_shape_i)

    flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)

    assert packed_tokens.dim() == 2
    assert packed_timestep_embeds.dim() == 2
    assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
    assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
    assert flattened_noisy_frame_indexes.dim() == 1
    assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
        -1,
        packed_tokens.shape[1],
    )

    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )
