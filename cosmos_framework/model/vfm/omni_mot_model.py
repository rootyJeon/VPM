# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import collections
import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.utils.flags import DEVICE, TRAINING, Device
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.count_params import count_params
from cosmos_framework.utils.timer import Timer
from cosmos_framework.model.vfm.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.vfm.algorithm.loss.load_balancing import compute_load_balancing_loss
from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig
from cosmos_framework.data.vfm.sequence_packing import (
    PackedSequence,
    SequencePlan,
    add_special_tokens,
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.vfm.diffusion.rectified_flow import RectifiedFlow
from cosmos_framework.model.vfm.diffusion.samplers.edm import EDMSampler
from cosmos_framework.model.vfm.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.vfm.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig
from cosmos_framework.model.vfm.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.vfm.tracking import MultiViewTrackingHead
from cosmos_framework.model.vfm.tracking.losses import multiview_track_loss
from cosmos_framework.model.vfm.mot.modeling_utils import has_noisy_tokens
from cosmos_framework.model.vfm.mot.parallelize_vfm_network import parallelize_vfm_network
from cosmos_framework.model.vfm.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
    build_dense_sound_schedule,
    unwrap_and_densify,
)
from cosmos_framework.model.vfm.utils.memory import MemoryState
from cosmos_framework.model.vfm.utils.safetensors_loader import load_language_model as load_language_model_safetensors
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.vfm.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.utils.vfm.data_utils import get_vision_data_resolution
from cosmos_framework.utils.vfm.dtensor_helper import DTensorFastEmaModelUpdater
from cosmos_framework.utils.vfm.model_weights_stats import WeightTrainingStat
from cosmos_framework.utils.vfm.parallelism import ParallelDims


class OmniMoTModel(ImaginaireModel):
    """
    Mixture of Transformers (MoT) model to be trained with the flow matching objective
    for visual / sound / action generation.
    """

    def __init__(self, config: OmniMoTModelConfig):
        super().__init__()
        self.config = config
        log.info(f"OmniMoTModel: config {self.config}")
        self.set_precision()

        self.set_up_data_key()

        self.set_up_tokenizers()

        self.set_up_parallelism()

        self.set_up_model()

        self.set_up_scheduler_and_sampler()

        self.log_enc_time_every_n = config.log_enc_time_every_n

    def set_precision(self) -> None:
        self.precision = getattr(torch, self.config.precision)
        self.tensor_kwargs = {"device": DEVICE, "dtype": self.precision}
        self.tensor_kwargs_fp32 = {"device": DEVICE, "dtype": torch.float32}
        log.warning(f"OmniMoTModel: precision {self.precision}")

        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False

    def set_up_data_key(self) -> None:

        self.input_video_key = self.config.input_video_key
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    @misc.timer("OmniMoTModel: set_up_tokenizers")
    def set_up_tokenizers(self) -> None:
        """
        Variable names follow the naming convention:
        - tokenizer_<modality_type>_gen if used for generation branch
        - tokenizer_<modality_type>_und if used for understanding branch
        """
        self.vlm_config = self.config.vlm_config


        self.vlm_processor: Any = lazy_instantiate(self.vlm_config.tokenizer)

        vlm_tokenizer = self.vlm_processor.tokenizer
        vlm_tokenizer, special_tokens = add_special_tokens(vlm_tokenizer)
        self.vlm_tokenizer = vlm_tokenizer


        self.llm_special_tokens = special_tokens
        self.llm_special_tokens["eos_token_id"] = vlm_tokenizer.eos_token_id

        self.tokenizer_vision_gen: VideoTokenizerInterface = lazy_instantiate(self.config.tokenizer)
        assert self.tokenizer_vision_gen.latent_ch == self.config.state_ch, (
            f"vision tokenizer latent_ch {self.tokenizer_vision_gen.latent_ch} != state_shape {self.config.state_ch}"
        )
        if hasattr(self.tokenizer_vision_gen, "reset_dtype"):
            self.tokenizer_vision_gen.reset_dtype()

        if self.config.sound_gen:
            assert self.config.sound_tokenizer is not None, "sound_tokenizer must be provided when sound_gen is True"
            self.tokenizer_sound_gen = lazy_instantiate(self.config.sound_tokenizer)
            assert self.config.sound_dim is not None, "sound_dim must be provided when sound_gen is True"
            assert self.tokenizer_sound_gen.latent_ch == self.config.sound_dim, (
                f"sound tokenizer latent_ch {self.tokenizer_sound_gen.latent_ch} != sound_dim {self.config.sound_dim}"
            )
            if hasattr(self.tokenizer_sound_gen, "reset_dtype"):
                self.tokenizer_sound_gen.reset_dtype()
            log.info(f"Sound tokenizer initialized: {type(self.tokenizer_sound_gen).__name__}")
        else:
            self.tokenizer_sound_gen = None



    def build_net(self, dtype: torch.dtype):
        with torch.device("meta"):
            assert self.vlm_config.model_instance is not None, "Model instance should be specified"

            language_model = lazy_instantiate(self.vlm_config.model_instance)


            num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
            network_config = Cosmos3VFMNetworkConfig(
                vlm_config=language_model.config,
                latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
                latent_downsample_factor=self.config.latent_downsample_factor,
                latent_channel_size=self.config.state_ch,
                max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
                max_latent_t=self.config.state_t,
                rope_h_extrapolation_ratio=self.config.diffusion_expert_config.rope_h_extrapolation_ratio,
                rope_w_extrapolation_ratio=self.config.diffusion_expert_config.rope_w_extrapolation_ratio,
                rope_t_extrapolation_ratio=self.config.diffusion_expert_config.rope_t_extrapolation_ratio,
                enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=self.config.diffusion_expert_config.base_fps,
                vision_gen=self.config.vision_gen,
                action_gen=self.config.action_gen,
                sound_gen=self.config.sound_gen,
                position_embedding_type=self.config.diffusion_expert_config.position_embedding_type,
                joint_attn_implementation=self.config.joint_attn_implementation,
                timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
                action_dim=self.config.max_action_dim,
                num_embodiment_domains=self.config.num_embodiment_domains,
                temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
                natten_parameter_list=self.config.natten_parameter_list,
                video_temporal_causal=self.config.video_temporal_causal,
                camera_conditioning_enabled=self.config.camera_conditioning_enabled,
                camera_code_dim=self.config.camera_code_dim,
                camera_ray_grid=self.config.camera_ray_grid,
                camera_roce_mode=self.config.camera_roce_mode,
                camera_roce_output_position=self.config.camera_roce_output_position,
                sound_dim=self.config.sound_dim,
                sound_latent_fps=self.config.sound_latent_fps,
            )
            network_config._attn_implementation_internal = "eager"
            net = Cosmos3VFMNetwork(
                language_model=language_model,
                config=network_config,
            )
            net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

            _rf = self.config.rectified_flow_training_config
            if getattr(_rf, "tracking_enabled", False):
                net.tracking_head = MultiViewTrackingHead(
                    in_dim=net.head_dim,
                    latent_dim=_rf.tracking_latent_dim,
                    corr_radius=_rf.tracking_corr_radius,
                    corr_levels=_rf.tracking_corr_levels,
                    num_virtual_tracks=_rf.tracking_num_virtual_tracks,
                    iters=_rf.tracking_iters,
                    view_att=True,
                    use_checkpoint=_rf.tracking_use_checkpoint,
                    use_camera=bool(getattr(_rf, "tracking_camera_enabled", False)),
                    upsample=int(getattr(_rf, "tracking_upsample", 1)),
                    predict_depth=bool(getattr(_rf, "tracking_depth_enabled", False)),
                )
                net.enable_track_capture(
                    _rf.tracking_layer,
                    per_head=float(getattr(_rf, "tracking_corr_weight", 0.0)) > 0,
                )
                if getattr(_rf, "tracking_feedback_enabled", False):
                    net.track_feedback_proj = torch.nn.Linear(
                        net.tracking_head.feature_dim, net.hidden_size
                    )
                    net.enable_track_feedback(_rf.tracking_feedback_layer)

            if getattr(self.config, "lora_enabled", False):
                net = self.add_lora(
                    net,
                    lora_rank=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_target_modules=self.config.lora_target_modules,
                )

        self.install_attention_dispatch(net)

        net = parallelize_vfm_network(
            net,
            parallel_dims=self.parallel_dims,
            compile_config=self.config.compile,
            ac_config=self.config.activation_checkpointing,
        )

        with misc.timer("meta to cuda and broadcast model states"):
            net = net.to(dtype=dtype)
            net.to_empty(device=DEVICE)
            if DEVICE == Device.CUDA:
                net.init_weights(buffer_device=DEVICE)
                if getattr(self.config, "lora_enabled", False):
                    self._init_lora_weights_post_materialization(net)
                if getattr(self.config.rectified_flow_training_config, "tracking_enabled", False) \
                        and hasattr(net, "tracking_head"):
                    net.tracking_head.reset_parameters()
                    if hasattr(net, "track_feedback_proj"):
                        torch.nn.init.zeros_(net.track_feedback_proj.weight)
                        torch.nn.init.zeros_(net.track_feedback_proj.bias)

        return net

    def load_pretrained_model_if_needed(
        self,
        *,
        has_resumable_checkpoint: bool,
        has_load_path: bool,
    ) -> None:
        """Load HF understanding-pathway weights, gated by runtime checkpoint state.

        Args:
            has_resumable_checkpoint: A latest_checkpoint.txt exists in the load
                directory; DCP has already populated the full model. Skip HF load entirely.
            has_load_path: ``checkpoint.load_path`` is set; DCP has loaded the full
                model from the warm-start path. Reload HF understanding-pathway weights
                (e.g. swap Qwen3-VL → Cosmos-Reason) but skip the understanding→generation
                copy since the generation pathway is already populated by load_path.

        Three cases the gates produce:
          1. Fresh init (neither): full HF load + understanding→generation copy.
          2. Warm-start (load_path only): reload understanding, skip the copy.
          3. Resume (resumable checkpoint): skip everything.
        """
        pretrained_weights = self.vlm_config.pretrained_weights
        if not pretrained_weights.enabled:
            return
        if has_resumable_checkpoint:
            log.info("Resumable checkpoint exists; skipping HF understanding-pathway load.")
            return

        def _load_language_model(net: torch.nn.Module):
            load_language_model_safetensors(
                model=net.language_model,
                checkpoint_path=pretrained_weights.backbone_path,
                credential_path=pretrained_weights.credentials_path,
                parallel_dims=self.parallel_dims,
                checkpoint_format=pretrained_weights.checkpoint_format,
            )

        log.info(f"Loading understanding pathway weights from {pretrained_weights.backbone_path}")
        _load_language_model(self.net)
        if self.config.ema.enabled:
            _load_language_model(self.net_ema)
        log.info("Successfully loaded understanding pathway weights.")

        if not self.config.diffusion_expert_config.load_weights_from_pretrained:
            return
        if has_load_path:
            log.info("Warm-start load_path active; skipping understanding→generation copy.")
            return

        log.info("Copying understanding pathway weights to generation pathway.")
        self.net.language_model.init_moe()
        if self.config.ema.enabled:
            self.net_ema.language_model.init_moe()
        log.info("Successfully copied understanding pathway weights to generation pathway.")

    @misc.timer("OmniMoTModel: set_up_model")
    def set_up_model(self):
        assert hasattr(self, "parallel_dims"), "parallel_dims must be set"
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.net = self.build_net(dtype=self.precision)
            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = self.build_net(dtype=torch.float32)
                self.net_ema.requires_grad_(False)

                self.net_ema_worker = DTensorFastEmaModelUpdater()


                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        self.set_up_memory()

        self._apply_tracking_freeze()

        if getattr(self.config.rectified_flow_training_config, "tracking_enabled", False) \
                and torch.cuda.is_available():
            prev = torch.backends.cuda.preferred_linalg_library()
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
                torch.linalg.solve(torch.eye(2, device="cuda"), torch.ones(2, device="cuda"))
                log.info("[mvtrack] set torch linalg backend -> magma (avoids cuSOLVER handle bug in UniPC)")
            except Exception as e:
                torch.backends.cuda.preferred_linalg_library(prev)
                log.info(f"[mvtrack] MAGMA unavailable ({type(e).__name__}); linalg backend left at "
                         f"{prev} — UniPC solves run on CPU, so cuSOLVER is not involved")

        torch.cuda.empty_cache()

    def _track_head_call(self):
        """The tracking-head entry point for TRAINING-time calls, chosen by freeze mode (cached).

        Head-ONLY training (every non-head/track_feedback param frozen — curriculum stage 2):
        the FSDP2-registered ``net.run_tracking_head``. The denoise forward's outputs carry no
        grad then, so no root post-backward is ever planted on them — the registered entry is
        the only thing that makes the root FSDP group reduce-scatter the head grads (otherwise
        the optimizer-held sharded params keep grad=None) and reshard after backward. Collective
        order stays uniform because the head calls are the ONLY grad-carrying root hooks.

        Backbone (even partially) trainable (stage 3, mvwm/droid recipes): the plain submodule
        call. net.forward's outputs carry grad, so the root post-backward planted on them fires
        AFTER every block's reduce — one deterministic collective order on all ranks, and it
        reduces the head grads too (head params live in the root group). The registered entry
        must NOT be used here: each call would plant an extra root post-backward inside the
        head-loss branch of autograd, whose position among the block reduces depends on the
        per-rank pack shape (#samples, #views) — ranks then emit reduces in different orders on
        the shared default PG and NCCL deadlocks (stage-3 step-1 hang, 2026-07-14)."""
        fn = getattr(self, "_track_head_call_fn", None)
        if fn is None:
            head_only = not any(
                p.requires_grad
                for n, p in self.net.named_parameters()
                if "tracking_head" not in n and "track_feedback" not in n
            )
            fn = self.net.run_tracking_head if head_only else self.net.tracking_head
            self._track_head_call_fn = fn
            log.info("[mvtrack] tracking-head call mode: "
                     + ("FSDP-registered (head-only training)" if head_only else "direct (backbone trainable)"))
        return fn

    def _apply_tracking_freeze(self) -> None:
        """Freeze ``net`` except params matching ``tracking_trainable_substrings`` (+ the head)."""
        rf = self.config.rectified_flow_training_config
        if not getattr(rf, "tracking_enabled", False) or not getattr(rf, "tracking_freeze_backbone", False):
            return
        subs = list(getattr(rf, "tracking_trainable_substrings", ["self_attn"]))
        n_train, n_total = 0, 0
        for name, p in self.net.named_parameters():
            keep = ("tracking_head" in name) or ("track_feedback" in name) or any(s in name for s in subs)
            p.requires_grad_(keep)
            n_total += p.numel()
            if keep:
                n_train += p.numel()
        log.info(
            f"[mvtrack] freeze backbone: trainable {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M net "
            f"params (substrings={subs} + tracking_head)"
        )

    @staticmethod
    def _unwrap_sample(x, b):
        """Pull sample ``b`` out of a (possibly nested-list collated) data_batch field."""
        if x is None:
            return None
        e = x[b]
        while isinstance(e, (list, tuple)):
            e = e[0]
        return e

    def _track_camera(self, data_batch: dict, b: int, wp: torch.Tensor, V: int):
        """Per-sample camera for Plücker conditioning: intrinsics [V,3,3] (image-normalized, time-const)
        + extrinsics remapped from pixel frames to the LATENT frames ``wp`` ([V,Tlat,4,4]). Returns
        (intr, extr) or (None, None) when camera conditioning is off / data absent / view-count mismatch."""
        if not getattr(self.config.rectified_flow_training_config, "tracking_camera_enabled", False):
            return None, None
        intr = self._unwrap_sample(data_batch.get("track_intrinsics"), b)
        extr = self._unwrap_sample(data_batch.get("track_extrinsics"), b)
        bad = (intr is None or extr is None or intr.dim() != 3 or extr.dim() != 4
               or intr.shape[0] != V or extr.shape[0] != V or extr.shape[1] <= int(wp.max()))
        if bad:
            if (intr is not None or extr is not None) and not getattr(self, "_track_cam_warned", False):
                self._track_cam_warned = True
                log.warning(
                    "[mvtrack] camera conditioning DISABLED for this run: track_intrinsics/extrinsics "
                    f"shape contract violated (intr={None if intr is None else tuple(intr.shape)}, "
                    f"extr={None if extr is None else tuple(extr.shape)}, V={V}, wp_max={int(wp.max())}). "
                    "Check custom_collate_fn list_collate_keys."
                )
            return None, None
        dev = wp.device
        return intr.to(dev).float(), extr.to(dev).float()[:, wp]

    def _scatter_vision_qk(self, out_net: dict, num_items):
        """Split the captured vision Q/K stream (item-major (t,h,w) row-major) into per-SAMPLE lists
        of per-view feature maps. Returns list[B] of (qf, kf) each [V, T_lat, C, H, W], or None."""
        if "track_q" not in out_net:
            return None
        vq, vk, shapes = out_net["track_q"], out_net["track_k"], out_net["track_token_shapes"]
        C = vq.shape[-1]
        items_q, items_k, off = [], [], 0
        for (t, h, w) in shapes:
            n = int(t) * int(h) * int(w)
            items_q.append(vq[off:off + n].reshape(int(t), int(h), int(w), C))
            items_k.append(vk[off:off + n].reshape(int(t), int(h), int(w), C))
            off += n
        num_items = num_items or [len(shapes)]
        out, item0 = [], 0
        for V in num_items:
            qv, kv = items_q[item0:item0 + V], items_k[item0:item0 + V]
            item0 += V
            qf = torch.stack(qv, 0).permute(0, 1, 4, 2, 3).contiguous()
            kf = torch.stack(kv, 0).permute(0, 1, 4, 2, 3).contiguous()
            out.append((qf, kf))
        return out

    def _scatter_grids_heads(self, out_net: dict, num_items):
        """Like _scatter_vision_qk but keeps the HEAD dim (for L_corr). Returns list[B] of
        (q_grid, k_grid) each [V, T_lat, h, w, H, D] (q: num_heads, k: num_kv_heads), or None."""
        if "track_q_heads" not in out_net:
            return None
        vq, vk, shapes = out_net["track_q_heads"], out_net["track_k_heads"], out_net["track_token_shapes"]
        Hq, D, Hk = vq.shape[1], vq.shape[2], vk.shape[1]
        iq, ik, off = [], [], 0
        for (t, h, w) in shapes:
            n = int(t) * int(h) * int(w)
            iq.append(vq[off:off + n].reshape(int(t), int(h), int(w), Hq, D))
            ik.append(vk[off:off + n].reshape(int(t), int(h), int(w), Hk, D))
            off += n
        num_items = num_items or [len(shapes)]
        out, i0 = [], 0
        for V in num_items:
            out.append((torch.stack(iq[i0:i0 + V], 0), torch.stack(ik[i0:i0 + V], 0)))
            i0 += V
        return out

    @staticmethod
    def _point_motion_weight(traj, vis, gamma: float, floor: float, eps: float = 0.02, cap: float = 0.0):
        """Per-point weight (~mean 1) as speed^gamma. traj [V,T,N,2], vis [V,T,N].
        gamma > 0 emphasizes FAST points (dynamic objects); gamma < 0 emphasizes STATIC points (the
        anti-jitter mode for static-camera data, where 2D speed IS object speed). ``eps`` keeps
        speed^gamma finite at speed=0; ``cap`` bounds the max weight after mean-1 normalization
        (0 = uncapped); ``floor`` keeps the de-emphasized regime supervised. Mean ~1 preserved."""
        d = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1)
        m = vis[:, 1:] * vis[:, :-1]
        speed = (d * m).sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-6)
        w = (speed.clamp_min(0) + eps).pow(gamma)
        w = w / (w.mean() + 1e-6)
        if cap > 0:
            w = w.clamp_max(cap)
        w = w.clamp_min(floor)
        return w / (w.mean() + 1e-6)

    def _maybe_motion_weight(self, traj_gt, vis_gt):
        """Returns the per-point motion weight if enabled, else None."""
        rf = self.config.rectified_flow_training_config
        if not getattr(rf, "tracking_motion_weight_enabled", False):
            return None
        return self._point_motion_weight(
            traj_gt, vis_gt, float(getattr(rf, "tracking_motion_gamma", 1.0)),
            float(getattr(rf, "tracking_motion_floor", 0.3)),
            eps=float(getattr(rf, "tracking_motion_eps", 0.02)),
            cap=float(getattr(rf, "tracking_motion_cap", 0.0)),
        )

    def _compute_corr_loss(self, out_net: dict, data_batch: dict, gen_data_clean):
        """MVTrack4Gen L_corr (multi-view correspondence loss). For each query point (sampled at every
        view's window-frame-0), supervise its head-averaged attention over EACH target (view, frame)'s
        latent grid to peak at the GT corresponding cell, with a visibility-weighted cross-entropy
        (single-label per the paper). Operates on the tapped layer's per-head post-RoPE Q/K; valid for
        all (v,frame) pairs since the vision attention is full (video_temporal_causal=False)."""
        if "track_q_heads" not in out_net or "point_tracks" not in data_batch:
            return None, {}
        per_sample = self._scatter_grids_heads(out_net, gen_data_clean.num_vision_items_per_sample)
        if per_sample is None:
            return None, {}
        tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
        total, nseen = None, 0
        for b, (qfh, kfh) in enumerate(per_sample):
            pt = self._unwrap_sample(data_batch.get("point_tracks"), b)
            tv = self._unwrap_sample(data_batch.get("track_visibility"), b)
            if pt is None or tv is None or qfh.shape[0] != pt.shape[0]:
                continue
            dev = qfh.device
            V, Tlat, h, w, Hq, D = qfh.shape
            Hk = kfh.shape[4]
            if Hk == 0 or Hq % Hk != 0:
                continue
            groups = Hq // Hk
            pt, tv = pt.to(dev).float(), tv.to(dev).float()
            Tpix, N = pt.shape[1], pt.shape[2]
            wp = torch.tensor([min(tcf * i, Tpix - 1) for i in range(Tlat)], device=dev)
            scl = torch.tensor([w, h], device=dev, dtype=torch.float32)
            traj = pt[:, wp] * scl
            vis = tv[:, wp]

            q0 = qfh[:, 0].permute(0, 3, 4, 1, 2).reshape(V, Hq * D, h, w).float()
            gx = traj[:, 0, :, 0] / max(w - 1, 1) * 2 - 1
            gy = traj[:, 0, :, 1] / max(h - 1, 1) * 2 - 1
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(2).float()
            q_anchor = torch.nn.functional.grid_sample(q0, grid, align_corners=True)
            q_anchor = q_anchor.squeeze(-1).reshape(V, Hq, D, N).permute(0, 3, 1, 2)

            keys = kfh.repeat_interleave(groups, dim=4).reshape(V, Tlat, h * w, Hq, D).float()
            logits = torch.einsum("anhd,bjphd->anbjp", q_anchor, keys) / (Hq * (D ** 0.5))
            log_p = torch.log_softmax(logits, dim=-1)

            tx = traj[..., 0].clamp(0, w - 1).round().long()
            ty = traj[..., 1].clamp(0, h - 1).round().long()
            tgt_cell = ty * w + tx
            tgt = tgt_cell.permute(2, 0, 1).unsqueeze(0).expand(V, N, V, Tlat)
            ce = -log_p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

            weight = vis[:, 0][:, :, None, None] * vis.permute(2, 0, 1).unsqueeze(0)
            mw = self._maybe_motion_weight(traj, vis)
            if mw is not None:
                weight = weight * mw[None, :, None, None]
            eye = torch.eye(V, device=dev, dtype=torch.bool)
            j0 = torch.arange(Tlat, device=dev) == 0
            self_m = eye[:, None, :, None] & j0[None, None, None, :]
            weight = weight * (~self_m).float()
            denom = weight.sum()
            if float(denom) < 1.0:
                continue
            l = (weight * ce).sum() / (denom + 1e-6)
            total = l if total is None else total + l
            nseen += 1
        if nseen == 0 or total is None:
            return None, {}
        loss = total / nseen
        return loss, {"corr_loss/ce": loss.detach()}

    def _compute_tracking_loss(self, out_net: dict, data_batch: dict, gen_data_clean,
                               frame_sigmas: "torch.Tensor | None" = None):
        """MVTrack4Gen L_track: scatter captured vision Q/K into per-view grids, run the head, score
        against pseudo-GT tracks. Returns (loss, log-dict) or (None, {}) when unavailable.

        ``frame_sigmas``: the training noise levels per vision item, ``[B_items, T]`` (T=1 for a
        single per-item sigma, T=T_latent under diffusion forcing) — enables the ROBUST-SEQ
        sigma-aware frame weighting (``tracking_seq_sigma_weight``)."""
        if "track_q" not in out_net or "point_tracks" not in data_batch:
            return None, {}
        per_sample = self._scatter_vision_qk(out_net, gen_data_clean.num_vision_items_per_sample)
        if per_sample is None:
            return None, {}
        tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
        _counts = list(gen_data_clean.num_vision_items_per_sample or [])
        total, agg, nseen = None, {}, 0
        for b, (qf, kf) in enumerate(per_sample):
            pt = self._unwrap_sample(data_batch.get("point_tracks"), b)
            tv = self._unwrap_sample(data_batch.get("track_visibility"), b)
            tval = self._unwrap_sample(data_batch.get("track_valid"), b)
            if pt is None or tv is None or tval is None or qf.shape[0] != pt.shape[0]:
                continue
            dev = qf.device
            pt, tv, tval = pt.to(dev).float(), tv.to(dev).float(), tval.to(dev).float()
            u = int(getattr(self.net.tracking_head, "upsample", 1))
            Tlat, H0, W0 = qf.shape[1], qf.shape[3] * u, qf.shape[4] * u
            Tpix = pt.shape[1]
            wp = torch.tensor([min(tcf * i, Tpix - 1) for i in range(Tlat)], device=dev)
            scale = torch.tensor([W0, H0], device=dev, dtype=torch.float32)
            traj_gt = pt[:, wp] * scale
            vis_gt, valid_gt = tv[:, wp], tval[:, wp]
            _rf_cfg = self.config.rectified_flow_training_config
            _min_ok = int(getattr(_rf_cfg, "tracking_min_valid_points", 0))
            if _min_ok > 0 and float((valid_gt * vis_gt).sum().item()) < _min_ok:
                continue
            V_, N_ = traj_gt.shape[0], traj_gt.shape[2]
            src_b = self._unwrap_sample(data_batch.get("track_src"), b)
            if src_b is not None and src_b.numel() == N_:
                covis = valid_gt.permute(2, 0, 1)[torch.arange(N_, device=dev), src_b.to(dev).long()]
                covis = (covis > 0.5).float()
            else:
                covis = (valid_gt > 0.5).all(dim=0).float().t()
            covis = torch.where(covis.sum(1, keepdim=True) > 0, covis, torch.ones_like(covis))
            qfi = torch.multinomial(covis, 1).squeeze(1)
            q_xy = torch.gather(traj_gt, 1, qfi.view(1, 1, N_, 1).expand(V_, 1, N_, 2)).squeeze(1)
            queries = torch.cat([qfi.view(1, N_, 1).expand(V_, N_, 1).float(), q_xy], dim=-1)
            intr, extr = self._track_camera(data_batch, b, wp, qf.shape[0])
            tvv = self._unwrap_sample(data_batch.get("track_vis_valid"), b)
            vis_mask = tvv.to(dev).float()[:, wp] if (tvv is not None and tvv.shape[0] == qf.shape[0]) else None
            rf = self.config.rectified_flow_training_config
            out = self._track_head_call()(qf, kf, queries, intrinsics=intr, extrinsics=extr)
            _cond = [int(c) for c in getattr(data_batch["sequence_plan"][b],
                                             "condition_frame_indexes_vision", []) if 0 <= int(c) < Tlat]
            past_mask = torch.zeros(qf.shape[0], Tlat, device=dev)
            if _cond:
                past_mask[:, _cond] = 1.0
            fw = None
            if bool(getattr(rf, "tracking_seq_sigma_weight", False)) and frame_sigmas is not None:
                off_b = int(sum(_counts[:b])) if _counts else b
                V_b = qf.shape[0]
                sig = frame_sigmas[off_b : off_b + V_b].to(dev).float()
                if sig.dim() == 1:
                    sig = sig[:, None]
                sig = sig.expand(V_b, Tlat) if sig.shape[1] == 1 else sig[:, :Tlat]
                sig = sig.clone()
                if _cond:
                    sig[:, _cond] = 0.0
                fw = (1.0 - sig).clamp_(0.0, 1.0)
            dep_kw = {}
            if "depth" in out:
                td = self._unwrap_sample(data_batch.get("track_depth"), b)
                if td is not None and td.shape[0] == qf.shape[0]:
                    z = td.to(dev).float()[:, wp]
                    dep_kw = dict(
                        depth_preds=out["depth"],
                        depth_gt_log=torch.log(z.clamp_min(1e-3)),
                        depth_valid=(z > 1e-3).float(),
                        w_depth=float(getattr(rf, "tracking_depth_weight", 1.0)),
                        depth_huber_delta=float(getattr(rf, "tracking_depth_huber_delta", 0.2)),
                    )
            l, logs = multiview_track_loss(
                out["coords"], out["vis"], out["conf"], traj_gt, vis_gt, valid_gt,
                point_weight=self._maybe_motion_weight(traj_gt, vis_gt),
                w_vel=float(getattr(rf, "tracking_vel_weight", 0.0)),
                vis_mask=vis_mask,
                seq_cap=float(getattr(rf, "tracking_seq_cap", 0.0)),
                frame_weight=fw,
                past_frames=past_mask,
                **dep_kw,
            )
            total = l if total is None else total + l
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nseen += 1
        if nseen == 0 or total is None:
            return None, {}
        return total / nseen, {k: v / nseen for k, v in agg.items()}

    @torch.no_grad()
    def _track_forward(self, data_batch: dict, seed: int = 0, sigma: float | None = None):
        """Shared no-grad denoise forward that captures the tapped layer's Q/K and scatters them into
        per-sample per-view feature grids [V, T_lat, C, h, w]. Returns (per_sample, gen_data_clean) or
        (None, None). Mirrors training_step's vision-only forward (random noise level)."""
        if not getattr(self.config.rectified_flow_training_config, "tracking_enabled", False):
            return None, None
        it = int(seed)
        vid = data_batch.get(self.input_video_key)
        if isinstance(vid, list) and vid and isinstance(vid[0], list):
            data_batch["image_size"] = [
                torch.tensor([itm.shape[-2], itm.shape[-1], itm.shape[-2], itm.shape[-1]],
                             dtype=torch.float32, device=itm.device)
                for sample in vid for itm in sample
            ]
        input_text_indexes = self._load_and_tokenize_text_data(data_batch, it)
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch, input_video_key=self.input_video_key, input_image_key=self.input_image_key,
        )
        gen_data_clean = self.get_data_and_condition(data_batch, iteration=it)
        gen_data_clean, memory_info = self.memory_init_training(gen_data_clean, data_batch, input_text_indexes)
        num_tokens = [x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision]
        num_frames = [x.shape[2] for x in gen_data_clean.x0_tokens_vision]
        data_resolutions = None
        if "image_size" in data_batch and len(data_batch["image_size"]) > 0:
            img0 = data_batch["image_size"][0]
            if hasattr(img0, "dim") and img0.dim() == 2:
                img0 = img0[0]
            res0 = get_vision_data_resolution((int(img0[0]), int(img0[1])))
            data_resolutions = [res0] * gen_data_clean.batch_size
        ts, sg = self._get_train_noise_level_vision(
            batch_size=gen_data_clean.batch_size, is_image_batch=gen_data_clean.is_image_batch,
            resolutions=data_resolutions, num_vision_latent_frames=num_frames, num_tokens=num_tokens, iteration=it,
        )
        max_t = float(self.rectified_flow_video.noise_scheduler.config.num_train_timesteps)
        if sigma is not None:
            sg = torch.full_like(sg, float(sigma))
            ts = sg * max_t
        ts = _expand_per_sample_to_per_vision_item(ts, gen_data_clean.num_vision_items_per_sample)
        sg = _expand_per_sample_to_per_vision_item(sg, gen_data_clean.num_vision_items_per_sample)
        packed = self._pack_input_sequence(sequence_plans, input_text_indexes, gen_data_clean, ts.cpu())
        sg_action = None
        rf_cfg = self.config.rectified_flow_training_config
        act_idx = [i for i, plan in enumerate(sequence_plans) if plan.has_action]
        if (rf_cfg.independent_action_schedule and act_idx and packed.action is not None
                and any(nfi.numel() > 0 for nfi in packed.action.noisy_frame_indexes)):
            ts_a, sg_a = self._get_train_noise_level_action(batch_size=gen_data_clean.batch_size, iteration=it)
            if sigma is not None:
                sg_a = torch.full_like(sg_a, float(sigma))
                ts_a = sg_a * max_t
            sel = torch.tensor(act_idx, dtype=torch.long)
            ts_a, sg_action = ts_a[sel], sg_a[sel]
            packed.action.timesteps = torch.cat([
                ts_a.squeeze(1).cpu()[i : i + 1].expand(nfi.numel())
                for i, nfi in enumerate(packed.action.noisy_frame_indexes)
            ]).to(dtype=torch.float32)
        noised = self._add_noise_to_input(gen_data_clean, packed, sg, sigmas_action=sg_action, iteration=it)
        self._replace_clean_with_noised(packed, noised)
        packed.to_cuda()
        out_net = self.denoise(
            data_batch_packed=packed, fps_vision=gen_data_clean.fps_vision,
            fps_action=gen_data_clean.fps_action, fps_sound=gen_data_clean.fps_sound,
            memory=self.build_memory_state(packed, memory_info),
        )
        return self._scatter_vision_qk(out_net, gen_data_clean.num_vision_items_per_sample), gen_data_clean

    @torch.no_grad()
    def predict_tracks(self, data_batch: dict, grid_size: int = 12, seed: int = 0,
                       use_gt_queries: bool = True, sigma: float | None = None):
        """Run ONE (no-grad) denoise forward on ``data_batch`` to capture the tracking layer's Q/K,
        then track query points across the clip. Returns list[B] of drawables:
        ``{"video": uint8[V,Tlat,H,W,3], "tracks": f32[V,Tlat,N,2] (pixel), "vis": f32[V,Tlat,N]}``.
        The video is the input clip subsampled to one pixel frame per latent frame. None if disabled.

        ``use_gt_queries``: when the batch carries GT ``point_tracks`` (object-centric on DROID —
        only ~10-15% of the frame), query at the SAME points (per-view frame-0 GT positions) so the
        predicted rows compare 1:1 (same point index -> same color) with the pseudo-GT row. Falls
        back to a regular ``grid_size``x``grid_size`` grid when tracks are absent."""
        per_sample, gen_data_clean = self._track_forward(data_batch, seed, sigma=sigma)
        if per_sample is None:
            return None
        tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
        vids = data_batch[self.input_video_key]
        results = []
        item_off = 0
        for b, (qf, kf) in enumerate(per_sample):
            dev = qf.device
            V, Tlat, _, H0, W0 = qf.shape
            u = int(getattr(self.net.tracking_head, "upsample", 1))
            H0, W0 = H0 * u, W0 * u
            gt_pt = self._unwrap_sample(data_batch.get("point_tracks"), b) if use_gt_queries else None
            if gt_pt is not None and gt_pt.shape[0] == V:
                scl = torch.tensor([W0, H0], device=dev, dtype=torch.float32)
                xy_v = gt_pt.to(dev).float()[:, 0] * scl
                N = xy_v.shape[1]
                queries = torch.cat([torch.zeros(V, N, 1, device=dev), xy_v], dim=-1)
            else:
                xs = torch.linspace(W0 / (grid_size + 1), W0 - W0 / (grid_size + 1), grid_size, device=dev)
                ys = torch.linspace(H0 / (grid_size + 1), H0 - H0 / (grid_size + 1), grid_size, device=dev)
                gx, gy = torch.meshgrid(xs, ys, indexing="ij")
                xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
                N = xy.shape[0]
                queries = torch.cat([torch.zeros(V, N, 1, device=dev), xy[None].expand(V, N, 2)], dim=-1)
            view_items = vids[item_off : item_off + V]
            item_off += V
            view_list = [(vi[0] if torch.is_tensor(vi) and vi.dim() == 5 else vi) for vi in view_items]
            Hpx, Wpx = view_list[0].shape[-2], view_list[0].shape[-1]
            Tpix = view_list[0].shape[1]
            wp = torch.tensor([min(tcf * i, Tpix - 1) for i in range(Tlat)], device=dev)
            intr, extr = self._track_camera(data_batch, b, wp, V)
            o = self.net.tracking_head(qf, kf, queries, intrinsics=intr, extrinsics=extr)
            coords = o["coords_final"].float()
            vis = (o["vis_final"] * o["conf_final"]).float()
            vid_u8 = torch.stack([
                (((vl.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)) for vl in view_list
            ], dim=0)
            sx, sy = Wpx / W0, Hpx / H0
            tracks_px = coords.clone()
            tracks_px[..., 0] *= sx
            tracks_px[..., 1] *= sy
            tracks_px = self._interp_time(tracks_px, Tpix)
            vis_px = self._interp_time(vis.unsqueeze(-1), Tpix).squeeze(-1)
            conf_px = self._interp_time(o["conf_final"].float().unsqueeze(-1), Tpix).squeeze(-1)
            res = {"video": vid_u8.cpu(), "tracks": tracks_px.cpu(), "vis": vis_px.cpu(), "conf": conf_px.cpu()}
            if "depth_final" in o:
                res["depth"] = self._interp_time(o["depth_final"].float().unsqueeze(-1), Tpix).squeeze(-1).cpu()
            results.append(res)
        return results

    @staticmethod
    def _interp_time(x: torch.Tensor, Tout: int) -> torch.Tensor:
        """Linearly resample the time axis of ``x`` [V, T, N, C] -> [V, Tout, N, C]."""
        V, T, N, C = x.shape
        if T == Tout:
            return x
        if T < 2:
            return x[:, :1].expand(V, Tout, N, C).contiguous()
        xr = x.permute(0, 2, 3, 1).reshape(V * N * C, 1, T).float()
        xr = torch.nn.functional.interpolate(xr, size=Tout, mode="linear", align_corners=True)
        return xr.reshape(V, N, C, Tout).permute(0, 3, 1, 2).to(x.dtype)

    @torch.no_grad()
    def generate_window(self, win_db: dict, clips0: list, P_px: int, F_px: int, T_win: int,
                        num_steps: int, guidance: float, seed: int, shift: float = 5.0) -> dict:
        """Generate the ``F_px`` frames (and the robot actions, if the plan has an action stream) that follow the
        ``P_px`` clean past frames of one window.

        Returns ``{"pred", "gt", "P_px", "action"}``: ``pred``/``gt`` are per-sample, per-view ``[3, T_win, H, W]``
        pixel clips (``pred`` = input past + generated future) and ``action`` is a per-sample list of
        ``[T_act, max_action_dim]`` tensors (None when the batch has no action stream)."""
        B = len(clips0)
        gt = [[item[:, :T_win].clone() for item in sample] for sample in clips0]
        pred = [[item[:, :T_win].clone() for item in sample] for sample in clips0]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            seeds = [int((seed + b) % 2147483647) for b in range(B)]
            out = self.generate_samples_from_batch(
                dict(win_db), guidance=float(guidance), n_sample=B, num_steps=int(num_steps),
                shift=float(shift), seed=seeds, return_all_generated=True,
            )
            for b in range(B):
                for j in range(len(pred[b])):
                    px = self.decode(out["vision"][b][j]).clamp(-1.0, 1.0)
                    px = px[0] if px.dim() == 5 else px
                    n_f = max(0, min(F_px, pred[b][j].shape[1] - P_px, px.shape[1]))
                    if n_f:
                        pred[b][j][:, P_px : P_px + n_f] = px[:, px.shape[1] - F_px :][:, :n_f].to(
                            dtype=pred[b][j].dtype, device=pred[b][j].device)
            actions = None
            if out.get("action"):
                actions = []
                for a in out["action"]:
                    a = a[0] if (a is not None and a.dim() == 3 and a.shape[0] == 1) else a
                    actions.append(None if a is None else a.float().cpu())
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return {"pred": pred, "gt": gt, "P_px": P_px, "action": actions}


    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        """Install a custom attention dispatch function on the network.

        Called during ``build_net()`` after the network is constructed but
        before parallelization.  The base implementation is a no-op;
        ``OmniMoTCausalModel`` overrides this to install
        ``dispatch_attention_with_memory`` on every attention layer.
        """
        pass

    def set_up_memory(self) -> None:
        """Initialize memory state used during training (e.g. KV caches).

        The base implementation is a no-op.  ``OmniMoTCausalModel`` overrides
        this to allocate a KV cache.
        """
        pass

    def set_up_parallelism(self) -> None:
        """Set up the fsdp for the model."""
        if not torch.distributed.is_initialized():
            self.parallel_dims = None
            return

        self.parallel_dims = ParallelDims(
            enable_inference_mode=self.config.parallelism.enable_inference_mode,
            world_size=torch.distributed.get_world_size(),
            dp_shard=self.config.parallelism.data_parallel_shard_degree,
            cfgp=self.config.parallelism.cfg_parallel_shard_degree,
            cp=self.config.parallelism.context_parallel_shard_degree,
        )
        self.parallel_dims.build_meshes(device_type=DEVICE)

    def set_up_scheduler_and_sampler(self):
        shift_config = self.config.rectified_flow_training_config.shift
        if isinstance(shift_config, int):
            shift = shift_config
        else:
            shift_dict = dict(shift_config)
            resolution = self.config.resolution
            if resolution not in shift_dict:
                raise ValueError(
                    f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                )
            shift = shift_dict[resolution]

        if self.config.vision_gen:
            self.rectified_flow_image = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_image_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
            self.rectified_flow_video = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_video_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.action_gen:
            self.rectified_flow_action = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_action_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )
        if self.config.sound_gen:
            self.rectified_flow_sound = RectifiedFlow(
                velocity_field=self.net,
                train_time_distribution=self.config.rectified_flow_training_config.train_time_sound_distribution,
                use_dynamic_shift=self.config.rectified_flow_training_config.use_dynamic_shift,
                shift=shift,
                train_time_weight_method=self.config.rectified_flow_training_config.train_time_weight,
                device=torch.device(DEVICE),
                dtype=self.tensor_kwargs_fp32["dtype"],
            )

        assert self.config.rectified_flow_inference_config.scheduler_type in ["unipc", "edm"]
        if self.config.rectified_flow_inference_config.scheduler_type == "unipc":
            unipc_sampler_config = UniPCSamplerConfig(
                num_train_timesteps=self.config.rectified_flow_inference_config.num_train_timesteps,
                shift=self.config.rectified_flow_inference_config.shift,
                use_dynamic_shifting=self.config.rectified_flow_inference_config.use_dynamic_shifting,
            )
            self.sampler = UniPCSampler(cfg=unipc_sampler_config, tensor_kwargs=self.tensor_kwargs)
        else:
            self.sampler = EDMSampler()

        if self.config.fixed_step_sampler_config is not None:
            cfg = self.config.fixed_step_sampler_config
            self.fixed_step_sampler = FixedStepSampler(
                t_list=list(cfg.t_list),
                sample_type=cfg.sample_type,
                num_train_timesteps=float(self.config.rectified_flow_inference_config.num_train_timesteps),
            )
        else:
            self.fixed_step_sampler = None

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            optimizer_config (LazyDict): The lazy config for the optimizer.
            scheduler_config (LazyDict): The lazy config for the learning rate scheduler.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        optimizer = lazy_instantiate(optimizer_config, model=self)
        scheduler = lazy_instantiate(scheduler_config, optimizer=optimizer)
        return optimizer, scheduler

    def _derive_include_end_of_generation_token(self) -> bool:
        impl = self.config.joint_attn_implementation
        assert impl in ("flex", "two_way", "three_way"), (
            f"Invalid joint_attn_implementation: {impl}. Must be 'flex', 'two_way', or 'three_way'."
        )
        return impl == "flex"

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.ema.enabled:
            if isinstance(self.net, FSDPModule):
                for m in self.net.modules():
                    if isinstance(m, FSDPModule):
                        m.reshard()
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)


    def _pack_input_sequence(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        input_timesteps: torch.Tensor,
        include_end_of_generation_token: bool = False,
        skip_text_tokens: bool = False,
        initial_mrope_temporal_offset: int | float = 0,
    ) -> PackedSequence:
        """Wrap ``pack_input_sequence`` with all config-derived args pre-filled.

        Centralises the 9 config-derived positional/embedding args so callers only
        supply the four per-call arguments (sequence_plans, text tokens, data, timesteps)
        plus three optional flags.
        """
        assert self.tokenizer_vision_gen is not None
        return pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            input_timesteps=input_timesteps,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
            skip_text_tokens=skip_text_tokens,
            include_end_of_generation_token=include_end_of_generation_token,
            position_embedding_type=self.config.diffusion_expert_config.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=float(self.config.diffusion_expert_config.base_fps),
            temporal_compression_factor=self.tokenizer_vision_gen.temporal_compression_factor,
            video_temporal_causal=self.config.video_temporal_causal,
            action_dim=self.config.max_action_dim,
            initial_mrope_temporal_offset=initial_mrope_temporal_offset,
        )


    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, torch.Tensor],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Prepare the memory for a single training step.

        Called at the start of ``training_step`` to give the causal subclass
        an injection point for memory-based segment handling (frame trimming,
        segment bookkeeping, cache resets, packing overrides).

        The base implementation returns *gen_data_clean* unmodified and a
        default memory_info dict that does not support memory-backed training.

        The ``skip_text`` and ``initial_temporal`` offset fields are required,
        and are used for both sequence packing and memory.

        Returns:
            ``(gen_data_clean, memory_info)`` where *memory_info* is a dict with keys:
            ``skip_text``, ``initial_temporal_offset``
        """
        return gen_data_clean, {
            "skip_text": False,
            "initial_temporal_offset": 0,
        }

    def build_memory_state(
        self,
        packed_seq: PackedSequence,
        memory_info: dict,
    ) -> MemoryState | None:
        """Construct a ``MemoryState`` from a packed sequence and context dict.

        Called after packing in ``training_step()``, and before ``denoise()``
        in AR inference.  The base implementation returns ``None`` (no
        persistent memory).  ``OmniMoTCausalModel`` overrides this to build
        the appropriate ``ARMemoryState`` or ``KVCacheTrainMemoryState``.

        Args:
            packed_seq: The packed multi-modal sequence produced by
                ``_pack_input_sequence``.
            memory_info: Context dict returned by ``memory_init_training()``
                (for the training path) or constructed by the AR inference
                caller.  See ``memory_init_training()`` for the base keys.
        """
        return None

    def pre_noise_memory_hook(
        self,
        packed_sequence: PackedSequence,
        gen_data_clean: GenerationDataClean,
        memory_info: dict,
    ) -> dict:
        """Hook called after sequence packing and before noising. Returns (possibly updated) memory_info.

        The packed sequence still contains clean tokens at this point.
        Override in subclasses to run a clean forward pass (e.g. for teacher forcing).
        """
        return memory_info

    def build_window_batch(self, base_db: dict, clips: list, w_start: int, T_win: int, cond_idx: list) -> dict:
        """Build a single 49-frame-window data_batch from full-clip pixel ``clips``.

        ``clips``: list[B] of list[V] pixel tensors ``[3, L_px, H, W]`` (already-spliced if rolling).
        Slices video + per-view camera action to the window ``[w_start : w_start+T_win]``, rebuilds a
        fresh ``num_source_views=0`` SequencePlan (past ``cond_idx`` clean, rest generated), AND rewrites
        the caption's duration to the WINDOW length (so train/inference stay consistent and, under
        variable-future training, duration is a real length signal). Other keys are copied verbatim."""
        cap_key = self.input_caption_key
        win = dict(base_db)
        win.pop("num_vision_items_per_sample", None)
        win[self.input_video_key] = [
            [item[:, w_start : w_start + T_win].contiguous() for item in sample] for sample in clips
        ]
        win_deltas = T_win - 1
        new_actions, plans, new_caps, new_tokids = [], [], [], []
        have_cap = isinstance(base_db.get(cap_key), list)
        have_tok = isinstance(base_db.get("text_token_ids"), list)
        base_plans = base_db.get("sequence_plan") or []
        for b in range(len(clips)):
            V = len(clips[b])
            act = base_db["action"][b]
            while isinstance(act, list):
                act = act[0]
            bp = base_plans[b] if b < len(base_plans) else None
            if bp is not None and getattr(bp, "action_single_stream", False):
                assert w_start == 0, f"robot action stream needs w_start=0, got {w_start}"
                off = int(bp.action_start_frame_offset)
                keep = max(0, min(act.shape[0], T_win - off))
                sl = act[:keep]
                new_actions.append(sl)
                plans.append(SequencePlan(
                    has_text=True, has_vision=True,
                    condition_frame_indexes_vision=list(cond_idx),
                    num_source_views=0,
                    has_action=True,
                    condition_frame_indexes_action=[i for i in bp.condition_frame_indexes_action if i < keep],
                    action_start_frame_offset=off,
                    action_single_stream=True,
                    action_view_index=int(getattr(bp, "action_view_index", 0)),
                ))
                if have_cap:
                    new_caps.append(base_db[cap_key][b])
                    if have_tok:
                        new_tokids.append(base_db["text_token_ids"][b])
                continue
            full_deltas = act.shape[0] // V
            sl = torch.cat(
                [act[v * full_deltas + w_start : v * full_deltas + w_start + win_deltas] for v in range(V)],
                dim=0,
            )
            new_actions.append(sl)
            plans.append(SequencePlan(
                has_text=True, has_vision=True,
                condition_frame_indexes_vision=list(cond_idx),
                num_source_views=0,
                has_action=True,
                condition_frame_indexes_action=list(range(sl.shape[0])),
                action_start_frame_offset=1,
            ))
            if have_cap:
                cap = base_db[cap_key][b]
                m = re.search(r"is of (\d+(?:\.\d+)?) FPS", str(cap))
                fps = float(m.group(1)) if m else 15.0
                cap2 = re.sub(r"The video is [\d.]+ seconds long",
                              f"The video is {T_win / fps:.1f} seconds long", str(cap))
                new_caps.append(cap2)
                if have_tok:
                    ids = tokenize_caption(cap2, self.vlm_tokenizer, is_video=True,
                                           use_system_prompt=self.vlm_config.use_system_prompt)
                    new_tokids.append([torch.tensor(ids)])
        win["action"] = new_actions
        win["sequence_plan"] = plans
        win["image_size"] = [
            torch.tensor([item.shape[-2], item.shape[-1], item.shape[-2], item.shape[-1]],
                         dtype=torch.float32, device=item.device)
            for sample in win[self.input_video_key] for item in sample
        ]
        if have_cap:
            win[cap_key] = new_caps
        if have_tok and new_tokids:
            win["text_token_ids"] = new_tokids
        if "num_frames" in base_db and torch.is_tensor(base_db["num_frames"]):
            win["num_frames"] = torch.full_like(base_db["num_frames"], T_win)
        for tk in ("point_tracks", "track_visibility", "track_valid", "track_vis_valid",
                   "track_extrinsics", "camera_c2w"):
            if isinstance(base_db.get(tk), list):
                sliced = []
                for b in range(len(clips)):
                    t = base_db[tk][b]
                    while isinstance(t, list):
                        t = t[0]
                    sliced.append(t[:, w_start : w_start + T_win].contiguous())
                win[tk] = sliced
        return win

    def nested_clips(self, data_batch: dict) -> "list | None":
        """The batch's videos as [B][V] lists of [C,T,H,W] tensors, or None.

        TRAINING batches reach the validation/viz paths already MUTATED by
        ``get_data_and_condition`` (the video is flattened IN-PLACE to a flat list of
        [1,C,T,H,W] items, sample-major view-minor, with ``num_vision_items_per_sample``
        recorded). The windowing helpers need the per-sample view lists, so re-nest here —
        without this every flattened batch silently skipped the whole rollout/track viz
        (the curriculum runs produced ZERO track videos until 2026-07-14)."""
        clips0 = data_batch.get(self.input_video_key)
        if (isinstance(clips0, list) and clips0 and torch.is_tensor(clips0[0])
                and data_batch.get("num_vision_items_per_sample") is not None):
            counts = data_batch["num_vision_items_per_sample"]
            counts = counts.tolist() if torch.is_tensor(counts) else list(counts)
            if sum(int(c) for c in counts) == len(clips0):
                flat = [(v[0] if v.dim() == 5 else v) for v in clips0]
                nested, off = [], 0
                for c in counts:
                    nested.append(flat[off : off + int(c)])
                    off += int(c)
                clips0 = nested
        if not (isinstance(clips0, list) and clips0 and isinstance(clips0[0], list) and clips0[0]):
            return None
        return clips0

    def _sf_split(self, data_batch: dict) -> dict | None:
        """Derive the FIXED per-window split of an SF clip, or None if it can't be windowed.

        The window is ``past_latents + num_future_latents`` latents (e.g. 9+4 = 13 = 49 px), a CONSTANT
        independent of clip length, so a longer clip simply holds MORE chunks (``chunks`` = how many
        49-frame windows tile it). ``F_lat`` is read back from the dataset's full-clip plan
        (cond = ``range(latent_t - F_lat)``); ``past_latents`` is a config constant matching inference."""
        clips0 = self.nested_clips(data_batch)
        if clips0 is None:
            return None
        rf = self.config.rectified_flow_training_config
        tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
        L_px = int(clips0[0][0].shape[1])
        cond_full = data_batch["sequence_plan"][0].condition_frame_indexes_vision
        F_lat = (L_px // tcf + 1) - len(cond_full)
        P_lat = int(rf.student_forcing_past_latents)
        F_px = F_lat * tcf
        T_win = (P_lat + F_lat - 1) * tcf + 1
        P_px = T_win - F_px
        if F_lat < 1 or T_win > L_px or (L_px - P_px) % F_px != 0:
            return None
        chunks = (L_px - P_px) // F_px
        if chunks < 1:
            return None
        return dict(clips0=clips0, B=len(clips0), F_px=F_px, F_lat=F_lat, T_win=T_win, P_px=P_px,
                    chunks=chunks, cond_idx=list(range(T_win // tcf + 1 - F_lat)))

    def _maybe_variable_future(self, data_batch: dict, iteration: int) -> dict:
        """VARIABLE FUTURE-LENGTH training: crop the clip to ``33 px past + F`` where F is a random
        choice from ``variable_future_choices_px`` (default 8..48). Single window (NO self-rollout) —
        the model just predicts a random-length future, learning to generate more per shot. No-op if
        not an SF clip or no valid choices. Past stays the canonical 9 latents (cond_idx)."""
        rf = self.config.rectified_flow_training_config
        _diag = (int(iteration) % 50 == 0) and (
            (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        )
        sp = self._sf_split(data_batch)
        if sp is None:
            if _diag:
                _vid = data_batch.get(self.input_video_key)
                log.warning(f"[varfut] it{iteration}: _sf_split returned None -> NO crop (full clip "
                            f"trained as-is). video={type(_vid).__name__} "
                            f"nested={isinstance(_vid, list) and bool(_vid) and isinstance(_vid[0], list)}")
            return data_batch
        tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
        P_px, clips0, cond_idx = sp["P_px"], sp["clips0"], sp["cond_idx"]
        L_px = int(clips0[0][0].shape[1])
        choices = [int(c) for c in (rf.variable_future_choices_px or [])
                   if c % tcf == 0 and P_px + int(c) <= L_px]
        if not choices:
            if _diag:
                log.warning(f"[varfut] it{iteration}: no valid choices "
                            f"(P_px={P_px}, L_px={L_px}, raw={list(rf.variable_future_choices_px or [])}) "
                            f"-> canonical {sp['T_win']}px window")
            return self.build_window_batch(data_batch, clips0, 0, sp["T_win"], cond_idx)
        F_px = choices[int(torch.randint(len(choices), ()).item())]
        if _diag:
            log.info(f"[varfut] it{iteration}: future={F_px}px ({F_px // tcf} latents), "
                     f"window={P_px + F_px}px past={P_px}px  (choices={choices})")
        return self.build_window_batch(data_batch, clips0, 0, P_px + F_px, cond_idx)

    def _maybe_student_forcing(self, data_batch: dict, iteration: int) -> dict:
        """Return the 49-frame-window data_batch the loss is computed on.

        With probability ``prob`` (after ``start_iter``): GENERATE the first ``R-1`` chunks with the
        current model (few-step, no_grad), decode + splice them back in PIXEL space (exactly as the
        inference rollout does), and return the LAST window — whose past now holds the model's OWN
        generations (closing the exposure-bias gap). Otherwise return the FIRST window (real past).
        No-op for non-multiview batches or clips too short for ``R`` windows."""
        sf = self.config.rectified_flow_training_config
        sp = self._sf_split(data_batch)
        if sp is None:
            return data_batch
        clips0, B, F_px, T_win, cond_idx = (sp["clips0"], sp["B"], sp["F_px"], sp["T_win"], sp["cond_idx"])
        R = min(int(sf.student_forcing_rollout_chunks), sp["chunks"])

        do_roll = (R >= 2 and self.training and iteration >= int(sf.student_forcing_start_iter)
                   and (torch.rand(()).item() < float(sf.student_forcing_prob)))
        if not do_roll:
            return self.build_window_batch(data_batch, clips0, 0, T_win, cond_idx)

        clips = [[item.clone() for item in sample] for sample in clips0]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with torch.no_grad():
                for s in range(R - 1):
                    w0 = s * F_px
                    win_db = self.build_window_batch(data_batch, clips, w0, T_win, cond_idx)
                    seeds = [int((iteration * 131 + s * 17 + b) % 2147483647) for b in range(B)]
                    out = self.generate_samples_from_batch(
                        win_db, guidance=float(sf.student_forcing_guidance), n_sample=B,
                        num_steps=int(sf.student_forcing_num_steps), seed=seeds, return_all_generated=True,
                    )
                    gen = out["vision"]
                    for b in range(B):
                        for j in range(len(clips[b])):
                            px = self.decode(gen[b][j]).clamp(-1.0, 1.0)
                            px = px[0] if px.dim() == 5 else px
                            gchunk = px[:, -F_px:].to(dtype=clips[b][j].dtype, device=clips[b][j].device)
                            lo = w0 + (T_win - F_px)
                            clips[b][j][:, lo : lo + F_px] = gchunk
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return self.build_window_batch(data_batch, clips, (R - 1) * F_px, T_win, cond_idx)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step for the rectified-flow (flow-matching) model.

        This method executes one iteration of the model's training. It involves:
        1. Tokenizing generation modalities (vision/action/sound) into latents (tokens).
        2. Sampling a training timestep (t) for each modality and constructing noised latents (xt)
           per the rectified-flow formulation.
        3. Packing text + generation tokens into a single sequence and running the MoT network to predict
           the flow field velocity at the given t.
        4. Computing flow-matching loss (plus optional auxiliary load-balancing losses).

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.

        Returns:
            tuple: A tuple containing two elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor: The computed loss for the training step as a PyTorch Tensor.

        """
        _rf_cfg = self.config.rectified_flow_training_config
        if _rf_cfg.student_forcing_enabled:
            data_batch = self._maybe_student_forcing(data_batch, iteration)
        elif getattr(_rf_cfg, "variable_future_enabled", False):
            data_batch = self._maybe_variable_future(data_batch, iteration)

        if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
            self._update_train_stats(data_batch)

        input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)

        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        gen_data_clean = self.get_data_and_condition(data_batch, iteration=iteration)

        gen_data_clean, memory_info = self.memory_init_training(gen_data_clean, data_batch, input_text_indexes)

        if "image_size" in data_batch:
            data_resolutions = []
            for i in range(gen_data_clean.batch_size):
                img_size = data_batch["image_size"][i]
                if img_size.dim() == 2:
                    img_size = img_size[0]
                target_h = int(img_size[0].item())
                target_w = int(img_size[1].item())
                data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
        else:
            data_resolutions = None

        assert all(x.shape[0] == 1 for x in gen_data_clean.x0_tokens_vision), (
            "Batch size must be 1 for individual samples"
        )
        num_tokens_per_sample = [x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision]

        num_vision_latent_frames = [x.shape[2] for x in gen_data_clean.x0_tokens_vision]
        timesteps_vision, sigmas_vision = self._get_train_noise_level_vision(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            resolutions=data_resolutions,
            num_vision_latent_frames=num_vision_latent_frames,
            num_tokens=num_tokens_per_sample,
            iteration=iteration,
        )

        rf_cfg = self.config.rectified_flow_training_config
        action_sample_indices = [i for i, plan in enumerate(sequence_plans) if plan.has_action]
        if rf_cfg.independent_action_schedule and action_sample_indices:
            ts_full, sg_full = self._get_train_noise_level_action(
                batch_size=gen_data_clean.batch_size, iteration=iteration
            )
            idx = torch.tensor(action_sample_indices, dtype=torch.long)
            timesteps_action = ts_full[idx]
            sigmas_action = sg_full[idx]
        else:
            timesteps_action, sigmas_action = (None, None)

        sound_sample_indices = [i for i, plan in enumerate(sequence_plans) if getattr(plan, "has_sound", False)]
        if getattr(rf_cfg, "independent_sound_schedule", False) and sound_sample_indices:
            ts_sound_full, sg_sound_full = self._get_train_noise_level_sound(
                batch_size=gen_data_clean.batch_size
            )
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                ts_sound_full,
                sg_sound_full,
            )
        else:
            timesteps_sound, sigmas_sound = (None, None)

        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            src_rank = 0
            cp_group = self.parallel_dims.cp_mesh.get_group()
            global_src_rank = torch.distributed.get_global_rank(cp_group, src_rank)
            timesteps_vision = timesteps_vision.contiguous()
            sigmas_vision = sigmas_vision.contiguous()
            torch.distributed.broadcast(timesteps_vision, src=global_src_rank, group=cp_group)
            torch.distributed.broadcast(sigmas_vision, src=global_src_rank, group=cp_group)
            if sigmas_action is not None:
                timesteps_action = timesteps_action.contiguous()
                sigmas_action = sigmas_action.contiguous()
                torch.distributed.broadcast(timesteps_action, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_action, src=global_src_rank, group=cp_group)
            if sigmas_sound is not None:
                timesteps_sound = timesteps_sound.contiguous()
                sigmas_sound = sigmas_sound.contiguous()
                torch.distributed.broadcast(timesteps_sound, src=global_src_rank, group=cp_group)
                torch.distributed.broadcast(sigmas_sound, src=global_src_rank, group=cp_group)

        if timesteps_sound is None:
            timesteps_sound, sigmas_sound = build_dense_sound_schedule(
                sequence_plans,
                gen_data_clean.x0_tokens_sound,
                timesteps_vision,
                sigmas_vision,
            )

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_vision.cpu(),
            skip_text_tokens=memory_info["skip_text"],
            initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
        )

        if timesteps_action is not None and packed_sequence.action is not None:
            action_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes)
            if action_has_noisy_tokens:
                sample_ts = timesteps_action.squeeze(1).cpu()
                packed_sequence.action.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)
            else:
                timesteps_action, sigmas_action = (None, None)

        if (
            getattr(rf_cfg, "independent_sound_schedule", False)
            and timesteps_sound is not None
            and packed_sequence.sound is not None
        ):
            sound_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.sound.noisy_frame_indexes)
            if sound_has_noisy_tokens:
                sample_ts = timesteps_sound.squeeze(1).cpu()
                packed_sequence.sound.timesteps = torch.cat(
                    [
                        sample_ts[i : i + 1].expand(nfi.numel())
                        for i, nfi in enumerate(packed_sequence.sound.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)
            else:
                timesteps_sound, sigmas_sound = (None, None)

        timesteps_vision = _expand_per_sample_to_per_vision_item(
            timesteps_vision, gen_data_clean.num_vision_items_per_sample
        )
        sigmas_vision = _expand_per_sample_to_per_vision_item(
            sigmas_vision, gen_data_clean.num_vision_items_per_sample
        )

        memory_info = self.pre_noise_memory_hook(packed_sequence, gen_data_clean, memory_info)

        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigmas_vision,
            sigmas_action=sigmas_action,
            sigmas_sound=sigmas_sound,
            iteration=iteration,
        )
        self._replace_clean_with_noised(packed_sequence, gen_data_noised)

        packed_sequence.to_cuda()

        memory = self.build_memory_state(packed_sequence, memory_info)
        out_net = self.denoise(
            data_batch_packed=packed_sequence,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=gen_data_clean.fps_action,
            fps_sound=gen_data_clean.fps_sound,
            memory=memory,
        )

        loss, losses_dict = self._compute_losses(
            out_net=out_net,
            data_batch_packed=packed_sequence,
            gen_data_noised=gen_data_noised,
            timesteps=timesteps_vision,
            is_image_batch=gen_data_clean.is_image_batch,
            timesteps_action=timesteps_action,
            timesteps_sound=timesteps_sound,
        )

        _trk_cfg = self.config.rectified_flow_training_config
        _tw = float(getattr(_trk_cfg, "tracking_weight", 0.0))
        _cw = float(getattr(_trk_cfg, "tracking_corr_weight", 0.0))
        if getattr(_trk_cfg, "tracking_enabled", False) and _tw > 0:
            track_loss, track_logs = self._compute_tracking_loss(
                out_net, data_batch, gen_data_clean, frame_sigmas=sigmas_vision)
            if track_loss is not None:
                loss = loss + _tw * track_loss
                losses_dict.update(track_logs)
            _head = getattr(self.net, "tracking_head", None)
            if _head is not None:
                _touch = [q.sum() for q in _head.parameters() if q.requires_grad]
                if _touch:
                    loss = loss + 0.0 * torch.stack(_touch).sum()
        if getattr(_trk_cfg, "tracking_enabled", False) and _cw > 0:
            corr_loss, corr_logs = self._compute_corr_loss(out_net, data_batch, gen_data_clean)
            if corr_loss is not None:
                loss = loss + _cw * corr_loss
                losses_dict.update(corr_logs)
        _vae_pixel_shapes: list[tuple[int, int, int]] = []
        if gen_data_clean.raw_state_vision is not None:
            for _v in gen_data_clean.raw_state_vision:
                if _v is not None:
                    assert _v.dim() in [4, 5], (
                        "Currently only [C, T, H, W] and [B, C, T, H, W] formats are supported for the VAE encoding."
                    )
                    t_h_w = (
                        (int(_v.shape[2]), int(_v.shape[3]), int(_v.shape[4]))
                        if _v.dim() == 5
                        else (int(_v.shape[1]), int(_v.shape[2]), int(_v.shape[3]))
                    )
                    _vae_pixel_shapes.append(t_h_w)

        _vision_tokens = len(packed_sequence.vision.sequence_indexes) if packed_sequence.vision else 0
        _action_tokens = len(packed_sequence.action.sequence_indexes) if packed_sequence.action else 0
        _sound_tokens = len(packed_sequence.sound.sequence_indexes) if packed_sequence.sound else 0

        output_batch = {
            "x0": gen_data_clean.x0_tokens_vision,
            "xt": gen_data_noised.xt_tokens_vision,
            "sigma": sigmas_vision,
            "model_pred": out_net["preds_vision"],
            "condition_mask_vision": packed_sequence.vision.condition_mask if packed_sequence.vision else None,
            "condition_mask_action": packed_sequence.action.condition_mask if packed_sequence.action else None,
            "und_token_length": packed_sequence.text_indexes.shape[0],
            "gen_token_length": packed_sequence.sequence_length - packed_sequence.text_indexes.shape[0],
            "vision_token_length": _vision_tokens,
            "action_token_length": _action_tokens,
            "sound_token_length": _sound_tokens,
            "is_image_batch": gen_data_clean.is_image_batch,
            "batch_size": gen_data_clean.batch_size,
            "split_lens": packed_sequence.split_lens,
            "attn_modes": packed_sequence.attn_modes,
            "vae_pixel_shapes": _vae_pixel_shapes,
            **losses_dict,
        }
        if sigmas_action is not None:
            output_batch["sigma_action"] = sigmas_action
        if getattr(rf_cfg, "independent_sound_schedule", False) and sigmas_sound is not None:
            output_batch["sigma_sound"] = sigmas_sound

        return output_batch, loss

    def _compute_flow_matching_loss(
        self,
        pred: list[torch.Tensor],
        target: list[torch.Tensor],
        condition_mask: list[torch.Tensor],
        timesteps: torch.Tensor,
        has_valid_tokens: bool,
        rectified_flow: RectifiedFlow,
        loss_scale: float | None = None,
        raw_action_dim: list[torch.Tensor] | None = None,
        normalize_by_active: bool = False,
    ) -> torch.Tensor:
        """Compute flow matching loss for a modality.

        Args:
            pred: Predicted velocity field (list of tensors, one per sample).
            target: Target velocity field (list of tensors, one per sample).
                Under rectified flow the target is ``v = eps - x0``.
            condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
            timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
                base/teacher_forcing (all frames share one timestep) or [B,T_max]
                for diffusion_forcing (per-frame independent timesteps). Time weights
                are applied per-frame before averaging, so non-uniform weight functions
                are handled correctly.
            has_valid_tokens: Whether this modality has valid noisy tokens.
            rectified_flow: The rectified flow object for time weighting.
            loss_scale: Optional per-modality loss scale. Falls back to the global
                ``rectified_flow_training_config.loss_scale`` when *None*.
            normalize_by_active: When True, normalize per-instance loss by the count of
                active (noisy) elements rather than all elements. Preserves the
                ``sum / active_count`` semantics needed for distillation critics where
                conditioned frames contribute no signal and should not dilute the
                denominator.

        Returns:
            tuple: A tuple containing two elements:
                - Flow matching loss (or dummy loss for gradient consistency).
                - Per-instance loss (or dummy loss for gradient consistency).
        """
        return compute_flow_matching_loss(
            pred=pred,
            target=target,
            condition_mask=condition_mask,
            timesteps=timesteps,
            has_valid_tokens=has_valid_tokens,
            rectified_flow=rectified_flow,
            tensor_kwargs_fp32=self.tensor_kwargs_fp32,
            loss_scale=loss_scale,
            raw_action_dim=raw_action_dim,
            normalize_by_active=normalize_by_active,
        )

    def _compute_losses(
        self,
        out_net: dict,
        data_batch_packed: PackedSequence,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        is_image_batch: bool,
        timesteps_action: torch.Tensor | None = None,
        timesteps_sound: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute flow matching loss and auxiliary load balancing losses.

        ``timesteps_action`` is an optional ``[n_action, 1]`` override for the action loss
        time-weighting — dense over action-bearing samples, matching ``data_batch_packed.action.*``.
        When None, action reuses ``timesteps`` (vision timesteps, legacy behavior). Set by
        ``training_step`` under ``independent_action_schedule=True``.

        ``timesteps_sound`` is an optional dense sound timestep tensor, matching
        ``data_batch_packed.sound.*``. When None, sound reuses ``timesteps``.
        """
        total_loss = 0.0
        losses_dict = {}
        ts_action = timesteps if timesteps_action is None else timesteps_action
        ts_sound = timesteps if timesteps_sound is None else timesteps_sound

        rf_cfg = self.config.rectified_flow_training_config
        normalize_by_active = rf_cfg.normalize_loss_by_active
        if self.config.vision_gen:
            assert data_batch_packed.vision is not None, "Vision packed data required when vision_gen is True"
            assert isinstance(data_batch_packed.vision.condition_mask, list), (
                "Vision condition mask must be a list of tensors for loss computation"
            )
            rectified_flow_vision = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

            fm_loss_vision, fm_loss_vision_per_instance = self._compute_flow_matching_loss(
                pred=out_net["preds_vision"],
                target=gen_data_noised.vt_target_vision,
                condition_mask=data_batch_packed.vision.condition_mask,
                timesteps=timesteps,
                has_valid_tokens=has_noisy_tokens(data_batch_packed.vision),
                rectified_flow=rectified_flow_vision,
                normalize_by_active=normalize_by_active,
            )
            loss_scale = (
                rf_cfg.image_loss_scale if is_image_batch and rf_cfg.image_loss_scale is not None else rf_cfg.loss_scale
            )
            total_loss += fm_loss_vision * loss_scale
            losses_dict["flow_matching_loss_vision"] = fm_loss_vision
            losses_dict["flow_matching_loss_vision_per_instance"] = fm_loss_vision_per_instance
        else:
            losses_dict["flow_matching_loss_vision"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        if self.config.action_gen:
            if data_batch_packed.action is not None:
                assert isinstance(data_batch_packed.action.condition_mask, list), (
                    "Action condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_action is not None, "Action targets required when action_gen is True"
                fm_loss_action, _ = self._compute_flow_matching_loss(
                    pred=out_net["preds_action"],
                    target=gen_data_noised.vt_target_action,
                    condition_mask=data_batch_packed.action.condition_mask,
                    timesteps=ts_action,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.action),
                    rectified_flow=self.rectified_flow_action,
                    raw_action_dim=data_batch_packed.action.raw_action_dim,
                    normalize_by_active=normalize_by_active,
                )

                total_loss += fm_loss_action * rf_cfg.action_loss_weight
                losses_dict["flow_matching_loss_action"] = fm_loss_action
            else:
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_action"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_action"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_action"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        if self.config.sound_gen:
            if data_batch_packed.sound is not None:
                assert isinstance(data_batch_packed.sound.condition_mask, list), (
                    "Sound condition mask must be a list of tensors for loss computation"
                )
                assert gen_data_noised.vt_target_sound is not None, "Sound targets required when sound_gen is True"
                fm_loss_sound, _ = self._compute_flow_matching_loss(
                    pred=out_net["preds_sound"],
                    target=gen_data_noised.vt_target_sound,
                    condition_mask=[m.T for m in data_batch_packed.sound.condition_mask],
                    timesteps=ts_sound,
                    has_valid_tokens=has_noisy_tokens(data_batch_packed.sound),
                    rectified_flow=self.rectified_flow_sound,
                    normalize_by_active=normalize_by_active,
                )
                loss_scale = rf_cfg.sound_loss_scale if rf_cfg.sound_loss_scale is not None else rf_cfg.loss_scale
                total_loss += fm_loss_sound * loss_scale
                losses_dict["flow_matching_loss_sound"] = fm_loss_sound
            else:
                dummy_loss = 0.0 * sum(p.sum() for p in out_net["preds_sound"])
                total_loss += dummy_loss
                losses_dict["flow_matching_loss_sound"] = dummy_loss
        else:
            losses_dict["flow_matching_loss_sound"] = torch.tensor(0.0, **self.tensor_kwargs_fp32)

        for load_balancing_type in ["und", "gen"]:
            lbl_metadata = out_net.get(f"lbl_metadata_{load_balancing_type}", None)
            if lbl_metadata is None:
                continue
            load_balancing_loss = compute_load_balancing_loss(
                lbl_metadata,
                coeff=getattr(self.config.lbl, f"coeff_{load_balancing_type}"),
                method=self.config.lbl.method,
                device_mesh=self.parallel_dims.dp_mesh if self.parallel_dims else None,
            )
            if load_balancing_loss is not None:
                total_loss += load_balancing_loss
                losses_dict[f"aux_loss_{load_balancing_type}"] = load_balancing_loss

        return total_loss, losses_dict

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_video_key
        if isinstance(self.net, WeightTrainingStat):
            val = data_batch[input_key]
            sample_count = len(val) if isinstance(val, list) else val.shape[0]
            if is_image:
                self.net.accum_image_sample_counter += sample_count
            else:
                self.net.accum_video_sample_counter += sample_count

    def _load_and_tokenize_text_data(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> list[list[int]]:
        """
        Load and tokenize the text data from the data batch.

        Args:
            data_batch (dict[str, torch.Tensor]): The data batch.
            iteration (int): The current iteration number.

        Returns:
            list[torch.Tensor]: The input text tokens.
        """
        input_text_tokens = data_batch["text_token_ids"]
        if isinstance(input_text_tokens, list):
            seqs = []
            for x in input_text_tokens:
                if torch.is_tensor(x) and x.dim() == 1:
                    seqs.append(x.tolist())
                else:
                    seqs.extend(t.tolist() if torch.is_tensor(t) else list(t) for t in x)
            input_text_tokens = seqs
        else:
            input_text_tokens = [tokens.squeeze(0).tolist() for tokens in input_text_tokens]

        return input_text_tokens

    def _get_train_noise_level_vision(
        self,
        batch_size: int,
        is_image_batch: bool,
        num_vision_latent_frames: list[int],
        resolutions: list[str] | str | None = None,
        num_tokens: list[int] | None = None,
        iteration: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the rectified flow interpolation coefficient (timesteps), optionally adjust the sampled
        timesteps with high sigma strategy, and obtain the corresponding normalized timestep.

        Args:
            batch_size: Batch size for sampling timesteps.
            is_image_batch: Whether this is an image batch (vs video).
            num_vision_latent_frames: Per-sample vision latent frame counts [T_0, ..., T_{B-1}].
                         For causal_training_strategy="diffusion_forcing", resamples B*T_max independent
                         times and returns tensors of shape [B,T_max]. For base/TF strategies, ignored —
                         returns shape [B,1] (all frames share the same sigma).
            resolutions: Resolution string(s) (e.g., "256", "512") for dict-based shift lookup.
                         Can be a single string (applied to all samples) or a list of strings (one per sample).
                         If None, defaults to self.config.resolution (can be used for other modalities).
            num_tokens: Number of tokens for each sample (before 2x2 merge). Needed for dynamic shift.

        Returns:
            (timesteps, sigmas): Both [B,1] for TF/base, or [B,T_max] for diffusion_forcing.
        """


        rectified_flow = self.rectified_flow_image if is_image_batch else self.rectified_flow_video

        assert not self.config.rectified_flow_training_config.use_discrete_rf, (
            "Discrete RF is not supported for Cosmos3"
        )
        max_timestep = rectified_flow.noise_scheduler.config.num_train_timesteps

        shift_config = self.config.rectified_flow_training_config.shift
        if isinstance(shift_config, int):
            shifts = torch.full((batch_size,), shift_config, dtype=torch.float32)
        else:
            shift_dict = dict(shift_config)
            if not is_image_batch and "dynamic_shift_base_num_tokens_video" in shift_dict:
                assert num_tokens is not None and len(num_tokens) == batch_size
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_video"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            elif is_image_batch and "dynamic_shift_base_num_tokens_image" in shift_dict:
                assert num_tokens is not None and len(num_tokens) == batch_size
                base_num_tokens = shift_dict["dynamic_shift_base_num_tokens_image"]
                shifts = torch.sqrt(torch.tensor(num_tokens, dtype=torch.float32) / base_num_tokens)
            else:
                if resolutions is None:
                    raise ValueError("Resolutions must be provided when shift is a dict")

                if isinstance(resolutions, str):
                    resolutions = [resolutions] * batch_size

                assert len(resolutions) == batch_size, (
                    f"Number of resolutions ({len(resolutions)}) must match batch_size ({batch_size})"
                )

                shifts_list = []
                for resolution in resolutions:
                    if resolution not in shift_dict:
                        raise ValueError(
                            f"Resolution '{resolution}' not found in shift dict. Available resolutions: {list(shift_dict.keys())}"
                        )
                    shifts_list.append(shift_dict[resolution])
                shifts = torch.tensor(shifts_list, dtype=torch.float32)

        if self.config.causal_training_strategy == "diffusion_forcing":
            T_max = max(num_vision_latent_frames)
            t_raw = (
                rectified_flow.sample_train_time(batch_size * T_max, iteration=iteration)
                .to(**self.tensor_kwargs_fp32)
                .reshape(batch_size, T_max)
            )
        else:
            t_raw = (
                rectified_flow.sample_train_time(batch_size, iteration=iteration)
                .to(**self.tensor_kwargs_fp32)
                .unsqueeze(1)
            )

        t = 1 - t_raw
        shifts_2d = shifts.unsqueeze(1).to(t_raw.device)
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep

        if self.config.rectified_flow_training_config.use_high_sigma_strategy:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)

        sigmas = timesteps / max_timestep
        return timesteps, sigmas

    def _apply_high_noise_strategy(self, timesteps: torch.Tensor, max_timestep: int) -> torch.Tensor:
        """
        Update the sampled RF timesteps to shift the distribution towards higher noise levels (high sigmas).

        Args:
            timesteps (torch.Tensor): Input timesteps. Shape [B,1] for base/TF or [B,T_max] for DF.
            max_timestep (int): The maximum timestep value.

        Returns:
            torch.Tensor: Timesteps with the same shape as input — [B,1] or [B,T_max].
        """
        mask = (
            torch.rand(timesteps.shape, device=timesteps.device)
            < self.config.rectified_flow_training_config.high_sigma_ratio
        )
        new_timesteps = (
            torch.rand(timesteps.shape, device=timesteps.device).type_as(timesteps)
            * (
                self.config.rectified_flow_training_config.high_sigma_timesteps_max
                - self.config.rectified_flow_training_config.high_sigma_timesteps_min
            )
            + self.config.rectified_flow_training_config.high_sigma_timesteps_min
        )
        timesteps = torch.where(mask, new_timesteps, timesteps)

        return timesteps

    def _get_train_noise_level_action(
        self, batch_size: int, iteration: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_action``.

        This helper is locally-scoped: it just draws ``batch_size`` independent σ values and
        applies action-specific shift / high-sigma config. The caller decides what ``batch_size``
        means semantically — ``training_step`` passes the full batch size and then reindexes to
        the dense action-bearing subset with ``action_sample_indices``.

        ``shift_action`` must be an int (or ``None`` to inherit ``shift``). Dict-keyed
        per-resolution shifts are vision-only — multi-resolution action training would need
        per-sample lookup, which this helper does not implement; if the global ``shift`` is a
        dict and ``shift_action`` is None, this raises so the user sets shift_action explicitly.
        ``use_high_sigma_strategy_action`` toggles the high-σ strategy for action; when on, the
        global ``high_sigma_ratio`` / ``_min`` / ``_max`` apply. σ is a shared scalar per input
        slot (no per-frame σ for action).
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_action
        max_timestep = rf.noise_scheduler.config.num_train_timesteps

        if rf_cfg.shift_action is not None:
            if not isinstance(rf_cfg.shift_action, int):
                raise ValueError(
                    f"shift_action must be an int; got {type(rf_cfg.shift_action).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_action
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift
        else:
            raise ValueError(
                "shift_action=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_action explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        t_raw = (
            rf.sample_train_time(batch_size, iteration=iteration).to(**self.tensor_kwargs_fp32).unsqueeze(1)
        )
        t = 1 - t_raw
        shifts_2d = torch.full((batch_size, 1), shift_val, dtype=torch.float32, device=t_raw.device)
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep

        if rf_cfg.use_high_sigma_strategy_action:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)

        sigmas = timesteps / max_timestep
        return timesteps, sigmas

    def _get_train_noise_level_sound(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(timesteps, sigmas)`` of shape ``[batch_size, 1]`` from ``rectified_flow_sound``.

        Sound uses a shared scalar sigma per audio-bearing sample, then training_step
        reindexes the full-batch samples to the dense sound tensor list.
        """
        rf_cfg = self.config.rectified_flow_training_config
        rf = self.rectified_flow_sound
        max_timestep = rf.noise_scheduler.config.num_train_timesteps

        if rf_cfg.shift_sound is not None:
            if not isinstance(rf_cfg.shift_sound, int):
                raise ValueError(
                    f"shift_sound must be an int; got {type(rf_cfg.shift_sound).__name__}. "
                    "Dict-keyed per-resolution shifts are vision-only."
                )
            shift_val = rf_cfg.shift_sound
        elif isinstance(rf_cfg.shift, int):
            shift_val = rf_cfg.shift
        else:
            raise ValueError(
                "shift_sound=None requires the global `shift` to be an int. When `shift` is a "
                f"dict (multi-resolution vision training), set shift_sound explicitly as an int. "
                f"Got shift={rf_cfg.shift!r}."
            )

        t_raw = rf.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32).unsqueeze(1)
        t = 1 - t_raw
        shifts_2d = torch.full((batch_size, 1), shift_val, dtype=torch.float32, device=t_raw.device)
        timesteps = shifts_2d * t / (1 + (shifts_2d - 1) * t) * max_timestep

        if rf_cfg.use_high_sigma_strategy_sound:
            timesteps = self._apply_high_noise_strategy(timesteps, max_timestep)

        sigmas = timesteps / max_timestep
        return timesteps, sigmas

    def _add_noise_to_input(
        self,
        gen_data_clean: GenerationDataClean,
        packed_sequence: PackedSequence,
        sigmas: torch.Tensor,
        sigmas_action: torch.Tensor | None = None,
        sigmas_sound: torch.Tensor | None = None,
        iteration: int | None = None,
    ) -> GenerationDataNoised:
        """
        Diffusion / Flow matching forward process: apply noise of given noise level (sigmas) to input data.

        Args:
            gen_data_clean (GenerationDataClean): The input dataclass containing the clean data *latents* (tokens).
            packed_sequence (PackedSequence): Packed sequence with condition masks attached to modalities.
            sigmas (torch.Tensor): The noise levels. Shape [B,1] for base/teacher_forcing (all video
                latent frames share the same sigma) or [B,T_max] for diffusion_forcing (per-latent-frame
                independent sigma). T_max is the number of video latent frames (temporally compressed
                tokens), not RGB frames. In all modes, sigmas are multiplied by (1 - condition_mask)
                so conditioning latent frames get sigma_eff=0 and only non-conditioned frames contribute
                to the loss.
            sigmas_action: Optional ``[n_action, 1]`` override for action noising — dense over
                action-bearing samples, matching ``packed_sequence.action.*``. When None, action
                reuses ``sigmas`` (vision σ, legacy behavior). Set by ``training_step`` when
                ``independent_action_schedule=True``.
            sigmas_sound: Optional dense sound sigma tensor matching ``packed_sequence.sound.*``.
                When None, sound reuses ``sigmas``.

        Returns:
            GenerationDataNoised: A dataclass containing the noise, noisy data (xt), and velocity field (vt).
        """
        sigmas_for_action = sigmas if sigmas_action is None else sigmas_action
        sigmas_for_sound = sigmas if sigmas_sound is None else sigmas_sound

        noise_gen: torch.Generator | None = None
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            noise_gen = torch.Generator(device=self.tensor_kwargs_fp32["device"])
            noise_gen.manual_seed(iteration * 65536 + rank + 32768)

        x0_vision = gen_data_clean.x0_tokens_vision
        assert x0_vision is not None, "Vision tokens are required for VFM noising."
        epsilon_vision = [
            torch.randn(x0_vision_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_vision_i in x0_vision
        ]
        context_parallel_broadcast_tensor_list(epsilon_vision, self.parallel_dims)

        assert packed_sequence.vision is not None, "Packed vision data required for noise scheduling"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for noise scheduling"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for noise scheduling"
        )

        num_vision_items = len(packed_sequence.vision.condition_mask)
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]

        x0_for_interp = x0_vision
        cond_aug_std = float(getattr(self.config.rectified_flow_training_config, "cond_noise_aug_std", 0.0) or 0.0)
        if self.training and cond_aug_std > 0.0 and not gen_data_clean.is_image_batch:
            aug_noise = [
                torch.randn(x0_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_i in x0_vision
            ]
            context_parallel_broadcast_tensor_list(aug_noise, self.parallel_dims)
            x0_for_interp = []
            for i in range(num_vision_items):
                cm = packed_sequence.vision.condition_mask[i]
                s = torch.rand((), generator=noise_gen, **self.tensor_kwargs_fp32) * cond_aug_std
                scale = (s * x0_vision[i].to(torch.float32).std()).to(x0_vision[i].dtype)
                x0_for_interp.append(
                    x0_vision[i] + scale * cm.to(x0_vision[i].dtype) * aug_noise[i].to(x0_vision[i].dtype)
                )

        sigmas_vision = [
            sigmas[i].view(-1, 1, 1)[: x0_vision[i].shape[2]] * noisy_mask_vision[i] for i in range(num_vision_items)
        ]
        rectified_flow_vision = (
            self.rectified_flow_image if gen_data_clean.is_image_batch else self.rectified_flow_video
        )
        xt_vision, vt_vision = rectified_flow_vision.get_interpolation(
            epsilon_vision, x0_for_interp, sigmas_vision
        )

        xt_vision = [
            xt_vision_i.to(**self.tensor_kwargs) for xt_vision_i in xt_vision
        ]

        x0_action = gen_data_clean.x0_tokens_action
        if self.config.action_gen and x0_action is not None and len(x0_action) > 0:
            assert packed_sequence.action is not None, "Packed action data required when action tokens exist"
            assert packed_sequence.action.condition_mask is not None, (
                "Action condition mask required when action tokens exist"
            )
            action_batch_size = len(packed_sequence.action.condition_mask)
            all_actions_are_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if all_actions_are_conditioning:
                epsilon_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]
                sigmas_action = [
                    torch.zeros_like(condition_mask, dtype=torch.float32, device=condition_mask.device)
                    for condition_mask in packed_sequence.action.condition_mask
                ]
                xt_action = [
                    x0_action_i.to(**self.tensor_kwargs) for x0_action_i in x0_action
                ]
                vt_action = [
                    torch.zeros(x0_action_i.size(), **self.tensor_kwargs_fp32) for x0_action_i in x0_action
                ]
            else:
                epsilon_action = [
                    torch.randn(x0_action_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32)
                    for x0_action_i in x0_action
                ]
                context_parallel_broadcast_tensor_list(epsilon_action, self.parallel_dims)
                sigmas_action = [
                    sigmas_for_action[i].view(-1, 1)[: x0_action[i].shape[0]]
                    * (1.0 - packed_sequence.action.condition_mask[i])
                    for i in range(action_batch_size)
                ]
                assert sigmas_action is not None
                xt_action, vt_action = self.rectified_flow_action.get_interpolation(
                    epsilon_action, x0_action, sigmas_action
                )
                xt_action = [
                    xt_action_i.to(**self.tensor_kwargs) for xt_action_i in xt_action
                ]
            for i in range(len(xt_action)):
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    xt_action[i][:, gen_data_clean.raw_action_dim[i] :] = 0

        else:
            epsilon_action = None
            sigmas_action = None
            xt_action = None
            vt_action = None

        x0_sound = gen_data_clean.x0_tokens_sound
        if x0_sound is not None and len(x0_sound) > 0:
            assert packed_sequence.sound is not None, "Packed sound data required when sound tokens exist"
            assert packed_sequence.sound.condition_mask is not None, (
                "Sound condition mask required when sound tokens exist"
            )
            sound_batch_size = len(packed_sequence.sound.condition_mask)
            epsilon_sound = [
                torch.randn(x0_i.size(), generator=noise_gen, **self.tensor_kwargs_fp32) for x0_i in x0_sound
            ]
            context_parallel_broadcast_tensor_list(epsilon_sound, self.parallel_dims)
            sigmas_sound = [
                sigmas_for_sound[i].view(-1, 1)[: x0_sound[i].shape[1]].T
                * (1.0 - packed_sequence.sound.condition_mask[i].T)
                for i in range(sound_batch_size)
            ]
            assert sigmas_sound is not None
            xt_sound, vt_sound = self.rectified_flow_sound.get_interpolation(epsilon_sound, x0_sound, sigmas_sound)
            xt_sound = [xt_i.to(**self.tensor_kwargs) for xt_i in xt_sound]
        else:
            epsilon_sound = None
            sigmas_sound = None
            xt_sound = None
            vt_sound = None

        gen_data_noised = GenerationDataNoised(
            batch_size=gen_data_clean.batch_size,
            epsilon_vision=epsilon_vision,
            xt_tokens_vision=xt_vision,
            vt_target_vision=vt_vision,
            sigmas_vision=sigmas_vision,
            epsilon_action=epsilon_action,
            xt_tokens_action=xt_action,
            vt_target_action=vt_action,
            sigmas_action=sigmas_action,
            raw_action_dim=gen_data_clean.raw_action_dim,
            epsilon_sound=epsilon_sound,
            xt_tokens_sound=xt_sound,
            vt_target_sound=vt_sound,
            sigmas_sound=sigmas_sound,
        )

        return gen_data_noised

    def _replace_clean_with_noised(
        self,
        packed_sequence: PackedSequence,
        gen_data_noised: GenerationDataNoised,
    ) -> None:
        """Replace packed clean tokens with noised tokens."""
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision
        if packed_sequence.action is not None and gen_data_noised.xt_tokens_action is not None:
            action_all_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if not action_all_conditioning:
                packed_sequence.action.tokens = gen_data_noised.xt_tokens_action
        if packed_sequence.sound is not None and gen_data_noised.xt_tokens_sound is not None:
            packed_sequence.sound.tokens = gen_data_noised.xt_tokens_sound

    def _get_inference_text_tokens(
        self, data_batch: dict, has_negative_prompt: bool
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Tokenize conditional and unconditional captions for inference.

        Delegates the per-caption chat-template tokenization to
        :meth:`_tokenize_captions` (the same helper backing the public
        :meth:`tokenize_text`) so there is a single source of truth for
        how raw captions become token ids.
        """
        use_system_prompt = self.vlm_config.use_system_prompt
        system_prompt: str | None = data_batch.get("system_prompt")

        cond_tokens = self._tokenize_captions(
            data_batch[self.input_caption_key],
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            is_video=False,
        )

        if has_negative_prompt:
            neg_key = "neg_" + self.input_caption_key
            if neg_key not in data_batch:
                raise ValueError(f"Negative prompt ({neg_key}) not found")
            uncond_captions = data_batch[neg_key]
        else:
            uncond_captions = [""] * len(cond_tokens)

        uncond_tokens = self._tokenize_captions(
            uncond_captions,
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            is_video=False,
        )
        return cond_tokens, uncond_tokens

    def _prepare_inference_data(
        self,
        data_batch: dict,
        seed: list[int],
        has_negative_prompt: bool = False,
    ) -> tuple[
        list[SequencePlan],
        GenerationDataClean,
        list[list[int]],
        list[list[int]],
        list[torch.Tensor],
    ]:
        """
        Prepare all data needed for inference sampling.
        Mirrors training_step's data preparation flow.

        This method:
        1. Builds sequence plans (conditioning information)
        2. Gets data and condition (encodes vision)
        3. Tokenizes text (conditional and unconditional for CFG)
        4. Builds a packed sequence to fetch conditioning masks
        5. Initializes noise with conditioning applied (as lists for variable shapes)
        6. If action_gen is True, concatenates action noise with vision noise

        Args:
            data_batch: Raw data batch from dataloader.
            seed: Random seed(s) for noise generation.
            has_negative_prompt: If True, use negative prompt for unconditional branch.

        Returns:
            Tuple of:
                - sequence_plans: List of SequencePlan objects
                - gen_data_clean: GenerationDataClean with encoded tokens
                - cond_text_tokens: Conditional text tokens
                - uncond_text_tokens: Unconditional text tokens (for CFG)
                - initial_noise: List of noise tensors (one per sample), each containing
                  flattened vision (and optionally action) noise concatenated
        """
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )

        gen_data_clean = self.get_data_and_condition(data_batch)

        num_items_per_sample = gen_data_clean.num_vision_items_per_sample

        cond_text_tokens, uncond_text_tokens = self._get_inference_text_tokens(data_batch, has_negative_prompt)

        mask_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            cond_text_tokens,
            gen_data_clean,
            mask_timesteps,
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
        )

        assert packed_sequence.vision is not None, "Packed vision data required for inference noise"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for inference noise"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for inference noise"
        )
        assert gen_data_clean.x0_tokens_vision is not None, "Vision data required for inference noise"
        n_sample = (
            len(gen_data_clean.x0_tokens_vision)
            if gen_data_clean.num_vision_items_per_sample is None
            else len(gen_data_clean.num_vision_items_per_sample)
        )

        assert len(seed) == n_sample, (
            f"Seed list length {len(seed)} must have the same length as the number of samples {n_sample}"
        )

        seed_dict = {"vision": [], "action": [], "sound": []}
        for sample_idx in range(n_sample):
            num_vision_items = num_items_per_sample[sample_idx] if num_items_per_sample is not None else 1
            seed_dict["vision"].extend([seed[sample_idx]] * num_vision_items)
            seed_dict["action"].append(seed[sample_idx])
            seed_dict["sound"].append(seed[sample_idx])

        noise_vision_list: list[torch.Tensor] = []
        for i, (x0_token, cond_mask) in enumerate(
            zip(gen_data_clean.x0_tokens_vision, packed_sequence.vision.condition_mask, strict=True)
        ):
            pure_noise_i = misc.arch_invariant_rand(
                tuple(x0_token.shape),
                self.tensor_kwargs["dtype"],
                self.tensor_kwargs["device"],
                seed_dict["vision"][i],
            )
            noise_i = cond_mask * x0_token.to(**self.tensor_kwargs) + (1.0 - cond_mask) * pure_noise_i
            noise_vision_list.append(noise_i)

        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        noise_action_list: list[torch.Tensor] | None = None

        if has_action:
            assert gen_data_clean.x0_tokens_action is not None, "Action data required when sequence plan has action"
            assert packed_sequence.action is not None, "Packed action data required when action_gen is True"
            assert packed_sequence.action.condition_mask is not None, "Action condition mask required"
            assert isinstance(packed_sequence.action.condition_mask, list), (
                "Action condition mask must be a list of tensors for inference noise"
            )

            noise_action_list = []
            for i, (x0_action, cond_mask_action) in enumerate(
                zip(gen_data_clean.x0_tokens_action, packed_sequence.action.condition_mask, strict=True)
            ):
                pure_noise_action_i = misc.arch_invariant_rand(
                    tuple(x0_action.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["action"][i],
                )
                noise_action_i = (
                    cond_mask_action * x0_action.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_action) * pure_noise_action_i
                )
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    noise_action_i[:, gen_data_clean.raw_action_dim[i] :] = 0
                noise_action_list.append(noise_action_i)

        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)
        noise_sound_list: list[torch.Tensor] | None = None

        if has_sound:
            assert gen_data_clean.x0_tokens_sound is not None, "Sound data required when sequence plan has sound"
            assert packed_sequence.sound is not None, "Packed sound data required when sound_gen is True"
            assert packed_sequence.sound.condition_mask is not None, "Sound condition mask required"
            assert isinstance(packed_sequence.sound.condition_mask, list), (
                "Sound condition mask must be a list of tensors for inference noise"
            )

            noise_sound_list = []
            for i, (x0_sound, cond_mask_sound) in enumerate(
                zip(gen_data_clean.x0_tokens_sound, packed_sequence.sound.condition_mask, strict=True)
            ):
                pure_noise_sound_i = misc.arch_invariant_rand(
                    tuple(x0_sound.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["sound"][i],
                )
                noise_sound_i = (
                    cond_mask_sound.T * x0_sound.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_sound.T) * pure_noise_sound_i
                )
                noise_sound_list.append(noise_sound_i)

        initial_noise: list[torch.Tensor] = []
        idx_vision = 0
        idx_action = 0
        idx_sound = 0

        for i in range(n_sample):
            parts = []

            num_vis = num_items_per_sample[i] if num_items_per_sample is not None else 1
            for _ in range(num_vis):
                parts.append(noise_vision_list[idx_vision].reshape(-1))
                idx_vision += 1

            if noise_action_list is not None and sequence_plans[i].has_action:
                parts.append(noise_action_list[idx_action].reshape(-1))
                idx_action += 1

            if noise_sound_list is not None and sequence_plans[i].has_sound:
                parts.append(noise_sound_list[idx_sound].reshape(-1))
                idx_sound += 1

            initial_noise.append(torch.cat(parts, dim=0))

        return (
            sequence_plans,
            gen_data_clean,
            cond_text_tokens,
            uncond_text_tokens,
            initial_noise,
        )

    def _get_velocity(
        self,
        *,
        net: torch.nn.Module | None = None,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        text_tokens: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        skip_text_tokens: bool = False,
    ) -> list[torch.Tensor]:
        """
        Compute velocity prediction for a single sampling step.

        This method handles the full pipeline for one denoising step:
        1. Splits flattened noise_x into vision (and action) parts per sample
        2. Packs the input sequence with current noisy latents
        3. Runs the network via self.denoise()
        4. Applies velocity masks (zeroes out conditioned parts)
        5. Returns flattened velocities (concatenated vision + action per sample)

        Args:
            noise_x: List of noisy latents, each containing concatenated
                     vision (and optionally action) noise.
                     len(noise_x) == B, noise_x[i] is shape (D)
            timestep: Current timestep for each sample
            text_tokens: Tokenized text for each sample
            sequence_plans: Pre-computed sequence plans (from _prepare_inference_data)
            gen_data_clean: Pre-computed clean data (from _prepare_inference_data)
            skip_text_tokens: If True, skip text tokens (for CFG unconditional branch)

        Returns:
            Stacked flattened velocity tensors (one per sample), each containing
            concatenated vision (and optionally action) velocity
        """
        n_samples = len(noise_x)
        is_image_batch = gen_data_clean.is_image_batch
        has_action = self.config.action_gen and any(plan.has_action for plan in sequence_plans)
        num_items = gen_data_clean.num_vision_items_per_sample
        has_sound = self.config.sound_gen and any(plan.has_sound for plan in sequence_plans)

        noise_x_vision: list[torch.Tensor] = []
        noise_x_action: list[torch.Tensor] | None = [] if has_action else None
        noise_x_sound: list[torch.Tensor] | None = [] if has_sound else None

        vision_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            n_vis = num_items[i] if num_items is not None else 1
            offset = 0
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[vision_offset + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                noise_vision_ij = noise_x[i][offset : offset + vision_dim].reshape(vision_shape)
                noise_x_vision.append(noise_vision_ij)
                offset += vision_dim
            vision_offset += n_vis

            if has_action and noise_x_action is not None:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                noise_x_action.append(noise_x[i][offset : offset + action_dim].reshape(action_shape))
                offset += action_dim
                idx_action += 1

            if has_sound and noise_x_sound is not None and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                noise_x_sound.append(
                    noise_x[i][offset : offset + sound_dim].reshape(sound_shape)
                )
                offset += sound_dim
                idx_sound += 1

        gen_data_for_packing = GenerationDataClean(
            batch_size=n_samples,
            is_image_batch=is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            camera_conditioning=gen_data_clean.camera_conditioning,
            raw_state_action=gen_data_clean.raw_state_action if has_action else None,
            x0_tokens_action=noise_x_action if has_action else None,
            action_domain_id=gen_data_clean.action_domain_id if has_action else None,
            fps_action=gen_data_clean.fps_action if has_action else None,
            raw_action_dim=gen_data_clean.raw_action_dim if has_action else None,
            raw_state_sound=gen_data_clean.raw_state_sound if has_sound else None,
            x0_tokens_sound=noise_x_sound if has_sound else None,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
            num_vision_items_per_sample=num_items,
        )

        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            text_tokens,
            gen_data_for_packing,
            timestep.cpu(),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=skip_text_tokens,
        )

        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]

        if has_action and noise_x_action is not None:
            assert packed_sequence.action is not None, "packed_sequence.action must exist when has_action is True"
            packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]
            packed_sequence.action.domain_id = gen_data_clean.action_domain_id

        if has_sound and noise_x_sound is not None:
            assert packed_sequence.sound is not None, "packed_sequence.sound must exist when has_sound is True"
            packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]

        packed_sequence.to_cuda()

        fps_action = gen_data_clean.fps_action if has_action else None
        fps_sound = gen_data_clean.fps_sound if has_sound else None
        out = self.denoise(
            net=net,
            data_batch_packed=packed_sequence,
            fps_vision=gen_data_clean.fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
        )

        assert packed_sequence.vision is not None, "packed_sequence.vision must exist for velocity masking"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for masking"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for masking"
        )
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]

        velocity_vision: list[torch.Tensor] = []
        for i, (pred, noisy_mask) in enumerate(zip(out["preds_vision"], noisy_mask_vision)):
            has_noisy_tokens_i = noisy_mask.sum() > 0
            if has_noisy_tokens_i:
                velocity_vision.append(pred * noisy_mask.to(dtype=pred.dtype, device=pred.device))
            else:
                velocity_vision.append(torch.zeros_like(pred))

        velocity_action: list[torch.Tensor] | None = None
        if (
            has_action
            and packed_sequence.action is not None
            and packed_sequence.action.condition_mask is not None
            and isinstance(packed_sequence.action.condition_mask, list)
        ):
            noisy_mask_action = [1.0 - cond_mask for cond_mask in packed_sequence.action.condition_mask]

            velocity_action = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_action"], noisy_mask_action)):
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    v = pred * noisy_mask.to(dtype=pred.dtype, device=pred.device)
                else:
                    v = torch.zeros_like(pred)
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    v[:, gen_data_clean.raw_action_dim[i] :] = 0
                velocity_action.append(v)

        velocity_sound: list[torch.Tensor] | None = None
        if (
            has_sound
            and packed_sequence.sound is not None
            and packed_sequence.sound.condition_mask is not None
            and isinstance(packed_sequence.sound.condition_mask, list)
        ):
            noisy_mask_sound = [1.0 - cond_mask for cond_mask in packed_sequence.sound.condition_mask]

            velocity_sound = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_sound"], noisy_mask_sound)):
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    velocity_sound.append(
                        pred * noisy_mask.T.to(dtype=pred.dtype, device=pred.device)
                    )
                else:
                    velocity_sound.append(torch.zeros_like(pred))

        velocity_output: list[torch.Tensor] = []
        vis_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            parts = []
            n_vis = num_items[i] if num_items is not None else 1

            for _ in range(n_vis):
                parts.append(velocity_vision[vis_offset].reshape(-1))
                vis_offset += 1

            if velocity_action is not None and sequence_plans[i].has_action:
                parts.append(velocity_action[idx_action].reshape(-1))
                idx_action += 1

            if velocity_sound is not None and sequence_plans[i].has_sound:
                parts.append(velocity_sound[idx_sound].reshape(-1))
                idx_sound += 1

            velocity_output.append(torch.cat(parts, dim=0))

        return velocity_output

    def _remove_padding_from_latent(
        self, x0_tokens_vision: list[torch.Tensor], frame_size: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Remove reflection padding from encoded latent vision tokens.

        Each sample in the batch may have different original dimensions, so we process
        each sample individually and return a list of latents with varying spatial sizes.

        The padding coordinates are scaled down by the spatial compression factor since
        we're operating in latent space.

        Args:
            x0_tokens_vision (list[torch.Tensor]): List of encoded latent tensors,
                each of shape (1, C, T, H_latent, W_latent)
                where H_latent, W_latent include scaled padding.
            frame_size (list[torch.Tensor]): List of tensors, each of shape (1,4) or (4,) containing
                [target_h, target_w, orig_h, orig_w] for each sample (in pixel space).

        Returns:
            list[torch.Tensor]: List of cropped latent tokens, each of shape (1, C, T, H_latent_cropped, W_latent_cropped).
                Each element may have different spatial sizes based on original image dimensions.
        """
        batch_size = len(x0_tokens_vision)
        spatial_factor = self.tokenizer_vision_gen.spatial_compression_factor
        cropped_latents = []
        for i in range(batch_size):
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())

            if orig_h // spatial_factor == 0 or orig_w // spatial_factor == 0:
                log.warning(
                    f"Zero-sized latent found: orig_h: {orig_h}, orig_w: {orig_w}, spatial_factor: {spatial_factor}"
                )

            orig_h_latent = max(orig_h // spatial_factor, 1)
            orig_w_latent = max(orig_w // spatial_factor, 1)

            cropped_latent = x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous()
            cropped_latents.append(cropped_latent)

        return cropped_latents

    def _run_classifier_free_guidance(
        self,
        cond_tokens: list[list[int]],
        uncond_tokens: list[list[int]],
        skip_text_tokens_for_cfg: bool,
        single_velocity_fn: Callable[[list[list[int]], bool], list[torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run classifier-free guidance, optionally in parallel via CFG parallelism.

        Args:
            cond_tokens: Tokenized text for the conditional branch.
            uncond_tokens: Tokenized text for the unconditional branch.
            skip_text_tokens_for_cfg: If True, skip text tokens in the
                unconditional branch.
            single_velocity_fn: Computes velocity for a given set of tokens.
                Accepts ``(tokens, skip_text_tokens)`` and returns a list of
                velocity tensors (one per sample).

        Returns:
            A tuple ``(cond_v, uncond_v)`` where each element is a list of
            velocity tensors (one per sample).
        """
        if self.parallel_dims is None or not self.parallel_dims.cfgp_enabled:
            return (
                single_velocity_fn(cond_tokens, False),
                single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg),
            )

        cfgp_rank = self.parallel_dims.cfgp_rank
        cfgp_size = self.parallel_dims.cfgp_size
        cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
        cfgp_peer = (cfgp_rank + 1) % cfgp_size

        if cfgp_rank == 0:
            v_list = single_velocity_fn(cond_tokens, False)
        else:
            v_list = single_velocity_fn(uncond_tokens, skip_text_tokens_for_cfg)

        other_v_list = [torch.empty_like(v_i) for v_i in v_list]

        ops: list[dist.P2POp] = []
        for v_i, other_v_i in zip(v_list, other_v_list):
            ops.append(dist.P2POp(op=dist.isend, tensor=v_i, group_peer=cfgp_peer, group=cfgp_group))
            ops.append(dist.P2POp(op=dist.irecv, tensor=other_v_i, group_peer=cfgp_peer, group=cfgp_group))

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        if cfgp_rank == 0:
            return v_list, other_v_list
        else:
            return other_v_list, v_list

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        net: torch.nn.Module | None = None,
        sampler: Any | None = None,
        guidance: float = 1.5,
        guidance_interval: Optional[list[float]] = None,
        seed: list[int] | int = 1,
        n_sample: int | None = None,
        has_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        sigma_max: float = 80.0,
        skip_text_tokens_for_cfg: bool = False,
        normalize_cfg: bool = False,
        upsample_task: str | None = None,
        upsample_max_new_tokens: int = 2048,
        upsample_temperature: float | None = 0.7,
        upsample_top_k: int | None = 20,
        upsample_top_p: float | None = 0.8,
        upsample_repetition_penalty: float = 1.0,
        upsample_presence_penalty: float = 0.0,
        upsample_seed: int | None = None,
        return_all_generated: bool = False,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Generate samples from the batch. Based on given batch, it will automatically
        determine whether to generate image or video samples.

        This method follows the same structure as training_step:
        1. Build sequence plans
        2. Get data and condition (encode vision)
        3. Initialize noise with conditioning (as lists for variable shapes)
        4. Run sampling loop with velocity function
        5. Return latents as lists (supports variable shapes)

        If ``upsample_task`` is not ``None``, the conditional captions are
        upsampled with the corresponding canonical task (``"t2i"``, ``"t2v"``,
        or ``"i2v"``) before sample generation.

        Args:
            data_batch (dict): Raw data batch from the dataloader.
            guidance (float): Classifier-free guidance weight.
            guidance_interval (list[float] | None): Optional timestep interval to apply guidance.
                For the timesteps (ranging between 0-1000) that fall between the interval, we perform CFG, otherwise, we skip the unconditional generation.
            seed (list[int] | int): Random seeds for noise generation. For all new use-cases,
                we use a list of seeds, one for each sample. The length of the list must match
                the number of samples. Legacy use-cases use a single integer seed which is
                incremented by 1 for each sample. But this is not supported anymore, and will
                raise an error if used.
            n_sample (int | None): Number of samples to generate; defaults to batch size.
            has_negative_prompt (bool): If True, use negative prompt for unconditional branch.
            num_steps (int): Number of sampling steps for the diffusion process.
            shift (float): Time shift parameter for the sampler.
            sigma_max (float): Maximum sigma for the EDM sampler.
            skip_text_tokens_for_cfg (bool): If True, skip text tokens in unconditional branch.
            normalize_cfg (bool): If True, normalize the CFG output.
            upsample_task (str | None): Canonical V4.2 task resolved by the
                inference caller. ``None`` disables native prompt upsampling.
                Otherwise, the conditional captions in
                ``data_batch[self.input_caption_key]`` are expanded via
                :meth:`upsample_captions` (which drives the reasoner tower's
                autoregressive loop) BEFORE downstream tokenization, so all
                subsequent code paths — cond/uncond build, sequence plans,
                velocity_fn, CFG — operate on the upsampled prompts uniformly.
                When ``upsample_task`` is ``"i2v"``, the
                anchor (first) raw frame of each sample's video — extracted
                from ``data_batch[self.input_video_key]`` via
                :meth:`_extract_upsample_conditioning_images` — is also
                forwarded to the upsampler's Qwen3-VL multimodal chat
                template so the rewritten caption is grounded in the
                actual conditioning frame, not just the original caption
                text.  The unconditional captions are NOT upsampled
                (negative prompts and empty strings pass through as-is).
            upsample_max_new_tokens (int): Maximum tokens generated per
                caption during upsampling.  Defaults to 2048.
            upsample_temperature, upsample_top_k, upsample_top_p: Native
                reasoner prompt upsampling sampling controls.
            upsample_repetition_penalty (float): CTRL/HF-style
                multiplicative logit penalty on tokens already seen
                in each caption's history.  ``>1.0`` discourages
                verbatim repetition, ``<1.0`` encourages it, ``1.0``
                (default) is identity and adds zero overhead.
            upsample_presence_penalty (float): OpenAI-style additive
                logit penalty (binary presence, not frequency) on
                tokens already seen.  ``>0`` discourages reuse,
                ``<0`` encourages it, ``0.0`` (default) is identity.
            upsample_seed (int | None): Optional integer seed for the
                native prompt upsampler's sampling RNG.  Forwarded to
                :meth:`upsample_captions` and through to
                :func:`unified_mot._impl_generate_reasoner_text`, which
                seeds a fresh device-local ``torch.Generator`` once and
                threads it into every ``torch.multinomial`` draw.
                ``None`` (default) uses the device's default RNG and is
                bit-identical to the pre-seed behavior.  Greedy
                upsampling (``upsample_temperature`` 0) never reads the
                generator, so the value has no effect in that case.
                NOTE: distinct from the per-sample ``seed`` argument
                above, which seeds the diffusion noise.

        Returns:
            Dict with keys:
                - "vision": List of vision latent tensors (one per sample, variable shapes)
                - "action": List of action latent tensors or None (only present when action_gen=True and has_action)

        Raises:
            ValueError: If the number of samples does not match the number of noise tensors or seeds.
            ValueError: If the seed is a single integer. This is not supported anymore: `seed` must be
                a list of integers, one for each sample.
        """
        if isinstance(seed, int):
            raise ValueError(
                "Single integer seed is not supported anymore: `seed` must be a list of integers, one for each sample."
            )
        assert isinstance(seed, list)

        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cp_mesh.get_group(), self.parallel_dims.cp_rank)

        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            seed = _broadcast_seed(seed, self.parallel_dims.cfgp_mesh.get_group(), self.parallel_dims.cfgp_rank)

        data_batch = self._maybe_apply_prompt_upsampling(
            data_batch,
            upsample_task=upsample_task,
            upsample_max_new_tokens=upsample_max_new_tokens,
            upsample_temperature=upsample_temperature,
            upsample_top_k=upsample_top_k,
            upsample_top_p=upsample_top_p,
            upsample_repetition_penalty=upsample_repetition_penalty,
            upsample_presence_penalty=upsample_presence_penalty,
            upsample_seed=upsample_seed,
        )

        (
            sequence_plans,
            gen_data_clean,
            cond_tokens,
            uncond_tokens,
            initial_noise,
        ) = self._prepare_inference_data(data_batch, seed, has_negative_prompt)

        if n_sample is not None:
            assert n_sample == len(initial_noise), (
                f"Number of samples {n_sample} must match number of noise tensors {len(initial_noise)}"
            )
        else:
            n_sample = len(initial_noise)

        assert n_sample == len(seed), f"Number of samples {n_sample} must match number of seeds {len(seed)}"

        if (
            self.parallel_dims is not None
            and self.parallel_dims.dp_shard_mesh is not None
            and torch.distributed.is_initialized()
            and self.parallel_dims.dp_shard_mesh.size() > 1
        ):
            _dp_shard_group = self.parallel_dims.dp_shard_mesh.get_group()
            _align_device = self.tensor_kwargs["device"]
        else:
            _dp_shard_group = None
            _align_device = None


        def velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
            torch.compiler.cudagraph_mark_step_begin()

            assert timestep.ndim == 2, f"timestep must be 2D, got {timestep.shape}"
            assert timestep.shape == (1, 1), f"timestep must be (1, 1), got {timestep.shape}"

            timestep = timestep.repeat(len(noise_x), 1)

            def _single_velocity_fn(tokens: list[list[int]], skip_text_tokens: bool):
                return self._get_velocity(
                    net=net,
                    noise_x=noise_x,
                    timestep=timestep,
                    text_tokens=tokens,
                    sequence_plans=sequence_plans,
                    gen_data_clean=gen_data_clean,
                    skip_text_tokens=skip_text_tokens,
                )

            _local_needs_cfg = guidance != 1.0
            if _local_needs_cfg and guidance_interval is not None:
                assert len(guidance_interval) == 2, f"guidance_interval must be [lo, hi], got {guidance_interval}"
                t_lo, t_hi = guidance_interval
                _local_needs_cfg = t_lo < timestep[0].item() < t_hi

            if _dp_shard_group is not None:
                _cfg_t = torch.tensor(
                    [1 if _local_needs_cfg else 0], device=_align_device, dtype=torch.int32
                )
                torch.distributed.all_reduce(
                    _cfg_t, op=torch.distributed.ReduceOp.MAX, group=_dp_shard_group
                )
                _any_needs_cfg = bool(_cfg_t.item())
            else:
                _any_needs_cfg = _local_needs_cfg

            if not _any_needs_cfg:
                return _single_velocity_fn(cond_tokens, skip_text_tokens=False)

            cond_v, uncond_v = self._run_classifier_free_guidance(
                cond_tokens=cond_tokens,
                uncond_tokens=uncond_tokens,
                skip_text_tokens_for_cfg=skip_text_tokens_for_cfg,
                single_velocity_fn=_single_velocity_fn,
            )

            if not _local_needs_cfg:
                return cond_v

            v_pred = [u_i + guidance * (c_i - u_i) for c_i, u_i in zip(cond_v, uncond_v)]

            if normalize_cfg:
                v_pred = [
                    v_i * (torch.norm(c_i) / (torch.norm(v_i) + 1e-8)).clamp(min=0.0, max=1.0)
                    for v_i, c_i in zip(v_pred, cond_v)
                ]

            return v_pred

        if _dp_shard_group is not None:
            _local_steps_t = torch.tensor([num_steps], device=_align_device, dtype=torch.int32)
            torch.distributed.all_reduce(
                _local_steps_t, op=torch.distributed.ReduceOp.MAX, group=_dp_shard_group
            )
            _max_num_steps = int(_local_steps_t.item())
        else:
            _max_num_steps = num_steps
        _extra_num_steps = _max_num_steps - num_steps

        sampler = sampler or self.sampler
        scheduler_type = self.config.rectified_flow_inference_config.scheduler_type
        if scheduler_type == "unipc":
            log.info(f"Using sampler: UniPC (shift={shift}, num_steps={num_steps})")
        else:
            log.info(f"Using sampler: EDM (sigma_max={sigma_max}, num_steps={num_steps})")

        if scheduler_type == "unipc":
            latents = sampler(
                velocity_fn,
                initial_noise,
                num_steps=num_steps,
                shift=shift,
                seed=seed,
            )
            if _extra_num_steps > 0:
                log.debug(
                    f"FSDP alignment: dummy sampler run with {_extra_num_steps} "
                    f"extra steps (local={num_steps}, max={_max_num_steps})"
                )
                _ = sampler(
                    velocity_fn,
                    latents,
                    num_steps=_extra_num_steps,
                    shift=shift,
                    seed=seed,
                )
        else:
            chunk_sizes = [_x.shape[0] for _x in initial_noise]
            initial_noise = torch.cat(initial_noise, dim=0)

            def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                assert sigma.ndim == 0, f"sigma must be 0D, got {sigma.shape}"
                timestep_rf = sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)

                _noise_x = list(torch.split(noise_x, chunk_sizes, dim=0))
                _velocity_pred = velocity_fn(_noise_x, timestep_rf.reshape(1, 1))
                velocity_pred = torch.cat(_velocity_pred, dim=0)

                x0_pred = noise_x - sigma * velocity_pred
                return x0_pred

            latents = sampler(
                x0_fn,
                initial_noise,
                num_steps=num_steps,
                sigma_max=sigma_max,
                sigma_min=0.002,
                solver_option="2ab",
            )
            if _extra_num_steps > 0:
                log.debug(
                    f"FSDP alignment: padding {_extra_num_steps} dummy x0_fn calls "
                    f"(local={num_steps}, max={_max_num_steps})"
                )
                _dummy_sigma = latents.new_tensor(sigma_max / (1.0 + sigma_max))
                for _ in range(_extra_num_steps):
                    _ = x0_fn(latents, _dummy_sigma)
            latents = list(torch.split(latents, chunk_sizes, dim=0))

        result_vision: list[torch.Tensor] = []
        result_action: list[torch.Tensor] = []
        result_sound: list[torch.Tensor] = []
        idx_vision = 0
        idx_action = 0
        idx_sound = 0
        num_vision_items = gen_data_clean.num_vision_items_per_sample

        for i in range(n_sample):
            offset = 0

            n_vis = num_vision_items[i] if num_vision_items is not None else 1
            num_source_views = getattr(sequence_plans[i], "num_source_views", None)
            group_size = getattr(sequence_plans[i], "vision_group_size", None)
            gen_items: list[torch.Tensor] = []
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[idx_vision + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                if num_source_views is not None:
                    is_target = j >= num_source_views
                elif group_size:
                    is_target = (j + 1) % group_size == 0
                else:
                    is_target = j == n_vis - 1
                if is_target:
                    gen_items.append(latents[i][offset : offset + vision_dim].reshape(vision_shape))
                offset += vision_dim
            idx_vision += n_vis
            result_vision.append(gen_items if return_all_generated else gen_items[-1])

            if self.config.action_gen and sequence_plans[i].has_action:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                result_action.append(latents[i][offset : offset + action_dim].reshape(action_shape))
                offset += action_dim
                idx_action += 1

            if self.config.sound_gen and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                result_sound.append(latents[i][offset : offset + sound_dim].reshape(sound_shape))
                offset += sound_dim
                idx_sound += 1

        result: dict[str, list[torch.Tensor]] = {"vision": result_vision}
        if self.config.action_gen and len(result_action) > 0:
            result["action"] = result_action
        if self.config.sound_gen and len(result_sound) > 0:
            result["sound"] = result_sound
        return result

    def _extract_condition_images_for_visualization(
        self,
        gen_data_clean: GenerationDataClean,
        sequence_plans: list[SequencePlan],
        n_samples: int,
    ) -> list[torch.Tensor | None]:
        """Extract condition images from gen_data_clean for visualization.

        For image editing, raw_state_vision is a flat list of individually-encoded
        images (e.g. [src1, tgt1, src2, tgt2, ...]).  The first vision item for
        each sample is the condition (source) image.  This method extracts it and
        resizes to match the target for side-by-side display.

        Args:
            gen_data_clean: Clean data containing raw vision states.
            sequence_plans: Sequence plans for each sample.
            n_samples: Number of samples to process.

        Returns:
            List of condition image tensors (one per sample with condition frames).
        """
        condition_images: list[torch.Tensor | None] = []

        if gen_data_clean.num_vision_items_per_sample is not None:
            vision_offset = 0
            for i in range(n_samples):
                num_items = gen_data_clean.num_vision_items_per_sample[i]
                if num_items >= 2:
                    cond_frame = gen_data_clean.raw_state_vision[vision_offset]
                    target_frame = gen_data_clean.raw_state_vision[vision_offset + 1]
                    if cond_frame.shape[-2:] != target_frame.shape[-2:]:
                        cond_frame = torch.nn.functional.interpolate(
                            cond_frame.squeeze(2),
                            size=target_frame.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).unsqueeze(2)
                    condition_images.append(cond_frame)
                else:
                    condition_images.append(None)
                vision_offset += num_items
        else:
            for i in range(n_samples):
                plan = sequence_plans[i]
                if len(plan.condition_frame_indexes_vision) > 0 and gen_data_clean.raw_state_vision is not None:
                    raw_vision = gen_data_clean.raw_state_vision[i]
                    condition_images.append(raw_vision[:, :, 0:1, :, :])
                else:
                    condition_images.append(None)

        return condition_images

    def _slice_gen_data_clean(self, gen_data_clean: GenerationDataClean, start: int, limit: int) -> GenerationDataClean:
        """Extract a subset of GenerationDataClean for inference.

        The samples in [start:limit] are extracted from the original GenerationDataClean.

        For image editing (``num_vision_items_per_sample`` is set), the sample index refers to
        the *real sample* index. The method computes the correct slice of the flat
        ``x0_tokens_vision`` / ``raw_state_vision`` lists using the item counts and
        preserves ``num_vision_items_per_sample`` on the returned subset so that
        downstream packing works correctly.

        Args:
            gen_data_clean: GenerationDataClean to slice.
            start: Start index of the slice.
            limit: Limit index of the slice.

        Returns:
            Sliced GenerationDataClean.
        """
        has_action = bool(gen_data_clean.x0_tokens_action)
        has_sound = bool(gen_data_clean.x0_tokens_sound)

        num_items = gen_data_clean.num_vision_items_per_sample
        if num_items is not None:
            vis_start = sum(num_items[:start])
            vis_end = sum(num_items[:limit])
            subset_x0_vision = gen_data_clean.x0_tokens_vision[vis_start:vis_end]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[vis_start:vis_end] if gen_data_clean.raw_state_vision else None
            )
            subset_num_items = num_items[start:limit]
        else:
            subset_x0_vision = gen_data_clean.x0_tokens_vision[start:limit]
            subset_raw_vision = (
                gen_data_clean.raw_state_vision[start:limit] if gen_data_clean.raw_state_vision else None
            )
            subset_num_items = None
        fps_vision = gen_data_clean.fps_vision[start:limit] if gen_data_clean.fps_vision is not None else None

        if has_action:
            subset_raw_action = (
                gen_data_clean.raw_state_action[start:limit] if gen_data_clean.raw_state_action else None
            )
            x0_tokens_action = gen_data_clean.x0_tokens_action[start:limit]
            fps_action = gen_data_clean.fps_action[start:limit] if gen_data_clean.fps_action is not None else None
            action_domain_id = gen_data_clean.action_domain_id[start:limit] if gen_data_clean.action_domain_id else None
            raw_action_dim = gen_data_clean.raw_action_dim[start:limit] if gen_data_clean.raw_action_dim else None
        else:
            subset_raw_action = None
            x0_tokens_action = None
            fps_action = None
            action_domain_id = None
            raw_action_dim = None

        if has_sound:
            subset_raw_sound = gen_data_clean.raw_state_sound[start:limit] if gen_data_clean.raw_state_sound else None
            x0_tokens_sound = gen_data_clean.x0_tokens_sound[start:limit]
            fps_sound = gen_data_clean.fps_sound[start:limit] if gen_data_clean.fps_sound is not None else None
        else:
            subset_raw_sound = None
            x0_tokens_sound = None
            fps_sound = None

        return GenerationDataClean(
            batch_size=limit - start,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=subset_raw_vision,
            raw_state_action=subset_raw_action,
            raw_state_sound=subset_raw_sound,
            x0_tokens_vision=subset_x0_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            camera_conditioning=(
                gen_data_clean.camera_conditioning[start:limit]
                if gen_data_clean.camera_conditioning is not None
                else None
            ),
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
            num_vision_items_per_sample=subset_num_items,
        )

    @torch.no_grad()
    def forward(self, xt, t):
        pass

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], iteration: int = 1) -> GenerationDataClean:
        """
        - Get raw data of different modalities from databatch
        - Tokenize into corresponding latents
        - Load other conditioning information if any (fps, etc.)
        """
        is_image_batch = self.is_image_batch(data_batch)
        sample_vision_list = data_batch[self.input_image_key if is_image_batch else self.input_video_key]


        if "num_vision_items_per_sample" not in data_batch:
            has_multiple_vision_per_sample = any(
                isinstance(v, (list, tuple)) for v in sample_vision_list
            )
            num_vision_items_per_sample: list[int] | None = (
                [len(v) for v in sample_vision_list] if has_multiple_vision_per_sample else None
            )

            data_batch["num_vision_items_per_sample"] = num_vision_items_per_sample

            if has_multiple_vision_per_sample:
                media_key = self.input_video_key if not is_image_batch else self.input_image_key
                data_batch[media_key] = [item.unsqueeze(0) for sublist in sample_vision_list for item in sublist]
                if data_batch[media_key][0].dtype == torch.float32 and not is_image_batch:
                    data_batch["is_preprocessed"] = (
                        True
                    )
        else:
            num_vision_items_per_sample = data_batch["num_vision_items_per_sample"]

        batch_size = (
            len(sample_vision_list) if num_vision_items_per_sample is None else len(num_vision_items_per_sample)
        )
        if hasattr(self, "net"):
            self.net._track_fb_num_items = num_vision_items_per_sample
            if getattr(self.config.rectified_flow_training_config, "tracking_camera_enabled", False):
                self.net._track_fb_intr = data_batch.get("track_intrinsics")
                self.net._track_fb_extr = data_batch.get("track_extrinsics")
                self.net._track_fb_tcf = int(self.tokenizer_vision_gen.temporal_compression_factor)
            else:
                self.net._track_fb_intr = self.net._track_fb_extr = None

        log_enc_time = False
        timer = None
        if TRAINING:
            import wandb

            log_enc_time = iteration % self.log_enc_time_every_n == 0 and wandb.run
            if log_enc_time:
                timer = Timer(unit="s")
                timer.start()
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        raw_state_vision = data_batch[self.input_image_key if is_image_batch else self.input_video_key]
        x0_tokens_vision = [
            self.encode(raw_state_vision_i).contiguous().float() for raw_state_vision_i in raw_state_vision
        ]

        frame_size = data_batch.get("image_size", None)
        if frame_size is not None:
            x0_tokens_vision = self._remove_padding_from_latent(x0_tokens_vision, frame_size)

        raw_state_action, action_domain_id = self._normalize_action_databatch(data_batch)
        x0_tokens_action = raw_state_action
        raw_action_dim = data_batch.get("raw_action_dim", None)

        camera_keys = (
            "camera_c2w",
            "camera_intrinsics",
            "camera_distortion",
            "camera_model",
            "camera_image_size",
        )
        present_camera_keys = [key for key in camera_keys if key in data_batch]
        camera_conditioning: list[dict[str, torch.Tensor]] | None = None
        if self.config.camera_conditioning_enabled:
            missing_camera_keys = sorted(set(camera_keys) - set(present_camera_keys))
            if missing_camera_keys:
                raise ValueError(
                    "Camera conditioning is incomplete; missing " + ", ".join(missing_camera_keys)
                )

            camera_conditioning = []
            for sample_idx in range(batch_size):
                sample_camera: dict[str, torch.Tensor] = {}
                for key in camera_keys:
                    value = data_batch[key]
                    if isinstance(value, (list, tuple)):
                        sample_value = value[sample_idx]
                    elif isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                        sample_value = value[sample_idx]
                    elif batch_size == 1 and isinstance(value, torch.Tensor):
                        sample_value = value
                    else:
                        raise ValueError(
                            f"Cannot align {key} with batch size {batch_size}: "
                            f"type={type(value).__name__}, shape={getattr(value, 'shape', None)}"
                        )
                    while isinstance(sample_value, (list, tuple)) and len(sample_value) == 1:
                        sample_value = sample_value[0]
                    if not isinstance(sample_value, torch.Tensor):
                        sample_value = torch.as_tensor(sample_value)
                    sample_camera[key] = sample_value
                camera_conditioning.append(sample_camera)

        self._normalize_sound_databatch_inplace(data_batch)
        raw_state_sound = data_batch.get("sound", None)
        if raw_state_sound is not None and self.tokenizer_sound_gen is not None:
            x0_tokens_sound = [self.encode_sound(s).contiguous().float() for s in raw_state_sound]
        else:
            x0_tokens_sound = None

        fps_raw = data_batch.get("conditioning_fps", None)
        if isinstance(fps_raw, list):
            fps_raw = torch.stack(fps_raw).flatten()
        fps_vision = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None
        fps_action = fps_raw.to(**self.tensor_kwargs) if fps_raw is not None else None

        if x0_tokens_sound is not None:
            sound_batch_size = len(x0_tokens_sound)
            fps_sound = torch.full(
                (sound_batch_size,),
                self._get_sound_fps_for_rope(),
                dtype=torch.float32,
            ).to(**self.tensor_kwargs)
        else:
            fps_sound = None

        if TRAINING and log_enc_time and timer is not None:
            timer.end()
            elapsed = timer.get_cuda_time()
            h, w = raw_state_vision[0].shape[-2], raw_state_vision[0].shape[-1]
            resolution_label = "unknown"
            for res_name, aspect_ratios in VIDEO_RES_SIZE_INFO.items():
                if (h, w) in aspect_ratios.values():
                    resolution_label = res_name
                    if res_name == "704":
                        if (h, w) in VIDEO_RES_SIZE_INFO.get("720", {}).values():
                            resolution_label = "720"
                    break
            wandb.log(
                {
                    f"timer/encoding_{resolution_label}p": elapsed,
                    "timer/encoding": elapsed,
                },
                step=iteration,
            )
        return GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=is_image_batch,
            raw_state_vision=raw_state_vision,
            raw_state_action=raw_state_action,
            raw_state_sound=raw_state_sound,
            x0_tokens_vision=x0_tokens_vision,
            x0_tokens_action=x0_tokens_action,
            x0_tokens_sound=x0_tokens_sound,
            fps_vision=fps_vision,
            camera_conditioning=camera_conditioning,
            fps_action=fps_action,
            fps_sound=fps_sound,
            action_domain_id=action_domain_id,
            num_vision_items_per_sample=num_vision_items_per_sample,
            raw_action_dim=raw_action_dim,
        )

    def _normalize_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str | None = None
    ) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.
            input_key (str | None): The key for the video tensor in the data_batch. Defaults to
                `self.input_video_key` if not provided.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"
        input_key = self.input_video_key if input_key is None else input_key
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i]), "Video data is not in float format."
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001)), (
                        f"Video data is not in the range [-1, 1]. get data range "
                        f"[{data_batch[input_key][i].min()}, {data_batch[input_key][i].max()}]"
                    )
            else:
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        item = [item]
                    assert item[0].dtype == torch.uint8, "Video data is not in uint8 format."
                    data_batch[input_key][i] = torch.stack(item).to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _normalize_action_databatch(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Extract dense action and domain_id lists from the data batch.

        The joint dataloader produces action and domain_id data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a
        single-element list).  This method unwraps inner lists and filters
        out ``None`` entries to produce dense lists suitable for the model,
        **without mutating** ``data_batch``.

        Returns:
            (dense_action, dense_domain_id): Each is a list of device tensors
            containing only non-None entries, or ``None`` if all entries are
            ``None`` / the key is absent.
        """
        dense_action = unwrap_and_densify(data_batch.get("action", None), self.tensor_kwargs)
        dense_domain_id = unwrap_and_densify(
            data_batch.get("domain_id", None), {"device": self.tensor_kwargs["device"]}
        )
        return dense_action, dense_domain_id

    def _normalize_sound_databatch_inplace(self, data_batch: dict[str, torch.Tensor]) -> None:
        """Flatten and densify nested sound lists in-place.

        The joint dataloader produces sound data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a single-element
        list).  This method:

        1. Unwraps inner lists: ``[[t], [None], [t]]`` -> ``[t, None, t]``
        2. Clears ``sequence_plan.has_sound`` for samples whose sound is ``None``
           (kept aligned by ``custom_collate_fn`` preserving ``None`` placeholders).
        3. Filters out None entries: ``[t, None, t]`` -> ``[t, t]``
        4. Moves tensors to the model device.
        5. Sets ``data_batch["sound"]`` to ``None`` if no valid sound data remains.

        Alignment invariant: ``custom_collate_fn`` keeps the ``"sound"`` key
        as a list with ``None`` placeholders for samples that lack audio (e.g.
        audio-extraction failures), so the unwrapped ``raw_state_sound`` is
        1:1 with ``sequence_plan``.  ``SoundSequencePlanBuilder`` already sets
        each plan's ``has_sound`` according to that sample's actual sound
        presence, so clearing flags for ``None`` slots here is just defensive.
        """
        raw_state_sound = data_batch.get("sound", None)
        sequence_plans = data_batch.get("sequence_plan", None)
        sound_enabled = self.tokenizer_sound_gen is not None

        def _disable_sound_on_plans() -> None:
            if isinstance(sequence_plans, list):
                for plan in sequence_plans:
                    if hasattr(plan, "has_sound"):
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []

        if not isinstance(raw_state_sound, list) or len(raw_state_sound) == 0:
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        if isinstance(raw_state_sound[0], list):
            raw_state_sound = [item[0] if isinstance(item, list) else item for item in raw_state_sound]

        if not sound_enabled:
            _disable_sound_on_plans()
            data_batch["sound"] = None
            return

        if isinstance(sequence_plans, list):
            if len(sequence_plans) == len(raw_state_sound):
                for plan, sound in zip(sequence_plans, raw_state_sound, strict=True):
                    if hasattr(plan, "has_sound") and sound is None:
                        plan.has_sound = False
                        plan.condition_frame_indexes_sound = []
            else:
                log.warning(
                    f"Sound/plan length mismatch ({len(sequence_plans)} plans vs "
                    f"{len(raw_state_sound)} sound entries). Disabling sound for "
                    "this batch.  Check that custom_collate_fn preserves the "
                    "'sound' key with None placeholders."
                )
                _disable_sound_on_plans()
                data_batch["sound"] = None
                return

        raw_state_sound = [
            s.to(self.tensor_kwargs["device"]) for s in raw_state_sound if s is not None
        ]

        if len(raw_state_sound) == 0:
            _disable_sound_on_plans()
            data_batch["sound"] = None
        else:
            data_batch["sound"] = raw_state_sound

    def _augment_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        """
        Augments image tensors by adding a temporal dimension (B, C, H, W) -> (B, C, 1, H, W).

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the image data.
            input_key (str | None): The key for the image tensor. Defaults to `self.input_image_key`.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"

        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert data_batch[input_key][i].shape[2] == 1, (
                        f"Image data is claimed be augmented while its shape is {data_batch[input_key][i].shape} for sample {i}"
                    )
                return
            else:
                new_image_tensor_list = []
                for i in range(len(data_batch[input_key])):
                    for img_tensor in data_batch[input_key][i]:
                        img_tensor = rearrange(img_tensor, "c h w -> 1 c 1 h w").contiguous()
                        if img_tensor.dtype == torch.uint8:
                            img_tensor = img_tensor.to(**self.tensor_kwargs) / 127.5 - 1.0
                        new_image_tensor_list.append(img_tensor)
                data_batch[input_key] = new_image_tensor_list
                data_batch[IS_PREPROCESSED_KEY] = True


    def state_dict(self, prefix: str = "", **kwargs) -> Dict[str, Any]:
        final_state_dict = self.net.state_dict(prefix=prefix + "net.", **kwargs)
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix=prefix + "net_ema.", **kwargs)
            final_state_dict.update(ema_state_dict)
        return final_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.

        Parameters:
            state_dict (Mapping[str, Any]): A dictionary containing separate state
                dictionaries for the model and potentially for an EMA version of the model
                under the keys 'net' and 'net_ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in
                the state dict match exactly those in the model and EMA model (if applicable).
                Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary
                directly rather than matching keys one-by-one. This is typically used when loading
                parts of state dicts or using customized loading procedures. Defaults to False.
        """
        if not strict:
            raise ValueError("Strict mode is required for OmniMoTModel load_state_dict")
        if assign:
            raise ValueError("Assign mode is not supported for OmniMoTModel load_state_dict")

        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        state_dict = _reg_state_dict

        reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=True, assign=False)
        missing_keys = reg_results.missing_keys
        unexpected_keys = reg_results.unexpected_keys

        if self.config.ema.enabled:
            ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(_ema_state_dict, strict=True, assign=False)
            missing_keys += ema_results.missing_keys
            unexpected_keys += ema_results.unexpected_keys
        else:
            assert len(_ema_state_dict) == 0, f"EMA is disabled but EMA state dict is not empty: {len(_ema_state_dict)}"

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)


    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """Check if the data_batch contains images (vs. videos).

        We handle two types of data_batch: one from a joint_dataloader where "dataset_name" can
        differentiate image_batch and video_batch, another from a single dataloader which we
        assume as video_data by default.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_video_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_video_key should be present in the data_batch."
        )
        return is_image

    def _maybe_apply_prompt_upsampling(self, data_batch: dict, upsample_task: str | None = None, **kwargs) -> dict:
        """Prompt upsampling is not part of VPM; the batch is returned unchanged."""
        if upsample_task:
            raise NotImplementedError("prompt upsampling is not included in VPM")
        return data_batch


    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        fps_vision: torch.Tensor | None = None,
        fps_action: torch.Tensor | None = None,
        fps_sound: torch.Tensor | None = None,
        memory: MemoryState | None = None,
    ) -> dict:
        """
        Runs the MoT network on a packed multi-modal sequence to predict velocity (v) targets.

        Args:
            data_batch_packed: PackedSequence from `pack_input_sequence(...)`.
            fps_vision: Optional FPS tensor used for RoPE FPS modulation (if enabled in config).
            fps_action: Optional FPS tensor used for action RoPE FPS modulation (if enabled in config).
            fps_sound: Optional FPS tensor for sound RoPE modulation (e.g., sound_latent_fps=25).
            memory: Optional pre-built MemoryState for autoregressive generation
                or KV-cache training.

        Returns:
            dict containing:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity prediction for action modality (if action_gen enabled).
                - "preds_sound": Velocity prediction for sound modality (if sound_gen enabled).
                - "lbl_metadata_und": Load balancing metadata for understanding pathway (if present).
                - "lbl_metadata_gen": Load balancing metadata for generation pathway (if present).
        """
        net = net or self.net
        out_net = net(
            packed_seq=data_batch_packed,
            fps_vision=fps_vision,
            fps_action=fps_action,
            fps_sound=fps_sound,
            memory=memory,
        )
        output_dict = dict()
        output_dict["preds_vision"] = out_net["preds_vision"]
        if self.config.action_gen and "preds_action" in out_net:
            output_dict["preds_action"] = out_net["preds_action"]
        if self.config.sound_gen and "preds_sound" in out_net:
            output_dict["preds_sound"] = out_net["preds_sound"]
        for key, value in out_net.items():
            if "lbl_metadata_" in key or key.startswith("track_"):
                output_dict[key] = value

        return output_dict

    def _tokenize_captions(
        self,
        captions: list[str],
        *,
        use_system_prompt: bool,
        system_prompt: str | None,
        is_video: bool,
    ) -> list[list[int]]:
        """Per-caption chat-template tokenization (ragged, no padding).

        Single source of truth for the per-caption loop shared by
        :meth:`tokenize_text` (which adds padding + tensor-wrapping) and
        :meth:`_get_inference_text_tokens` (which feeds the diffusion
        sampling pipeline that natively consumes ``list[list[int]]``).

        Args:
            captions: Raw text prompts.
            use_system_prompt: Already-resolved boolean (callers handle any
                ``None`` fallback against ``self.vlm_config``).
            system_prompt: Explicit system prompt that overrides
                ``use_system_prompt`` / ``is_video`` when supplied.
            is_video: Selects the video vs image default system prompt
                when ``use_system_prompt=True`` and no explicit
                ``system_prompt`` is given.
        """
        return [
            tokenize_caption(
                c,
                self.vlm_tokenizer,
                is_video=is_video,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            for c in captions
        ]

    @torch.no_grad()
    def tokenize_text(
        self,
        text: str | list[str],
        *,
        use_system_prompt: bool | None = None,
        system_prompt: str | None = None,
        is_video: bool = False,
        pad_token_id: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Tokenize one or more text prompts into a ``[B, T]`` long tensor.

        Wraps each prompt in the VLM's chat template via
        :func:`tokenize_caption` (the same helper used by
        :meth:`_get_inference_text_tokens` during conditional sampling), so
        the resulting ids match what the model sees during training /
        inference (e.g. ``<|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n``
        for Qwen3-style chat models).

        When multiple prompts of different lengths are supplied, the result
        is right-padded with ``pad_token_id``.  Right-padding is the natural
        choice for prefill into :meth:`generate_reasoner_text`, which reads
        the last position's logits to predict the first new token; for that
        to be meaningful, callers should normally pass either a single
        prompt or a list of equal-length prompts.

        Args:
            text: A single string or a list of strings, one per batch item.
            use_system_prompt: Override for ``self.vlm_config.use_system_prompt``.
                When ``None``, falls back to the VLM config flag.
            system_prompt: When supplied, this exact string is used as the
                system prompt and overrides ``use_system_prompt`` /
                ``is_video``.
            is_video: Selects the video vs image default system prompt
                when ``use_system_prompt=True`` and no explicit
                ``system_prompt`` is given.
            pad_token_id: Pad id for right-padding.  Defaults to
                ``self.vlm_tokenizer.pad_token_id`` when set, else
                ``self.llm_special_tokens["eos_token_id"]``, else ``0``.
            device: Optional device for the returned tensor.

        Returns:
            A ``torch.long`` tensor of shape ``[B, T_max]`` ready to be fed
            into :meth:`generate_reasoner_text` or any other token-driven
            entry point.
        """
        prompts: list[str] = [text] if isinstance(text, str) else list(text)
        if len(prompts) == 0:
            raise ValueError("tokenize_text requires at least one prompt.")
        resolved_use_system_prompt: bool = (
            bool(getattr(self.vlm_config, "use_system_prompt", False))
            if use_system_prompt is None
            else use_system_prompt
        )

        token_lists = self._tokenize_captions(
            prompts,
            use_system_prompt=resolved_use_system_prompt,
            system_prompt=system_prompt,
            is_video=is_video,
        )

        if pad_token_id is None:
            candidate = getattr(self.vlm_tokenizer, "pad_token_id", None)
            if candidate is None:
                candidate = self.llm_special_tokens.get("eos_token_id")
            pad_id: int = int(candidate) if candidate is not None else 0
        else:
            pad_id = int(pad_token_id)

        max_len = max(len(ids) for ids in token_lists)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_lists]
        out = torch.tensor(padded, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out

    @torch.no_grad()
    def detokenize_text(
        self,
        token_ids: torch.Tensor | list[int] | list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> str | list[str]:
        """Decode token ids produced by :meth:`generate_reasoner_text` back to text.

        Accepts the natural output shapes:
          - 1D tensor / ``list[int]``  → returns a single ``str``.
          - 2D tensor / ``list[list[int]]`` → returns ``list[str]`` (one
            per row).

        Args:
            token_ids: Ids to decode.
            skip_special_tokens: Forwarded to
                ``self.vlm_tokenizer.decode``; when ``True`` (default),
                strips chat-template / vision boundary specials so the
                returned string contains only model-generated content.
        """
        raw: Any = token_ids
        if hasattr(raw, "tolist"):
            if hasattr(raw, "dim") and raw.dim() not in (1, 2):
                raise ValueError(f"token_ids tensor must be 1D or 2D, got shape {tuple(raw.shape)}")
            ids_list: list = raw.tolist()
        elif isinstance(raw, list):
            ids_list = raw
        else:
            raise TypeError(f"Unsupported token_ids type: {type(token_ids)}")

        if len(ids_list) == 0:
            return ""

        decode = self.vlm_tokenizer.decode
        if isinstance(ids_list[0], list):
            return [decode(row, skip_special_tokens=skip_special_tokens) for row in ids_list]
        return decode(ids_list, skip_special_tokens=skip_special_tokens)

    def _broadcast_outputs_to_parallel_groups(self, outputs: list[str]) -> list[str]:
        """Pin ``outputs`` to the lowest-numbered rank's value within each CP / CFGP group.

        Under context parallelism (CP) or CFG parallelism (CFGP), every
        rank within a parallel group is processing the same logical
        batch — CP splits the context dimension across ranks, CFGP splits
        the cfg/uncfg pair.  In a deterministic AR decode (e.g. greedy)
        the per-rank token streams should agree exactly, but numerical
        noise (kernel selection, FSDP all-gather ordering, RNG state in
        the sampling controls, EMA scope) can let those ranks emit
        token streams that differ at one or two greedy-tied positions.
        The public contract for ``generate_reasoner_text`` is "list[str]
        in, list[str] out" with a single canonical response per prompt
        across the parallel group, so we resolve any divergence by
        broadcasting rank 0's strings to every other rank within the
        group via :func:`torch.distributed.broadcast_object_list`.

        Order matches the seed-broadcast pattern used elsewhere in this
        class (CP first, then CFGP) so the final outputs are consistent
        across both groups when both are enabled.  No-op when neither CP
        nor CFGP is enabled (the typical case for pure-FSDP runs) — and
        also when ``parallel_dims`` is unset, e.g. CPU-only smoke tests.
        """
        if self.parallel_dims is None:
            return outputs
        if self.parallel_dims.cp_enabled and self.parallel_dims.cp_mesh is not None:
            cp_group = self.parallel_dims.cp_mesh.get_group()
            cp_bucket: list[Any] = [outputs]
            dist.broadcast_object_list(cp_bucket, group=cp_group, group_src=0)
            outputs = cp_bucket[0]
        if self.parallel_dims.cfgp_enabled and self.parallel_dims.cfgp_mesh is not None:
            cfgp_group = self.parallel_dims.cfgp_mesh.get_group()
            cfgp_bucket: list[Any] = [outputs]
            dist.broadcast_object_list(cfgp_bucket, group=cfgp_group, group_src=0)
            outputs = cfgp_bucket[0]
        return outputs

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.decode(latent)

    @torch.no_grad()
    def encode_sound(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform into latent tokens.

        Args:
            waveform: Audio tensor of shape (C, N). A batch dim is added/removed
                      internally since AVAE expects (B, C, N).
                      Mono audio is duplicated to stereo if the tokenizer expects 2 channels.
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        expected_channels = self.tokenizer_sound_gen.audio_channels
        if waveform.shape[0] == 1 and expected_channels == 2:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > expected_channels:
            waveform = waveform[:expected_channels]
        latent = self.tokenizer_sound_gen.encode(waveform.unsqueeze(0))
        return latent.squeeze(0)

    @torch.no_grad()
    def decode_sound(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode sound latent tokens back to waveform.

        Args:
            latent: Sound latent tensor of shape (C, T). A batch dim is added/removed
                    internally since AVAE expects (B, C, T).
        """
        assert self.tokenizer_sound_gen is not None, "Sound tokenizer not initialized"
        waveform = self.tokenizer_sound_gen.decode(latent.unsqueeze(0))
        return waveform.squeeze(0)

    def _get_sound_fps_for_rope(self) -> float:
        """Compute the sound FPS to pass to RoPE for temporal alignment with video.

        Returns the sound tokenizer's latent rate (e.g., 25 Hz for 48kHz/1920 hop).
        This is passed as input_fps to the sound RoPE's generate_embeddings(), where
        the FPS modulation formula aligns sound indices with video indices.
        """
        return float(self.config.sound_latent_fps)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return (
            height // self.tokenizer_vision_gen.spatial_compression_factor,
            width // self.tokenizer_vision_gen.spatial_compression_factor,
        )

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def add_lora(
        self,
        network: torch.nn.Module,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
    ) -> torch.nn.Module:
        """Pre-FSDP LoRA injection — see :func:`inject_lora_pre_fsdp` for details."""
        from cosmos_framework.utils.vfm.lora import inject_lora_pre_fsdp

        self.lora_alpha = lora_alpha
        return inject_lora_pre_fsdp(
            network,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )

    def _init_lora_weights_post_materialization(self, network: torch.nn.Module) -> None:
        """Post-materialization LoRA init — see :func:`init_lora_weights_post_materialization`."""
        from cosmos_framework.utils.vfm.lora import init_lora_weights_post_materialization

        init_lora_weights_post_materialization(network)


def _broadcast_seed(seed: list[int], group: dist.ProcessGroup, rank: int) -> list[int]:
    if rank == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.int64, device=DEVICE)
    else:
        seed_tensor = torch.zeros(len(seed), dtype=torch.int64, device=DEVICE)

    dist.broadcast(seed_tensor, group=group, group_src=0)
    return seed_tensor.tolist()
