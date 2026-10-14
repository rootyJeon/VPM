# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


from typing import Any

import attrs

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.ema import EMAConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.vlm import VLMConfig


@attrs.define(slots=False)
class DiffusionExpertConfig:
    timestep_range: float = 1.0
    load_weights_from_pretrained: bool = True

    patch_spatial: int = 2
    max_vae_latent_side_after_patchify: int = (
        20
    )
    position_embedding_type: str = "3d_rope"
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    enable_fps_modulation: bool = False
    base_fps: int = 24
    unified_3d_mrope_reset_spatial_ids: bool = True
    unified_3d_mrope_temporal_modality_margin: int = 0


@attrs.define(slots=False)
class LBLConfig:
    method: str = "local"

    coeff_und: float | None = None
    coeff_gen: float | None = None


@attrs.define(slots=False)
class RectifiedFlowTrainingConfig:
    shift: Any = 5
    use_dynamic_shift: bool = False
    train_time_image_distribution: str = "logitnormal"
    train_time_video_distribution: str = "logitnormal"
    train_time_action_distribution: str = "logitnormal"
    train_time_sound_distribution: str = "logitnormal"
    train_time_weight: str = "uniform"
    loss_scale: float = 1.0
    image_loss_scale: float | None = None
    sound_loss_scale: float | None = None
    use_high_sigma_strategy: bool = False
    high_sigma_ratio: float = 0.05
    high_sigma_timesteps_min: int = 995
    high_sigma_timesteps_max: int = 1000
    use_discrete_rf: bool = False

    action_loss_weight: float = 10.0

    independent_action_schedule: bool = False
    shift_action: int | None = None
    use_high_sigma_strategy_action: bool = False

    independent_sound_schedule: bool = False
    shift_sound: int | None = None
    use_high_sigma_strategy_sound: bool = False

    normalize_loss_by_active: bool = False

    cond_noise_aug_std: float = 0.0

    student_forcing_enabled: bool = False
    student_forcing_prob: float = 0.5
    student_forcing_rollout_chunks: int = 2
    student_forcing_num_steps: int = 6
    student_forcing_guidance: float = 1.0
    student_forcing_start_iter: int = 0
    student_forcing_past_latents: int = 9
    variable_future_enabled: bool = False
    variable_future_choices_px: list = attrs.field(factory=list)

    tracking_enabled: bool = False
    tracking_layer: int = 18
    tracking_weight: float = 0.01
    tracking_corr_weight: float = 0.0
    tracking_motion_weight_enabled: bool = False
    tracking_motion_gamma: float = 1.0
    tracking_motion_floor: float = 0.3
    tracking_motion_eps: float = 0.02
    tracking_motion_cap: float = 0.0
    tracking_vel_weight: float = 0.0
    tracking_seq_cap: float = 0.0
    tracking_seq_sigma_weight: bool = False
    tracking_min_valid_points: int = 0
    tracking_feedback_enabled: bool = False
    tracking_feedback_layer: int = 22
    tracking_camera_enabled: bool = False
    tracking_upsample: int = 1
    tracking_depth_enabled: bool = False
    tracking_depth_weight: float = 1.0
    tracking_depth_huber_delta: float = 0.2
    tracking_num_points: int = 256
    tracking_latent_dim: int = 128
    tracking_iters: int = 4
    tracking_corr_radius: int = 3
    tracking_corr_levels: int = 4
    tracking_num_virtual_tracks: int = 64
    tracking_use_checkpoint: bool = True
    tracking_freeze_backbone: bool = True
    tracking_trainable_substrings: list = attrs.field(
        factory=lambda: ["self_attn"]
    )


@attrs.define(slots=False)
class RectifiedFlowInferenceConfig:
    scheduler_type: str = "unipc"
    num_train_timesteps: int = 1000
    shift: int = 1
    use_dynamic_shifting: bool = False


@attrs.define(slots=False)
class FixedStepSamplerConfig:
    """Config for the fixed-step sampler used by distilled models.

    Uses a fixed sigma schedule instead of a smooth multi-step solver.

    Mirrors the constructor args of ``FixedStepSampler``.
    """

    t_list: list[float] = [0.999, 0.75, 0.5, 0.25]
    sample_type: str = "ode"


@attrs.define(slots=False)
class OmniMoTModelConfig:
    """
    Config for Omni MoT model.
    """

    tokenizer: LazyDict = None
    net: LazyDict = None
    ema: EMAConfig = EMAConfig()

    parallelism: ParallelismConfig = ParallelismConfig()

    compile: CompileConfig = CompileConfig()

    activation_checkpointing: ActivationCheckpointingConfig = ActivationCheckpointingConfig()

    precision: str = "bfloat16"

    lora_enabled: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"

    camera_conditioning_enabled: bool = False
    camera_code_dim: int = 512
    camera_ray_grid: str = "latent16"
    camera_roce_mode: str = "none"
    camera_roce_output_position: str = "post_o"

    action_expert: dict | None = None

    rectified_flow_training_config: RectifiedFlowTrainingConfig = RectifiedFlowTrainingConfig()
    rectified_flow_inference_config: RectifiedFlowInferenceConfig = RectifiedFlowInferenceConfig()

    fixed_step_sampler_config: FixedStepSamplerConfig | None = None

    vlm_config: VLMConfig = VLMConfig()
    diffusion_expert_config: DiffusionExpertConfig = DiffusionExpertConfig()
    input_video_key: str = "video"
    input_image_key: str = "images"
    input_caption_key: str = "ai_caption"

    state_ch: int = 16
    state_t: int = 8
    latent_downsample_factor: int = 8
    resolution: str = "512"
    max_num_tokens_after_packing: int = 13312

    joint_attn_implementation: str = (
        "two_way"
    )

    natten_parameter_list: list | None = None

    video_temporal_causal: bool = False
    causal_training_strategy: str = attrs.field(
        default="none",
        validator=attrs.validators.in_({"none", "teacher_forcing", "diffusion_forcing", "teacher_forcing_dcm"}),
    )

    lbl: LBLConfig = LBLConfig()

    vision_gen: bool = True

    action_gen: bool = False
    max_action_dim: int = 32
    num_embodiment_domains: int = 32

    sound_gen: bool = False
    sound_tokenizer: LazyDict | None = None
    sound_dim: int | None = None
    sound_latent_fps: int = 25

    log_enc_time_every_n: int = 100
