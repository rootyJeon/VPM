# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.model.vfm.mot.parallelize_unified_mot import parallelize_unified_mot
from cosmos_framework.utils import log
from cosmos_framework.utils.vfm.parallelism import ParallelDims


def apply_compile(model: torch.nn.Module, config: CompileConfig):
    """Apply torch.compile to the VFM encode/decode heads.

    The MoT-side ``compile_dynamic`` knob on ``CompileConfig`` intentionally
    does **not** propagate here.  The VFM encode/decode paths have no graph
    breaks and their input shapes are stable across a prompt, so we always
    trace them as a single dynamic graph (``fullgraph=True, dynamic=True``).
    This keeps AR inference (which sets ``compile_dynamic=False`` on MoT for
    shape-specialized kernels) from accidentally regressing the VFM compile.
    """

    inductor_options = {}
    if config.max_autotune_pointwise:
        inductor_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        inductor_options["coordinate_descent_tuning"] = True

    compile_options = {
        "fullgraph": True,
        "dynamic": True,
        "mode": "reduce-overhead" if config.use_cuda_graphs else None,
        "options": inductor_options or None,
    }

    model._encode_text = torch.compile(model._encode_text, **compile_options)
    model._encode_vision = torch.compile(model._encode_vision, **compile_options)
    model._encode_action = torch.compile(model._encode_action, **compile_options)
    model._decode_vision = torch.compile(model._decode_vision, **compile_options)
    model._decode_action = torch.compile(model._decode_action, **compile_options)
    return model


def apply_compile_action_expert(action_dit: torch.nn.Module, config: CompileConfig) -> None:
    """Per-block ``torch.compile`` of the ActionDiT trunk (option B expert).

    Mirrors ``parallelize_unified_mot.apply_compile``: each ``ActionExpertBlock``
    traces fullgraph — ``action_joint_attention`` uses the same varlen-attention
    + LSE-merge primitives as the compile-verified backbone two_way path.  Only
    ``forward`` (the one-way begin+finish fusion) is compiled; the joint
    co-step's direct ``begin``/``finish`` calls proxy through OptimizedModule to
    the eager methods, unchanged.
    """
    if not hasattr(action_dit, "blocks"):
        log.info(
            f"[compile] skipping per-block action-expert compile: {type(action_dit).__name__} has no .blocks"
        )
        return

    compile_options = {}
    if config.max_autotune_pointwise:
        compile_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        compile_options["coordinate_descent_tuning"] = True

    for block_id, block in action_dit.blocks.named_children():
        block = torch.compile(
            block,
            fullgraph=True,
            dynamic=config.compile_dynamic,
            mode="reduce-overhead" if config.use_cuda_graphs else None,
            options=compile_options or None,
        )
        action_dit.blocks.register_module(block_id, block)


def parallelize_vfm_network(
    model: torch.nn.Module,
    parallel_dims: ParallelDims | None,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
) -> torch.nn.Module:
    """Optimize the model using FSDP, CP, activation checkpointing, and torch.compile.

    FSDP reduces memory usage by sharding the model parameters across multiple GPUs.
    Activation checkpointing reduces memory usage by selectively checkpointing only
    the outputs of each layer. Torch.compile compiles the model for faster training.

    Args:
        model: The Cosmos3 VFM network.
        parallel_dims: Device mesh / parallelism descriptor.
        compile_config: Compile switches (enabled, compiled_region, etc.).
        ac_config: Selective activation-checkpointing policy, typically
            ``OmniMoTModelConfig.sac``. Forwarded to
            ``parallelize_unified_mot``; ``None`` falls back to the
            ``ActivationCheckpointingConfig`` defaults.
    """
    if parallel_dims is not None and parallel_dims.cp_enabled:
        model.parallel_dims = parallel_dims

    model.language_model = parallelize_unified_mot(
        model.language_model,
        parallel_dims=parallel_dims,
        compile_config=compile_config,
        ac_config=ac_config,
    )

    if compile_config.enabled and compile_config.compiled_region == "all":
        model = apply_compile(model, compile_config)

    if compile_config.enabled and hasattr(model, "action_dit"):
        apply_compile_action_expert(model.action_dit, compile_config)

    if parallel_dims is not None and parallel_dims.dp_enabled:
        ignored_params = set()
        if model.latent_pos_embed is not None:
            ignored_params.update(model.latent_pos_embed.parameters())

        model = fully_shard(
            module=model,
            mesh=parallel_dims.dp_mesh,
            ignored_params=ignored_params,
        )

        register_fsdp_forward_method(model, "generate_reasoner_text")

        if getattr(model, "tracking_head", None) is not None:
            register_fsdp_forward_method(model, "run_tracking_head")

    return model
