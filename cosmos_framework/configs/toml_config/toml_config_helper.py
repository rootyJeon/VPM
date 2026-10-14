# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Schema-agnostic helpers for the structured-TOML flow.

``build_hydra_overrides(toml_dict)`` walks the TOML dict and emits a
``["--", "experiment=<name>", "dotted.path=value", ...]`` list compatible
with ``cosmos_framework.utils.config_helper.override``. Per-task path remapping lives
in ``PATH_REMAPS``.

This module knows nothing about the pydantic schema — the entry point that
ties validation + override-build + Hydra-compose together lives in
``sft_config.py``. TOML → schema validation goes through pydantic's
``BaseModel.model_validate`` directly there.
"""

from __future__ import annotations

from typing import Any


TASK_TO_BASE_CONFIG: dict[str, str] = {
    "vfm": "cosmos_framework/configs/base/config.py",
    "vlm": "cosmos_framework/configs/base/vlm/config.py",
}


PATH_REMAPS: dict[str, dict[tuple[str, ...], "tuple[str, ...] | None"]] = {
    "vfm": {
        ("model", "attn_implementation"): None,
        ("model", "backbone"): None,
        ("model",): ("model", "config"),
    },
    "vlm": {
        ("model", "max_num_tokens_after_packing"): None,
        ("model", "joint_attn_implementation"): None,
        ("model", "lora_enabled"): None,
        ("model", "lora_rank"): None,
        ("model", "lora_alpha"): None,
        ("model", "lora_target_modules"): None,
        ("model", "tokenizer"): None,
        ("dataloader_train", "seed"): None,
        ("optimizer", "eps"): None,
        ("scheduler", "verbosity_interval"): None,
        ("trainer", "callbacks", "compile_tokenizer"): None,
        ("model", "attn_implementation"): ("model", "config", "policy", "attn_implementation"),
        ("model", "ema"): ("model", "config", "ema"),
        ("model", "backbone"): ("model", "config", "policy", "backbone"),
        ("dataloader_train", "max_samples_per_batch"): ("dataloader_train", "max_batch_size"),
        ("dataloader_train", "max_sequence_length"): ("dataloader_train", "max_tokens"),
        ("model",): ("model", "config"),
    },
}


def _apply_remap(
    rules: dict[tuple[str, ...], "tuple[str, ...] | None"],
    path: list[str],
) -> "list[str] | None":
    """Greedy longest-prefix lookup against ``rules``.

    Returns the rewritten path (replacement + tail), ``None`` if a matched
    rule says skip, or the original path when no rule matches.
    """
    for n in range(len(path), 0, -1):
        key = tuple(path[:n])
        if key in rules:
            replacement = rules[key]
            if replacement is None:
                return None
            return list(replacement) + path[n:]
    return path


def build_hydra_overrides(toml_dict: dict) -> list[str]:
    """Walk a TOML dict and produce a Hydra override list compatible with
    ``cosmos_framework.utils.config_helper.override``.

    Each leaf path is routed through ``PATH_REMAPS[task]`` so dataclass
    paths land at the correct Hydra location for VFM vs VLM. ``job.task``
    and ``job.experiment`` are meta-fields — consumed here, not emitted.
    """
    overrides: list[str] = ["--"]

    job = dict(toml_dict.get("job", {}))
    task = job.pop("task", "vfm")
    experiment_name = job.pop("experiment", None)
    if not experiment_name:
        raise ValueError("[job].experiment is required in the TOML")
    overrides.append(f"experiment={experiment_name}")

    if task not in PATH_REMAPS:
        raise ValueError(
            f"[job].task={task!r} has no remap rules. "
            f"Valid values: {sorted(PATH_REMAPS)}"
        )
    rules = PATH_REMAPS[task]

    overlay = dict(toml_dict)
    overlay["job"] = job

    for top_key, val in overlay.items():
        _emit_with_remap(overrides, [top_key], val, rules)
    return overrides


def _emit_with_remap(
    out: list[str],
    prefix: list[str],
    value: Any,
    rules: dict[tuple[str, ...], "tuple[str, ...] | None"],
) -> None:
    if isinstance(value, dict):
        if not value:
            new_path = _apply_remap(rules, prefix)
            if new_path is not None:
                out.append(f"{'.'.join(new_path)}={{}}")
            return
        for k, v in value.items():
            _emit_with_remap(out, prefix + [k], v, rules)
        return
    if value == "???":
        return
    new_path = _apply_remap(rules, prefix)
    if new_path is None:
        return
    out.append(f"{'.'.join(new_path)}={_hydra_format(value)}")


def _hydra_format(v: Any, in_list: bool = False) -> str:
    """Convert a Python value to a Hydra CLI override RHS.

    ``in_list=True`` indicates the value is being emitted inside a list
    literal (``[a,b,c]``); strings then get single-quoted unconditionally
    so numeric-looking entries like ``"480"`` stay strings rather than
    being coerced to int by Hydra's list parser.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ",".join(_hydra_format(x, in_list=True) for x in v) + "]"
    if isinstance(v, str):
        if in_list or "," in v or " " in v:
            return f"'{v}'"
        return v
    return str(v)


