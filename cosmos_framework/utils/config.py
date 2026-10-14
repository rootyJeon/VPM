# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training config system for Imaginare4"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Dict, Optional, Type, TypeVar, Union

import attrs
import torch
from loguru import logger as logging

from cosmos_framework.utils.flags import TRAINING

try:
    from megatron.core import ModelParallelConfig

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils import distributed
from cosmos_framework.utils.misc import Color

T = TypeVar("T")


def _is_attrs_instance(obj: object) -> bool:
    """
    Helper function to check if an object is an instance of an attrs-defined class.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is an instance of an attrs-defined class, False otherwise.
    """
    return hasattr(obj, "__attrs_attrs__")


def make_freezable(cls: T) -> T:
    """
    A decorator that adds the capability to freeze instances of an attrs-defined class.

    NOTE: This requires the wrapped attrs to be defined with attrs.define(slots=False) because we need
    to hack on a "_is_frozen" attribute.

    This decorator enhances an attrs-defined class with the ability to be "frozen" at runtime.
    Once an instance is frozen, its attributes cannot be changed. It also recursively freezes
    any attrs-defined objects that are attributes of the class.

    Usage:
        @make_freezable
        @attrs.define(slots=False)
        class MyClass:
            attribute1: int
            attribute2: str

        obj = MyClass(1, 'a')
        obj.freeze()  # Freeze the instance
        obj.attribute1 = 2  # Raises AttributeError

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with added freezing capability.
    """

    if not hasattr(cls, "__dict__"):
        raise TypeError(
            "make_freezable cannot be used with classes that do not define __dict__. Make sure that the wrapped "
            "class was defined with `@attrs.define(slots=False)`"
        )

    original_setattr = cls.__setattr__

    def setattr_override(self, key, value) -> None:
        """
        Override __setattr__ to allow modifications during initialization
        and prevent modifications once the instance is frozen.
        """
        if hasattr(self, "_is_frozen") and self._is_frozen and key != "_is_frozen":
            raise AttributeError("Cannot modify frozen instance")
        original_setattr(self, key, value)

    cls.__setattr__ = setattr_override

    def freeze(self: object) -> None:
        """
        Freeze the instance and all its attrs-defined attributes.
        """
        for _, value in attrs.asdict(self, recurse=False).items():
            if _is_attrs_instance(value) and hasattr(value, "freeze"):
                value.freeze()
        self._is_frozen = True

    cls.freeze = freeze

    return cls


def _pretty_print_attrs_instance(obj: object, indent: int = 0, use_color: bool = False) -> str:
    """
    Recursively pretty prints attrs objects with color.
    """

    assert attrs.has(obj.__class__)

    lines: list[str] = []
    for attribute in attrs.fields(obj.__class__):
        value = getattr(obj, attribute.name)
        if attrs.has(value.__class__):
            if use_color:
                lines.append("   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ":")
            else:
                lines.append("   " * indent + "* " + attribute.name + ":")
            lines.append(_pretty_print_attrs_instance(value, indent + 1, use_color))
        else:
            if use_color:
                lines.append(
                    "   " * indent + Color.cyan("* ") + Color.green(attribute.name) + ": " + Color.yellow(value)
                )
            else:
                lines.append("   " * indent + "* " + attribute.name + ": " + str(value))
    return "\n".join(lines)


def pretty_print_overrides(overrides: Optional[list[str]] = None, use_color: bool = False) -> str:
    """
    Pretty prints overrides.
    """

    lines: list[str] = []
    lines.append(Color.cyan("* ") + Color.green("overrides") + ": ")
    for override in overrides:
        if override == "--":
            continue
        if override.startswith("~"):
            attribute_name = override[1:]
            attribute_value = None
        else:
            attribute_name, attribute_value = override.split("=")
        if use_color:
            lines.append("   " + Color.cyan("* ") + Color.green(attribute_name) + ": " + Color.yellow(attribute_value))
        else:
            lines.append("   " + "* " + attribute_name + ": " + str(attribute_value))

    return "\n".join(lines)


@make_freezable
@attrs.define(slots=False)
class ObjectStoreConfig:
    enabled: bool = False
    credentials: str = ""
    bucket: str = ""


@make_freezable
@attrs.define(slots=False)
class HFExportConfig:
    enabled: bool = False
    hf_repo_id: Optional[str] = None
    upload_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)
    export_every_n: int = attrs.field(default=1, validator=attrs.validators.ge(1))


@make_freezable
@attrs.define(slots=False)
class JobConfig:
    project: str = ""
    group: str = ""
    name: str = ""
    wandb_mode: str = "online"
    cluster: Optional[Any] = None

    @property
    def path(self) -> str:
        return f"{self.project}/{self.group}/{self.name}"

    @property
    def path_local(self) -> str:
        local_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output")
        return f"{local_root}/{self.path}"


@make_freezable
@attrs.define(slots=False)
class EMAConfig:
    enabled: bool = False
    beta: float = 0.9999
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class PowerEMAConfig:
    enabled: bool = False
    s: float = 0.1
    torch_compile_buffer_renaming: bool = False


@make_freezable
@attrs.define(slots=False)
class DDPConfig:
    find_unused_parameters: bool = False
    static_graph: bool = True
    broadcast_buffers: bool = True


@make_freezable
@attrs.define(slots=False)
class CuDNNConfig:
    deterministic: bool = False
    benchmark: bool = True


@make_freezable
@attrs.define(slots=False)
class JITConfig:
    enabled: bool = False
    input_shape: Union[list[int], None] = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    strict: bool = True


@make_freezable
@attrs.define(slots=False)
class CheckpointConfig:
    type: Optional[Dict] = None

    dcp_async_mode_enabled: bool = False

    save_to_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)

    save_iter: int = 999999999

    strict_resume: bool = True

    load_from_object_store: ObjectStoreConfig = attrs.field(factory=ObjectStoreConfig)

    load_path: str = ""


    load_training_state: bool = False

    only_load_scheduler_state: bool = False

    keys_to_skip_loading: list[str] = []

    jit: JITConfig = attrs.field(factory=JITConfig)

    verbose: bool = True

    keys_not_to_resume: list[str] = []

    broadcast_via_filesystem: bool = False
    load_ema_to_reg: bool = False

    enable_gcs_patch_in_boto3: bool = False

    hf_export: HFExportConfig = attrs.field(factory=HFExportConfig)


@make_freezable
@attrs.define(slots=False)
class NVTXConfig:
    """Config for NVTX ranges used in the main training loop.

    See tutorials/nanogpt for more details on how to integrate profiling into your model."""

    enabled: bool = False
    cuda_synchronize: bool = False


@make_freezable
@attrs.define(slots=False)
class StragglerDetectionConfig:
    """Config for the Straggler detection tool."""

    enabled: bool = False
    report_freq: int = 100
    profile_freq: int = 1
    max_diff: float = 2.0
    raise_error: bool = True
    analyze_forward: bool = True
    analyze_backward: bool = True
    analyze_optimizer: bool = True
    analyze_dataloading: bool = True
    save_s3: bool = False


@make_freezable
@attrs.define(slots=False)
class Profiling:
    enable_profiling: bool = False
    enable_nsys: bool = False
    enable_memory_snapshot: bool = False
    save_s3: bool = False
    profile_freq: int = 1
    profile_warmup: int = 3
    target_ranks: list[int] = list(range(8))
    record_shape: bool = False
    profile_memory: bool = False
    with_stack: bool = True
    with_modules: bool = True


@make_freezable
@attrs.define(slots=False)
class CompileConfig:
    """
    torch.compile config options passed to set_torch_compile_options function.
    """

    recompile_limit: int = 8
    use_duck_shape: bool = True


@make_freezable
@attrs.define(slots=False)
class TrainerConfig:
    if TRAINING:
        from cosmos_framework.trainer import ImaginaireTrainer
        from cosmos_framework.utils import callback

        type: Type[ImaginaireTrainer] = ImaginaireTrainer
        callbacks: LazyDict = LazyDict(
            dict(
                ema=L(callback.EMAModelCallback)(),
                progress_bar=L(callback.ProgressBarCallback)(),
                wandb=L(callback.WandBCallback)(),
            )
        )

    distributed_parallelism: str = "ddp"
    ddp: DDPConfig = attrs.field(factory=DDPConfig)
    cudnn: CuDNNConfig = attrs.field(factory=CuDNNConfig)
    seed: int = 0
    grad_scaler_args: dict = attrs.field(factory=lambda: dict(enabled=False))
    max_iter: int = 999999999
    logging_iter: int = 100
    timeout_period: int = 999999999
    memory_format: torch.memory_format = torch.preserve_format
    grad_accum_iter: int = 1
    straggler_detection: StragglerDetectionConfig = attrs.field(factory=StragglerDetectionConfig)
    profiling: Profiling = attrs.field(factory=Profiling)
    compile_config: CompileConfig = attrs.field(factory=CompileConfig)

    save_zero_checkpoint: bool = False


@make_freezable
@attrs.define(slots=False)
class Config:
    """Config for an imaginaire4 job.

    See /README.md/Configuration System for more info.
    """

    model: LazyDict
    optimizer: LazyDict
    scheduler: LazyDict
    dataloader_train: LazyDict | None

    job: JobConfig = attrs.field(factory=JobConfig)

    trainer: TrainerConfig = attrs.field(factory=TrainerConfig)

    if USE_MEGATRON:
        model_parallel: ModelParallelConfig = attrs.field(factory=ModelParallelConfig)
    else:
        model_parallel: None = None

    checkpoint: CheckpointConfig = attrs.field(factory=CheckpointConfig)

    upload_reproducible_setup: bool = False

    def pretty_print(self, use_color: bool = False) -> str:
        return _pretty_print_attrs_instance(self, 0, use_color)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    def validate(self) -> None:
        """Validate that the config has all required fields."""

        job_name_tensor = torch.ByteTensor(bytearray(self.job.name, "utf-8")).cuda()
        distributed.broadcast(job_name_tensor, 0)
        self.job.name = job_name_tensor.cpu().numpy().tobytes().decode("utf-8")


        assert self.job.project != ""
        assert self.job.group != ""
        assert self.job.name != ""


def _reload_make_config_for_registrations(root_cfg: "Config") -> None:
    """Run ``make_config()`` once for import-time registrations (same intent as loading ``config.py``).

    Deserialized YAML/TOML instantiates attrs ``Config`` with ``__class__.__module__`` set to the
    module that defines the class (often ``…defaults.config``). ``load_callable`` splits on the
    last dot, which turns that into ``import …defaults`` + ``getattr(..., "config")`` — the
    ``defaults.config`` submodule, which often has no ``make_config``. The entrypoint with
    ``make_config`` is typically the sibling module ``….config``.
    """
    from cosmos_framework.utils.serialization import load_callable

    cls_mod = type(root_cfg).__module__

    def _try_make_config(mod: object) -> bool:
        mk = getattr(mod, "make_config", None)
        if mk is None:
            return False
        _ = mk()
        return True

    if cls_mod.endswith(".defaults.config"):
        sibling = cls_mod[: -len(".defaults.config")] + ".config"
        try:
            if _try_make_config(importlib.import_module(sibling)):
                return
        except ModuleNotFoundError:
            pass

    try:
        if _try_make_config(load_callable(cls_mod)):
            return
    except (AssertionError, AttributeError, ModuleNotFoundError):
        pass

    try:
        if _try_make_config(importlib.import_module(cls_mod)):
            return
    except ModuleNotFoundError:
        pass

    raise AttributeError(
        f"No make_config() found for Config class module {cls_mod!r}. "
        "YAML/TOML export must match a tree whose Python package exposes make_config "
        "(e.g. cosmos_framework.configs.base.vlm.config next to cosmos_framework.configs.base.vlm.defaults.config)."
    )


def load_config(config_path: str, opts: list[str], enable_one_logger: bool = False) -> Config:
    from cosmos_framework.utils.serialization import from_toml, from_yaml

    t1 = time.monotonic_ns()
    if config_path.endswith(".yaml"):
        config = from_yaml(config_path)
        _reload_make_config_for_registrations(config)

        from cosmos_framework.utils.config_helper import override

        config = override(config, opts, remove_defaults=True)
    elif config_path.endswith(".toml"):
        config = from_toml(config_path)
        _reload_make_config_for_registrations(config)

        from cosmos_framework.utils.config_helper import override

        config = override(config, opts, remove_defaults=True)
    else:
        config = _load_py_config(config_path, opts, validate=False)

    if enable_one_logger:
        try:
            from cosmos_framework.utils.one_logger.one_logger_override_utils import override_one_logger_callback

            ol_t1 = time.monotonic_ns()
            config = override_one_logger_callback(config)
            ol_t2 = time.monotonic_ns()
            logging.debug(f"override_one_logger_callback: took {(ol_t2 - ol_t1) / 1e6:.2f}ms")
        except ImportError:
            pass

    t2 = time.monotonic_ns()
    logging.debug(f"total time to load config: {(t2 - t1) / 1e6:.2f}ms")
    return config


def _load_py_config(config_path: str, opts: list[str], validate: bool = True) -> Config:

    from cosmos_framework.utils.config_helper import get_config_module, override

    t1 = time.monotonic_ns()
    config_module = get_config_module(config_path)
    t2 = time.monotonic_ns()
    logging.debug(f"get_config_module: took {(t2 - t1) / 1e6:.2f}ms")

    t1 = time.monotonic_ns()
    config = importlib.import_module(config_module).make_config()
    t2 = time.monotonic_ns()
    logging.debug(f"importlib.import_module: took {(t2 - t1) / 1e6:.2f}ms")

    t1 = time.monotonic_ns()
    config = override(config, opts)
    t2 = time.monotonic_ns()
    logging.debug(f"override: took {(t2 - t1) / 1e6:.2f}ms")

    if validate:
        t1 = time.monotonic_ns()
        config.validate()
        t2 = time.monotonic_ns()
        logging.debug(f"config.validate: took {(t2 - t1) / 1e6:.2f}ms")

    return config
