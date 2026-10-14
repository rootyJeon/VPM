# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""SFT training entrypoint backed by the structured TOML dataclass.

Sole input is ``--sft-toml <path>`` — no ``--config`` or interface_toml flow.

Usage::

    torchrun --nproc_per_node=<N> -m cosmos_framework.scripts.train \\
        --sft-toml=examples/toml/sft_config/<experiment>.toml \\
        -- optimizer.lr=1e-5 trainer.max_iter=200

The TOML is loaded via ``SFTExperimentConfig.from_toml`` (structural validation,
raises on unknown keys), then
``cosmos_framework.configs.toml_config.sft_config.load_experiment_from_toml`` picks the
base ``config.py`` from ``[job].task`` (``vfm`` → ``cosmos_framework/configs/base/config.py``,
``vlm`` → ``cosmos_framework/configs/base/vlm/config.py``), resolves ``[job].experiment``
against the Hydra ``ConfigStore``, and overlays every other TOML key as a Hydra
override. Trailing ``key.path=value`` positionals are applied last (so they
win over TOML).
"""

from __future__ import annotations

import argparse
import os
import traceback

import torch
from loguru import logger as logging

from cosmos_framework.utils.config import Config
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate
from cosmos_framework.utils.serialization import to_yaml
from cosmos_framework.utils import distributed
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.launch import log_reproducible_setup
from cosmos_framework.utils.training_telemetry import telemetry
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml


def _setup_deterministic_env_and_backends() -> None:
    """Set determinism env vars + torch backend flags. Call at script entry, pre-CUDA init."""
    if "PYTHONHASHSEED" not in os.environ:
        logging.warning(
            "PYTHONHASHSEED is not set; --deterministic is best-effort without it. "
            "For full reproducibility, prepend `PYTHONHASHSEED=42` (or any fixed value) "
            "to your launch command — Python's hash seed is fixed at interpreter startup "
            "and cannot be set retroactively."
        )
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    logging.info("Deterministic mode enabled.")


def _apply_deterministic_config_overrides(config: Config) -> None:
    """Apply config mutations. Call after load_config, before config.freeze().

    Forces:
      - trainer.cudnn.deterministic=True, trainer.cudnn.benchmark=False
      - trainer.seed=42 when at its default (0)
      - model.config.compile.enabled=False (any node with the key)
      - dataloader num_workers=0, prefetch_factor=None, dataset.detshuffle=True
        on every dataloader-shaped node in dataloader_train.

    Only existing keys are mutated; projects without these fields are unaffected.
    """
    from omegaconf import DictConfig, ListConfig

    config.trainer.cudnn.deterministic = True
    config.trainer.cudnn.benchmark = False
    if config.trainer.seed == 0:
        config.trainer.seed = 42

    def _walk(cfg, mutations: dict) -> int:
        if cfg is None:
            return 0
        n = 0
        if isinstance(cfg, DictConfig):
            for k in list(cfg.keys()):
                if k in mutations:
                    target = mutations[k]
                    try:
                        if cfg[k] != target:
                            cfg[k] = target
                            n += 1
                    except Exception as e:
                        logging.warning(f"--deterministic: failed to set {k}={target!r}: {e}")
                    continue
                try:
                    v = cfg[k]
                except Exception:
                    continue
                if isinstance(v, (DictConfig, ListConfig)):
                    n += _walk(v, mutations)
        elif isinstance(cfg, ListConfig):
            for item in cfg:
                if isinstance(item, (DictConfig, ListConfig)):
                    n += _walk(item, mutations)
        return n

    dl_overrides = {
        "num_workers": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "detshuffle": True,
    }
    n_dl = _walk(config.dataloader_train, dl_overrides)

    def _force_compile_disabled(cfg) -> int:
        """Force ``compile.enabled=False`` on every CompileConfig subtree in cfg.

        Scoped to ``compile`` parents (not a generic ``enabled`` walk) because
        ``enabled`` is a common key shared by unrelated configs (e.g. EMA).
        """
        if cfg is None:
            return 0
        n = 0
        if isinstance(cfg, DictConfig):
            for k in list(cfg.keys()):
                try:
                    v = cfg[k]
                except Exception:
                    continue
                if k == "compile" and isinstance(v, DictConfig) and "enabled" in v:
                    try:
                        if v["enabled"] is not False:
                            v["enabled"] = False
                            n += 1
                    except Exception as e:
                        logging.warning(f"--deterministic: failed to set compile.enabled=False: {e}")
                elif isinstance(v, (DictConfig, ListConfig)):
                    n += _force_compile_disabled(v)
        elif isinstance(cfg, ListConfig):
            for item in cfg:
                if isinstance(item, (DictConfig, ListConfig)):
                    n += _force_compile_disabled(item)
        return n

    n_tc = _force_compile_disabled(config.model)
    logging.info(
        f"--deterministic: applied {n_dl} dataloader override(s), "
        f"{n_tc} compile.enabled override(s); trainer.seed={config.trainer.seed}"
    )


@logging.catch(reraise=True)
@telemetry.monitor
def launch(config: Config, args: argparse.Namespace) -> None:
    with distributed_init():
        distributed.init()

    if args.deterministic:
        _apply_deterministic_config_overrides(config)
    config.validate()
    config.freeze()
    trainer = config.trainer.type(config)
    log_reproducible_setup(config, args)

    if args.attach_vscode_debugger:
        print(f"RANK: {os.environ['RANK']}")
        if os.environ["RANK"] == "0":
            import debugpy

            debugpy.listen(3002)
            print("Waiting for debugger to attach. Listening on port 3002...")
            debugpy.wait_for_client()

    with model_init():
        model = instantiate(config.model)

    with data_loader_init():
        dataloader_train = instantiate(config.dataloader_train)

    trainer.train(
        model,
        dataloader_train,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training (structured TOML)")
    parser.add_argument(
        "--sft-toml",
        required=True,
        help=(
            "Path to an SFT structured-dataclass TOML — see "
            "cosmos_framework/configs/toml_config/sft_config.py "
            "(SFTExperimentConfig)."
        ),
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra Hydra-style dotted-path overrides applied AFTER the TOML "
            "values (so they win). Use the standard Hydra syntax, e.g. "
            "'optimizer.lr=1e-5 trainer.max_iter=200 "
            "model.config.parallelism.data_parallel_shard_degree=4'. "
            "Prefix with '--' to make argparse stop interpreting the rest as "
            "flags."
        ),
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--attach_vscode_debugger",
        action="store_true",
        help="Debug mode. Will start a debugpy server at 0.0.0.0:3002.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic mode (mirrors launch_vfm.sh). Auto-applies env: "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8, FLASH_ATTENTION_DETERMINISTIC=1; torch backends: "
            "cudnn.deterministic=True, cudnn.benchmark=False, "
            "use_deterministic_algorithms(warn_only=True); config: trainer.cudnn.{deterministic, "
            "benchmark}, trainer.seed=42 (when at default 0), "
            "model.config.compile.enabled=False, and for every dataloader in "
            "dataloader_train: num_workers=0, prefetch_factor=None, "
            "dataset.detshuffle=True. PYTHONHASHSEED must be set externally (e.g. "
            "`PYTHONHASHSEED=42 torchrun ...`) since Python locks it in at interpreter startup."
        ),
    )
    args = parser.parse_args()

    if args.deterministic:
        _setup_deterministic_env_and_backends()

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)

    args.config = args.sft_toml

    if args.dryrun:
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        launch(config, args)
