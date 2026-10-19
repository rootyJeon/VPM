# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert the Hugging Face ``nvidia/Cosmos3-Nano`` checkpoint into the VPM warm-start checkpoint.

Writes a torch distributed checkpoint (DCP) to ``<out>/model`` with keys ``net.<native name>`` in bf16,
the format expected by ``checkpoint.load_path`` (``BASE_CHECKPOINT_PATH=<out>``). Only the ``transformer/``
and ``vision_encoder/`` weights of the pinned revision are needed (~31.5 GB). CPU only, single process.

    python scripts/convert_cosmos3_nano.py --out checkpoints/Cosmos3-Nano
    python scripts/convert_cosmos3_nano.py --hf-dir /path/to/Cosmos3-Nano/snapshot --out checkpoints/Cosmos3-Nano
"""

import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemWriter

# This repository's cosmos_framework first, also when another copy of the package is installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos_framework.utils.vfm import diffusers_keys  # noqa: E402

REPO_ID = "nvidia/Cosmos3-Nano"
REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
ALLOW_PATTERNS = (
    diffusers_keys.DIFFUSERS_ROOT_INDEX,
    "transformer/*.safetensors",
    "vision_encoder/*.safetensors",
)
NET_PREFIX = "net."
VIEW_INV_FREQ_KEY = "language_model.model.rotary_emb.view_inv_freq"
HEAD_DIM_KEY = "language_model.model.layers.0.self_attn.q_norm.weight"
MAX_SHARD_BYTES = 5 * 1024**3

warnings.filterwarnings("ignore", message=r"torch\.distributed is .*(unavailable|uninitialized)")
warnings.filterwarnings("ignore", message=r"TypedStorage is deprecated")


def download_snapshot(repo_id: str, revision: str, cache_dir: str | None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=list(ALLOW_PATTERNS),
        cache_dir=cache_dir,
    )
    return Path(path)


def load_net_state_dict(hf_dir: Path, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Read the model tensors of a diffusers snapshot (memory-mapped) under native ``net.`` keys."""
    key_map = diffusers_keys.diffusers_to_net_key_map(diffusers_keys.diffusers_weight_map(hf_dir))
    by_file: dict[str, list[tuple[str, str]]] = {}
    for net_key, (diff_key, rel_path) in key_map.items():
        by_file.setdefault(rel_path, []).append((net_key, diff_key))

    state_dict: dict[str, torch.Tensor] = {}
    for rel_path in sorted(by_file):
        shard = hf_dir / rel_path
        if not shard.exists():
            raise FileNotFoundError(f"Missing weight file {shard}")
        with safe_open(str(shard), framework="pt") as f:
            available = set(f.keys())
            for net_key, diff_key in by_file[rel_path]:
                if diff_key not in available:
                    raise KeyError(f"{diff_key!r} is listed in the index but missing from {shard}")
                tensor = f.get_tensor(diff_key)
                state_dict[NET_PREFIX + net_key] = tensor if tensor.dtype == dtype else tensor.to(dtype)

    view_key = NET_PREFIX + VIEW_INV_FREQ_KEY
    if view_key not in state_dict:
        head_dim = state_dict[NET_PREFIX + HEAD_DIM_KEY].shape[0]
        state_dict[view_key] = torch.zeros(head_dim // 2, dtype=dtype)
    return state_dict


def save_dcp(state_dict: dict[str, torch.Tensor], model_dir: Path) -> None:
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    writer = FileSystemWriter(str(model_dir), thread_count=max(1, math.ceil(total_bytes / MAX_SHARD_BYTES)))
    dcp.save(state_dict=state_dict, storage_writer=writer, planner=dcp.DefaultSavePlanner(), no_dist=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="Output directory; the DCP is written to <out>/model.")
    parser.add_argument("--hf-dir", type=Path, default=None, help="Local snapshot of the HF repo (skips download).")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=REVISION, help="HF revision to download (pinned by default).")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory (default: $HF_HOME/hub).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing <out>/model checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.out / "model"
    if (model_dir / ".metadata").exists() and not args.overwrite:
        raise SystemExit(f"{model_dir} already holds a checkpoint; pass --overwrite to replace it.")

    t0 = time.time()
    hf_dir = args.hf_dir if args.hf_dir is not None else download_snapshot(args.repo_id, args.revision, args.cache_dir)
    print(f"Reading {hf_dir}")
    state_dict = load_net_state_dict(hf_dir)
    n_params = sum(t.numel() for t in state_dict.values())
    print(f"{len(state_dict)} tensors, {n_params:,} parameters")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {model_dir}")
    save_dcp(state_dict, model_dir)
    print(f"Done in {time.time() - t0:.0f}s. Use BASE_CHECKPOINT_PATH={args.out}")


if __name__ == "__main__":
    main()
