# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export the model weights of a training checkpoint to a sharded safetensors release.

Reads ``<src>/model`` (the DCP written by training, ``<src>`` = an ``iter_*`` directory) and writes::

    <out>/model/model-00001-of-0000N.safetensors ...   keys net.*, bf16
    <out>/model/model.safetensors.index.json            Hugging Face index (metadata.total_size + weight_map)
    <out>/release.json                                  tensor/parameter counts, sha256 per shard, source iteration

``--ema`` exports the EMA weights (``net_ema.*`` renamed to ``net.*``) instead of the regular ones.
CPU only, no process group; tensors are read one shard at a time. The result loads with
``torch.distributed.checkpoint.load(..., storage_reader=HuggingFaceStorageReader("<out>/model"))``.

    python scripts/export_checkpoint.py --src <run>/checkpoints/iter_<N> --out release/VPM-DROID
"""

import argparse
import hashlib
import json
import random
import re
import time
import warnings
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

try:
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
except ImportError:
    _load_state_dict_from_keys = None

NET_PREFIX = "net."
EMA_PREFIX = "net_ema."
INDEX_NAME = "model.safetensors.index.json"
DTYPE = torch.bfloat16
READ_BATCH_BYTES = 2 * 1024**3

warnings.filterwarnings("ignore", message=r"torch\.distributed is .*(unavailable|uninitialized)")
warnings.filterwarnings("ignore", message=r"TypedStorage is deprecated")


def resolve_model_dir(src: Path) -> Path:
    if (src / "model" / ".metadata").exists():
        return src / "model"
    if (src / ".metadata").exists():
        return src
    raise FileNotFoundError(f"No DCP checkpoint found at {src}/model")


def source_iteration(src: Path) -> int | None:
    for part in (src.name, src.parent.name):
        match = re.fullmatch(r"iter_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _load_with_public_api(keys: list[str], tensor_md: dict[str, TensorStorageMetadata], model_dir: Path) -> dict:
    state_dict = {k: torch.empty(tensor_md[k].size, dtype=tensor_md[k].properties.dtype) for k in keys}
    dcp.load(state_dict, storage_reader=FileSystemReader(str(model_dir)), no_dist=True)
    return state_dict


def load_tensors(keys: list[str], tensor_md: dict[str, TensorStorageMetadata], model_dir: Path) -> dict:
    """Load full (unsharded) tensors for ``keys`` from a DCP directory without a process group."""
    if _load_state_dict_from_keys is not None:
        try:
            return _load_state_dict_from_keys(set(keys), checkpoint_id=str(model_dir))
        except TypeError:
            pass
    return _load_with_public_api(keys, tensor_md, model_dir)


def plan_shards(names: list[str], nbytes: dict[str, int], shard_bytes: int) -> list[list[str]]:
    shards: list[list[str]] = [[]]
    size = 0
    for name in names:
        if shards[-1] and size + nbytes[name] > shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += nbytes[name]
    return shards


def batched(names: list[str], nbytes: dict[str, int], limit: int):
    batch: list[str] = []
    size = 0
    for name in names:
        if batch and size + nbytes[name] > limit:
            yield batch
            batch, size = [], 0
        batch.append(name)
        size += nbytes[name]
    if batch:
        yield batch


def sha256_file(path: Path, chunk: int = 64 * 1024**2) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_export(
    out_model_dir: Path,
    src_model_dir: Path,
    tensor_md: dict,
    rename: dict[str, str],
    samples: int,
    seed: int = 0,
) -> None:
    """Reload a random subset through ``HuggingFaceStorageReader`` and compare with the bf16-cast source."""
    from torch.distributed.checkpoint import HuggingFaceStorageReader

    rng = random.Random(seed)
    picked = rng.sample(sorted(rename), min(samples, len(rename)))
    exported = {n: torch.empty(tensor_md[rename[n]].size, dtype=DTYPE) for n in picked}
    dcp.load(exported, storage_reader=HuggingFaceStorageReader(str(out_model_dir)), no_dist=True)
    source = load_tensors([rename[n] for n in picked], tensor_md, src_model_dir)
    for name in picked:
        if not torch.equal(exported[name], source[rename[name]].to(DTYPE)):
            raise RuntimeError(f"Exported tensor {name} does not match the source checkpoint")
    print(f"Verified {len(picked)} tensors through HuggingFaceStorageReader")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="Training checkpoint directory (iter_*) with model/.")
    parser.add_argument("--out", required=True, type=Path, help="Release directory to create.")
    parser.add_argument("--ema", action="store_true", help="Export net_ema.* (renamed to net.*) instead of net.*.")
    parser.add_argument("--shard-gb", type=float, default=5.0, help="Maximum shard size in GB (1e9 bytes).")
    parser.add_argument("--keys-regex", default=None, help="Only export net.* keys matching this regex (re.search).")
    parser.add_argument("--verify-samples", type=int, default=16, help="Tensors to re-read and compare (0 = skip).")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty --out.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    src_model_dir = resolve_model_dir(args.src)
    out_model_dir = args.out / "model"
    if out_model_dir.exists() and any(out_model_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_model_dir} is not empty; pass --overwrite to replace its contents.")

    metadata = FileSystemReader(str(src_model_dir)).read_metadata().state_dict_metadata
    prefix = EMA_PREFIX if args.ema else NET_PREFIX
    tensor_md = {k: v for k, v in metadata.items() if isinstance(v, TensorStorageMetadata)}
    skipped = sorted(k for k in metadata if k.startswith(prefix) and k not in tensor_md)
    if skipped:
        print(f"Skipping {len(skipped)} non-tensor entries, e.g. {skipped[:3]}")

    rename = {NET_PREFIX + k[len(prefix) :]: k for k in tensor_md if k.startswith(prefix)}
    if args.keys_regex:
        pattern = re.compile(args.keys_regex)
        rename = {n: k for n, k in rename.items() if pattern.search(n)}
    if not rename:
        raise SystemExit(f"No tensors with prefix {prefix!r} selected from {src_model_dir}")

    names = list(rename)
    out_bytes = {n: tensor_md[rename[n]].size.numel() * DTYPE.itemsize for n in names}
    src_bytes = {n: tensor_md[rename[n]].size.numel() * tensor_md[rename[n]].properties.dtype.itemsize for n in names}
    shards = plan_shards(names, out_bytes, int(args.shard_gb * 1e9))
    n_shards = len(shards)
    total_size = sum(out_bytes.values())
    n_params = sum(tensor_md[rename[n]].size.numel() for n in names)
    print(f"Exporting {len(names)} tensors ({n_params:,} params, {total_size / 1e9:.2f} GB bf16) in {n_shards} shards")

    out_model_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_model_dir.glob("model-*-of-*.safetensors"):
        stale.unlink()

    weight_map: dict[str, str] = {}
    shard_info = []
    for i, shard in enumerate(shards, start=1):
        file_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        for batch in batched(shard, src_bytes, READ_BATCH_BYTES):
            loaded = load_tensors([rename[n] for n in batch], tensor_md, src_model_dir)
            for name in batch:
                tensors[name] = loaded.pop(rename[name]).to(DTYPE).contiguous()
            del loaded
        path = out_model_dir / file_name
        save_file(tensors, str(path), metadata={"format": "pt"})
        del tensors
        weight_map.update({name: file_name for name in shard})
        shard_info.append({"file": f"model/{file_name}", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        print(f"[{i}/{n_shards}] {file_name} ({len(shard)} tensors, {path.stat().st_size / 1e9:.2f} GB)")

    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (out_model_dir / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

    release = {
        "format": "safetensors",
        "dtype": "bfloat16",
        "weights": "ema" if args.ema else "regular",
        "key_prefix": NET_PREFIX,
        "source_iteration": source_iteration(args.src),
        "num_tensors": len(names),
        "num_parameters": n_params,
        "total_size": total_size,
        "shards": shard_info,
    }
    if args.keys_regex:
        release["keys_regex"] = args.keys_regex
    (args.out / "release.json").write_text(json.dumps(release, indent=2) + "\n")

    if args.verify_samples > 0:
        verify_export(out_model_dir, src_model_dir, tensor_md, rename, args.verify_samples)
    print(f"Done in {time.time() - t0:.0f}s: {args.out}")


if __name__ == "__main__":
    main()
