# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Key mapping from the Hugging Face (diffusers) Cosmos3 checkpoint layout to native ``OmniMoTModel.net`` names."""

import json
import re
from pathlib import Path

DIFFUSERS_ROOT_INDEX = "model.safetensors.index.json"
DIFFUSERS_MODEL_INDEX = "model_index.json"
DIFFUSERS_MODEL_SUBFOLDERS: tuple[str, ...] = ("transformer", "vision_encoder")

_DROP_WEIGHT_PATH_RES: tuple[re.Pattern[str], ...] = (re.compile(r"^(?!transformer/|vision_encoder/)"),)
_DROP_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(
            r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"
        ),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_NET_KEY_PREFIXES: tuple[str, ...] = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "llm2action.",
    "llm2sound.",
    "llm2vae.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_NET_KEYS: frozenset[str] = frozenset({"action_modality_embed", "latent_pos_embed", "sound_modality_embed"})


def should_drop_weight_path(path: str) -> bool:
    """True if a weight file (path relative to the snapshot root) holds no ``net`` tensors."""
    path = path.replace("\\", "/")
    return bool(path) and any(pattern.search(path) is not None for pattern in _DROP_WEIGHT_PATH_RES)


def should_drop_key(name: str) -> bool:
    """True if a key belongs to a non-``net`` pipeline component (VAE, tokenizer, scheduler, ...)."""
    return any(pattern.search(name) is not None for pattern in _DROP_KEY_RES)


def is_model_weight_path(path: str) -> bool:
    return bool(path) and not should_drop_weight_path(path)


def apply_key_mapping(name: str) -> str:
    for pattern, replacement in _KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def is_loadable_net_key(name: str) -> bool:
    return name in _NET_KEYS or name.startswith(_NET_KEY_PREFIXES)


def diffusers_to_net_key(name: str, weight_path: str = "") -> str | None:
    """Map a diffusers key (and the file holding it) to its ``OmniMoTModel.net`` key; None if not a net tensor."""
    if should_drop_weight_path(weight_path) or should_drop_key(name):
        return None
    net_key = apply_key_mapping(name)
    if should_drop_key(net_key) or not is_loadable_net_key(net_key):
        return None
    return net_key


def read_safetensors_index(index_path: Path) -> dict[str, str]:
    """Return the ``weight_map`` (key -> relative file) of a safetensors index json."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a safetensors weight_map.")
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{index_path} weight_map must contain string keys and values.")
    return dict(weight_map)


def diffusers_weight_map(checkpoint_path: Path) -> dict[str, str]:
    """Root weight map of a diffusers snapshot (key -> path relative to the snapshot root)."""
    index_path = Path(checkpoint_path) / DIFFUSERS_ROOT_INDEX
    if not index_path.exists():
        raise FileNotFoundError(f"Diffusers safetensors index not found: {index_path}")
    return read_safetensors_index(index_path)


def diffusers_files_to_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Group the model keys of a root weight map by file, skipping non-model files."""
    files_to_keys: dict[str, list[str]] = {}
    for diff_key, rel_path in weight_map.items():
        if should_drop_weight_path(rel_path):
            continue
        files_to_keys.setdefault(rel_path, []).append(diff_key)
    return files_to_keys


def diffusers_to_net_key_map(weight_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Map every model tensor of a root weight map to ``{net_key: (diffusers_key, relative_file)}``.

    Raises if a model-file key has no native name or two keys collide.
    """
    result: dict[str, tuple[str, str]] = {}
    for diff_key, rel_path in sorted(weight_map.items()):
        net_key = diffusers_to_net_key(diff_key, rel_path)
        if net_key is None:
            if is_model_weight_path(rel_path):
                raise KeyError(f"Diffusers model key {diff_key!r} from {rel_path!r} has no native mapping.")
            continue
        if net_key in result:
            raise KeyError(f"Multiple diffusers keys map to native key {net_key!r}.")
        result[net_key] = (diff_key, rel_path)
    return result
