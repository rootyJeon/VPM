# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Core masking functions extracted from transformers.masking_utils for BAGEL compatibility
# Original Copyright 2025 HuggingFace Inc. team. Licensed under the Apache License, Version 2.0

from typing import Any, Callable, ClassVar, Optional, cast

import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging

from transformers.utils.import_utils import is_torch_greater_or_equal

logger = logging.get_logger(__name__)

_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."
_SYSTEM_PROMPT_TRANSFER = (
    "You are a helpful assistant that generates images or videos following the user's instructions"
    " and control signals (edge maps, blur, depth, or segmentation)."
)
_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."


def tokenize_caption(
    caption: str,
    tokenizer: PreTrainedTokenizerBase,
    is_video: bool = False,
    use_system_prompt: bool = False,
    system_prompt: Optional[str] = None,
) -> list[int]:
    """Tokenize a text caption into token IDs using the Qwen2 chat template.

    Wraps the caption in a chat-style conversation (with a "user" role) and applies
    the tokenizer's chat template to produce the final token ID sequence, including
    any special tokens (e.g., BOS, role markers, generation prompt).

    Args:
        caption: The text caption to tokenize.
        tokenizer: A HuggingFace ``PreTrainedTokenizerBase`` (e.g. Qwen2Tokenizer or Fast tokenizer).
        is_video: If True (and use_system_prompt=True), uses the video system prompt;
            otherwise uses the image system prompt. Ignored when ``system_prompt`` is
            provided.
        use_system_prompt: If True, prepends a system prompt message to the conversation
            before the user caption. Ignored when ``system_prompt`` is provided.
        system_prompt: When supplied, this exact string is used as the system prompt,
            overriding both ``is_video`` and ``use_system_prompt``.

    Returns:
        List of token IDs representing the full chat-formatted caption.
    """
    conversations = []
    if system_prompt is not None:
        conversations.append({"role": "system", "content": system_prompt})
    elif use_system_prompt:
        _system_prompt = _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE
        conversations.append({"role": "system", "content": _system_prompt})
    conversations.append({"role": "user", "content": caption})

    tokenizer_output = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=False,
    )
    return cast(list[int], tokenizer_output)


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx > q_idx - sliding_window

    return inner_mask


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        return result

    return and_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs.
    """
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[torch.Tensor]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
            local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        if _slice:
            mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
            mask_indices += kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def eager_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
    the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
    it should not.
    """
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    causal_mask = _vmap_for_bhqkv(mask_function, bh_indices=False)(
        None, None, cache_position, kv_arange
    )
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, -1, -1, -1)
    if padding_mask is not None:
        causal_mask = causal_mask * padding_mask[:, None, None, :]

    min_dtype = torch.finfo(dtype).min
    mask = torch.where(
        causal_mask, torch.tensor(0.0, device=causal_mask.device, dtype=dtype), min_dtype
    )
    return mask


class AttentionMaskInterface:
    _global_mapping: ClassVar = {
        "eager": eager_mask,
    }


ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[torch.Tensor], None, int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code.
    """
    if isinstance(attention_mask, torch.Tensor) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=cache_position.device, dtype=torch.bool)

    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    return False, attention_mask, None, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config).
    """
    layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    if attention_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, padding_mask_function(attention_mask))

    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        dtype=dtype,
        config=config,
    )
    return causal_mask


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config).
    """
    layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    if attention_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, padding_mask_function(attention_mask))

    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        dtype=dtype,
        config=config,
    )
    return causal_mask


def get_rope_index(
    model: Any,
    input_ids: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL multimodal RoPE positions and deltas."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = model.config.vision_config.spatial_merge_size
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    vision_start_token_id = model.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = (
                    torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                )
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    return position_ids, mrope_position_deltas


def get_image_features(
    model: Any,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
    """Encode images with the Qwen3-VL visual module."""
    pixel_values = pixel_values.type(model.visual.dtype)
    image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds


def get_placeholder_mask(
    model: Any,
    input_ids: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    image_features: Optional[torch.Tensor] = None,
    video_features: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expanded image/video placeholder masks and validate feature lengths."""
    if input_ids is None:
        special_image_mask = inputs_embeds == model.get_input_embeddings()(
            torch.tensor(model.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_image_mask = special_image_mask.all(-1)
        special_video_mask = inputs_embeds == model.get_input_embeddings()(
            torch.tensor(model.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        special_video_mask = special_video_mask.all(-1)
    else:
        special_image_mask = input_ids == model.config.image_token_id
        special_video_mask = input_ids == model.config.video_token_id

    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    n_video_tokens = special_video_mask.sum()
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        raise ValueError(
            f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        )

    return special_image_mask, special_video_mask


def prepare_multimodal_reasoner_inputs(
    causal_lm: Any,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Build the I2V prefill inputs for the reasoner-only autoregressive path.

    Bundles the standard Qwen3-VL multimodal preprocessing recipe — text
    embed → visual encode → ``masked_scatter`` of image features →
    derive ``visual_pos_masks`` → align deepstack embeds to the
    embeddings device/dtype → multimodal rope index — into the single
    helper that the reasoner-only AR generation path
    (``unified_mot._impl_generate_reasoner_text``) needs for its
    image-conditioned prefill step.

    Functionally mirrors the inline preprocessing in HuggingFace's
    ``Qwen3VLModel.forward`` (the ``pixel_values`` branch plus the
    ``image_mask``-only aggregation branch and the prefill
    ``get_rope_index`` call) but stops short of running the language
    model — the caller feeds the returned tensors into
    ``*TextModel.reasoner_forward`` instead of HF's full
    ``self.language_model(...)`` forward, so HF's
    ``past_key_values`` / ``cache_position`` lifecycle is replaced by
    the AR loop's :class:`ReasonerKVCache` lifecycle.  Videos and
    dual image+video paths are not supported here; only
    ``image_grid_thw`` is consumed — matching the public
    ``generate_reasoner_text`` API, which has no
    ``pixel_values_videos`` / ``video_grid_thw`` parameters.

    Validation: ``get_placeholder_mask`` raises ``ValueError`` if the
    number of image placeholder tokens in ``input_ids`` does not match
    the number of visual features produced by ``causal_lm.visual``.

    Args:
        causal_lm: A ``*ForCausalLM`` instance providing
            ``model.embed_tokens`` / ``visual`` / ``config``.  Must
            already have a vision tower attached (``causal_lm.visual``);
            language-only or combined checkpoints without one should be
            rejected by the caller before invoking this helper.
        input_ids: ``[B, T_prompt]`` integer token ids of the prompt.
            Must contain ``causal_lm.config.image_token_id`` placeholder
            tokens (one per patch *after* spatial merging) where the
            image features get scattered in.
        pixel_values: ``[N_patches, C, H, W]`` image patches
            concatenated across the whole batch (the layout
            ``Qwen3VLProcessor`` emits — pass it through unchanged).
            Moved to the embeddings device internally.
        image_grid_thw: ``[num_images, 3]`` long tensor giving
            ``(t, h, w)`` per image as produced by
            ``Qwen3VLProcessor``.  ``num_images`` is the total image
            count across the batch.  Moved to the embeddings device
            internally.
        attention_mask: Optional ``[B, T_prompt]`` attention mask
            forwarded to ``get_rope_index`` so positions of pad tokens
            are clamped correctly under multimodal rope.

    Returns:
        ``(inputs_embeds, visual_pos_masks, deepstack_visual_embeds,
        position_ids, mrope_position_deltas)``:
            inputs_embeds: ``[B, T_prompt, hidden_size]`` text
                embeddings with image features scattered into the
                placeholder positions.
            visual_pos_masks: ``[B, T_prompt]`` bool mask of visual
                token positions (consumed by per-layer deepstack
                additions inside ``reasoner_forward``).
            deepstack_visual_embeds: list of per-deepstack-layer
                tensors aligned to the embeddings device/dtype.
            position_ids: Multimodal rope position ids in the layout
                produced by ``get_rope_index``.
            mrope_position_deltas: Per-sample rope delta used by the
                caller to extend positions during decode.
    """
    inputs_embeds = causal_lm.model.embed_tokens(input_ids).clone()
    pixel_values = pixel_values.to(device=inputs_embeds.device)
    image_grid_thw = image_grid_thw.to(device=inputs_embeds.device)

    image_embeds, deepstack_visual_embeds = get_image_features(causal_lm, pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    image_mask, _video_mask = get_placeholder_mask(
        causal_lm,
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_embeds,
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    visual_pos_masks = image_mask[..., 0]

    deepstack_visual_embeds = [
        embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) for embed in deepstack_visual_embeds
    ]

    position_ids, mrope_position_deltas = get_rope_index(
        causal_lm,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids, mrope_position_deltas
