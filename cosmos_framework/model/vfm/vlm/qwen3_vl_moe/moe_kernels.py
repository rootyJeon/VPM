# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Callable, Literal

import torch
import triton
import triton.language as tl


TOKEN_GROUP_ALIGN_SIZE_M = 16
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def _permute(
    x: torch.Tensor,
    num_tokens_per_expert: int,
    num_experts: int,
    alignment: int = TOKEN_GROUP_ALIGN_SIZE_M,
):
    x_padded_size = x.shape[0] + num_experts * alignment
    padded_max_len = ((x_padded_size + alignment - 1) // alignment) * alignment

    with torch.no_grad():
        (
            permuted_indices,
            padded_num_tokens_per_expert,
        ) = _generate_permute_indices(
            num_tokens_per_expert=num_tokens_per_expert,
            num_experts=num_experts,
            max_len=padded_max_len,
            alignment=alignment,
        )

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, padded_num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted[:-1]


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = num_tokens_per_expert.shape[0]

        input_shape, x, permuted_indices, padded_num_tokens_per_expert = _permute(x, num_tokens_per_expert, num_experts)

        out = func(gate_up_proj, down_proj, act_fn, x, padded_num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)
        return out

    return wrapper


@triton.jit
def _fill_indices_kernel(
    num_tokens_per_expert_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for expert_id in range(pid, num_experts, num_programs):
        write_offset = tl.load(write_offsets_ptr + expert_id)

        start_index = tl.load(start_index_values_ptr + expert_id)
        length = tl.load(num_tokens_per_expert_ptr + expert_id)

        offsets = tl.arange(0, BLOCK_SIZE)

        for chunk_start in range(0, length, BLOCK_SIZE):
            chunk_offsets = chunk_start + offsets

            mask = chunk_offsets < length

            values = start_index + chunk_offsets

            dest_indices = write_offset + chunk_offsets

            tl.store(output_ptr + dest_indices, values, mask=mask)


def _fill_indices_wrapper(
    num_tokens_per_expert: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    num_experts: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
):
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32, device=num_tokens_per_expert.device)

    num_blocks = min(num_experts, max_blocks)
    grid = (num_blocks,)

    _fill_indices_kernel[grid](
        num_tokens_per_expert,
        start_index_values,
        write_offsets,
        permuted_indices,
        num_experts,
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


def _generate_permute_indices(
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    max_len: int,
    alignment: int,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        num_tokens_per_expert: number of tokens for each expert.
        num_experts: number of experts.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.

    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """
    start_index_values = torch.cumsum(num_tokens_per_expert, dim=0) - num_tokens_per_expert

    m_sizes = torch.clamp_min(num_tokens_per_expert, alignment)

    m_sizes = (m_sizes.to(torch.int32) + alignment - 1) // alignment * alignment

    write_offsets = torch.cumsum(m_sizes, dim=0) - m_sizes

    permuted_indices = _fill_indices_wrapper(
        num_tokens_per_expert=num_tokens_per_expert,
        start_index_values=start_index_values,
        write_offsets=write_offsets,
        num_experts=num_experts,
        max_len=max_len,
    )

    return permuted_indices, m_sizes
