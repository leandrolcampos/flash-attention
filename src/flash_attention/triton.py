# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0
#
# Some of the code in this file is inspired by:
#
# triton-lang/triton:
# Copyright 2020-2022 OpenAI
# Licensed under the MIT License.
#
# References:
#
# Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022).
# FlashAttention: Fast and Memory-Efficient Exact Attention
#   with IO-Awareness.
# Advances in Neural Information Processing Systems, 35,
#   16344-16359.
# https://dl.acm.org/doi/10.5555/3600270.3601459
#
# Dao, T. (2023). FlashAttention-2: Faster Attention with
#   Better Parallelism and Work Partitioning.
# arXiv preprint arXiv:2307.08691.
# https://doi.org/10.48550/arXiv.2307.08691

"""FlashAttention forward pass implementation in Triton."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_seq_len,
    k_seq_len,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    global_head_idx = tl.program_id(1)

    q_offset = global_head_idx * q_seq_len * HEAD_DIM
    k_offset = global_head_idx * k_seq_len * HEAD_DIM

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(q_seq_len, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(block_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(HEAD_DIM, k_seq_len),
        strides=(1, HEAD_DIM),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + k_offset,
        shape=(k_seq_len, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + q_offset,
        shape=(q_seq_len, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(block_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    NEG_INF = -float("inf")
    LOG2_E = 1.442695040889

    row_max = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_block = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    q_block = tl.load(q_block_ptr)
    q_block = (q_block * (softmax_scale * LOG2_E)).to(tl.float16)

    for _ in range(0, k_seq_len, BLOCK_N):
        k_block = tl.load(k_block_ptr)
        tl.static_assert(k_block.dtype == tl.float16, "k_block dtype mismatch")

        attn_score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        attn_score += tl.dot(q_block, k_block)

        row_max_new = tl.maximum(row_max, tl.max(attn_score, axis=1))
        attn_weight_unnormalized = tl.exp2(attn_score - row_max_new[:, None])

        attn_score_scale = tl.exp2(row_max - row_max_new)

        row_sum *= attn_score_scale
        row_sum_new = row_sum + tl.sum(attn_weight_unnormalized, axis=1)

        v_block = tl.load(v_block_ptr)

        o_block *= attn_score_scale[:, None]
        o_block += tl.dot(attn_weight_unnormalized.to(tl.float16), v_block)

        row_max = row_max_new
        row_sum = row_sum_new

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    o_block /= tl.where(row_sum == 0.0, 1.0, row_sum)[:, None]
    tl.store(o_block_ptr, o_block.to(tl.float16))


def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert query.is_cuda, "Input tensor 'query' must be on a CUDA device."

    assert (
        key.device == query.device
    ), "Input tensor 'key' must be on the same device as 'query'."
    assert (
        value.device == query.device
    ), "Input tensor 'value' must be on the same device as 'query'."

    assert (
        query.dtype == torch.float16
    ), "Input tensor 'query' must be of type torch.float16."
    assert (
        key.dtype == torch.float16
    ), "Input tensor 'key' must be of type torch.float16."
    assert (
        value.dtype == torch.float16
    ), "Input tensor 'value' must be of type torch.float16."

    assert query.is_contiguous(), "Input tensor 'query' must be contiguous."
    assert key.is_contiguous(), "Input tensor 'key' must be contiguous."
    assert value.is_contiguous(), "Input tensor 'value' must be contiguous."

    batch_size, num_heads, query_length, head_dim = query.shape
    key_length = key.shape[2]

    assert (
        key.shape[0] == batch_size
    ), "Input tensor 'key' must have the same batch_size as 'query'."
    assert (
        key.shape[1] == num_heads
    ), "Input tensor 'key' must have the same num_heads as 'query'."
    assert (
        key.shape[3] == head_dim
    ), "Input tensor 'key' must have the same head_dim as 'query'."

    assert (
        key.shape == value.shape
    ), "Input tensors 'key' and 'value' must have the same shape."

    assert head_dim in {
        64,
        128,
    }, f"Unsupported head_dim: {head_dim}. Must be 64 or 128."

    output = torch.empty_like(query)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    block_size = 128 if head_dim == 64 else 64
    grid = (triton.cdiv(query_length, block_size), batch_size * num_heads, 1)

    _mha_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        query_length,
        key_length,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        num_warps=4,
        num_stages=2,
    )

    return output
