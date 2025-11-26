# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""Naive Multi-Head Attention implementation in PyTorch."""

import math

import torch


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    assert (
        key.device == query.device
    ), "Input tensor 'key' must be on the same device as 'query'."
    assert (
        value.device == query.device
    ), "Input tensor 'value' must be on the same device as 'query'."

    assert (
        query.is_floating_point()
    ), f"Input tensor 'query' must be a floating point type."
    assert (
        key.dtype == query.dtype
    ), f"Input tensor 'key' must have the same dtype as 'query'."
    assert (
        value.dtype == query.dtype
    ), f"Input tensor 'value' must have the same dtype as 'query'."

    batch_size, num_heads, _, head_dim = query.shape

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

    # b: batch_size, t: num_tokens, h: num_heads, d: head_dim

    # (b, h, t_q, d) @ (b, h, d, t_k) -> (b, h, t_q, t_k)
    attn_score = query @ key.transpose(2, 3)

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)
    # (b, h, t_q, t_k) -> (b, h, t_q, t_k)
    attn_weight = torch.softmax(attn_score * softmax_scale, dim=-1)

    # (b, h, t_q, t_k) @ (b, h, t_k, d) -> (b, h, t_q, d)
    output = attn_weight @ value

    return output
