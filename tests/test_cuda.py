# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""Tests the FlashAttention forward pass implementation in CUDA."""

import pytest
import torch

from flash_attention import cuda
from flash_attention.utils import baseline
from flash_attention.utils.testing import all_close


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    yield
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "batch_size, num_tokens", [(32, 512), (16, 1024), (8, 2048), (4, 4096)]
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mha_fwd_triton(batch_size, num_tokens, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    emb_dim = 512
    dtype = torch.float16
    device = "cuda"

    num_heads = emb_dim // head_dim

    query, key, value = [
        torch.randn(
            batch_size, num_heads, num_tokens, head_dim, dtype=dtype, device=device
        )
        for _ in range(3)
    ]

    output_triton = cuda.multi_head_attention_forward(query, key, value)
    output_baseline = baseline.multi_head_attention(
        query.to(torch.float32),
        key.to(torch.float32),
        value.to(torch.float32),
    )

    assert all_close(actual=output_triton, desired=output_baseline.to(dtype))
