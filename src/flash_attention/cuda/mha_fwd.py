# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""JIT compilation and wrapper for the CUDA FlashAttention kernel."""

import os
import torch
from torch.utils.cpp_extension import load


__all__ = [
    "multi_head_attention_forward",
]


# -------------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------------

# Current directory: src/flash_attention/cuda
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root directory: src/flash_attention/cuda/../../.. => Repo Root
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))

# Build directory at the repository root
_BUILD_DIR = os.path.join(_ROOT_DIR, "build")
os.makedirs(_BUILD_DIR, exist_ok=True)

# Cutlass paths (inside src/flash_attention/cuda/third_party/cutlass)
_CUTLASS_DIR = os.path.join(_CURRENT_DIR, "third_party", "cutlass")
_INCLUDE_DIRS = [
    os.path.join(_CUTLASS_DIR, "include"),
    os.path.join(_CUTLASS_DIR, "tools", "util", "include"),
]

# -------------------------------------------------------------------------
# Compilation Configuration
# -------------------------------------------------------------------------

# The kernel uses SM80 specific instructions, so we target Ampere (8.0)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

# Remove flags that might conflict with half-precision compilation
_REMOVE_NVCC_FLAGS = [
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

for flag in _REMOVE_NVCC_FLAGS:
    try:
        torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass

# -------------------------------------------------------------------------
# JIT Compilation
# -------------------------------------------------------------------------

_mha_fwd_native = load(
    name="mha_fwd_native",
    sources=[
        os.path.join(_CURRENT_DIR, "mha_fwd.cpp"),
        os.path.join(_CURRENT_DIR, "mha_fwd.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        *[f"-I{inc}" for inc in _INCLUDE_DIRS],
    ],
    build_directory=_BUILD_DIR,
    verbose=False,
)


def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _mha_fwd_native.multi_head_attention_forward(query, key, value)
