# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""FlashAttention forward pass implementations using CUDA and Triton."""

import torch
import warnings


def _check_cuda_availability():
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. FlashAttention kernels will fail to build or run.",
            UserWarning,
            stacklevel=2,
        )
        return

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        warnings.warn(
            f"Some FlashAttention kernels require Ampere (SM80) or newer GPUs. "
            f"Detected: {props.name} (SM{props.major}{props.minor}).",
            UserWarning,
            stacklevel=2,
        )


_check_cuda_availability()
