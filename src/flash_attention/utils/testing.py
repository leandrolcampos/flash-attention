# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities."""

import torch


__all__ = [
    "all_close",
]


def _max_error_bound(dtype: torch.dtype) -> float:
    if dtype == torch.float64 or dtype == torch.float32:
        return 1e-6
    elif dtype == torch.float16:
        return 1e-3
    elif dtype == torch.bfloat16:
        return 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def all_close(
    *,
    actual: torch.Tensor,
    desired: torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> bool:
    assert actual.dtype == desired.dtype

    default_tol = _max_error_bound(desired.dtype)
    return torch.allclose(
        actual,
        desired,
        rtol=rtol or default_tol,
        atol=atol or default_tol,
        equal_nan=equal_nan,
    )
