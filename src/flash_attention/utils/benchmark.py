# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0
#
# Some of the code in this file is inspired by:
#
# Dao-AILab/flash-attention:
# Copyright (c) 2022-2025, Tri Dao.
# Licensed under BSD 3 clause.

"""Runtime and memory benchmarking utilities."""

from typing import Any

import torch
import torch.utils.benchmark as benchmark


__all__ = [
    "memory",
    "runtime",
]


def _create_timer(
    func,
    *inputs,
    convergence_threshold: float = 0.001,
    min_run_time: float = 0.01,
    max_run_time: float = 10,
    **kwinputs: dict[str, Any],
) -> benchmark.Measurement:
    timer = benchmark.Timer(
        stmt="func(*inputs, **kwinputs)",
        globals={"func": func, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    return timer.adaptive_autorange(
        threshold=convergence_threshold,
        min_run_time=min_run_time,
        max_run_time=max_run_time,
    )


def runtime(func, *args, **kwargs) -> tuple[float, float]:
    S_TO_MS = 1e3
    result = _create_timer(func, *args, **kwargs)

    median_in_ms = result.median * S_TO_MS
    iqr_in_ms = result.iqr * S_TO_MS

    return median_in_ms, iqr_in_ms


def memory(func, *inputs, **kwinputs) -> float:
    BYTE_TO_MEBIBYTE = 1 / 1024**2

    torch.accelerator.memory.empty_cache()
    torch.accelerator.memory.reset_peak_memory_stats()

    torch.accelerator.synchronize()
    func(*inputs, **kwinputs)
    torch.accelerator.synchronize()

    peak_memory_in_mib = (
        torch.accelerator.memory.max_memory_allocated() * BYTE_TO_MEBIBYTE
    )
    torch.accelerator.memory.empty_cache()

    return peak_memory_in_mib
