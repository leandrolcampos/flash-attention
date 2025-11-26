# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks the peak memory usage of FlashAttention implementations."""

import argparse
import sys

import pandas as pd
import torch

from flash_attention import cuda
from flash_attention import triton
from flash_attention.utils import baseline
from flash_attention.utils import benchmark


def print_results(*, memory_df: pd.DataFrame, markdown: bool = False):
    if markdown:
        floatfmt = [".0f", ".0f", ".0f", ".3f", ".3f", ".3f"]

        print("Peak Memory Usage (MiB)\n")
        print(memory_df.to_markdown(index=False, floatfmt=floatfmt))
    else:
        print("-" * 70)
        print("Peak Memory Usage (MiB)")
        print("-" * 70)
        print(memory_df.to_string(index=False))


def run_benchmarks(args: argparse.Namespace):
    device = "cuda"
    dtype = torch.float16
    emb_dim = 512

    head_dim_vals = [64, 128]
    bsz_ntok_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096)]

    memory_results = []

    print("Running memory benchmarks...", file=sys.stderr)

    for head_dim in head_dim_vals:
        for batch_size, num_tokens in bsz_ntok_vals:
            num_heads = emb_dim // head_dim

            query, key, value = [
                torch.randn(
                    batch_size,
                    num_heads,
                    num_tokens,
                    head_dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(3)
            ]

            record = {
                "head_dim": head_dim,
                "batch_size": batch_size,
                "num_tokens": num_tokens,
            }

            # --- Benchmark: Baseline ---
            # Baseline often consumes O(N^2) memory, so we catch potential OOMs
            try:
                baseline_mem = benchmark.memory(
                    baseline.multi_head_attention, query, key, value
                )
                record["baseline"] = baseline_mem
            except torch.cuda.OutOfMemoryError:
                record["baseline"] = float("inf")

            # --- Benchmark: CUDA ---
            cuda_mem = benchmark.memory(
                cuda.multi_head_attention_forward, query, key, value
            )
            record["cuda"] = cuda_mem

            # --- Benchmark: Triton ---
            triton_mem = benchmark.memory(
                triton.multi_head_attention_forward, query, key, value
            )
            record["triton"] = triton_mem

            memory_results.append(record)

    print_results(
        memory_df=pd.DataFrame.from_records(memory_results),
        markdown=args.markdown,
    )


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Memory Benchmark")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print the output as markdown tables instead of plain text.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmarks.", file=sys.stderr)
        sys.exit(1)

    run_benchmarks(args)


if __name__ == "__main__":
    main()
