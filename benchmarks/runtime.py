# Copyright 2025 Cayro Neto, Leandro Campos.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks the runtime of FlashAttention implementations."""

import argparse
import sys

import pandas as pd
import torch

from flash_attention import cuda
from flash_attention import triton
from flash_attention.utils import baseline
from flash_attention.utils import benchmark


def print_results(
    *, median_df: pd.DataFrame, iqr_df: pd.DataFrame, markdown: bool = False
):
    if markdown:
        floatfmt = [".0f", ".0f", ".0f", ".6f", ".6f", ".6f"]

        print("Median Runtime (ms)\n")
        print(median_df.to_markdown(index=False, floatfmt=floatfmt))
        print("")
        print("Runtime Interquartile Range (ms)\n")
        print(iqr_df.to_markdown(index=False, floatfmt=floatfmt))
    else:
        print("-" * 70)
        print("Median Runtime (ms)")
        print("-" * 70)
        print(median_df.to_string(index=False))
        print("\n")
        print("-" * 70)
        print("Runtime Interquartile Range (ms)")
        print("-" * 70)
        print(iqr_df.to_string(index=False))


def run_benchmarks(args: argparse.Namespace):
    device = "cuda"
    dtype = torch.float16
    emb_dim = 512

    head_dim_vals = [64, 128]
    bsz_ntok_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096)]

    median_results = []
    iqr_results = []

    print("Running benchmarks (this may take a few minutes)...", file=sys.stderr)

    for head_dim in head_dim_vals:
        for batch_size, num_tokens in bsz_ntok_vals:
            torch.cuda.empty_cache()

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

            base_record = {
                "head_dim": head_dim,
                "batch_size": batch_size,
                "num_tokens": num_tokens,
            }
            median_record = base_record.copy()
            iqr_reccord = base_record.copy()

            # --- Benchmark: Baseline ---
            baseline_med, baseline_iqr = benchmark.runtime(
                baseline.multi_head_attention, query, key, value
            )
            median_record["baseline"] = baseline_med
            iqr_reccord["baseline"] = baseline_iqr

            # --- Benchmark: CUDA ---
            cuda_med, cuda_iqr = benchmark.runtime(
                cuda.multi_head_attention_forward, query, key, value
            )
            median_record["cuda"] = cuda_med
            iqr_reccord["cuda"] = cuda_iqr

            # --- Benchmark: Triton ---
            triton_med, triton_iqr = benchmark.runtime(
                triton.multi_head_attention_forward, query, key, value
            )
            median_record["triton"] = triton_med
            iqr_reccord["triton"] = triton_iqr

            median_results.append(median_record)
            iqr_results.append(iqr_reccord)

    print_results(
        median_df=pd.DataFrame.from_records(median_results),
        iqr_df=pd.DataFrame.from_records(iqr_results),
        markdown=args.markdown,
    )


def main():
    parser = argparse.ArgumentParser(description="FlashAttention Runtime Benchmark")
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
