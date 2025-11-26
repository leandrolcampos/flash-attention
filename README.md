# FlashAttention

**FlashAttention forward pass implementations using CUDA and Triton.**

This repository provides implementations of the FlashAttention-1 algorithm (forward pass), comparing a high-level Triton kernel against a low-level CUDA kernel.

## Benchmarks

Performance measured on an **NVIDIA GeForce RTX 4070 Laptop GPU** (Ampere Architecture) for `emb_dim` 512.

### Median Runtime (ms)

Lower is better.

|   head_dim |   batch_size |   num_tokens |   baseline |     cuda |   triton |
|-----------:|-------------:|-------------:|-----------:|---------:|---------:|
|         64 |           32 |          512 |   3.903739 | 0.525061 | 0.659767 |
|         64 |           16 |         1024 |   7.445763 | 0.989893 | 1.117725 |
|         64 |            8 |         2048 |  14.473714 | 1.923763 | 2.152186 |
|         64 |            4 |         4096 |  28.570170 | 3.785288 | 4.211343 |
|        128 |           32 |          512 |   2.066309 | 0.559682 | 0.776051 |
|        128 |           16 |         1024 |   3.941361 | 1.050164 | 1.444576 |
|        128 |            8 |         2048 |   7.515268 | 2.004821 | 2.781371 |
|        128 |            4 |         4096 |  14.708045 | 3.908829 | 5.411558 |

### Peak Memory Usage (MiB)

Lower is better.

|   head_dim |   batch_size |   num_tokens |   baseline |   cuda |   triton |
|-----------:|-------------:|-------------:|-----------:|-------:|---------:|
|         64 |           32 |          512 |    440.125 | 72.125 |   72.125 |
|         64 |           16 |         1024 |    824.125 | 72.125 |   72.125 |
|         64 |            8 |         2048 |   1592.125 | 72.125 |   72.125 |
|         64 |            4 |         4096 |   3128.125 | 72.125 |   72.125 |
|        128 |           32 |          512 |    248.125 | 72.125 |   72.125 |
|        128 |           16 |         1024 |    440.125 | 72.125 |   72.125 |
|        128 |            8 |         2048 |    824.125 | 72.125 |   72.125 |
|        128 |            4 |         4096 |   1592.125 | 72.125 |   72.125 |

-----

## Getting Started

This project uses **Pixi** for dependency management and task automation.

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/leandrolcampos/flash-attention.git
cd flash-attention
```

*Note: The `--recursive` flag is required to fetch the CUTLASS submodule for the CUDA kernel.*

### 2. Run Tests

Verify the correctness of both Triton and CUDA implementations against the PyTorch baseline.

```bash
# Runs pytest in the 'tst' environment
pixi run -e tst test
```

### 3. Execute Benchmarks

Execute the runtime and memory benchmarks to get tables similar to those above.

**Runtime Benchmark:**

```bash
# Runs benchmarks/runtime.py (supports --markdown flag for markdown output)
pixi run -e tst bench-runtime
```

**Memory Benchmark:**

```bash
# Runs benchmarks/memory.py (supports --markdown flag for markdown output)
pixi run -e tst bench-memory
```

### Additional Information

For a full list of dependencies, available tasks, and development tools (such as `tree-repo` or `ninja-compdb`), please refer to the `pyproject.toml` file.
