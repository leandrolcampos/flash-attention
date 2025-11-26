// Copyright 2025 Cayro Neto, Leandro Campos.
// SPDX-License-Identifier: Apache-2.0

/// Python bindings for the CUDA-based FlashAttention forward pass.

#include <torch/extension.h>

torch::Tensor multi_head_attention_forward(torch::Tensor query,
                                           torch::Tensor key,
                                           torch::Tensor value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_head_attention_forward",
        torch::wrap_pybind_function(multi_head_attention_forward),
        "multi_head_attention_forward");
}
