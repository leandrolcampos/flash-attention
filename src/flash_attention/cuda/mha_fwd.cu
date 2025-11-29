// Copyright 2025 Cayro Neto, Leandro Campos.
// SPDX-License-Identifier: Apache-2.0
//
// Some of the code in this file is adapted from:
//
// luliyucoordinate/cute-flash-attention
//
// NVIDIA/cutlass:
// Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES.
// Licensed under BSD 3 clause.
//
// References:
//
// Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022).
// FlashAttention: Fast and Memory-Efficient Exact Attention
//   with IO-Awareness.
// Advances in Neural Information Processing Systems, 35,
//   16344-16359.
// https://dl.acm.org/doi/10.5555/3600270.3601459
//
// Dao, T. (2023). FlashAttention-2: Faster Attention with
//   Better Parallelism and Work Partitioning.
// arXiv preprint arXiv:2307.08691.
// https://doi.org/10.48550/arXiv.2307.08691

/// CUDA kernel implementation of the FlashAttention forward pass using CuTe.

#include <cmath>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cute/tensor.hpp>

template <class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  cute::ArrayEngine<cute::half_t, cute::cosize_v<SmemLayoutQ>> Q;
  cute::ArrayEngine<cute::half_t, cute::cosize_v<SmemLayoutK>> K;
  cute::ArrayEngine<cute::half_t, cute::cosize_v<SmemLayoutV>> V;
};

template <typename config>
__global__ void mha_fwd_kernel(cute::half_t *O, cute::half_t const *Q,
                               cute::half_t const *K, cute::half_t const *V,
                               int q_seq_len, int k_seq_len,
                               float softmax_scale) {
  using namespace cute;
  using X = Underscore;

  //
  // Type Aliases & Constants
  //
  // Extract architecture-specific types and constants from the Config struct
  // to keep the kernel code more generic and clean.
  //

  using SmemLayoutQ = typename config::SmemLayoutQ;
  using SmemLayoutK = typename config::SmemLayoutK;
  using SmemLayoutV = typename config::SmemLayoutV;
  using SmemLayoutO = typename config::SmemLayoutO;

  using SmemLayoutVtNoSwizzle = typename config::SmemLayoutVtNoSwizzle;
  using SmemLayoutVt = typename config::SmemLayoutVt;

  using Gmem2SmemCopyQKV = typename config::Gmem2SmemCopyQKV;

  using Smem2RmemCopyAtom = typename config::Smem2RmemCopyAtom;
  using Smem2RmemCopyAtomT = typename config::Smem2RmemCopyAtomT;
  using Rmem2SmemCopyAtomO = typename config::Rmem2SmemCopyAtomO;

  using TiledMma = typename config::TiledMma;

  using Smem2GmemCopyO = typename config::Smem2GmemCopyO;

  //
  // Tensor Creation (Global Memory)
  //
  // Define logical views (tensors) over the raw pointers in global memory.
  // These represent the full problem size.
  //

  constexpr int kHeadDim = config::kHeadDim;
  constexpr int kBlockM = config::kBlockM;
  constexpr int kBlockN = config::kBlockN;

  const int block_m_idx = blockIdx.x;
  const int global_head_idx = blockIdx.y;
  const int thread_idx = threadIdx.x;

  const int q_offset = global_head_idx * q_seq_len * kHeadDim;
  const int k_offset = global_head_idx * k_seq_len * kHeadDim;

  Tensor mQ = make_tensor(make_gmem_ptr(Q + q_offset),
                          make_shape(q_seq_len, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor mK = make_tensor(make_gmem_ptr(K + k_offset),
                          make_shape(k_seq_len, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor mV = make_tensor(make_gmem_ptr(V + k_offset),
                          make_shape(k_seq_len, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor mO = make_tensor(make_gmem_ptr(O + q_offset),
                          make_shape(q_seq_len, Int<kHeadDim>{}),
                          make_stride(Int<kHeadDim>{}, Int<1>{}));

  // Slice the global tensors to get the tiles for the current CTA/Block
  Tensor gQ = local_tile(mQ, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}),
                         make_coord(block_m_idx, _));
  Tensor gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                         make_coord(0, _));
  Tensor gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                         make_coord(0, _));

  //
  // Tensor Creation (Shared Memory)
  //
  // Allocate shared memory and define layouts for Q, K, V tiles used
  // during the main loop.
  //

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor sQ = make_tensor(make_smem_ptr(smem.Q.begin()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(smem.K.begin()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(smem.V.begin()), SmemLayoutV{});
  Tensor sVtNoSwizzle =
      make_tensor(make_smem_ptr(smem.V.begin()), SmemLayoutVtNoSwizzle{});
  Tensor sVt = make_tensor(make_smem_ptr(smem.V.begin()), SmemLayoutVt{});

  //
  // Partitioning (Copy & Compute)
  //
  // Divide the work of copying (GMEM->SMEM) and computing (MMA) among threads.
  // Each thread gets a specific slice of the data.
  //

  // GMEM -> SMEM copy partitioning
  Gmem2SmemCopyQKV g2s_copy_QKV;
  ThrCopy g2s_thr_copy_QKV = g2s_copy_QKV.get_thread_slice(thread_idx);
  Tensor tQgQ = g2s_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = g2s_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = g2s_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = g2s_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = g2s_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = g2s_thr_copy_QKV.partition_D(sV);

  // MMA partitioning for the first GEMM (S = Q @ K.T)
  TiledMma mma;
  ThrMMA thr_mma = mma.get_slice(thread_idx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ); // (MMA_A,MMA1_M,MMA1_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK); // (MMA_B,MMA1_N,MMA1_K)
  Tensor tSrS = partition_fragment_C(             // ((2,2),MMA1_M,MMA1_N)
      mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

  static_assert(std::is_same_v<typename TiledMma::ValTypeC, float>);
  CUTE_STATIC_ASSERT_V(shape<0>(tSrS) == make_shape(Int<2>{}, Int<2>{}));

  // MMA partitioning for the second GEMM (O = P @ V)
  // MMA2_M == MMA1_M, MMA2_N == MMA1_K * 2, MMA2_K == MMA1_N / 2
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA_B,MMA2_N,MMA2_K)
  Tensor tOrO = partition_fragment_C(             // ((2,2),MMA2_M,MMA2_N)
      mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  CUTE_STATIC_ASSERT_V(shape<0>(tOrO) == make_shape(Int<2>{}, Int<2>{}));
  CUTE_STATIC_ASSERT_V(size<1>(tSrS) == size<1>(tOrO));

  //
  // Epilogue Setup
  //
  // Initialize accumulators for softmax statistics and output.
  //

  clear(tOrO);

  Tensor rScoreMax =
      make_tensor<float>(Shape<Int<2 * size<1>(tSrS)>>{}); // (2*MMA1_M)
  Tensor rScoreSum = make_fragment_like(rScoreMax);

  CUTE_UNROLL
  for (int i = 0; i < size(rScoreMax); i++) {
    rScoreMax(i) = -INFINITY;
    rScoreSum(i) = 0;
  }

  // Reshape tOrO (accumulator) to match softmax/normalization access pattern
  // ((2,2),MMA2_M,MMA2_N) to ((2,MMA2_M),(2,MMA2_N))
  auto tOrO_layout_split = logical_divide(tOrO.layout(), Shape<Int<2>>{});
  auto tOrO_norm_layout = make_layout(
      make_layout(get<1>(get<0>(tOrO_layout_split)), get<1>(tOrO_layout_split)),
      make_layout(get<0>(get<0>(tOrO_layout_split)),
                  get<2>(tOrO_layout_split)));
  Tensor tOrO_norm_view = make_tensor(tOrO.data(), tOrO_norm_layout);

  //
  // SMEM -> RMEM Retiling
  //
  // Create partitioned views for loading data from SMEM to Registers
  // during the main loop (S2R).
  //

  TiledCopy s2r_copy_Q = make_tiled_copy_A(Smem2RmemCopyAtom{}, mma);
  ThrCopy s2r_thr_copy_Q = s2r_copy_Q.get_thread_slice(thread_idx);
  Tensor tXsQ = s2r_thr_copy_Q.partition_S(sQ);
  Tensor tXrQ = s2r_thr_copy_Q.retile_D(tSrQ);

  CUTE_STATIC_ASSERT_V(size<2>(tXsQ) == size<2>(tSrQ));
  CUTE_STATIC_ASSERT_V(size<2>(tXrQ) == size<2>(tSrQ));

  TiledCopy s2r_copy_K = make_tiled_copy_B(Smem2RmemCopyAtom{}, mma);
  ThrCopy s2r_thr_copy_K = s2r_copy_K.get_thread_slice(thread_idx);
  Tensor tXsK = s2r_thr_copy_K.partition_S(sK);
  Tensor tXrK = s2r_thr_copy_K.retile_D(tSrK);

  CUTE_STATIC_ASSERT_V(size<2>(tXsK) == size<2>(tSrQ));
  CUTE_STATIC_ASSERT_V(size<2>(tXrK) == size<2>(tSrQ));

  TiledCopy s2r_copy_V = make_tiled_copy_B(Smem2RmemCopyAtomT{}, mma);
  ThrCopy s2r_thr_copy_V = s2r_copy_V.get_thread_slice(thread_idx);
  Tensor tXsVt = s2r_thr_copy_V.partition_S(sVt);
  Tensor tXrVt = s2r_thr_copy_V.retile_D(tOrVt);

  //
  // Main Loop Prologue
  //
  // Prepare input data for the loop.
  //

  // Copy Q tile from GMEM to SMEM
  copy(g2s_copy_QKV, tQgQ, tQsQ);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // Apply scaling to Q
  // The scaling factor includes M_LOG2E because we use exp2f (base 2)
  // instead of expf (natural base) for performance
  const half2 sm_scale_half2 = __float2half2_rn(softmax_scale * M_LOG2E);

  static_assert((size(tQsQ) % 8) == 0);
  Tensor tQsQ_128b = recast<uint128_t>(tQsQ);

  CUTE_UNROLL
  for (int i = 0; i < size(tQsQ_128b); i++) {
    uint128_t raw_chunk = tQsQ_128b(i);
    half2 *vec_chunk = reinterpret_cast<half2 *>(&raw_chunk);

    CUTE_UNROLL
    for (int j = 0; j < 4; j++) {
      vec_chunk[j] = __hmul2_rn(sm_scale_half2, vec_chunk[j]);
    }

    tQsQ_128b(i) = raw_chunk;
  }

  // Pre-load first K and V tiles
  copy(g2s_copy_QKV, tKgK, tKsK);
  cp_async_fence();
  copy(g2s_copy_QKV, tVgV, tVsV);
  cp_async_fence();

  int n_block_max = ceil_div(k_seq_len, kBlockN);

  //
  // Main Loop
  //
  // Iterate over K and V tiles to compute O = softmax(Q @ K.T) @ V
  //

  CUTE_NO_UNROLL
  for (int n_block = 0; n_block < n_block_max; n_block++) {
    clear(tSrS);

    /// Wait for K tile to arrive in SMEM
    cp_async_wait<1>();
    __syncthreads();

    //
    // GEMM 1: Compute S = Q @ K.T
    //

    copy(s2r_copy_Q, tXsQ(_, _, Int<0>{}), tXrQ(_, _, Int<0>{}));
    copy(s2r_copy_K, tXsK(_, _, Int<0>{}), tXrK(_, _, Int<0>{}));

    CUTE_UNROLL
    for (int k_block = 0; k_block < size<2>(tSrQ); k_block++) {
      if (k_block < (size<2>(tSrQ) - 1)) {
        copy(s2r_copy_Q, tXsQ(_, _, k_block + 1), tXrQ(_, _, k_block + 1));
        copy(s2r_copy_K, tXsK(_, _, k_block + 1), tXrK(_, _, k_block + 1));
      }

      gemm(mma, tSrQ(_, _, k_block), tSrK(_, _, k_block), tSrS);
    }

    //
    // Update of the online softmax statistics
    //

    // Reshape tSrS to facilitate row-wise reduction
    // ((2,2),MMA1_M,MMA1_N) -> ((2,MMA1_M),(2,MMA1_N))
    auto tSrS_layout_split = logical_divide(tSrS.layout(), Shape<Int<2>>{});
    auto tSrS_sm_layout =
        make_layout(make_layout(get<1>(get<0>(tSrS_layout_split)),
                                get<1>(tSrS_layout_split)),
                    make_layout(get<0>(get<0>(tSrS_layout_split)),
                                get<2>(tSrS_layout_split)));
    Tensor tSrS_sm_view = make_tensor(tSrS.data(), tSrS_sm_layout);

    CUTE_STATIC_ASSERT_V(size(rScoreMax) == size<0>(tSrS_sm_view));
    CUTE_STATIC_ASSERT_V(size(rScoreSum) == size<0>(tSrS_sm_view));
    CUTE_STATIC_ASSERT_V(size<0>(tOrO_norm_view) == size<0>(tSrS_sm_view));

    Tensor rScoreMax_old = make_fragment_like(rScoreMax);
    copy(rScoreMax, rScoreMax_old);

    CUTE_UNROLL
    for (int i = 0; i < size<0>(tSrS_sm_view); i++) {
      float &score_max_row = rScoreMax(i);
      float &score_sum_row = rScoreSum(i);

      CUTE_UNROLL
      for (int j = 0; j < size<1>(tSrS_sm_view); j++) {
        score_max_row = max(score_max_row, tSrS_sm_view(i, j));
      }
      score_max_row =
          max(score_max_row, __shfl_xor_sync(0xffffffff, score_max_row, 0x2));
      score_max_row =
          max(score_max_row, __shfl_xor_sync(0xffffffff, score_max_row, 0x1));

      float score_scale = exp2f(rScoreMax_old(i) - score_max_row);

      CUTE_UNROLL
      for (int j = 0; j < size<1>(tOrO_norm_view); j++) {
        tOrO_norm_view(i, j) *= score_scale;
      }

      float score_sum_row_local = 0;

      CUTE_UNROLL
      for (int j = 0; j < size<1>(tSrS_sm_view); j++) {
        tSrS_sm_view(i, j) = exp2f(tSrS_sm_view(i, j) - score_max_row);
        score_sum_row_local += tSrS_sm_view(i, j);
      }
      score_sum_row_local +=
          __shfl_xor_sync(0xffffffff, score_sum_row_local, 0x2);
      score_sum_row_local +=
          __shfl_xor_sync(0xffffffff, score_sum_row_local, 0x1);
      score_sum_row = score_sum_row * score_scale + score_sum_row_local;
    }

    __syncthreads();
    // Trigger prefetch for the next tile of K
    if (n_block != n_block_max - 1) {
      gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                      make_coord(n_block + 1, _));
      tKgK = g2s_thr_copy_QKV.partition_S(gK(_, _, 0));
      copy(g2s_copy_QKV, tKgK, tKsK);
    }
    cp_async_fence();

    // Wait for V tile to arrive in SMEM
    cp_async_wait<1>();
    __syncthreads();

    //
    // GEMM 2: Compute O = exp2f(S) @ V
    //

    // Convert scores from float to half for GEMM
    Tensor rScore_half = make_tensor_like<half_t>(tSrS_sm_view);
    Tensor rScore_half2 = recast<half2>(rScore_half);
    Tensor rScoreAcc_float2 = recast<float2>(tSrS_sm_view);

    CUTE_UNROLL
    for (int i = 0; i < size(rScore_half2); i++) {
      rScore_half2(i) = __float22half2_rn(rScoreAcc_float2(i));
    }

    // Reshape exp2f(S) to match GEMM 2 access pattern
    // ((2,MMA1_M),(2,MMA1_N)) to ((2,2,2),MMA1_M,MMA1_N/2)
    //   where ((2,2,2),MMA1_M,MMA1_N/2) == (MMA_A,MMA2_M,MMA2_K)
    auto tSrS_sm_layout_split =
        logical_divide(tSrS_sm_view.layout(), Shape<X, Shape<X, Int<2>>>{});
    auto tOrS_layout =
        make_layout(make_layout(get<0>(get<1>(tSrS_sm_layout_split)),
                                get<0>(get<0>(tSrS_sm_layout_split)),
                                get<0>(get<1>(get<1>(tSrS_sm_layout_split)))),
                    get<1>(get<0>(tSrS_sm_layout_split)),
                    get<1>(get<1>(get<1>(tSrS_sm_layout_split))));
    Tensor tOrS = make_tensor(rScore_half.data(), tOrS_layout);

    CUTE_STATIC_ASSERT_V(size<2>(tXsVt) == size<2>(tOrS));
    CUTE_STATIC_ASSERT_V(size<2>(tXrVt) == size<2>(tOrS));

    copy(s2r_copy_V, tXsVt(_, _, Int<0>{}), tXrVt(_, _, Int<0>{}));

    CUTE_UNROLL
    for (int k_block = 0; k_block < size<2>(tOrS); k_block++) {
      if (k_block < (size<2>(tOrS) - 1)) {
        copy(s2r_copy_V, tXsVt(_, _, k_block + 1), tXrVt(_, _, k_block + 1));
      }
      gemm(mma, tOrS(_, _, k_block), tOrVt(_, _, k_block), tOrO);
    }

    __syncthreads();
    // Trigger prefetch for the next tile of V
    if (n_block != n_block_max - 1) {
      gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}),
                      make_coord(n_block + 1, _));
      tVgV = g2s_thr_copy_QKV.partition_S(gV(_, _, 0));
      copy(g2s_copy_QKV, tVgV, tVsV);
    }
    cp_async_fence();
  }

  //
  // Epilogue
  //
  // Finalize normalization of O and write result to global memory
  //

  // Normalize the tile of O by dividing by sum(exp2f(S), axis=1)
  CUTE_UNROLL
  for (int i = 0; i < size<0>(tOrO_norm_view); i++) {
    float score_sum_inv_row = __frcp_rn(rScoreSum(i));

    CUTE_UNROLL
    for (int j = 0; j < size<1>(tOrO_norm_view); j++) {
      tOrO_norm_view(i, j) *= score_sum_inv_row;
    }
  }

  // Convert the tile of O back to half
  Tensor tOrO_half = make_tensor_like<half_t>(tOrO);
  Tensor tOrO_half2 = recast<half2>(tOrO_half);
  Tensor tOrO_float2 = recast<float2>(tOrO);

  CUTE_UNROLL
  for (int i = 0; i < size(tOrO_half2); i++) {
    tOrO_half2(i) = __float22half2_rn(tOrO_float2(i));
  }

  static_assert(cosize_v<SmemLayoutO> == cosize_v<SmemLayoutQ>);
  Tensor sO = make_tensor(sQ.data(), SmemLayoutO{}); // Reuse Q SMEM for O

  TiledCopy r2s_copy_O = make_tiled_copy_C(Rmem2SmemCopyAtomO{}, mma);
  ThrCopy r2s_thr_copy_O = r2s_copy_O.get_thread_slice(thread_idx);
  Tensor tXrO = r2s_thr_copy_O.retile_S(tOrO_half);
  Tensor tXsO = r2s_thr_copy_O.partition_D(sO);

  // Copy the tile of O from RMEM to SMEM
  copy(r2s_copy_O, tXrO, tXsO);

  Tensor gO = local_tile(mO, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}),
                         make_coord(block_m_idx, _));
  Smem2GmemCopyO s2g_copy_O;
  ThrCopy s2g_thr_copy_O = s2g_copy_O.get_thread_slice(thread_idx);
  Tensor tOsO = s2g_thr_copy_O.partition_S(sO);
  Tensor tOgO = s2g_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();
  // Copy the tile of O from SMEM to GMEM
  copy(s2g_copy_O, tOsO, tOgO);
}

namespace config {
using namespace cute;

template <int kHeadDim_ = 64, int kBlockM_ = 64, int kBlockN_ = 64>
struct FlashAttentionConfig {
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = 64;

  static_assert((kHeadDim % 64) == 0);
  static_assert((kBlockM % 64) == 0);
  static_assert((kBlockN % 64) == 0);

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockK>>, Stride<Int<kBlockK>, Int<1>>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutAtomVtNoSwizzle =
      Layout<Shape<Int<kBlockK>, Int<kBlockN>>, Stride<Int<1>, Int<kBlockK>>>;
  using SmemLayoutVtNoSwizzle = decltype(tile_to_shape(
      SmemLayoutAtomVtNoSwizzle{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  using SmemLayoutAtomVt =
      decltype(composition(Swizzle<3, 3, 3>{}, SmemLayoutAtomVtNoSwizzle{}));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  using Gmem2SmemCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
      Layout<Shape<_1, _8>>{}));                // Val layout  1x8 m-major

  using Smem2RmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
  using Smem2RmemCopyAtomT = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;
  using Rmem2SmemCopyAtomO = Copy_Atom<DefaultCopy, half_t>;

  using TiledMma = decltype(make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<_4, _1, _1>>{}, // 4x1x1 MMA Atoms
      Tile<_64, _16, _16>{}));     // 64x16x16 Tiled MMA for LDSM

  using Smem2GmemCopyO = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // Thr layout 16x8 k-major
      Layout<Shape<_1, _8>>{}));                // Val layout  1x8 m-major

  CUTE_STATIC_ASSERT_V(size(Gmem2SmemCopyQKV{}) ==
                       size(TiledMma{})); // NumThreads
  CUTE_STATIC_ASSERT_V(size(Smem2GmemCopyO{}) ==
                       size(TiledMma{})); // NumThreads

  static constexpr int kNumThreads = size(TiledMma{});
  static_assert(kNumThreads <= 1024);

  static constexpr int kSmemSize =
      int(sizeof(SharedStorage<SmemLayoutQ, SmemLayoutK, SmemLayoutV>));
};

} // namespace config

torch::Tensor multi_head_attention_forward(at::Tensor query, at::Tensor key,
                                           at::Tensor value) {
  using namespace cute;

  TORCH_CHECK(query.is_cuda(),
              "Input tensor 'query' must be on a CUDA device.");

  TORCH_CHECK(key.get_device() == query.get_device(),
              "Input tensor 'key' must be on the same device as 'query'.");
  TORCH_CHECK(value.get_device() == query.get_device(),
              "Input tensor 'value' must be on the same device as 'query'.");

  TORCH_CHECK(query.scalar_type() == torch::kFloat16,
              "Input tensor 'query' must be of type torch.float16.");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16,
              "Input tensor 'key' must be of type torch.float16.");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16,
              "Input tensor 'value' must be of type torch.float16.");

  TORCH_CHECK(query.is_contiguous(),
              "Input tensor 'query' must be contiguous.");
  TORCH_CHECK(key.is_contiguous(), "Input tensor 'key' must be contiguous.");
  TORCH_CHECK(value.is_contiguous(),
              "Input tensor 'value' must be contiguous.");

  int batch_size = query.size(0);
  int num_heads = query.size(1);
  int query_length = query.size(2);
  int key_length = key.size(2);
  int head_dim = query.size(3);

  TORCH_CHECK(key.size(0) == batch_size,
              "Input tensor 'key' must have the same batch_size as 'query'.");
  TORCH_CHECK(key.size(1) == num_heads,
              "Input tensor 'key' must have the same num_heads as 'query'.");
  TORCH_CHECK(key.size(3) == head_dim,
              "Input tensor 'key' must have the same head_dim as 'query'.");

  TORCH_CHECK(key.sizes() == value.sizes(),
              "Input tensors 'key' and 'value' must have the same shape.");

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Unsupported head_dim: ", head_dim, ". Must be 64 or 128.");

  at::Tensor output = torch::empty_like(query);

  float softmax_scale = 1.0 / sqrt(head_dim);

  if (head_dim == 64) {
    config::FlashAttentionConfig<64, 128, 128> config;

    TORCH_CHECK(query_length % config.kBlockM == 0,
                "Query sequence length must be a multiple of ", config.kBlockM,
                " for head_dim=64. Got: ", query_length);

    TORCH_CHECK(key_length % config.kBlockN == 0,
                "Key/Value sequence length must be a multiple of ",
                config.kBlockN, " for head_dim=64. Got: ", key_length);

    dim3 dimBlock = config.kNumThreads;
    dim3 dimGrid(ceil_div(query_length, config.kBlockM),
                 batch_size * num_heads);

    auto kernel_fptr = mha_fwd_kernel<decltype(config)>;
    int smem_size = config.kSmemSize;

    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel_fptr<<<dimGrid, dimBlock, smem_size>>>(
        static_cast<half_t *>(output.data_ptr()),
        static_cast<const half_t *>(query.data_ptr()),
        static_cast<const half_t *>(key.data_ptr()),
        static_cast<const half_t *>(value.data_ptr()), query_length, key_length,
        softmax_scale);
  } else { // head_dim == 128
    config::FlashAttentionConfig<128, 64, 64> config;

    TORCH_CHECK(query_length % config.kBlockM == 0,
                "Query sequence length must be a multiple of ", config.kBlockM,
                " for head_dim=128. Got: ", query_length);

    TORCH_CHECK(key_length % config.kBlockN == 0,
                "Key/Value sequence length must be a multiple of ",
                config.kBlockN, " for head_dim=128. Got: ", key_length);

    dim3 dimBlock = config.kNumThreads;
    dim3 dimGrid(ceil_div(query_length, config.kBlockM),
                 batch_size * num_heads);

    auto kernel_fptr = mha_fwd_kernel<decltype(config)>;
    int smem_size = config.kSmemSize;

    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel_fptr<<<dimGrid, dimBlock, smem_size>>>(
        static_cast<half_t *>(output.data_ptr()),
        static_cast<const half_t *>(query.data_ptr()),
        static_cast<const half_t *>(key.data_ptr()),
        static_cast<const half_t *>(value.data_ptr()), query_length, key_length,
        softmax_scale);
  }

  return output;
}
