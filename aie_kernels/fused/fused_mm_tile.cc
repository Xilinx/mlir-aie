// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One call runs the whole tile, so it is timed as one interval.
#define MM_FUSED_WHOLE_TILE_MARKERS
#include "mm_fused.h"

#if !ACTIVATIONS_NATIVE_TANH
#include "lut_based_ops.cpp"
#endif

// B's k chunks are CT_K * N elements apart. Prepacked bfp16ebs8 B stores 8
// elements in 9 bytes, and pointer arithmetic on it counts bytes
// (llvm-aie#1232), so the stride is stepped in bytes for either storage.
#ifdef MM_FUSED_BFP16_B
constexpr int b_chunk_bytes = MM_FUSED_CT_K * MM_FUSED_TILE_N / 8 * 9;
#else
constexpr int b_chunk_bytes =
    MM_FUSED_CT_K * MM_FUSED_TILE_N * sizeof(bfloat16);
#endif

// mode and the clamp bounds reach the epilogue as runtime words, so one
// compiled tile serves every activation the mode mask admits. The bounds are
// int32 bit patterns because npu_write_rtp only writes i32. b is bfp16ebs8
// blocks, not bf16, under MM_FUSED_BFP16_B; the pointer is opaque either way.
extern "C" void fused_mm_tile(bfloat16 *a, bfloat16 *b, bfloat16 *c,
                              int32_t mode, int32_t clamp_min_bits,
                              int32_t clamp_max_bits) {
  event0();
  alignas(aie::vector_decl_align) float acc[MM_FUSED_TILE_M * MM_FUSED_TILE_N];
  mm_fused_acc_init(acc);
  for (int k = 0; k < MM_FUSED_TILE_K / MM_FUSED_CT_K; ++k)
    for (int band = 0; band < MM_FUSED_TILE_M / MM_FUSED_TILE_MA; ++band)
      mm_fused_k_step(
          a + (k * MM_FUSED_TILE_M + band * MM_FUSED_TILE_MA) * MM_FUSED_CT_K,
          reinterpret_cast<mm_fused_b_elem_t *>(reinterpret_cast<uint8_t *>(b) +
                                                k * b_chunk_bytes),
          acc, band);
  for (int outer = 0;
       outer < MM_FUSED_TILE_M * MM_FUSED_TILE_N / (2 * MM_FUSED_OUT_CHUNK);
       ++outer)
    for (int half = 0; half < 2; ++half)
      mm_fused_epilogue_chunk(c + (outer * 2 + half) * MM_FUSED_OUT_CHUNK, acc,
                              outer, half, mode, clamp_min_bits,
                              clamp_max_bits);
  event1();
}
