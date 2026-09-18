// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mm_fused.cc"

#if __AIE_ARCH__ == 20
#include "lut_based_ops.cpp"
#endif

extern "C" void fused_mm_tile(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  alignas(aie::vector_decl_align) float acc[MM_FUSED_TILE_M * MM_FUSED_TILE_N];
  mm_fused_acc_init(acc);
  for (int k = 0; k < MM_FUSED_TILE_K / MM_FUSED_CT_K; ++k)
    for (int band = 0; band < MM_FUSED_TILE_M / MM_FUSED_TILE_MA; ++band)
      mm_fused_k_step(a + (k * MM_FUSED_TILE_M + band * MM_FUSED_TILE_MA) *
                              MM_FUSED_CT_K,
                      b + k * MM_FUSED_CT_K * MM_FUSED_TILE_N, acc, band);
  for (int outer = 0;
       outer < MM_FUSED_TILE_M * MM_FUSED_TILE_N / (2 * MM_FUSED_OUT_CHUNK);
       ++outer)
    for (int half = 0; half < 2; ++half)
      mm_fused_epilogue_chunk(c + (outer * 2 + half) * MM_FUSED_OUT_CHUNK, acc,
                              outer, half);
}
