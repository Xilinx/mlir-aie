//===- decode_swa_attn_kv_core.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One swa_attn_kv_round without the v lock, for the kernel harness.
// _ATTN_KV_L_SLOT in python/iron/kernels/flm_gemma4.py describes sv and ly.
#include "decode_swa_attn_kv.cc"

extern "C" void swa_attn_kv_round_core(bf16 *sv, float *ly_in, float *ly_out) {
  bf16 *s = sv;
  bf16 *v = sv + Q_HEADS_PADDED_PER_CU * 16 + 32;
  event0();
  copy_vectorized<float, 8 * SWA_DH + 16>(ly_out, ly_in);
  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  calculate_l(s, c, ly_out + 8 * SWA_DH);
  calculate_y<8 * SWA_DH, 64>(ly_out, c);
  attn_fv<SWA_DH / 8, GQA_R, GQA_S, GQA_T>(s, v, ly_out);
  event1();
}
