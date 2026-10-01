//===- decode_attn_kv_kvh2_core.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// attn_kv_s_begin, then attn_kv_v_half's arithmetic for KV head kv_head,
// without the v lock, for the kernel harness. One call folds in one head; see
// flm_gemma4_attn_kv_kvh2_core.
// _ATTN_KV_L_SLOT in python/iron/kernels/flm_gemma4.py describes sv and ly.
#include "decode_attn_kv_kvh2.cc"

extern "C" void attn_kv_kvh2_round_core(bf16 *sv, float *ly_in, float *ly_out,
                                        int kv_head) {
  bf16 *s = sv;
  bf16 *v = sv + Q_HEADS_PADDED_PER_CU * 16 + 32;
  event0();
  copy_vectorized<float, 8 * DH + 16>(ly_out, ly_in);
  attn_kv_s_begin(s, ly_out, ly_out + 8 * DH);
  attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v, ly_out, kv_head == 0);
  event1();
}
