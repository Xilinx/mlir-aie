//===- prefill_core.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The copies of the running state sit outside the event0/event1 pair, so a
// trace interval covers the same arithmetic the production entry point runs.
#include "prefill.cc"

#include <stdint.h>

extern "C" {

// out = y | c | l | prev_m | new_m, bytes concatenated. Every part starts at
// a multiple of its store width from the buffer start.
void attn_round_begin_core(uint8_t *out) {
  event0();
  float *y = (float *)out;
  float *c = y + LQ * DH;
  float *l = c + LQ;
  bfloat16 *prev_m = (bfloat16 *)(l + LQ);
  attn_round_begin(prev_m, prev_m + LQ, c, l, y);
  event1();
}

// out = s | m: S for one key chunk, masked, and m folded with its row maxima.
void attn_qk_core(bfloat16 *q, bfloat16 *k, bfloat16 *m, int inner_k,
                  int inner_q, int inner_k_current, bfloat16 *out) {
  bfloat16 *s = out;
  bfloat16 *m_out = out + LQ * LK;
  copy_vectorized<bfloat16, LQ * LK>(m_out, m);
  event0();
  G::attn_qk(s, q, k);
  apply_mask_and_get_max<LQ, LK>(s, m_out, inner_k, inner_q, inner_k_current);
  event1();
}

// sv = s | v. y_out = y + s * v.
void attn_fv_core(float *y, bfloat16 *sv, float *y_out) {
  copy_vectorized<float, LQ * DH>(y_out, y);
  event0();
  flm_attn_fv(y_out, sv, sv + LQ * LK);
  event1();
}

// in_bf16 = s | m | prev_m and in_f32 = y | l. out_bf16 = s | new_m and
// out_f32 = y | l | c, each after attn_block_mid.
void attn_block_mid_core(bfloat16 *in_bf16, float *in_f32, bfloat16 *out_bf16,
                         float *out_f32) {
  bfloat16 *m = in_bf16 + LQ * 128;
  bfloat16 *prev_m = m + LQ * LK;
  float *y = out_f32;
  float *l = y + LQ * DH;
  copy_vectorized<bfloat16, LQ * 128>(out_bf16, in_bf16);
  copy_vectorized<float, LQ * DH + LQ>(out_f32, in_f32);
  event0();
  attn_block_mid(out_bf16, m, out_bf16 + LQ * 128, prev_m, l + LQ, l, y);
  event1();
}

} // extern "C"
