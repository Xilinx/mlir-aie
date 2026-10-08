//===- decode_attn_kv_kvh2.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_attn_kv_common.h"
#include "decode_geometry.h"
#include "decode_layout.h"
#include "utils.h"
#if ATTN_IMPL == ATTN_IMPL_2x4x1

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, bool is_up);


extern "C" {}

// IS_UP is a template parameter, so the half-select in the loop is resolved at
// compile time. A runtime branch in the loop blocks back-to-back vector loads
// and MACs and causes spills.
template <unsigned colQ, unsigned r, unsigned s, unsigned t, bool IS_UP>
void attn_fv_impl(bf16 *__restrict pS, bf16 *__restrict pV,
                  y_acc_dtype *__restrict pY) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  y_acc_dtype *__restrict pY1 = pY;
  aie::vector<bf16, MMUL::size_A> S0, S1;
  load_split_s(pS, S0, S1);

  bf16 *__restrict pV1 = pV;

  // Keep this loop rolled, with one accumulator per iteration. With the
  // extract<32> below, it pipelines at II=21 with 12 stages. An unroll count of
  // 2 fails to schedule, and 4 or 8 exceed the MII limit.
  for (unsigned j = 0; j < colQ; ++j) {
    bf16 *__restrict pV0 = pV1 + j * MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV0);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y0 =
        aie::load_v<MMUL::size_C>(pY1);

    MMUL Y0(acc_Y0);
    Y0.mac(S0, V0);

    V0 = aie::load_v<MMUL::size_B>(pV0 + MMUL::size_B);
    Y0.mac(S1, V0);

    // extract<32> selects the same half as filter_even/filter_odd(., 32).
    // filter_* spills the accumulator to a stack array and copies it back in a
    // second basic block, and the pipeliner rejects a loop with two blocks.
    if constexpr (IS_UP) {
      aie::store_v(
          pY1, Y0.template to_vector<y_acc_dtype>().template extract<32>(0));
    } else {
      aie::store_v(
          pY1 + 32,
          Y0.template to_vector<y_acc_dtype>().template extract<32>(1));
    }
    pY1 += MMUL::size_C;
  }
}

// Resolves the runtime half-select to a template argument once per call.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, bool is_up) {
  if (is_up)
    attn_fv_impl<colQ, r, s, t, true>(pS, pV, pY);
  else
    attn_fv_impl<colQ, r, s, t, false>(pS, pV, pY);
}

// ---------------------------------------------------------------------------
// Entry points. See decode_attn_kv.cc. This variant has two KV heads per
// core, so one s round consumes two v buffers. The Worker calls attn_kv_v_half
// once per head.
static PingPong v_pingpong;

extern "C" {

void attn_kv_begin(float *y, float *l) {
  attn_kv_begin_impl<8 * DH>(y, l);
}

// Start of an s round: fold the new scores into l and rescale the accumulator.
// The Worker calls it once per s object, before either v half.
void attn_kv_s_begin(bf16 *s, float *y, float *l) {
  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  calculate_l(s, c, l);
  calculate_y<8 * DH, 128>(y, c);
}

// Half a round: one of the two KV heads.
void attn_kv_v_half(bf16 *s, bf16 *v_ping, bf16 *v_pong, float *y, int j) {
  bf16 *v = v_pingpong.next(v_ping, v_pong);
  attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y, j == 0);
}

void attn_kv_finish(float *y, bf16 *o, float *l) {
  attn_kv_finish_impl<8 * DH>(y, o, l);
}
}

#endif