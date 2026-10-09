//===- decode_attn_qk_kvh2.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_attn_qk_common.h"
#include "decode_geometry.h"
#include "decode_lut_exp.h"
#include "utils.h"
#if ATTN_IMPL == ATTN_IMPL_2x4x1

inline aie::vector<bf16, 16> update(bf16 *m, float *c,
                                    aie::vector<bf16, 16> &out,
                                    aie::mask<16> &mask, bool &is_first);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first, bool &is_up);


// Written for a head dim of 512.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first, bool &is_up) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

  bf16 *__restrict pQ1 = pQ;

  // The loop schedules only with AIE_TRY_INITIATION_INTERVAL, at II=8 with 34
  // stages. colQ is DH/8 = 64, so the pipeline fills.
  MMUL C00(aie::zeros<bf16, MMUL::size_C>());
  MMUL C01(aie::zeros<bf16, MMUL::size_C>());

  AIE_TRY_INITIATION_INTERVAL(8)
  for (unsigned i = 0; i < colQ; ++i) {
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;

    aie::vector<bf16, MMUL::size_B> B0 =
        aie::transpose(aie::load_v<MMUL::size_B>(pK1), 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B1 =
        aie::transpose(aie::load_v<MMUL::size_B>(pK2), 8, 8);
    pK2 += MMUL::size_B * 2;

    C00.mac(A0, B0);
    C01.mac(A0, B1);
  }
  auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                   C01.template to_vector<bf16>(), 8);

  aie::vector<bf16, 32> mout2, mout3 = aie::zeros<bf16, 32>();

  if (is_up) {
    mout2 = aie::filter_even(mout0.first, 32);
    mout3 = aie::filter_odd(mout0.first, 32);
  } else {
    mout2 = aie::filter_even(mout0.second, 32);
    mout3 = aie::filter_odd(mout0.second, 32);
  }

  aie::vector<bf16, 16> out[4];
  out[0] = aie::filter_even(mout2, 16);
  out[1] = aie::filter_odd(mout2, 16);
  out[2] = aie::filter_even(mout3, 16);
  out[3] = aie::filter_odd(mout3, 16);

  AIE_LOOP_UNROLL_FULL
  for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
    aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
    aie::store_v(pY1, vec);
    pY1 += 16;
  }
  pY1 += 16 * ATTN_GROUPS_PADDING;
}

// ---------------------------------------------------------------------------
// Entry points. See decode_attn_qk.cc. This variant has two KV heads per
// core, so one s round consumes two k buffers. The Worker calls attn_qk_half
// once per head. This file uses the 2-argument lock form, which carries no
// memory operand; see utils.h.
static PingPong k_pingpong;

extern "C" {

// See attn_qk_begin in decode_attn_qk.cc.
void attn_qk_begin(bf16 *m) {
  static const aie::vector<bf16, 16> neg_inf =
      aie::broadcast<bf16, 16>(-0x1.FEp127f);
  aie::store_v(m, neg_inf);
}

// Half a round: one of the two KV heads. j selects which quarter of s/m/c_local
// it lands in and which half of the mmul result _attn_qk keeps (is_up).
void attn_qk_half(bf16 *q, bf16 *k_ping, bf16 *k_pong, bf16 *s, bf16 *m,
                  float *c_local, int j, int iter, int L0) {
  static const aie::vector<int, 16> idx = aie::vector<int, 16>(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  bf16 *k = k_pingpong.next(k_ping, k_pong);
  int i = L0 - 16 * iter;
  bool is_first = (iter == 0);
  aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
  bool is_up = (j == 0);
  _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s + j * 4 * 16, m + j * 4,
                                        c_local + j * 4, mask, is_first, is_up);
}

// Both halves done: publish c into the tail of the s object.
void attn_qk_store_c(bf16 *s, float *c_local) {
  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
  aie::store_v(c, c_vec);
}
}

#endif