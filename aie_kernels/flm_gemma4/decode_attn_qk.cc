//===- decode_attn_qk.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_attn_qk_common.h"
#include "decode_layout.h"
#include "decode_lut_exp.h"
#include "utils.h"

// FLM_GEMMA4_DECODE_ATTN_QK_SWA picks the head dim, as the RoPE flag does in
// decode_rope.cc. Only its =1 build takes the 2x4x1 path below:
// decode_attn_qk_kvh2.cc serves two KV heads at the other head dim.
#ifndef FLM_GEMMA4_DECODE_ATTN_QK_SWA
#define FLM_GEMMA4_DECODE_ATTN_QK_SWA 0
#endif
constexpr int QK_DH = FLM_GEMMA4_DECODE_ATTN_QK_SWA ? SWA_DH : DH;

inline aie::vector<bf16, 16> update(bf16 *m, float *c,
                                    aie::vector<bf16, 16> &out,
                                    aie::mask<16> &mask, bool &is_first);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bf16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first);

constexpr int k_prod_lock = FLM_GEMMA4_DECODE_ATTN_QK_K_PROD_LOCK;
constexpr int k_cons_lock = FLM_GEMMA4_DECODE_ATTN_QK_K_CONS_LOCK;
constexpr int l_cons_lock = FLM_GEMMA4_DECODE_ATTN_QK_L_CONS_LOCK;

// ---------------------------------------------------------------------------
// Entry points. See decode_attn_kv.cc; k has the role of v there. q arrives
// through an ObjectFifo, so this kernel has no q lock. The k parity persists
// across dispatches and never resets. One round covers every KV head of the
// core.
static PingPong k_pingpong;

extern "C" {

// Primes the running max. The Worker acquires q before this call, and q
// arrives only after the host writes the RTPs, so the Worker may read L only
// after this function returns. The kv tile waits for the l_cons_lock release.
void attn_qk_begin(bf16 *m) {
  static const aie::vector<bf16, 16> neg_inf =
      aie::broadcast<bf16, 16>(-0x1.FEp127f);
  aie::store_v(m, neg_inf);
  _lock_release_p(m, l_cons_lock);
}

// One round. iter is the round index and sets the causal mask. s is the
// ObjectFifo object.
void attn_qk_round(bf16 *q, bf16 *k_ping, bf16 *k_pong, bf16 *s, bf16 *m,
                   float *c_local, int iter, int L0) {
  bf16 *k = k_pingpong.acquire(k_ping, k_pong, k_cons_lock);

  int i = L0 - 16 * iter;
  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  bool is_first = (iter == 0);

  aie::vector<int, 16> _idxv;
  for (int _z = 0; _z < 16; _z++)
    _idxv.set(_z + 1, _z);
  aie::mask<16> mask = aie::le(_idxv, (i < 16) ? i : 16);

  _attn_qk<QK_DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s, m, c_local, mask, is_first);

  // c goes in the tail of the s object, after the scores.
  aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
  aie::store_v(c, c_vec);

  _lock_release_p(k, k_prod_lock);
}
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

  {
    bf16 *__restrict pQ1 = pQ;

    // Do not add AIE_TRY_INITIATION_INTERVAL here. colQ is SWA_DH/8 = 32 here,
    // too few iterations to fill the pipeline. See decode_attn_qk_kvh2.cc.
    MMUL C00(aie::zeros<bf16, MMUL::size_C>());
    MMUL C01(aie::zeros<bf16, MMUL::size_C>());

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

    aie::vector<bf16, 32> mout2, mout3;

    mout2 = aie::filter_even(mout0.first, 32);
    mout3 = aie::filter_odd(mout0.first, 32);

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
  {
    bf16 *__restrict pQ1 = pQ;

    // No AIE_TRY_INITIATION_INTERVAL; see the loop above.
    MMUL C00(aie::zeros<bf16, MMUL::size_C>());
    MMUL C01(aie::zeros<bf16, MMUL::size_C>());

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

    aie::vector<bf16, 32> mout2, mout3;
    mout2 = aie::filter_even(mout0.second, 32);
    mout3 = aie::filter_odd(mout0.second, 32);

    aie::vector<bf16, 16> out[4];
    out[0] = aie::filter_even(mout2, 16);
    out[1] = aie::filter_odd(mout2, 16);
    out[2] = aie::filter_even(mout3, 16);
    out[3] = aie::filter_odd(mout3, 16);
    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
      aie::vector<bf16, 16> vec =
          update(m + 4 + h, c + 4 + h, out[h], mask, is_first);
      aie::store_v(pY1, vec);
      pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1

// The QK mmul for one KV head per core.
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
  {
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    for (unsigned i = 1; i < colQ; ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 64> mout2 = mout0.first;
    aie::vector<bf16, 64> mout3 = mout0.second;

    aie::vector<bf16, 16> out[8];

    out[0] = mout2.extract<16>(0);
    out[1] = mout2.extract<16>(1);
    out[2] = mout2.extract<16>(2);
    out[3] = mout2.extract<16>(3);
    out[4] = mout3.extract<16>(0);
    out[5] = mout3.extract<16>(1);
    out[6] = mout3.extract<16>(2);
    out[7] = mout3.extract<16>(3);

    for (int h = 0; h < Q_HEADS_PER_CU; h++) {
      aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
      aie::store_v(pY1, vec);
      pY1 += 16;
    }
    pY1 += ATTN_GROUPS_PADDING * 16;
  }
}

#endif
