//===- decode_swa_attn_kv.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_attn_kv_common.h"
#include "decode_layout.h"
#include "utils.h"

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY);


extern "C" {

// ---------------------------------------------------------------------------
// Entry points. See decode_attn_kv.cc.
//
// l holds 8 floats, because calculate_l and scale_div_aie use only l[0..7].
// A 64-byte l ties with the RTP buffer in the allocator's size order and
// displaces the RTP buffer.
static PingPong v_pingpong;

void swa_attn_kv_begin(float *y, float *l) {
  attn_kv_begin_impl<8 * SWA_DH>(y, l);
}

void swa_attn_kv_round(bf16 *s, bf16 *v_ping, bf16 *v_pong, float *y,
                       float *l) {
  bf16 *v = v_pingpong.next(v_ping, v_pong);

  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  calculate_l(s, c, l);
  // 64 lanes, not 128 as in decode_attn_kv_kvh2.cc. This translation unit
  // has more live state, and a 128-lane (512-byte) accumulator makes it spill.
  calculate_y<8 * SWA_DH, 64>(y, c);
  attn_fv<SWA_DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y);
}

void swa_attn_kv_finish(float *y, bf16 *o, float *l) {
  attn_kv_finish_impl<8 * SWA_DH>(y, o, l);
}
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  y_acc_dtype *__restrict pY1 = pY;
  aie::vector<bf16, MMUL::size_A> S0, S1;
  load_split_s(pS, S0, S1);

  bf16 *__restrict pV1 = pV;
  bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;

  // The CORRECT rescale is in calculate_y, a separate function. The optimizer
  // fuses two adjacent loops over the same trip count and pointer, so a
  // separate loop here would put the fp32 multiply back into this loop.
  //
  // Each V stream has its own loop and accumulator. An mmul initializes its
  // accumulator element-wise, so the low 32 lanes of Y0 depend only on the low
  // 32 lanes of the tile, and the high 32 lanes of Y1 only on the high 32.
  for (unsigned j = 0; j < colQ; ++j) {
    bf16 *__restrict pV0p = pV1 + j * MMUL::size_B;
    y_acc_dtype *__restrict pYj = pY + j * MMUL::size_C;
    aie::accum<accfloat, MMUL::size_C> A;
    A.from_vector(aie::load_v<MMUL::size_C>(pYj));
    MMUL Y0(A);
    Y0.mac(S0, aie::load_v<MMUL::size_B>(pV0p));
    Y0.mac(S1, aie::load_v<MMUL::size_B>(pV0p + MMUL::size_B * colQ * 2));
    aie::store_v(pYj,
                 Y0.template to_vector<y_acc_dtype>().template extract<32>(0));
  }

  for (unsigned j = 0; j < colQ; ++j) {
    bf16 *__restrict pV1p = pV2 + j * MMUL::size_B;
    y_acc_dtype *__restrict pYj = pY + j * MMUL::size_C;
    aie::accum<accfloat, MMUL::size_C> A;
    A.from_vector(aie::load_v<MMUL::size_C>(pYj));
    MMUL Y1(A);
    Y1.mac(S0, aie::load_v<MMUL::size_B>(pV1p));
    Y1.mac(S1, aie::load_v<MMUL::size_B>(pV1p + MMUL::size_B * colQ * 2));
    aie::store_v(pYj + 32,
                 Y1.template to_vector<y_acc_dtype>().template extract<32>(1));
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  aie::vector<bf16, MMUL::size_A> S0, S1;
  load_split_s(pS, S0, S1);

  bf16 *__restrict pV1 = pV;

  // The CORRECT rescale is in calculate_y. See the 2x4x1 attn_fv above for why
  // it is a separate function.
  for (unsigned j = 0; j < colQ; ++j) {
    bf16 *__restrict pV0 = pV1 + j * MMUL::size_B;
    y_acc_dtype *__restrict pYj = pY + j * MMUL::size_C;

    aie::accum<accfloat, MMUL::size_C> ACC_Y;
    ACC_Y.from_vector(aie::load_v<MMUL::size_C>(pYj));
    MMUL Y(ACC_Y);
    Y.mac(S0, aie::load_v<MMUL::size_B>(pV0));
    Y.mac(S1, aie::load_v<MMUL::size_B>(pV0 + MMUL::size_B * colQ));

    aie::store_v(pYj, Y.template to_vector<y_acc_dtype>());
  }
}

#endif