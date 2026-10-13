//===- decode_attn_kv.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_attn_kv_common.h"
#include "decode_layout.h"
#include "utils.h"
#if ATTN_IMPL == ATTN_IMPL_1x8x1
typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, float *__restrict c);

constexpr int v_prod_lock = FLM_GEMMA4_DECODE_ATTN_KV_V_PROD_LOCK;
constexpr int v_cons_lock = FLM_GEMMA4_DECODE_ATTN_KV_V_CONS_LOCK;
constexpr int o_prod_lock = FLM_GEMMA4_DECODE_ATTN_KV_O_PROD_LOCK;
constexpr int o_cons_lock = FLM_GEMMA4_DECODE_ATTN_KV_O_CONS_LOCK;
constexpr int l_cons_lock = FLM_GEMMA4_DECODE_ATTN_KV_L_CONS_LOCK;

extern "C" {}
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, float *__restrict c) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  y_acc_dtype *__restrict pY1 = pY;
  aie::vector<bf16, MMUL::size_A> S0, S1;
  load_split_s(pS, S0, S1);

  aie::vector<y_acc_dtype, 64> CORRECT = broadcast_c(c);

  bf16 *__restrict pV1 = pV;

  for (unsigned j = 0; j < colQ; j += 4) {
    bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
    bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
    bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
    bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

    aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
    pV01 += MMUL::size_B * colQ;
    aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
    pV02 += MMUL::size_B * colQ;
    aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
    pV03 += MMUL::size_B * colQ;
    aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
    pV04 += MMUL::size_B * colQ;

    aie::vector<y_acc_dtype, MMUL::size_C> acc_y00 =
        aie::load_v<MMUL::size_C>(pY1);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_y01 =
        aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_y02 =
        aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_y03 =
        aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

    aie::accum<accfloat, MMUL::size_C> ACC_Y00;
    aie::accum<accfloat, MMUL::size_C> ACC_Y01;
    aie::accum<accfloat, MMUL::size_C> ACC_Y02;
    aie::accum<accfloat, MMUL::size_C> ACC_Y03;
    ACC_Y00 = aie::mul(CORRECT, acc_y00);
    ACC_Y01 = aie::mul(CORRECT, acc_y01);
    ACC_Y02 = aie::mul(CORRECT, acc_y02);
    ACC_Y03 = aie::mul(CORRECT, acc_y03);

    MMUL Y00(ACC_Y00);
    MMUL Y01(ACC_Y01);
    MMUL Y02(ACC_Y02);
    MMUL Y03(ACC_Y03);

    Y00.mac(S0, V00);
    Y01.mac(S0, V01);
    Y02.mac(S0, V02);
    Y03.mac(S0, V03);

    V00 = aie::load_v<MMUL::size_B>(pV01);
    V01 = aie::load_v<MMUL::size_B>(pV02);
    V02 = aie::load_v<MMUL::size_B>(pV03);
    V03 = aie::load_v<MMUL::size_B>(pV04);

    Y00.mac(S1, V00);
    Y01.mac(S1, V01);
    Y02.mac(S1, V02);
    Y03.mac(S1, V03);

    aie::store_v(pY1, Y00.template to_vector<y_acc_dtype>());
    pY1 += MMUL::size_C;
    aie::store_v(pY1, Y01.template to_vector<y_acc_dtype>());
    pY1 += MMUL::size_C;
    aie::store_v(pY1, Y02.template to_vector<y_acc_dtype>());
    pY1 += MMUL::size_C;
    aie::store_v(pY1, Y03.template to_vector<y_acc_dtype>());
    pY1 += MMUL::size_C;
  }
}

// ---------------------------------------------------------------------------
// Entry points. The design's Worker runs the round loop, and `s` arrives
// through an ObjectFifo, so these functions do not touch the s locks. The
// design owns the y and l accumulators.
//
// v and o use hand-written lock and BD edges. This kernel holds their locks and
// the v ping/pong parity, which must stay in lockstep with the BD chain.
static PingPong v_pingpong;

extern "C" {

void attn_kv_begin(float *y, float *l) {
  attn_kv_begin_impl<8 * DH, l_cons_lock>(y, l);
}

// One round. s is the ObjectFifo object.
void attn_kv_round(bf16 *s, bf16 *v_ping, bf16 *v_pong, float *y, float *l) {
  bf16 *v = v_pingpong.acquire(v_ping, v_pong, v_cons_lock);

  float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
  calculate_l(s, c, l);
  attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y, c);

  _lock_release_p(v, v_prod_lock);
}

void attn_kv_finish(float *y, bf16 *o, float *l) {
  attn_kv_finish_impl<8 * DH, o_prod_lock, o_cons_lock, O_DOWN_REPEATS>(y, o,
                                                                        l);
}
}
#endif