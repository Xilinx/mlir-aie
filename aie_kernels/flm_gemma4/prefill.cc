//===- prefill.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FastFlowLM's Gemma 4 prefill attention on linalg/flash_attn_prefill.h, whose
// q * k^T, mask, row sums, S reorder and block bookkeeping it uses. The head
// dim is FLM_GEMMA4_PREFILL_HEAD_DIM, 512 or 256.
//
// It keeps FastFlowLM's numerics: the softmax scales by log2(e) alone (Gemma
// 4's attention scale is 1), c and y round as FastFlowLM's kernel does, S * V
// runs through the mmul, and the epilogue multiplies by a bf16 1/l. k and v
// arrive in one ping-pong buffer pair behind two of this core's locks,
// FLM_GEMMA4_PREFILL_IN_{PROD,CONS}_LOCK.
#include "../linalg/flash_attn_prefill.h"
#include "utils.h"

#include <aie_api/aie.hpp>

#ifndef FLM_GEMMA4_PREFILL_HEAD_DIM
#define FLM_GEMMA4_PREFILL_HEAD_DIM 512
#endif

// The sliding-window design's stack size counts on attn_fv and calculate_y
// keeping their own frames.
#if FLM_GEMMA4_PREFILL_HEAD_DIM == 256
#define FLM_PREFILL_OUT_OF_LINE __attribute__((noinline))
#else
#define FLM_PREFILL_OUT_OF_LINE
#endif

namespace {

using G = PrefillGeom<FLM_GEMMA4_PREFILL_HEAD_DIM>;
constexpr int LQ = G::LQ; // query rows per core
constexpr int LK = G::LK; // key rows per step
constexpr int DH = G::DH; // head dim
constexpr bool WINDOWED = DH == 256;

constexpr int IN_PROD_LOCK = FLM_GEMMA4_PREFILL_IN_PROD_LOCK;
constexpr int IN_CONS_LOCK = FLM_GEMMA4_PREFILL_IN_CONS_LOCK;

constexpr bfloat16 LOG2_E = (bfloat16)1.4426950408889634f;
constexpr float NEG_INF = -0x1.FEp127f;

// attn_qk_step and attn_fv_step alternate one flag, because k and v share the
// buffer pair.
PingPong in_pingpong;

// S = exp(S - row max), in place.
void flm_softmax(bfloat16 *__restrict pS, bfloat16 *__restrict new_m_local) {
  for (int b = 0; b < 128 / LK; b++) {
    for (int q = 0; q < LQ; q++) {
      aie::vector<bfloat16, LK> s_vec = aie::load_v<LK>(pS);
      aie::vector<bfloat16, LK> Vec = aie::sub(s_vec, *(new_m_local + q));
      aie::accum<accfloat, LK> Vec_acc = aie::mul(Vec, LOG2_E);
      Vec = aie::exp2<bfloat16>(Vec_acc.template to_vector<float>());
      aie::store_v(pS, Vec);
      pS += LK;
    }
  }
}

// c = exp(previous row max - new row max).
void flm_calculate_c(float *c, bfloat16 *prev_m_local, bfloat16 *new_m_local) {
  for (int i = 0; i < LQ; i++) {
    bfloat16 inner_correct = aie::sub(*(prev_m_local + i), *(new_m_local + i));
    inner_correct = aie::mul(inner_correct, LOG2_E);
    aie::vector<bfloat16, LK> correct_vec =
        aie::broadcast<bfloat16, LK>(inner_correct);
    aie::accum<accfloat, LK> correct_acc;
    correct_acc.from_vector(correct_vec);
    correct_vec = aie::exp2<bfloat16>(correct_acc.template to_vector<float>());
    *(c + i) = (float)correct_vec.get(0);
  }
}

// y = c * y.
FLM_PREFILL_OUT_OF_LINE void flm_calculate_y(float *y, float *c) {
  for (int i = 0; i < LQ / 8; i++) {
    aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[i * 8]);
    aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[i * 8 + 1]);
    aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[i * 8 + 2]);
    aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[i * 8 + 3]);
    aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[i * 8 + 4]);
    aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[i * 8 + 5]);
    aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[i * 8 + 6]);
    aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[i * 8 + 7]);
    aie::vector<float, 64> CORRECT =
        aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

    float *pY = y + i * 8 * DH;
    for (unsigned j = 0; j < DH / 8; j += 1) {
      aie::vector<float, 64> Y = aie::load_v<64>(pY);
      aie::accum<accfloat, 64> ACC_Y = aie::mul(CORRECT, Y);
      aie::store_v(pY, ACC_Y.template to_vector<float>());
      pY += 64;
    }
  }
}

// y += S * V for one key chunk, through the mmul: under
// AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 that rounds its operands to bfp16,
// where G::attn_fv's vector macs keep them bf16, so the two differ.
FLM_PREFILL_OUT_OF_LINE void flm_attn_fv(float *__restrict pY,
                                         bfloat16 *__restrict pS,
                                         bfloat16 *__restrict pV) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  constexpr unsigned colA = LK / 8, colB = DH / 8;
  // The query row blocks of S, and each one's y: one for the global geometry,
  // two for the sliding-window one.
  constexpr unsigned rowA = LQ / 8;
  aie::vector<bfloat16, 64> S[2][2];
  for (unsigned z = 0; z < rowA; z++)
    for (unsigned k = 0; k < colA; k++)
      S[z][k] = aie::load_v<64>(pS + (z * colA + k) * 64);

  for (unsigned j = 0; j < colB; j += 2) {
    for (unsigned z = 0; z < rowA; z++) {
      float *__restrict pY1 = pY + z * colB * MMUL::size_C + j * MMUL::size_C;
      MMUL Y0(aie::load_v<MMUL::size_C>(pY1));
      MMUL Y1(aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C));
      for (unsigned k = 0; k < colA; k++) {
        Y0.mac(S[z][k],
               aie::load_v<MMUL::size_B>(pV + (j * colA + k) * MMUL::size_B));
        Y1.mac(S[z][k], aie::load_v<MMUL::size_B>(pV + ((j + 1) * colA + k) *
                                                           MMUL::size_B));
      }
      aie::store_v(pY1, Y0.template to_vector<float>());
      aie::store_v(pY1 + MMUL::size_C, Y1.template to_vector<float>());
    }
  }
}

// o = y * inv_l for one 64-element chunk, both narrowed to bf16 first.
void flm_scale_by_inv_l(bfloat16 *o, bfloat16 *inv_l, float *y) {
  constexpr int vec_factor = 64;

  aie::vector<bfloat16, 8> L0 = aie::broadcast<bfloat16, 8>(inv_l[0]);
  aie::vector<bfloat16, 8> L1 = aie::broadcast<bfloat16, 8>(inv_l[1]);
  aie::vector<bfloat16, 8> L2 = aie::broadcast<bfloat16, 8>(inv_l[2]);
  aie::vector<bfloat16, 8> L3 = aie::broadcast<bfloat16, 8>(inv_l[3]);
  aie::vector<bfloat16, 8> L4 = aie::broadcast<bfloat16, 8>(inv_l[4]);
  aie::vector<bfloat16, 8> L5 = aie::broadcast<bfloat16, 8>(inv_l[5]);
  aie::vector<bfloat16, 8> L6 = aie::broadcast<bfloat16, 8>(inv_l[6]);
  aie::vector<bfloat16, 8> L7 = aie::broadcast<bfloat16, 8>(inv_l[7]);

  auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

  aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
  aie::accum<accfloat, vec_factor> Y00_acc;
  Y00_acc.from_vector(Y00);
  aie::accum<accfloat, vec_factor> AL00 =
      aie::mul(Y00_acc.template to_vector<bfloat16>(), LL00);
  aie::store_v(o, AL00.template to_vector<bfloat16>());
}

} // namespace

extern "C" {

// The round and block counts are computed here: a shift in the core body
// lowers to a vector srs intrinsic that the core's link step cannot resolve.
void attn_rounds(int *L_begin_buffer, int *L_end_buffer, int *n_out) {
  event0();
  n_out[0] = (L_end_buffer[0] >> 7) - (L_begin_buffer[0] >> 7);
  event1();
}

#if FLM_GEMMA4_PREFILL_HEAD_DIM == 256
void attn_blocks(int *L_begin_buffer, int *window_size_buffer, const int i,
                 int *n_out) {
  event0();
  const int pointer_block_q = L_begin_buffer[0] + i * 128;
  int pointer_block_k = pointer_block_q - window_size_buffer[0];
  pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
  n_out[0] = ((pointer_block_q - pointer_block_k) >> 7) + 1;
  event1();
}
#else
void attn_blocks(int *L_begin_buffer, const int i, int *n_out) {
  event0();
  n_out[0] = (L_begin_buffer[0] >> 7) + i + 1;
  event1();
}
#endif

void attn_round_begin(bfloat16 *prev_m, bfloat16 *new_m, float *c, float *l,
                      float *y) {
  static const aie::vector<bfloat16, LK> neg_inf =
      aie::broadcast<bfloat16, LK>(NEG_INF);
  static const aie::vector<float, LQ> one = aie::broadcast<float, LQ>(1.0f);
  aie::store_v(prev_m, neg_inf);
  aie::store_v(new_m, neg_inf);
  aie::store_v(c, one);
  zero_256<float, LQ>(l);
  zero_256<float, LQ * DH>(y);
}

void attn_block_begin(bfloat16 *m, bfloat16 *prev_m) {
  event0();
  block_begin_impl<DH>(m, prev_m);
  event1();
}

void attn_qk_step(bfloat16 *s, bfloat16 *__restrict q,
                  bfloat16 *__restrict in_ping, bfloat16 *__restrict in_pong,
                  bfloat16 *m, int *L_begin_buffer, int *window_size_buffer,
                  const int row, const int col, const int i,
                  const int block_idx, const int j) {
  const int pointer_block_q = L_begin_buffer[0] + i * 128;
  const int pointer_block_inner_q = G::inner_q(pointer_block_q, row, col);
  const int window_size = window_size_buffer[0];
  int pointer_block_k = pointer_block_q - window_size;
  pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
  int pointer_block_inner_k = pointer_block_inner_q - window_size;
  pointer_block_inner_k =
      pointer_block_inner_k > -LK ? pointer_block_inner_k : -LK;
  const int block_location = block_idx * (128 / LK) + j;
  const int pointer_block_inner_k_current =
      pointer_block_k + block_location * LK;
  bfloat16 *k = in_pingpong.acquire(in_ping, in_pong, IN_CONS_LOCK);
  G::attn_qk(s + j * LQ * LK, q + G::q_offset(col), k);
  _lock_release_p(k, IN_PROD_LOCK);
  apply_mask_and_get_max<LQ, LK>(s + j * LQ * LK, m, pointer_block_inner_k,
                                 pointer_block_inner_q,
                                 pointer_block_inner_k_current);
}

void attn_block_mid(bfloat16 *s, bfloat16 *m, bfloat16 *new_m, bfloat16 *prev_m,
                    float *c, float *l, float *y) {
  for (int j = 0; j < LQ; j++) {
    aie::vector<bfloat16, LK> m_vec = aie::load_v<LK>(m + j * LK);
    bfloat16 vm = aie::reduce_max(m_vec);
    *(new_m + j) = (bfloat16)vm;
  }
  flm_softmax(s, new_m);
  flm_calculate_c(c, prev_m, new_m);
  G::reorder_s(s);
  G::calculate_l(l, c, s);
  flm_calculate_y(y, c);
}

void attn_fv_step(float *y, bfloat16 *s, bfloat16 *__restrict in_ping,
                  bfloat16 *__restrict in_pong, const int j) {
  bfloat16 *v = in_pingpong.acquire(in_ping, in_pong, IN_CONS_LOCK);
  flm_attn_fv(y, s + j * LQ * LK, v);
  _lock_release_p(v, IN_PROD_LOCK);
}

void attn_block_end(bfloat16 *prev_m, bfloat16 *new_m) {
  event0();
  aie::vector<bfloat16, LQ> v = aie::load_v<LQ>(new_m);
  aie::store_v(prev_m, v);
  event1();
}

void attn_finalize(float *l, bfloat16 *inv_l) {
  event0();
  aie::vector<float, LQ> l_vec = aie::load_v<LQ>(l);
  l_vec = aie::inv(l_vec);
  aie::accum<accfloat, LQ> l_acc;
  l_acc.from_vector(l_vec);
  aie::store_v(inv_l, l_acc.template to_vector<bfloat16>());
  event1();
}

// Chunk c of LQ * DH / 64 of this round's output, row block c / (DH / 8).
void attn_epilogue(bfloat16 *__restrict o, bfloat16 *inv_l, float *y,
                   const int c) {
  event0();
  if constexpr (LQ == 8) {
    flm_scale_by_inv_l(o, inv_l, y + c * 64);
  } else {
    const int o_row = c / (DH / 8);
    const int o_col = c % (DH / 8);
    flm_scale_by_inv_l(o, inv_l + o_row * 8, y + o_row * 8 * DH + o_col * 64);
  }
  event1();
}

} // extern "C"
