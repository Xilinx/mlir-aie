//===- decode_attn_kv_common.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Helpers shared by decode_attn_kv.cc, decode_attn_kv_kvh2.cc and
// decode_swa_attn_kv.cc.
#ifndef AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_KV_COMMON_H
#define AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_KV_COMMON_H

#include "decode_geometry.h"
#include "lut_based_ops.h" // getInvBf16, from aie_runtime_lib
#include "utils.h"         // lock helpers, zero_256, narrow_to_bf16

/// \brief Load a round's 128 scores and split them into the mmul's A
/// operands: S0 takes the even and S1 the odd 8-score groups of each half.
inline void load_split_s(bf16 *__restrict pS, aie::vector<bf16, 64> &S0,
                         aie::vector<bf16, 64> &S1) {
  aie::vector<bf16, 64> S_up = aie::load_v<64>(pS);
  aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);

  aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
  aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);

  S0 = aie::concat(S10, S20);
  S1 = aie::concat(S11, S21);
}

/// \brief c[0..7], each broadcast across an 8-lane group.
inline aie::vector<float, 64> broadcast_c(const float *c) {
  aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[0]);
  aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[1]);
  aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[2]);
  aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[3]);
  aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[4]);
  aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[5]);
  aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[6]);
  aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[7]);
  return aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);
}

/// \brief y = c * y over N floats, LANES (64 or 128) at a time.
///
/// The emulated fp32 multiply costs about the same per iteration at any width,
/// so a wider vector means fewer iterations. The multiply is elementwise, so
/// the width does not change the result.
template <int N, int LANES>
void calculate_y(float *__restrict y, float *__restrict c) {
  const aie::vector<float, 64> C8 = broadcast_c(c);
  aie::vector<float, LANES> CORRECT;
  if constexpr (LANES == 128)
    CORRECT = aie::concat(C8, C8); // the 64-lane pattern twice
  else
    CORRECT = C8;

  for (int i = 0; i < N / LANES; i++) {
    aie::vector<float, LANES> y_vec = aie::load_v<LANES>(y);
    aie::accum<accfloat, LANES> y_acc = aie::mul(y_vec, CORRECT);
    aie::store_v(y, y_acc.template to_vector<float>());
    y += LANES;
  }
}

/// \brief l = c * l + the row sums of the round's scores.
///
/// One 8-wide vector multiply-add after the loop updates l. Do not add
/// AIE_TRY_INITIATION_INTERVAL: it pipelines the loop at II=7 with 78 stages,
/// but the loop runs 8 iterations, so the pipeline never fills, and
/// calculate_l doubles in size and spills more.
static void calculate_l(bf16 *__restrict pS, float *__restrict c,
                        float *__restrict l) {
  aie::vector<float, 8> sums;
  for (int i = 0; i < 8; i++) {
    aie::vector<bf16, 16> s_vec = aie::load_v<16>(pS);
    sums.set(aie::reduce_add(s_vec), i);
    pS += 16;
  }
  aie::vector<float, 8> l_v = aie::load_v<8>(l);
  aie::vector<float, 8> c_v = aie::load_v<8>(c);
  aie::accum<accfloat, 8> prod = aie::mul(l_v, c_v);
  aie::store_v(l, aie::add(prod.to_vector<float>(), sums));
}

/// \brief o = a * (1/l), broadcasting each of the 8 per-row reciprocals across
/// an 8-lane group, over N elements.
///
/// This function inverts l itself with the getInvBf16 table. The prefill
/// kernels' flm_scale_by_inv_l (prefill.cc) expects l already inverted.
template <unsigned N>
void scale_div_aie(bf16 *a, bf16 *o, float *l) {

  constexpr int vec_factor = 64;

  bf16 *pA1 = a;
  bf16 *pO1 = o;
  aie::vector<bf16, 8> L_chunks[8];
  for (int i = 0; i < 8; i++) {
    bf16 l_i = (bf16)getInvBf16(l[i]);
    L_chunks[i] = aie::broadcast<bf16, 8>(l_i);
  }
  auto L = aie::concat(L_chunks[0], L_chunks[1], L_chunks[2], L_chunks[3],
                       L_chunks[4], L_chunks[5], L_chunks[6], L_chunks[7]);

  for (int d = 0; d < N / vec_factor; d++) {
    aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
    aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
    aie::store_v(pO1, AL_bf16);
    pO1 += vec_factor;
    pA1 += vec_factor;
  }
}

// ---------------------------------------------------------------------------
// Round prologue and epilogue of the three kv kernels. The lock ids are
// template parameters because each kernel has its own lock contract with the
// design.

/// \brief Zero the y/l accumulators.
///
/// The Worker acquires the qk tile's handshake lock after this call, so it
/// may read L only once that acquire returns.
template <unsigned N>
void attn_kv_begin_impl(float *y, float *l) {
  zero_256<float, N>(y);
  const aie::vector<float, 8> zero = aie::broadcast<float, 8>(0);
  aie::store_v(l, zero);
}

/// \brief Narrow the f32 accumulator in place, then scale by 1/l into o.
///
/// The Worker holds the o locks across this call; see decode_layout.h for the
/// O_REPEATS the down projection reads o with.
template <unsigned N>
void attn_kv_finish_impl(float *y, bf16 *o, float *l) {
  narrow_to_bf16<N>((bf16 *)y, y);
  scale_div_aie<N>((bf16 *)y, o, l);
}

#endif // AIE_KERNELS_FLM_GEMMA4_DECODE_ATTN_KV_COMMON_H
