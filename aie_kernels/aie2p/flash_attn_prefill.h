//===- flash_attn_prefill.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_AIE2P_FLASH_ATTN_PREFILL_H
#define AIE_KERNELS_AIE2P_FLASH_ATTN_PREFILL_H

#include "../generic/zero.cc" // zero_vectorized

#include <aie_api/aie.hpp>

// Flash-attention prefill (online softmax) in two geometries: global (head_dim
// 512, per-round chunk 8) and sliding-window (head_dim 256, chunk 16), both
// over a 128-key block. PrefillGeom<DH> carries the geometry and the four steps
// tuned per variant; everything else here is shared. Those four are separately
// tuned rather than incidental copies: attn_qk/attn_fv decompose 1x1 vs 2x2,
// calculate_l sums through one MMUL accumulator vs two with
// mac_elem_16_accuracy_safe, and reorder_s has no global equivalent.
//
// Pure compute. Synchronization, and the arithmetic mapping cores onto the
// query grid, belong to the design driving this.

using bf16 = bfloat16;

// The softmax runs on exp2, so scores are pre-scaled by log2(e).
constexpr bf16 exp_scale = (bf16)1.4426950408889634f;

// bf16 -inf, the mask fill value.
constexpr bf16 kNegInf = bf16(-0x1.FEp127f);

//===----------------------------------------------------------------------===//
// Steps shared by both geometries
//===----------------------------------------------------------------------===//

/// f = exp(s - rowmax), in place over the whole S tile. The outer trip count is
/// 128/LK, so the same S tile is covered either way, chunked by LK.
template <int LQ, int LK>
void apply_softmax(bf16 *__restrict pS, bf16 *__restrict new_m_local) {
  for (int b = 0; b < 128 / LK; b++) {
    for (int q = 0; q < LQ; q++) {
      aie::vector<bf16, LK> s_vec = aie::load_v<LK>(pS);
      aie::vector<bf16, LK> Vec = aie::sub(s_vec, *(new_m_local + q));
      aie::accum<accfloat, LK> Vec_acc = aie::mul(Vec, exp_scale);
      Vec = aie::exp2<bf16>(Vec_acc.template to_vector<float>());
      aie::store_v(pS, Vec);
      pS += LK;
    }
  }
}

/// Descending lane indices LK-1 .. 0. A constexpr constructor rather than a
/// literal list, because the two instantiations need different lengths.
template <int LK>
struct DescendingIdx {
  alignas(aie::vector_decl_align) int v[LK];
  constexpr DescendingIdx() : v() {
    for (int i = 0; i < LK; ++i)
      v[i] = LK - 1 - i;
  }
};
template <int LK>
inline constexpr DescendingIdx<LK> descending_idx{};

/// Apply the causal + sliding-window mask to S, folding the result into the
/// running row max.
template <int LQ, int LK>
void apply_mask_and_get_max(bf16 *__restrict pS, bf16 *__restrict m,
                            int pointer_block_inner_k,
                            int pointer_block_inner_q,
                            int pointer_block_inner_k_current) {
  aie::vector<int, LK> idx_vec = aie::load_v<LK>(descending_idx<LK>.v);
  aie::vector<int, LK> l_vec = aie::broadcast<int, LK>(pointer_block_inner_k);
  l_vec = aie::add(l_vec, idx_vec);
  aie::vector<int, LK> r_vec = aie::broadcast<int, LK>(pointer_block_inner_q);
  r_vec = aie::add(r_vec, idx_vec);

  for (int i = 0; i < LQ; i++) {
    int current_idx = pointer_block_inner_k_current + (LK - 1) - i;
    aie::mask<LK> mask_l = aie::gt(current_idx, l_vec);
    aie::mask<LK> mask_r = aie::le(current_idx, r_vec);
    aie::vector<bf16, LK> In_vec = aie::load_v<LK>(pS);
    aie::vector<bf16, LK> s_vec = aie::select(kNegInf, In_vec, mask_r);
    s_vec = aie::select(kNegInf, s_vec, mask_l);

    aie::vector<bf16, LK> m_vec = aie::load_v<LK>(m);
    aie::mask<LK> mask_m = aie::gt(s_vec, m_vec);
    aie::vector<bf16, LK> m_vec_new = aie::select(m_vec, s_vec, mask_m);
    aie::store_v(pS, s_vec);
    aie::store_v(m, m_vec_new);
    pS += LK;
    m += LK;
  }
}

/// correct = exp(m_prev - m_new), one per query row.
template <int LQ, int LK>
void calculate_c(float *c, bf16 *prev_m_local, bf16 *new_m_local) {
  for (int i = 0; i < LQ; i++) {
    bf16 inner_correct = aie::sub(*(prev_m_local + i), *(new_m_local + i));
    inner_correct = aie::mul(inner_correct, exp_scale);
    aie::vector<bf16, LK> correct_vec = aie::broadcast<bf16, LK>(inner_correct);
    aie::accum<accfloat, LK> correct_acc;
    correct_acc.from_vector(correct_vec);
    correct_vec = aie::exp2<bf16>(correct_acc.template to_vector<float>());
    *(c + i) = (float)correct_vec.get(0);
  }
}

/// Rescale the running y accumulator by the per-row correction factor.
template <int LQ, int DH>
void calculate_y(float *y, float *c) {
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

/// o = y * l for one 64-element output chunk. l arrives already inverted, from
/// finalize_impl.
inline void scale_by_inv_l(bf16 *o, bf16 *l, float *y) {
  constexpr int vec_factor = 64;

  aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l[0]);
  aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l[1]);
  aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l[2]);
  aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l[3]);
  aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l[4]);
  aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l[5]);
  aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l[6]);
  aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l[7]);

  auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

  aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
  aie::accum<accfloat, vec_factor> Y00_acc;
  Y00_acc.from_vector(Y00);
  aie::accum<accfloat, vec_factor> AL00 =
      aie::mul(Y00_acc.template to_vector<bf16>(), LL00);
  aie::store_v(o, AL00.template to_vector<bf16>());
}

//===----------------------------------------------------------------------===//
// Per-geometry steps
//===----------------------------------------------------------------------===//

template <int DH_>
struct PrefillGeom;

/// head_dim 512, 1x1 MMUL decomposition.
template <>
struct PrefillGeom<512> {
  static constexpr int DH = 512;
  static constexpr int LQ = 8;
  static constexpr int LK = 8;

  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

  static constexpr unsigned QK_rowA = LQ / 8;
  static constexpr unsigned QK_colA = DH / 8;
  static constexpr unsigned QK_colB = LK / 8;
  static constexpr unsigned FV_colA = LK / 8;
  static constexpr unsigned FV_colB = DH / 8;

  /// The global variant has no S reordering step.
  static inline void reorder_s(bf16 *__restrict) {}

  /// Query-grid mapping: which query a (row, col) core position covers, and
  /// where its Q tile starts. This variant pairs columns, so col's low bit
  /// selects the half.
  static inline int inner_q(int block_q, int row, int col) {
    return block_q + (col >> 1) * 64 + (col & 0x1) * 8 + row * 16;
  }
  static inline int q_offset(int col) { return (col & 0x1) * LQ * DH; }

  static void calculate_l(float *l, float *__restrict c, bf16 *__restrict pS) {
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>();
    MMUL C0(acc_C0);

    for (int b = 0; b < 16; b++) {
      bf16 *pS1 = pS + b * 64;
      aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);

      C0.mac(S0, Ones);
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    aie::accum<accfloat, 8> sum;
    sum.from_vector(sum01);

    aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
    aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
    sum = aie::mac(sum, c_float32, l_float32);
    aie::store_v(l, sum.template to_vector<float>());
  }

  static void attn_fv(float *__restrict pY, bf16 *__restrict pS,
                      bf16 *__restrict pV) {
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);

    float *__restrict pY1 = pY;

    for (unsigned j = 0; j < FV_colB; j += 2) {
      bf16 *__restrict pV1 = pV + j * MMUL::size_B * FV_colA;
      bf16 *__restrict pV2 = pV + (j + 1) * MMUL::size_B * FV_colA;

      aie::vector<bf16, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
      aie::vector<bf16, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);

      aie::vector<float, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
      aie::vector<float, MMUL::size_C> acc_Y01 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);

      MMUL Y00(acc_Y00);
      MMUL Y01(acc_Y01);

      Y00.mac(S0, V0);
      Y01.mac(S0, V1);

      aie::store_v(pY1, Y00.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y01.template to_vector<float>());
      pY1 += MMUL::size_C;
    }
  }

  static void attn_qk(bf16 *__restrict pS, bf16 *__restrict pQ,
                      bf16 *__restrict pK) {
    for (unsigned z = 0; z < QK_rowA; z += 1) {
      bf16 *__restrict pS1 = pS + (z * QK_colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < QK_colB; j += 1) {
        const bf16 *__restrict pQ1 = pQ + (z * QK_colA + 0) * MMUL::size_A;
        const bf16 *__restrict pK1 = pK + (0 * QK_colB + j) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
        aie::vector<bf16, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
        pK1 += MMUL::size_B * QK_colB;

        aie::vector<bf16, MMUL::size_C> acc_C00 =
            aie::zeros<bf16, MMUL::size_C>();

        MMUL C00(acc_C00);

        C00.mac(Q0, K0);

        for (unsigned i = 1; i < QK_colA; ++i) {
          Q0 = aie::load_v<MMUL::size_A>(pQ1);
          pQ1 += MMUL::size_A;
          K00 = aie::load_v<MMUL::size_B>(pK1);
          K0 = aie::transpose(K00, 8, 8);
          pK1 += MMUL::size_B * QK_colB;

          C00.mac(Q0, K0);
        }

        aie::store_v(pS1, C00.template to_vector<bf16>());
        pS1 += MMUL::size_C;
      }
    }
  }
};

/// head_dim 256, 2x2 MMUL decomposition.
template <>
struct PrefillGeom<256> {
  static constexpr int DH = 256;
  static constexpr int LQ = 16;
  static constexpr int LK = 16;

  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

  static constexpr unsigned QK_rowA = LQ / 8;
  static constexpr unsigned QK_colA = DH / 8;
  static constexpr unsigned QK_colB = LK / 8;
  static constexpr unsigned FV_colA = LK / 8;
  static constexpr unsigned FV_colB = DH / 8;

  static inline int inner_q(int block_q, int row, int col) {
    return block_q + col * 16 + row * 32;
  }
  static inline int q_offset(int col) { return col * LQ * DH; }

  /// Deinterleave S from the 2x2 attn_qk output order into the order attn_fv
  /// consumes.
  static void reorder_s(bf16 *__restrict pS) {
    for (int b = 0; b < 8; b++) {
      bf16 *pS1 = pS + b * 256;
      aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
      aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
      aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
      aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
      aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
      aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
      aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
      aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
      aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
      aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
      aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
      aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

      S0 = aie::concat(S00, S10);
      S1 = aie::concat(S01, S11);
      S2 = aie::concat(S20, S30);
      S3 = aie::concat(S21, S31);

      aie::store_v(pS1, S0);
      aie::store_v(pS1 + 64, S1);
      aie::store_v(pS1 + 128, S2);
      aie::store_v(pS1 + 192, S3);
    }
  }

  static void calculate_l(float *l, float *__restrict c, bf16 *__restrict pS) {
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>();
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);

    for (int b = 0; b < 8; b++) {
      bf16 *pS1 = pS + b * 256;
      aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
      aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
      aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
      aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);

      C0.mac(S0, Ones);
      C1.mac(S2, Ones);

      C0.mac(S1, Ones);
      C1.mac(S3, Ones);
    }

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sum2 = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum2);

    aie::vector<float, 16> c_float32 = aie::load_v<16>(c);
    aie::vector<float, 16> l_float32 = aie::load_v<16>(l);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());
  }

  static void attn_fv(float *__restrict pY, bf16 *__restrict pS,
                      bf16 *__restrict pV) {
    aie::vector<bf16, 64> S0 = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S1 = aie::load_v<64>(pS + 64);
    aie::vector<bf16, 64> S2 = aie::load_v<64>(pS + 128);
    aie::vector<bf16, 64> S3 = aie::load_v<64>(pS + 192);

    float *__restrict pY1 = pY;
    float *__restrict pY2 = pY + FV_colB * MMUL::size_C;

    for (unsigned j = 0; j < FV_colB; j += 2) {
      bf16 *__restrict pV1 = pV + j * MMUL::size_B * FV_colA;
      bf16 *__restrict pV2 = pV + (j + 1) * MMUL::size_B * FV_colA;

      aie::vector<bf16, MMUL::size_B> V0 = aie::load_v<MMUL::size_B>(pV1);
      pV1 += MMUL::size_B;
      aie::vector<bf16, MMUL::size_B> V1 = aie::load_v<MMUL::size_B>(pV2);
      pV2 += MMUL::size_B;

      aie::vector<float, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1);
      aie::vector<float, MMUL::size_C> acc_Y01 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
      aie::vector<float, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
      aie::vector<float, MMUL::size_C> acc_Y11 =
          aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);

      MMUL Y00(acc_Y00);
      MMUL Y01(acc_Y01);
      MMUL Y10(acc_Y10);
      MMUL Y11(acc_Y11);

      Y00.mac(S0, V0);
      Y01.mac(S0, V1);
      Y10.mac(S2, V0);
      Y11.mac(S2, V1);

      V0 = aie::load_v<MMUL::size_B>(pV1);
      pV1 += MMUL::size_B;
      V1 = aie::load_v<MMUL::size_B>(pV2);
      pV2 += MMUL::size_B;

      Y00.mac(S1, V0);
      Y01.mac(S1, V1);
      Y10.mac(S3, V0);
      Y11.mac(S3, V1);

      aie::store_v(pY1, Y00.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y01.template to_vector<float>());
      pY1 += MMUL::size_C;
      aie::store_v(pY2, Y10.template to_vector<float>());
      pY2 += MMUL::size_C;
      aie::store_v(pY2, Y11.template to_vector<float>());
      pY2 += MMUL::size_C;
    }
  }

  static void attn_qk(bf16 *__restrict pS, bf16 *__restrict pQ,
                      bf16 *__restrict pK) {
    for (unsigned z = 0; z < QK_rowA; z += 2) {
      bf16 *__restrict pS1 = pS + (z * QK_colB + 0) * MMUL::size_C;
      bf16 *__restrict pS2 = pS + ((z + 1) * QK_colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < QK_colB; j += 2) {
        const bf16 *__restrict pQ1 = pQ + (z * QK_colA + 0) * MMUL::size_A;
        const bf16 *__restrict pQ2 =
            pQ + ((z + 1) * QK_colA + 0) * MMUL::size_A;
        const bf16 *__restrict pK1 = pK + (0 * QK_colB + j) * MMUL::size_B;
        const bf16 *__restrict pK2 =
            pK + (0 * QK_colB + (j + 1)) * MMUL::size_B;

        aie::vector<bf16, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
        pQ1 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_A> Q1 = aie::load_v<MMUL::size_A>(pQ2);
        pQ2 += MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
        aie::vector<bf16, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
        pK1 += MMUL::size_B * QK_colB;
        aie::vector<bf16, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
        aie::vector<bf16, MMUL::size_B> K1 = aie::transpose(K01, 8, 8);
        pK2 += MMUL::size_B * QK_colB;

        aie::vector<bf16, MMUL::size_C> acc_C00 =
            aie::zeros<bf16, MMUL::size_C>();
        aie::vector<bf16, MMUL::size_C> acc_C01 =
            aie::zeros<bf16, MMUL::size_C>();
        aie::vector<bf16, MMUL::size_C> acc_C10 =
            aie::zeros<bf16, MMUL::size_C>();
        aie::vector<bf16, MMUL::size_C> acc_C11 =
            aie::zeros<bf16, MMUL::size_C>();

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        C00.mac(Q0, K0);
        C01.mac(Q0, K1);
        C10.mac(Q1, K0);
        C11.mac(Q1, K1);

        for (unsigned i = 1; i < QK_colA; ++i) {
          Q0 = aie::load_v<MMUL::size_A>(pQ1);
          pQ1 += MMUL::size_A;
          Q1 = aie::load_v<MMUL::size_A>(pQ2);
          pQ2 += MMUL::size_A;
          K00 = aie::load_v<MMUL::size_B>(pK1);
          K0 = aie::transpose(K00, 8, 8);
          pK1 += MMUL::size_B * QK_colB;
          K01 = aie::load_v<MMUL::size_B>(pK2);
          K1 = aie::transpose(K01, 8, 8);
          pK2 += MMUL::size_B * QK_colB;

          C00.mac(Q0, K0);
          C01.mac(Q0, K1);
          C10.mac(Q1, K0);
          C11.mac(Q1, K1);
        }
        auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                         C01.template to_vector<bf16>(), 8);
        auto mout1 = aie::interleave_zip(C10.template to_vector<bf16>(),
                                         C11.template to_vector<bf16>(), 8);

        aie::store_v(pS1, mout0.first);
        pS1 += MMUL::size_C;
        aie::store_v(pS1, mout0.second);
        pS1 += MMUL::size_C;
        aie::store_v(pS2, mout1.first);
        pS2 += MMUL::size_C;
        aie::store_v(pS2, mout1.second);
        pS2 += MMUL::size_C;
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Round and block bookkeeping, shared across geometries
//===----------------------------------------------------------------------===//

/// Rounds in this dispatch. The shift is computed here rather than in the
/// caller because a shift in an IRON-generated core wrapper lowers to a vector
/// srs intrinsic the wrapper link step cannot resolve.
inline void rounds_impl(int *L_begin_buffer, int *L_end_buffer, int *n_out) {
  n_out[0] = (L_end_buffer[0] >> 7) - (L_begin_buffer[0] >> 7);
}

/// Blocks in round i under a sliding window.
inline void blocks_windowed_impl(int *L_begin_buffer, int window_size,
                                 const int i, int *n_out) {
  const int pointer_block_q = L_begin_buffer[0] + i * 128;
  int pointer_block_k = pointer_block_q - window_size;
  pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
  n_out[0] = ((pointer_block_q - pointer_block_k) >> 7) + 1;
}

/// Blocks in round i when the window spans the whole sequence: the lower bound
/// pins to 0 and the count above collapses to this.
inline void blocks_unbounded_impl(int *L_begin_buffer, const int i,
                                  int *n_out) {
  n_out[0] = (L_begin_buffer[0] >> 7) + i + 1;
}

/// Per-round init of the online-softmax accumulators.
template <int DH>
__attribute__((always_inline)) inline void
round_begin_impl(bf16 *prev_m, bf16 *new_m, float *c, float *l, float *y) {
  using G = PrefillGeom<DH>;
  static const aie::vector<bf16, G::LK> neg_inf =
      aie::broadcast<bf16, G::LK>(kNegInf);
  static const aie::vector<float, G::LQ> one =
      aie::broadcast<float, G::LQ>(1.0f);
  aie::store_v(prev_m, neg_inf);
  aie::store_v(new_m, neg_inf);
  aie::store_v(c, one);
  // l holds LQ floats, which is not a multiple of the 512-bit lane count for
  // LQ=8, so this one zeroes at 256.
  zero_vectorized<float, G::LQ, 1, 256>(l);
  zero_vectorized<float, G::LQ, DH>(y);
}

/// Start of one block: broadcast prev_m across each row of m.
template <int DH>
__attribute__((always_inline)) inline void block_begin_impl(bf16 *m,
                                                            bf16 *prev_m) {
  using G = PrefillGeom<DH>;
  for (int j = 0; j < G::LQ; j++) {
    aie::vector<bf16, G::LK> m_vec = aie::broadcast<bf16, G::LK>(*(prev_m + j));
    aie::store_v(m + j * G::LK, m_vec);
  }
}

/// Middle of one block: row max, softmax, correction, then the running sums.
template <int DH>
__attribute__((always_inline)) inline void
block_mid_impl(bf16 *s, bf16 *m, bf16 *new_m, bf16 *prev_m, float *c, float *l,
               float *y) {
  using G = PrefillGeom<DH>;
  for (int j = 0; j < G::LQ; j++) {
    aie::vector<bf16, G::LK> m_vec = aie::load_v<G::LK>(m + j * G::LK);
    bf16 vm = aie::reduce_max(m_vec);
    *(new_m + j) = (bf16)vm;
  }
  apply_softmax<G::LQ, G::LK>(s, new_m);
  calculate_c<G::LQ, G::LK>(c, prev_m, new_m);
  G::reorder_s(s);
  G::calculate_l(l, c, s);
  calculate_y<G::LQ, DH>(y, c);
}

/// S = QK^T for one key chunk, then masked and folded into the running row max.
/// An unbounded window is spelled as a window_size at least the sequence
/// length, which pins both k pointers to the start of the block.
template <int DH>
__attribute__((always_inline)) inline void
qk_step_impl(bf16 *s, bf16 *__restrict q, bf16 *__restrict k, bf16 *m,
             int *L_begin_buffer, int window_size, const int row, const int col,
             const int i, const int block_idx, const int j) {
  using G = PrefillGeom<DH>;
  const int pointer_block_q = L_begin_buffer[0] + i * 128;
  const int pointer_block_inner_q = G::inner_q(pointer_block_q, row, col);
  int pointer_block_k = pointer_block_q - window_size;
  pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
  int pointer_block_inner_k = pointer_block_inner_q - window_size;
  pointer_block_inner_k =
      pointer_block_inner_k > -G::LK ? pointer_block_inner_k : -G::LK;
  const int block_location = block_idx * (128 / G::LK) + j;
  const int pointer_block_inner_k_current =
      pointer_block_k + block_location * G::LK;

  G::attn_qk(s + j * G::LQ * G::LK, q + G::q_offset(col), k);
  apply_mask_and_get_max<G::LQ, G::LK>(
      s + j * G::LQ * G::LK, m, pointer_block_inner_k, pointer_block_inner_q,
      pointer_block_inner_k_current);
}

/// y += S*V for one key chunk.
template <int DH>
__attribute__((always_inline)) inline void
fv_step_impl(float *y, bf16 *s, bf16 *__restrict v, const int j) {
  using G = PrefillGeom<DH>;
  G::attn_fv(y, s + j * G::LQ * G::LK, v);
}

/// End of one block: this block's max becomes the next block's prev.
template <int DH>
__attribute__((always_inline)) inline void block_end_impl(bf16 *prev_m,
                                                          bf16 *new_m) {
  using G = PrefillGeom<DH>;
  aie::vector<bf16, G::LQ> v = aie::load_v<G::LQ>(new_m);
  aie::store_v(prev_m, v);
}

/// End of the round: 1/l, consumed by epilogue_impl.
template <int DH>
__attribute__((always_inline)) inline void finalize_impl(float *l,
                                                         bf16 *l_bf16) {
  using G = PrefillGeom<DH>;
  aie::vector<float, G::LQ> l_vec = aie::load_v<G::LQ>(l);
  l_vec = aie::inv(l_vec);
  aie::accum<accfloat, G::LQ> l_acc;
  l_acc.from_vector(l_vec);
  aie::store_v(l_bf16, l_acc.template to_vector<bf16>());
}

/// One 64-element output chunk of o = y/l.
template <int DH>
__attribute__((always_inline)) inline void
epilogue_impl(bf16 *__restrict o, bf16 *l_bf16, float *y, const int c) {
  using G = PrefillGeom<DH>;
  if constexpr (G::LQ == 8) {
    // A single row of 8 queries, so the row index is always 0 and the row/col
    // split below would be dead arithmetic.
    scale_by_inv_l(o, l_bf16, y + c * 64);
  } else {
    const int o_row = c / (DH / 8);
    const int o_col = c % (DH / 8);
    scale_by_inv_l(o, l_bf16 + o_row * 8, y + o_row * 8 * DH + o_col * 64);
  }
}

#endif // AIE_KERNELS_AIE2P_FLASH_ATTN_PREFILL_H
