//===- flash_attn_prefill.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_LINALG_FLASH_ATTN_PREFILL_H
#define AIE_KERNELS_LINALG_FLASH_ATTN_PREFILL_H

#include "../aie_kernel_utils.h"
#include "../common/zero.h" // zero_vectorized

#include <aie_api/aie.hpp>

#include "../common/exp2_bf16.h"

// Flash-attention prefill (online softmax) in two geometries: global (head_dim
// 512, chunk 8) and sliding-window (head_dim 256, chunk 16), over a 128-key
// block. PrefillGeom<DH> carries the four steps tuned per variant: attn_qk and
// attn_fv decompose 1x1 vs 2x2, calculate_l sums through one MMUL accumulator
// vs two, reorder_s has no global equivalent.
//
// Pure compute; synchronization belongs to the design driving this.
//
// PRECONDITION: j ascends from 0 within a block, c within a round.

using bf16 = bfloat16;

// log2(e) / sqrt(head_dim), for scaled attention through the exp2 the softmax
// runs on. LQ uniquely identifies the two supported geometries.
//
// The scale multiplies s - m, never s, so its bf16 rounding error grows only as
// s - m goes negative -- exactly where the weight itself is vanishing.
template <int LQ>
constexpr bf16 exp_scale =
    (bf16)(1.4426950408889634f * (LQ == 8 ? 0.04419417382415922f : 0.0625f));

// bf16 -inf, the mask fill value.
constexpr bf16 kNegInf = bf16(-0x1.FEp127f);

//===----------------------------------------------------------------------===//
// Steps shared by both geometries
//===----------------------------------------------------------------------===//

/// f = exp(s - rowmax), in place over the whole S tile.
///
/// Each 64-lane step spans 64/LK query rows, so the row maxima arrive as a
/// broadcast pattern built once per group.
template <int LQ, int LK>
void apply_softmax(bf16 *__restrict pS, bf16 *__restrict new_m_local) {
  constexpr int kRows = 64 / LK; // query rows covered by one 64-lane vector
  constexpr int kGroups = LQ / kRows;

  for (int g = 0; g < kGroups; g++) {
    aie::vector<bf16, 64> m_bcast;
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < kRows; r++)
      m_bcast.insert(r, aie::broadcast<bf16, LK>(new_m_local[g * kRows + r]));

    // Group g owns lanes [g*64, g*64+64) of every LQ*LK chunk of S.
    bf16 *__restrict pSg = pS + g * 64;
    for (int b = 0; b < 128 / LK; b++) {
      aie::vector<bf16, 64> s_vec = aie::load_v<64>(pSg);
      // The multiply doubles as the widening exp2 needs, so the scale is free
      // here: the bf16 product lands in a float accumulator either way.
      aie::vector<bf16, 64> Vec = aie::sub(s_vec, m_bcast);
      aie::accum<accfloat, 64> Vec_acc = aie::mul(Vec, exp_scale<LQ>);
      aie::store_v(pSg, exp2_bf16(Vec_acc.template to_vector<float>()));
      pSg += kGroups * 64;
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
///
/// All LQ rows are one vector. They used to be a scalar loop that broadcast
/// each row across an LK-wide vector, exponentiated it, and kept lane 0 --
/// LQ exp2 calls, LK-1 lanes of each discarded.
///
/// exp2 lands in bf16 and widens: on XDNA2 aie::exp2 has no float result form
/// (aie_api/aie.hpp:8107 admits one only on AIE_MLv2), so this rounding is the
/// hardware's, not a choice. It is also the benign one -- c rescales y and l
/// alike, and o = y/l divides most of it back out.
template <int LQ>
void calculate_c(float *c, bf16 *prev_m_local, bf16 *new_m_local) {
  aie::vector<bf16, LQ> prev = aie::load_v<LQ>(prev_m_local);
  aie::vector<bf16, LQ> next = aie::load_v<LQ>(new_m_local);
  aie::accum<accfloat, LQ> arg = aie::mul(aie::sub(prev, next), exp_scale<LQ>);
  aie::accum<accfloat, LQ> e;
  e.from_vector(exp2_bf16(arg.template to_vector<float>()));
  aie::store_v(c, e.template to_vector<float>());
}

/// Broadcast eight consecutive floats across the eight 8-lane groups of one
/// 64-lane vector: lane i takes p[i / 8]. Both callers pair the result with a
/// 64-lane slice of an LQ x DH tile, which spans eight rows of eight columns,
/// so "one value per group" is one value per query row.
inline aie::vector<float, 64> broadcast_by_row(const float *p) {
  aie::vector<float, 64> v;
  AIE_LOOP_UNROLL_FULL
  for (int r = 0; r < 8; r++)
    v.insert(r, aie::broadcast<float, 8>(p[r]));
  return v;
}

/// Rescale the running y accumulator by the per-row correction factor.
///
/// There is no fp32 multiplier. c is exact in bf16 (aie::exp2<bf16> produced
/// it), so only y is split into bf16 limbs. Two limbs give roughly 16 mantissa
/// bits, far below a bf16 output ulp.
template <int LQ, int DH>
void calculate_y(float *y, float *c) {
  aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
  for (int i = 0; i < LQ / 8; i++) {
    aie::accum<accfloat, 64> corr_acc;
    corr_acc.from_vector(broadcast_by_row(c + i * 8));
    aie::vector<bf16, 64> CORRECT = corr_acc.template to_vector<bf16>();

    float *pY = y + i * 8 * DH;
    for (unsigned j = 0; j < DH / 8; j += 1) {
      aie::accum<accfloat, 64> Y;
      Y.from_vector(aie::load_v<64>(pY));
      aie::vector<bf16, 64> y_hi = Y.template to_vector<bf16>();
      // Y - y_hi in the accumulator, rounded down to bf16: the next limb.
      aie::vector<bf16, 64> y_lo =
          aie::msc(Y, y_hi, Ones).template to_vector<bf16>();
      aie::accum<accfloat, 64> ACC_Y = aie::mul(y_hi, CORRECT);
      ACC_Y = aie::mac(ACC_Y, y_lo, CORRECT);
      aie::store_v(pY, ACC_Y.template to_vector<float>());
      pY += 64;
    }
  }
}

/// o = y * l over 64 elements, l arriving already inverted.
///
/// Both operands stay float to the last moment. o is bf16 and so must round
/// once; rounding y and 1/l on the way in as well would spend three roundings
/// where one is owed, for no saving -- this is the same float multiply
/// calculate_y already does, minus the two conversions.
inline void scale_by_inv_l(bf16 *o, float *l, float *y) {
  constexpr int vec_factor = 64;

  aie::vector<float, vec_factor> LL00 = broadcast_by_row(l);
  aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
  aie::accum<accfloat, vec_factor> AL00 = aie::mul(LL00, Y00);
  aie::store_v(o, AL00.template to_vector<bf16>());
}

/// Y += S * V across one 8-row block of y: kTiles 8x8 output tiles, the i-th
/// taking its V tile from pV + i * kVStride.
#if AIE_TUNED_AIE2P
/// Two neighbouring output tiles fill one 64-lane accumulator, rows 0-3 of
/// both in L and rows 4-7 in H, so one native mac per k advances both: the S
/// side is S[r][k] repeated across the row, and the V side is row k of each
/// tile's V broadcast over its four rows, one vextbcst per tile. Every output
/// still sums its products in ascending k, so y is bit-identical to the
/// mmul's.
///
/// y is loaded one tile pair ahead, so the next pair's load need not wait for
/// this pair's store, which the pipeliner cannot tell apart.
template <unsigned kTiles, unsigned kVStride>
void sv_row_block(float *__restrict pY, const bf16 *__restrict pS,
                  const bf16 *__restrict pV) {
  aie::vector<bf16, 64> St = aie::transpose(aie::load_v<64>(pS), 8, 8);
  aie::vector<bf16, 64> AL[8], AH[8];
  AIE_LOOP_UNROLL_FULL
  for (int k = 0; k < 8; k++) {
    // Lane (r, c) of A holds S[r][k].
    aie::vector<bf16, 32> b = (v32bfloat16)::broadcast_elem_128(
        (v16int32)St.template extract<32>(k / 4), k % 4);
    aie::vector<bf16, 64> A = aie::transpose(aie::concat(b, b), 8, 8);
    AL[k] = aie::concat(A.template extract<32>(0), A.template extract<32>(0));
    AH[k] = aie::concat(A.template extract<32>(1), A.template extract<32>(1));
  }

  auto load_pair = [](aie::accum<accfloat, 64> &L, aie::accum<accfloat, 64> &H,
                      const float *y) {
    L.from_vector(aie::concat(aie::load_v<32>(y), aie::load_v<32>(y + 64)));
    H.from_vector(
        aie::concat(aie::load_v<32>(y + 32), aie::load_v<32>(y + 96)));
  };

  aie::accum<accfloat, 64> L, H;
  load_pair(L, H, pY);
  for (unsigned j = 0; j < kTiles; j += 2) {
    float *pYn = j + 2 < kTiles ? pY + 128 : pY;
    aie::accum<accfloat, 64> Ln, Hn;
    load_pair(Ln, Hn, pYn);

    aie::vector<bf16, 64> V0 = aie::load_v<64>(pV);
    aie::vector<bf16, 64> V1 = aie::load_v<64>(pV + kVStride);
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < 8; k++) {
      aie::vector<bf16, 32> b0 = (v32bfloat16)::broadcast_elem_128(
          (v16int32)V0.template extract<32>(k / 4), k % 4);
      aie::vector<bf16, 32> b1 = (v32bfloat16)::broadcast_elem_128(
          (v16int32)V1.template extract<32>(k / 4), k % 4);
      aie::vector<bf16, 64> B = aie::concat(b0, b1);
      L = aie::mac(L, AL[k], B);
      H = aie::mac(H, AH[k], B);
    }

    aie::vector<float, 64> l = L.template to_vector<float>();
    aie::vector<float, 64> h = H.template to_vector<float>();
    aie::store_v(pY, l.template extract<32>(0));
    aie::store_v(pY + 64, l.template extract<32>(1));
    aie::store_v(pY + 32, h.template extract<32>(0));
    aie::store_v(pY + 96, h.template extract<32>(1));
    L = Ln;
    H = Hn;
    pY = pYn;
    pV += 2 * kVStride;
  }
}
#else
// One mmul<8, 8, 8> per tile, for an architecture without the vextbcst the
// branch above builds its operands from. It sums in a different order from
// ascending k, which the aie2 tolerance in linalg.py allows for. y is loaded
// one tile pair ahead so that the next pair's load need not wait for this
// pair's store.
template <unsigned kTiles, unsigned kVStride>
void sv_row_block(float *__restrict pY, const bf16 *__restrict pS,
                  const bf16 *__restrict pV) {
  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
  const aie::vector<bf16, 64> S = aie::load_v<64>(pS);
  aie::vector<float, 64> y0 = aie::load_v<64>(pY);
  aie::vector<float, 64> y1 = aie::load_v<64>(pY + 64);
  for (unsigned j = 0; j < kTiles; j += 2) {
    float *pYn = j + 2 < kTiles ? pY + 128 : pY;
    aie::vector<float, 64> yn0 = aie::load_v<64>(pYn);
    aie::vector<float, 64> yn1 = aie::load_v<64>(pYn + 64);
    MMUL Y0(y0), Y1(y1);
    Y0.mac(S, aie::load_v<64>(pV));
    Y1.mac(S, aie::load_v<64>(pV + kVStride));
    aie::store_v(pY, Y0.template to_vector<float>());
    aie::store_v(pY + 64, Y1.template to_vector<float>());
    y0 = yn0;
    y1 = yn1;
    pY = pYn;
    pV += 2 * kVStride;
  }
}
#endif

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
    sv_row_block<FV_colB, MMUL::size_B>(pY, pS, pV);
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
#if AIE_TUNED_AIE2P
    l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
#else
    l_out = aie::mac(sum, c_float32, l_float32);
#endif
    aie::store_v(l, l_out.template to_vector<float>());
  }

  /// S is a 2x2 grid of 8x8 tiles, one per (query row block, key half),
  /// stored row block major. Each tile is one pass over its row block of y,
  /// the first key half ahead of the second, so every output still sums its
  /// sixteen products in ascending k.
  static void attn_fv(float *__restrict pY, bf16 *__restrict pS,
                      bf16 *__restrict pV) {
    for (unsigned t = 0; t < 4; t++)
      sv_row_block<FV_colB, MMUL::size_B * FV_colA>(
          pY + (t / 2) * FV_colB * MMUL::size_C, pS + t * MMUL::size_A,
          pV + (t % 2) * MMUL::size_B);
  }

  /// One 8-query row block of S against both key columns: two live C tiles
  /// fit alongside the hoisted broadcasts, four spill.
  static void attn_qk_half(bf16 *__restrict pS1, const bf16 *__restrict pQ1,
                           bf16 *__restrict pK) {
    const bf16 *__restrict pK1 = pK;
    const bf16 *__restrict pK2 = pK + MMUL::size_B;

    aie::vector<bf16, MMUL::size_A> Q0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> K00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> K0 = aie::transpose(K00, 8, 8);
    pK1 += MMUL::size_B * QK_colB;
    aie::vector<bf16, MMUL::size_B> K01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> K1 = aie::transpose(K01, 8, 8);
    pK2 += MMUL::size_B * QK_colB;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(Q0, K0);
    C01.mac(Q0, K1);

    for (unsigned i = 1; i < QK_colA; ++i) {
      Q0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;
      K00 = aie::load_v<MMUL::size_B>(pK1);
      K0 = aie::transpose(K00, 8, 8);
      pK1 += MMUL::size_B * QK_colB;
      K01 = aie::load_v<MMUL::size_B>(pK2);
      K1 = aie::transpose(K01, 8, 8);
      pK2 += MMUL::size_B * QK_colB;

      C00.mac(Q0, K0);
      C01.mac(Q0, K1);
    }

    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);
    aie::store_v(pS1, mout0.first);
    aie::store_v(pS1 + MMUL::size_C, mout0.second);
  }

  static void attn_qk(bf16 *__restrict pS, bf16 *__restrict pQ,
                      bf16 *__restrict pK) {
    attn_qk_half(pS, pQ, pK);
    attn_qk_half(pS + QK_colB * MMUL::size_C, pQ + QK_colA * MMUL::size_A, pK);
  }
};

//===----------------------------------------------------------------------===//
// Round and block bookkeeping, shared across geometries
//===----------------------------------------------------------------------===//

// The round and block trip counts are a shift and a clamp on runtime scalars,
// with no AIE work in them, so the driving design's runtime sequence computes
// them rather than a core kernel.

/// Per-round init of the online-softmax accumulators.
template <int DH>
__attribute__((always_inline)) inline void
round_begin_impl(bf16 *prev_m, bf16 *new_m, float *c, float *l, float *y) {
  using G = PrefillGeom<DH>;
  static const aie::vector<bf16, G::LK> neg_inf =
      aie::broadcast<bf16, G::LK>(kNegInf);
  static const aie::vector<float, G::LQ> one =
      aie::broadcast<float, G::LQ>(1.0f);
  static const aie::vector<float, G::LQ> zero = aie::zeros<float, G::LQ>();
  aie::store_v(prev_m, neg_inf);
  aie::store_v(new_m, neg_inf);
  aie::store_v(c, one);
  // l is LQ floats, the same width as c above, so it stores the same way.
  // zero_vectorized would take its scalar tail here: LQ=8 is under the native
  // float lane count, so its vector body would never run.
  aie::store_v(l, zero);
  zero_vectorized<float, G::LQ, DH>(y);
}

/// Start of one block: broadcast prev_m across each row of m. Folded into
/// qk_step_impl's first key chunk rather than exposed as its own entry point.
template <int DH>
__attribute__((always_inline)) inline void block_begin_impl(bf16 *m,
                                                            bf16 *prev_m) {
  using G = PrefillGeom<DH>;
  for (int j = 0; j < G::LQ; j++) {
    aie::vector<bf16, G::LK> m_vec = aie::broadcast<bf16, G::LK>(*(prev_m + j));
    aie::store_v(m + j * G::LK, m_vec);
  }
}

/// Middle of one block: row max, softmax, correction, then the running sums,
/// and finally this block's max becomes the next block's prev.
///
/// That last step used to be its own entry point, called after the fv loop.
/// It is one LQ-wide copy, and nothing between here and there reads prev_m --
/// calculate_c is its last reader, and the fv loop touches only y, s and v --
/// so it runs here instead.
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
  calculate_c<G::LQ>(c, prev_m, new_m);
  G::reorder_s(s);
  G::calculate_l(l, c, s);
  calculate_y<G::LQ, DH>(y, c);
  aie::store_v(prev_m, aie::load_v<G::LQ>(new_m));
}

/// S = QK^T for one key chunk, then masked and folded into the running row max.
/// An unbounded window is spelled as a window_size at least the sequence
/// length, which pins both k pointers to the start of the block.
///
/// j == 0 also seeds m from prev_m (what block_begin was). The test is at
/// function entry, outside attn_qk's MAC loops.
template <int DH>
__attribute__((always_inline)) inline void
qk_step_impl(bf16 *s, bf16 *__restrict q, bf16 *__restrict k, bf16 *m,
             bf16 *prev_m, int *L_begin_buffer, int window_size, const int row,
             const int col, const int i, const int block_idx, const int j) {
  using G = PrefillGeom<DH>;
  if (j == 0) {
    block_begin_impl<DH>(m, prev_m);
  }
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

/// l becomes 1/l in place; epilogue_impl below says when.
///
/// The reciprocal stays float rather than landing in a bf16 side buffer, which
/// would cost a rounding the output cannot absorb -- o rounds to bf16 anyway
/// afterwards -- for a buffer nothing else wants.
template <int DH>
__attribute__((always_inline)) inline void finalize_impl(float *l) {
  using G = PrefillGeom<DH>;
  aie::store_v(l, aie::inv(aie::load_v<G::LQ>(l)));
}

/// One 64-element output chunk of o = y/l. c == 0 runs finalize_impl first, so
/// the round needs no separate closing call -- which is what makes l
/// read-write here, and why c must ascend from 0.
template <int DH>
__attribute__((always_inline)) inline void
epilogue_impl(bf16 *__restrict o, float *l, float *y, const int c) {
  using G = PrefillGeom<DH>;
  if (c == 0) {
    finalize_impl<DH>(l);
  }
  if constexpr (G::LQ == 8) {
    // A single row of 8 queries, so the row index is always 0 and the row/col
    // split below would be dead arithmetic.
    scale_by_inv_l(o, l, y + c * 64);
  } else {
    const int o_row = c / (DH / 8);
    const int o_col = c % (DH / 8);
    scale_by_inv_l(o, l + o_row * 8, y + o_row * 8 * DH + o_col * 64);
  }
}

#endif // AIE_KERNELS_LINALG_FLASH_ATTN_PREFILL_H
