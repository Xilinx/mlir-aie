//===- gated_delta_rule.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Gated DeltaNet's recurrence for one value head over a block of rows.
//
// The state is kept as three bf16 limbs (f32_split.h), and each f32 product is
// summed from exact limb products and rounded once, as the scalar f32 product
// is. The per-row scalars (norms, decay, beta) are computed for every row at
// once, a row per lane; the state's columns are independent, so each
// 16-column block then runs a row's whole update on its own.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../common/exp_f32.h"
#include "../common/f32_split.h"
#include "softplus.h"

#ifndef GDR_ROWS
#define GDR_ROWS 16
#endif
#ifndef GDR_KEY_DIM
#define GDR_KEY_DIM 32
#endif
#ifndef GDR_VALUE_DIM
#define GDR_VALUE_DIM 16
#endif
#ifndef GDR_BLOCKS
#define GDR_BLOCKS 1
#endif

namespace {
constexpr int R = GDR_ROWS, Dk = GDR_KEY_DIM, Dv = GDR_VALUE_DIM;
// Columns of the state per vector, and rows per vector for the row scalars.
constexpr int L = 16, W = f32_lanes;
constexpr int P = (Dv + L - 1) / L * L, KP = (Dk + L - 1) / L * L;
constexpr int RP = (R + W - 1) / W * W;
// AIE2 keeps the scalar paths at key_dim 2 only: there Peano's aie2
// MachineLICM asserts on the paired update ("Machine instr not mapped for
// this vreg?!").
constexpr bool Vec = AIE_ARCH_AIE2P || Dk > 2;
// Where a row scalar's second set starts (b after a, k's norm after q's):
// with at most 16 rows, both sets share one vector.
constexpr int RS = Vec && R <= L ? L : RP;
template <unsigned N>
using acc_t = aie::accum<accfloat, N>;
template <unsigned N>
using bf_t = aie::vector<bfloat16, N>;

alignas(64) bfloat16 s_hi[Dk * P], s_mid[Dk * P], s_lo[Dk * P];
alignas(64) bfloat16 row[KP], v_row[P], out[P];
// One per row: a then b; the squared norms of q then k, then their
// reciprocal roots; exp's arguments, then decay and exp(-b).
alignas(64) bfloat16 ab[2 * RP];
alignas(64) float sq[2 * RP], ex[2 * RP];
alignas(64) float beta[RP], qn[RP], e[RP], f[RP];
alignas(64) float lanes[L], wide[P];

template <unsigned N>
inline __attribute__((always_inline)) Limbs<N> limbs(float x) {
  return bf16_limbs(acc_t<N>(aie::broadcast<float, N>(x)));
}

template <unsigned N>
inline __attribute__((always_inline)) Limbs<N> limbs(const float *x) {
  return bf16_limbs(acc_t<N>(aie::load_v<N>(x)));
}

inline __attribute__((always_inline)) Limbs<L> state(int i, int c) {
  return {aie::load_v<L>(s_hi + i * P + c), aie::load_v<L>(s_mid + i * P + c),
          aie::load_v<L>(s_lo + i * P + c)};
}

inline __attribute__((always_inline)) void set_state(int i, int c, Limbs<L> s) {
  aie::store_v(s_hi + i * P + c, s.hi);
  aie::store_v(s_mid + i * P + c, s.mid);
  aie::store_v(s_lo + i * P + c, s.lo);
}

// With one column block, rows i and i + 1 of the state side by side.
inline __attribute__((always_inline)) Limbs<W> state2(int i) {
  return {aie::load_v<W>(s_hi + i * L), aie::load_v<W>(s_mid + i * L),
          aie::load_v<W>(s_lo + i * L)};
}

inline __attribute__((always_inline)) void set_state2(int i, Limbs<W> s) {
  aie::store_v(s_hi + i * L, s.hi);
  aie::store_v(s_mid + i * L, s.mid);
  aie::store_v(s_lo + i * L, s.lo);
}

// x[i] in the low half's lanes and x[i + 1] in the high half's.
inline __attribute__((always_inline)) bf_t<W> pair(const bfloat16 *x, int i) {
  return aie::concat(aie::broadcast<bfloat16, L>(x[i]),
                     aie::broadcast<bfloat16, L>(x[i + 1]));
}

template <unsigned N>
inline __attribute__((always_inline)) Limbs<2 * N> twice(const Limbs<N> &x) {
  return {aie::concat(x.hi, x.hi), aie::concat(x.mid, x.mid),
          aie::concat(x.lo, x.lo)};
}

// Pairs of state rows a trip of the update. Peano orders a trip's state
// loads after the previous trip's stores, so a trip loads all its pairs
// before storing any (#1337).
constexpr int PB = 3;

// One pair of the state's rows decayed by G and given outer(k, d), added to
// q @ state and to the next row's k @ state, with even rows in the low lanes
// and odd rows in the high.
inline __attribute__((always_inline)) void
update_pair(acc_t<W> &y2, acc_t<W> &k2, const Limbs<W> &s, int i,
            const bfloat16 *kt, const bfloat16 *qt, const bfloat16 *kn,
            const Limbs<W> &G2, const Limbs<W> &d2) {
  const Limbs<W> n =
      bf16_limbs(aie::add(mul_limbs(s, G2), mul_limbs(d2, pair(kt, i))));
  set_state2(i, n);
  y2 = aie::add(y2, mul_limbs(n, pair(qt, i)));
  k2 = aie::add(k2, mul_limbs(n, pair(kn, i)));
}

// 1 / sqrt(x) for x > 0: the bit-trick estimate and three Newton steps.
constexpr float rsqrt_newton(float x) {
  uint32_t i = __builtin_bit_cast(uint32_t, x);
  i = 0x5f3759df - (i >> 1);
  float y = __builtin_bit_cast(float, i);
  for (int n = 0; n < 3; n++)
    y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

// rsqrt_newton per lane, its operations in the same order.
inline __attribute__((always_inline)) acc_t<W> rsqrt_f32(acc_t<W> x) {
  const auto xi = x.to_vector<float>().cast_to<int32_t>();
  acc_t<W> y(
      aie::sub(aie::broadcast<int32_t, W>(0x5f3759df), aie::downshift(xi, 1))
          .cast_to<float>());
  const Limbs<W> half_x =
      bf16_limbs(mul_limbs(bf16_limbs(x), bf16_splat<W>(0x3f00)));
  for (int n = 0; n < 3; n++) {
    const Limbs<W> yl = bf16_limbs(y);
    const acc_t<W> xyy = mul_limbs(bf16_limbs(mul_limbs(half_x, yl)), yl);
    y = mul_limbs(yl, bf16_limbs(aie::sub(
                          acc_t<W>(aie::broadcast<float, W>(1.5f)), xyy)));
  }
  return y;
}

// Exact bf16 squares summed per lane, then the lanes in four chains.
float sum_squares() {
  acc_t<L> a = aie::zeros<accfloat, L>();
  for (int i = 0; i < KP; i += L) {
    const bf_t<L> x = aie::load_v<L>(row + i);
    a = aie::mac(a, x, x);
  }
  aie::store_v(lanes, a.to_vector<float>());
  float s0 = lanes[0], s1 = lanes[1], s2 = lanes[2], s3 = lanes[3];
  for (int j = 4; j < L; j += 4) {
    s0 += lanes[j];
    s1 += lanes[j + 1];
    s2 += lanes[j + 2];
    s3 += lanes[j + 3];
  }
  return (s0 + s1) + (s2 + s3);
}

// Exact bf16 squares of x[0, KP) summed per lane.
acc_t<L> lane_squares(const bfloat16 *x) {
  acc_t<L> a = aie::zeros<accfloat, L>();
#pragma clang loop unroll(full)
  for (int i = 0; i < KP; i += L) {
    const bf_t<L> xi = aie::load_v<L>(x + i);
    a = aie::mac(a, xi, xi);
  }
  return a;
}

// Each half's lanes summed in four chains, chain j adding lanes j, j + 4,
// j + 8 and j + 12 in turn, then the chains as (0 + 1) + (2 + 3), into its
// first lane. The shifts never bring a lane into another half's first four.
inline __attribute__((always_inline)) aie::vector<float, W>
fold_lanes(acc_t<W> a) {
  const aie::vector<float, W> x = a.to_vector<float>();
#pragma clang loop unroll(full)
  for (int m = 4; m < L; m += 4)
    a = aie::add(a, acc_t<W>(aie::shuffle_down(x, m)));
#pragma clang loop unroll(full)
  for (int m = 1; m < 4; m *= 2)
    a = aie::add(a, acc_t<W>(aie::shuffle_down(a.to_vector<float>(), m)));
  return a.to_vector<float>();
}
} // namespace

// block is q (R x Dk), k (R x Dk), v (R x Dv), then a (R) and b (R); head is
// (dt_bias, A), then the state (Dk x Dv) before the head's first block. b_idx
// counts blocks, GDR_BLOCKS to a head, and a head's first block loads the
// state. Each block runs its first valid rows: q and k are l2-normalized
// (eps 1e-6) and q scaled by 1 / sqrt(Dk), g = A * softplus(a + dt_bias),
// beta = sigmoid(b), state = exp(g) * state, then state += outer(k, beta *
// (v - k @ state)), and y = q @ state. The rows after valid leave the state
// alone and write zeros; final is the state after this block.
extern "C" void gated_delta_rule(bfloat16 *restrict block, float *restrict head,
                                 float *restrict final, bfloat16 *restrict y,
                                 int32_t valid, int32_t b_idx) {
  event0();
  const aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *q = block, *k = q + R * Dk, *v = k + R * Dk;
  const bfloat16 *a = v + R * Dv, *b = a + R;
  const bf_t<L> one = aie::broadcast<bfloat16, L>(1.0f);
  if (b_idx % GDR_BLOCKS == 0) {
    int i = 0;
    // head's state starts two floats in, so it is loaded unaligned.
    if constexpr (Vec && Dv == L)
      for (; i + 1 < Dk; i += 2)
        set_state2(i, bf16_limbs(acc_t<W>(
                          aie::load_unaligned_v<W>(head + 2 + i * Dv, 2))));
    for (; i < Dk; i++) {
      for (int j = 0; j < Dv; j++)
        wide[j] = head[2 + i * Dv + j];
      for (int c = 0; c < P; c += L)
        set_state(i, c, limbs<L>(wide + c));
    }
  }

  for (int t = 0; t < R; t++) {
    ab[t] = a[t];
    ab[RS + t] = b[t];
    if constexpr (!Vec) {
      for (int i = 0; i < Dk; i++)
        row[i] = q[t * Dk + i];
      sq[t] = sum_squares();
      for (int i = 0; i < Dk; i++)
        row[i] = k[t * Dk + i];
      sq[RS + t] = sum_squares();
      continue;
    }
    acc_t<L> aq, ak;
    if constexpr (Dk == KP) {
      aq = lane_squares(q + t * Dk);
      ak = lane_squares(k + t * Dk);
    } else {
      // row's tail past Dk stays zero.
      for (int i = 0; i < Dk; i++)
        row[i] = q[t * Dk + i];
      aq = lane_squares(row);
      for (int i = 0; i < Dk; i++)
        row[i] = k[t * Dk + i];
      ak = lane_squares(row);
    }
    const aie::vector<float, W> sums = fold_lanes(aie::concat(aq, ak));
    sq[t] = sums.get(0);
    sq[RS + t] = sums.get(L);
  }
  // exp's arguments, g then -b, so one exp_f32 serves both; then the norms'
  // reciprocals, q's then k's, from one rsqrt_f32.
  const Limbs<W> A = limbs<W>(head[1]);
  for (int t0 = 0; t0 < RP; t0 += W) {
    const acc_t<W> z = aie::add(
        aie::mul(aie::load_v<W>(ab + t0), aie::broadcast<bfloat16, W>(1.0f)),
        acc_t<W>(aie::broadcast<float, W>(head[0])));
    aie::store_v(ex + t0,
                 mul_limbs(bf16_limbs(softplus_f32(z)), A).to_vector<float>());
    // ab + RS is only 32-byte aligned when RS is 16.
    if constexpr (RS == L)
      aie::store_v(ex + RS, aie::mul(aie::load_v<L>(ab + RS),
                                     aie::broadcast<bfloat16, L>(-1.0f))
                                .to_vector<float>());
    else
      aie::store_v(ex + RS + t0, aie::mul(aie::load_v<W>(ab + RS + t0),
                                          aie::broadcast<bfloat16, W>(-1.0f))
                                     .to_vector<float>());
  }
  const acc_t<W> eps(aie::broadcast<float, W>(1e-6f));
  for (int t0 = 0; t0 < 2 * RS; t0 += W) {
    aie::store_v(ex + t0,
                 exp_f32(acc_t<W>(aie::load_v<W>(ex + t0))).to_vector<float>());
    aie::store_v(sq + t0,
                 rsqrt_f32(aie::add(acc_t<W>(aie::load_v<W>(sq + t0)), eps))
                     .to_vector<float>());
  }
  constexpr float q_root = rsqrt_newton((float)Dk);
  // AIE2 keeps the base kernel's run-time form: folding it changes AIE2's
  // code at key_dim 2, which no npu1 run has checked.
  const Limbs<W> q_scale = limbs<W>(Vec ? q_root : rsqrt_newton((float)Dk));
  for (int t0 = 0; t0 < RP; t0 += W) {
    aie::store_v(qn + t0,
                 mul_limbs(limbs<W>(sq + t0), q_scale).to_vector<float>());
    aie::store_v(beta + t0, aie::add(acc_t<W>(aie::load_v<W>(ex + RS + t0)),
                                     acc_t<W>(aie::broadcast<float, W>(1.0f)))
                                .to_vector<float>());
  }
  for (int t = 0; t < R; t++)
    beta[t] = 1.0f / beta[t];
  // With k the unnormalized row, the update adds outer(k, d) for d = kn delta
  // = e v - f (k @ state), e = kn beta and f = e kn decay.
  for (int t0 = 0; t0 < RP; t0 += W) {
    const Limbs<W> kn = limbs<W>(sq + RS + t0);
    const acc_t<W> et = mul_limbs(kn, limbs<W>(beta + t0));
    aie::store_v(e + t0, et.to_vector<float>());
    aie::store_v(f + t0, mul_limbs(bf16_limbs(mul_limbs(bf16_limbs(et), kn)),
                                   limbs<W>(ex + t0))
                             .to_vector<float>());
  }

  // With one column block and paired rows, a row's update also takes the next
  // row's k @ state, in the same order as its own pass would.
  acc_t<W> k2 = aie::zeros<accfloat, W>();
  for (int t = 0; t < R; t++) {
    bfloat16 *yt = y + t * Dv;
    if (t >= valid) {
      if constexpr (Vec && Dv == P)
        for (int c = 0; c < P; c += L)
          aie::store_v(yt + c, aie::zeros<bfloat16, L>());
      else
        for (int j = 0; j < Dv; j++)
          yt[j] = 0.0f;
      continue;
    }
    const bfloat16 *kt = k + t * Dk, *qt = q + t * Dk;
    // v's row in place when it fills whole vectors, else padded with zeros.
    const bfloat16 *vr = v_row;
    if constexpr (Vec && Dv == P)
      vr = v + t * Dv;
    else
      for (int j = 0; j < Dv; j++)
        v_row[j] = v[t * Dv + j];
    const Limbs<L> G = limbs<L>(ex[t]), E = limbs<L>(e[t]);
    const Limbs<L> F = limbs<L>(f[t]), Q = limbs<L>(qn[t]);
    for (int c = 0; c < P; c += L) {
      // Even rows in kv and ya, odd rows in kw and yb. With one column
      // block, a vector of twice the width carries an even and an odd row.
      acc_t<L> kv = aie::zeros<accfloat, L>(), kw = aie::zeros<accfloat, L>();
      int r = 0;
      if constexpr (Vec && P == L) {
        if (t == 0)
          for (; r + 1 < Dk; r += 2)
            k2 = aie::add(k2, mul_limbs(state2(r), pair(kt, r)));
        r = Dk / 2 * 2;
        kv = k2.template extract<L>(0);
        kw = k2.template extract<L>(1);
      } else {
        for (; r + 1 < Dk; r += 2) {
          kv = aie::add(
              kv, mul_limbs(state(r, c), aie::broadcast<bfloat16, L>(kt[r])));
          kw = aie::add(kw, mul_limbs(state(r + 1, c),
                                      aie::broadcast<bfloat16, L>(kt[r + 1])));
        }
      }
      if (r < Dk)
        kv = aie::add(
            kv, mul_limbs(state(r, c), aie::broadcast<bfloat16, L>(kt[r])));
      const bf_t<L> vt = aie::load_v<L>(vr + c);
      const Limbs<L> d = bf16_limbs(aie::sub(
          mul_limbs(E, vt), mul_limbs(bf16_limbs(aie::add(kv, kw)), F)));
      acc_t<L> ya = aie::zeros<accfloat, L>(), yb = aie::zeros<accfloat, L>();
      int i = 0;
      if constexpr (Vec && P == L) {
        const Limbs<W> G2 = twice(G), d2 = twice(d);
        acc_t<W> y2 = aie::zeros<accfloat, W>();
        k2 = aie::zeros<accfloat, W>();
        // The last row's next is itself; nothing reads that sum.
        const bfloat16 *kn = kt + (t + 1 < R ? Dk : 0);
        for (; i + 2 * PB - 1 < Dk; i += 2 * PB) {
          Limbs<W> s[PB];
#pragma clang loop unroll(full)
          for (int u = 0; u < PB; u++)
            s[u] = state2(i + 2 * u);
#pragma clang loop unroll(full)
          for (int u = 0; u < PB; u++)
            update_pair(y2, k2, s[u], i + 2 * u, kt, qt, kn, G2, d2);
        }
        for (; i + 1 < Dk; i += 2)
          update_pair(y2, k2, state2(i), i, kt, qt, kn, G2, d2);
        ya = y2.template extract<L>(0);
        yb = y2.template extract<L>(1);
      } else {
        // Two rows a trip, so the scheduler interleaves independent chains.
        for (; i + 1 < Dk; i += 2) {
          const Limbs<L> n0 = bf16_limbs(
              aie::add(mul_limbs(state(i, c), G),
                       mul_limbs(d, aie::broadcast<bfloat16, L>(kt[i]))));
          const Limbs<L> n1 = bf16_limbs(
              aie::add(mul_limbs(state(i + 1, c), G),
                       mul_limbs(d, aie::broadcast<bfloat16, L>(kt[i + 1]))));
          set_state(i, c, n0);
          set_state(i + 1, c, n1);
          ya = aie::add(ya, mul_limbs(n0, aie::broadcast<bfloat16, L>(qt[i])));
          yb = aie::add(yb,
                        mul_limbs(n1, aie::broadcast<bfloat16, L>(qt[i + 1])));
        }
      }
      if (i < Dk) {
        const Limbs<L> n0 = bf16_limbs(
            aie::add(mul_limbs(state(i, c), G),
                     mul_limbs(d, aie::broadcast<bfloat16, L>(kt[i]))));
        set_state(i, c, n0);
        ya = aie::add(ya, mul_limbs(n0, aie::broadcast<bfloat16, L>(qt[i])));
      }
      aie::store_v(
          (Vec && Dv == P ? yt : out) + c,
          mul_limbs(bf16_limbs(aie::add(ya, yb)), Q).to_vector<bfloat16>());
    }
    if constexpr (!Vec || Dv != P)
      for (int j = 0; j < Dv; j++)
        yt[j] = out[j];
  }
  int i = 0;
  if constexpr (Vec && Dv == L) {
    const bf_t<W> one2 = aie::broadcast<bfloat16, W>(1.0f);
    for (; i + 1 < Dk; i += 2)
      aie::store_v(final + i * Dv,
                   mul_limbs(state2(i), one2).to_vector<float>());
  }
  for (; i < Dk; i++) {
    for (int c = 0; c < P; c += L)
      aie::store_v(wide + c, mul_limbs(state(i, c), one).to_vector<float>());
    for (int j = 0; j < Dv; j++)
      final[i * Dv + j] = wide[j];
  }
  aie::set_rounding(saved_rounding);
  event1();
}
