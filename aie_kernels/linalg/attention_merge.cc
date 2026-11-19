//===- attention_merge.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Split-KV attention's merge. Each part attended the same ROWS query rows over
// its own keys, leaving o, its attention normalized by its own sum, and per
// row its scores' maximum m and l = sum exp(s - m). Attention over all the
// keys is
//   out = sum_j w_j o_j / sum_j w_j,  w_j = l_j exp(m_j - M),  M = max_j m_j.
// A call folds one part in, online: the running maximum M, total L and
// acc = sum w_j o_j are rescaled by exp(M - M') when a part raises M to M'.
// o is ROWS x DIM float32 and stats ROWS x 2 float32, each row's m and l, as
// a float32 softmax with stats writes them a row at a time. acc is the
// core's: ROWS x DIM sums, then M, then L. The FIRST call starts it without
// reading it, and the LAST writes out, ROWS x DIM bf16. A part with l = 0
// attended no key and weighs nothing (its m may be -inf), and a row no part
// attended is 0.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <limits>
#include <stdint.h>

#include "../aie_arch.h"
#include "../aie_kernel_utils.h"
#include "../common/exp2_bf16.h"
#include "../common/f32_split.h"

#ifndef ROWS
#error Please define ROWS, the query rows, at compile time.
#endif
#ifndef DIM
#error Please define DIM, the head dimension, at compile time.
#endif

static_assert(ROWS % 16 == 0, "rows come in vectors of 16 or 32");
static_assert(DIM % f32_lanes == 0, "a row is whole vectors of 32");

#define ATTENTION_MERGE_FIRST 1
#define ATTENTION_MERGE_LAST 2

// Past this, exp underflows float32: a clamp keeps the sentinel maximum's
// difference, times log2(e), finite.
static constexpr float kFloor = -100.0f;

// Each row's rescale of the running sums (alpha) and the part's weight
// (beta), L rows a vector, and the state moved on to the new maximum.
template <int L>
static inline void fold_weights(const float *__restrict stats,
                                float *__restrict state, bool first,
                                float *__restrict alpha,
                                float *__restrict beta) {
  using acc_t = aie::accum<accfloat, L>;
  const auto zeros = aie::zeros<float, L>();
  const auto lowest =
      aie::broadcast<float, L>(std::numeric_limits<float>::lowest());
  const auto floor = aie::broadcast<float, L>(kFloor);
  const Limbs<L> log2e =
      bf16_limbs(acc_t(aie::broadcast<float, L>(1.44269504088896341f)));
  // exp(x) for x in [kFloor, 0], 32 lanes at a time.
  auto exp = [&](aie::vector<float, L> x) {
    acc_t xa;
    xa.from_vector(aie::max(x, floor));
    const auto e2 = mul_limbs(bf16_limbs(xa), log2e).template to_vector<float>();
    aie::vector<float, 32> x32;
    if constexpr (L == 32)
      x32 = e2;
    else
      x32 = e2.template grow<32>();
    const Exp2F32 e = exp2_f32_32(x32);
    const auto power = aie::select(e.bits.cast_to<float>(),
                                   aie::zeros<float, 32>(), e.below);
    if constexpr (L == 32)
      return power;
    else
      return power.template extract<L>(0);
  };
  auto mul = [](aie::vector<float, L> a, aie::vector<float, L> b) {
    acc_t aa, ba;
    aa.from_vector(a);
    ba.from_vector(b);
    return mul_limbs(bf16_limbs(aa), bf16_limbs(ba)).template to_vector<float>();
  };
  for (int32_t i = 0; i < ROWS; i += L) {
    const auto [maxima, l] = aie::interleave_unzip(
        aie::load_v<L>(stats + 2 * i), aie::load_v<L>(stats + 2 * i + L), 1);
    const auto empty = aie::eq(l, zeros);
    const auto m = aie::select(maxima, lowest, empty);
    const auto M = first ? lowest : aie::load_v<L>(state + i);
    const auto total = first ? zeros : aie::load_v<L>(state + ROWS + i);
    const auto M_new = aie::max(M, m);
    const auto a = exp(aie::sub(M, M_new));
    const auto b = aie::select(mul(l, exp(aie::sub(m, M_new))), zeros, empty);
    acc_t t;
    t.from_vector(mul(total, a));
    aie::store_v(state + i, M_new);
    aie::store_v(state + ROWS + i,
                 aie::add(t, b).template to_vector<float>());
    aie::store_v(alpha + i, a);
    aie::store_v(beta + i, b);
  }
}

extern "C" {

void attention_merge_f32_bf16(int32_t flags, const float *__restrict o,
                              const float *__restrict stats,
                              float *__restrict acc,
                              bfloat16 *__restrict out) {
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  const bool first = flags & ATTENTION_MERGE_FIRST;
  float *__restrict state = acc + ROWS * DIM;
  alignas(64) float alpha[ROWS], beta[ROWS];
  // 16 lanes on AIE2: at 32 its instruction selection asserts (llvm-aie
  // 22.0.0.2026102001).
  fold_weights<(AIE_ARCH_AIE2 || ROWS % 32) ? 16 : 32>(stats, state, first,
                                                       alpha, beta);

  for (int32_t r = 0; r < ROWS; r++) {
    const Limbs<f32_lanes> b =
        bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(beta[r])));
    const Limbs<f32_lanes> a =
        bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(alpha[r])));
    for (int32_t d = 0; d < DIM; d += f32_lanes) {
      float *__restrict row = acc + r * DIM + d;
      f32_acc v;
      v.from_vector(aie::load_v<f32_lanes>(o + r * DIM + d));
      f32_acc sum = mul_limbs(bf16_limbs(v), b);
      if (!first) {
        f32_acc s;
        s.from_vector(aie::load_v<f32_lanes>(row));
        sum = mac_limbs(sum, bf16_limbs(s), a);
      }
      aie::store_v(row, sum.template to_vector<float>());
    }
  }

  if (flags & ATTENTION_MERGE_LAST) {
    alignas(64) float inv[ROWS];
    const auto zeros = aie::zeros<float, 16>();
    for (int32_t i = 0; i < ROWS; i += 16) {
      const auto total = aie::load_v<16>(state + ROWS + i);
      aie::store_v(inv + i, aie::select(aie::inv(total), zeros,
                                        aie::eq(total, zeros)));
    }
    for (int32_t r = 0; r < ROWS; r++) {
      const Limbs<f32_lanes> s =
          bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(inv[r])));
      for (int32_t d = 0; d < DIM; d += f32_lanes) {
        f32_acc v;
        v.from_vector(aie::load_v<f32_lanes>(acc + r * DIM + d));
        aie::store_v(out + r * DIM + d,
                     mul_limbs(bf16_limbs(v), s).template to_vector<bfloat16>());
      }
    }
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
