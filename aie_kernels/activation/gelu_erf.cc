//===- gelu_erf.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exact GELU, x * Phi(x) = 0.5 x erfc(-x / sqrt 2), where gelu.cc takes the
// tanh approximation. The two round differently on 405 of the 65,280 finite
// bf16 inputs, 280 of them below x = -3, where tanh's tail decays too fast.
//
// With a = |x|, z = a / sqrt 2 and t = 1 / (1 + z / 2), erfc(z) is
// t exp(-z^2 + P(t)) for a degree-9 P (Numerical Recipes' erfcc, relative
// error under 1.2e-7 for every z >= 0). h = a erfc(z) / 2 is then x Phi(-a),
// and gelu is x - h for x >= 0 and -h below. Each f32 product multiplies
// [hi | lo] bf16 limbs into the f32 accumulator, as exp2_f32_32 does. A model
// of this arithmetic is within one bf16 ulp of the float64 GELU on every
// finite bf16 input, or under the smallest normal bf16 of it.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/exp2_bf16.h" // exp2_f32_32

using acc_t = aie::accum<accfloat, 32>;
using bf_t = aie::vector<bfloat16, 32>;

struct Limbs {
  bf_t hi, lo;
};

static inline __attribute__((always_inline)) acc_t acc(float c) {
  return acc_t(aie::broadcast<float, 32>(c));
}

static inline __attribute__((always_inline)) Limbs limbs(acc_t v) {
  const bf_t hi = v.to_vector<bfloat16>();
  return {hi, aie::msc(v, hi, aie::broadcast<bfloat16, 32>(1.0f))
                  .to_vector<bfloat16>()};
}

// c + u v, dropping the lo x lo term.
static inline __attribute__((always_inline)) acc_t mac3(acc_t c, Limbs u,
                                                        Limbs v) {
  return aie::mac(aie::mac(aie::mac(c, u.hi, v.hi), u.hi, v.lo), u.lo, v.hi);
}

static inline __attribute__((always_inline)) acc_t msc3(acc_t c, Limbs u,
                                                        Limbs v) {
  return aie::msc(aie::msc(aie::msc(c, u.hi, v.hi), u.hi, v.lo), u.lo, v.hi);
}

static inline __attribute__((always_inline)) bf_t gelu_erf_32(bf_t x) {
  static constexpr float kErfc[] = {
      -1.26551223f, 1.00002368f,  0.37409196f, 0.09678418f,  -0.18628806f,
      0.27886807f,  -1.13520398f, 1.48851587f, -0.82215223f, 0.17087277f};
  const auto xi = x.cast_to<int16_t>();
  // From 13.5 on, h is below the smallest normal f32.
  const bf_t a = aie::min(aie::bit_and(xi, aie::broadcast<int16_t, 32>(0x7fff)),
                          aie::broadcast<int16_t, 32>(0x4158))
                     .cast_to<bfloat16>();

  // d = 1 + a / (2 sqrt 2), and t = 1 / d by Newton's r (2 - d r) from the
  // integer-subtraction estimate, which is within 5.1% on [1, 5.8].
  const Limbs k = limbs(acc(0.35355339f));
  const acc_t d = aie::mac(aie::mac(acc(1.0f), a, k.hi), a, k.lo);
  const Limbs dl = limbs(d);
  acc_t r = acc_t(aie::sub(aie::broadcast<int32_t, 32>(0x7ef311c7),
                           d.to_vector<float>().cast_to<int32_t>())
                      .cast_to<float>());
  for (int i = 0; i < 3; i++) {
    const Limbs rl = limbs(r);
    r = mac3(acc(0.0f), rl, limbs(msc3(acc(2.0f), dl, rl)));
  }
  const Limbs t = limbs(r);

  acc_t p = acc(kErfc[9]);
  for (int i = 8; i >= 0; i--)
    p = mac3(acc(kErfc[i]), limbs(p), t);
  // -z^2 = -a (a / 2), exact.
  const bf_t half_a = aie::mul(a, bfloat16(0.5f)).to_vector<bfloat16>();
  const acc_t e = aie::msc(p, a, half_a);

  // h = t 2^(e log2(e) + 2) (a / 8). The 2 keeps 2^y normal for every h at or
  // above the smallest normal f32; below that, h is flushed to 0.
  const Exp2F32 ex = exp2_f32_32(
      mac3(acc(2.0f), limbs(e), limbs(acc(1.44269504f))).to_vector<float>());
  const Limbs u =
      limbs(mac3(acc(0.0f), t, limbs(acc_t(ex.bits.cast_to<float>()))));
  const bf_t eighth_a = aie::mul(a, bfloat16(0.125f)).to_vector<bfloat16>();
  const auto h = aie::select(
      aie::mac(aie::mul(u.hi, eighth_a), u.lo, eighth_a).to_vector<float>(),
      aie::zeros<float, 32>(), ex.below);

  acc_t xa;
  xa.from_vector(x);
  const auto x_or_0 =
      aie::select(xa.to_vector<float>(), aie::zeros<float, 32>(),
                  aie::lt(xi, aie::zeros<int16_t, 32>()));
  return acc_t(aie::sub(x_or_0, h)).to_vector<bfloat16>();
}

extern "C" {

// vector_size must be a multiple of 32.
void gelu_erf_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                        int32_t vector_size) {
  event0();
  // Every limb and the output round to bf16 in this mode.
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < vector_size / 32; i++) {
    aie::store_v(output, gelu_erf_32(aie::load_v<32>(input)));
    input += 32;
    output += 32;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
