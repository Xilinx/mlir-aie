//===- gelu_erf.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exact GELU, x * Phi(x), where gelu.cc takes the tanh approximation: the two
// round differently on 405 finite bf16 inputs, most below x = -3. Within one
// bf16 ulp of the float64 GELU wherever that is a normal number; subnormal
// results flush to 0, as the AIE's floating point does.
//
// With a = |x|, z = a / sqrt 2 and t = 1 / (1 + z / 2), erfc(z) = t exp(-z^2 +
// P(t)) as in Numerical Recipes' erfcc, with P refit as a degree-6 minimax.
// h = a erfc(z) / 2 is computed in f32 products of bf16 limbs, and gelu is
// x - h for x >= 0, else -h.
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
  // log2(e) P(t), P the minimax of log(erfc(z) / t) + z^2 on t in [0.1732, 1]
  // (a <= 13.5), within 6.1e-6; kP[0] also carries the + 8 below. The degree-6
  // coefficient, -0.0238606557, is the two bf16 limbs in c6.
  static constexpr float kP[] = {6.17218924f,  1.4762423f,   0.328846067f,
                                 0.781043708f, -1.15243161f, 0.417979658f};
  const bf_t one = aie::broadcast<bfloat16, 32>(1.0f);
  const auto xi = x.cast_to<int16_t>();
  // From 13.5 on, h is below the smallest normal f32.
  const auto ai = aie::min(aie::bit_and(xi, aie::broadcast<int16_t, 32>(0x7fff)),
                           aie::broadcast<int16_t, 32>(0x4158));
  const bf_t a = ai.cast_to<bfloat16>();

  // d = 1 + a / (2 sqrt 2). t = 1 / d from the integer-subtraction estimate
  // (within 5.1% on [1, 5.8]) by r + r (1 - d r), whose correction needs one
  // limb: once in bf16, then twice on limbs of r.
  const Limbs k = limbs(acc(0.35355339f));
  const acc_t d = aie::mac(aie::mac(acc(1.0f), a, k.hi), a, k.lo);
  const Limbs dl = limbs(d);
  const bf_t r0 = acc_t(aie::sub(aie::broadcast<int32_t, 32>(0x7ef311c7),
                                 d.to_vector<float>().cast_to<int32_t>())
                            .cast_to<float>())
                      .to_vector<bfloat16>();
  const bf_t e0 =
      aie::msc(aie::msc(acc(1.0f), dl.hi, r0), dl.lo, r0).to_vector<bfloat16>();
  acc_t r = aie::mac(aie::mul(r0, one), r0, e0);
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < 2; i++) {
    const Limbs rl = limbs(r);
    const bf_t e = msc3(acc(1.0f), dl, rl).to_vector<bfloat16>();
    r = aie::mac(aie::mac(r, rl.hi, e), rl.lo, e);
  }
  const Limbs t = limbs(r);

  // -z^2 log2(e) = -a^2 log2(e) / 2, with a^2 exact in two bf16 limbs (a has
  // 8 significant bits) and log2(e) / 2 in three. It starts the accumulator
  // of Horner's last step, off the polynomial's dependency chain.
  const acc_t sq = aie::mul(a, a);
  const bf_t sq_hi = sq.to_vector<bfloat16>();
  const bf_t sq_lo = aie::msc(sq, sq_hi, one).to_vector<bfloat16>();
  const bf_t l2e0 = aie::broadcast<int16_t, 32>(0x3f39).cast_to<bfloat16>();
  const bf_t l2e1 =
      aie::broadcast<int16_t, 32>(int16_t(0xbaac)).cast_to<bfloat16>();
  const bf_t l2e2 = aie::broadcast<int16_t, 32>(0x366d).cast_to<bfloat16>();
  acc_t q = aie::msc(aie::msc(acc(kP[0]), sq_hi, l2e0), sq_hi, l2e1);
  q = aie::msc(aie::msc(aie::msc(q, sq_lo, l2e0), sq_hi, l2e2), sq_lo, l2e1);

  const Limbs c6 = {
      aie::broadcast<int16_t, 32>(int16_t(0xbcc3)).cast_to<bfloat16>(),
      aie::broadcast<int16_t, 32>(int16_t(0xb86f)).cast_to<bfloat16>()};
  acc_t p = mac3(acc(kP[5]), c6, t);
  AIE_LOOP_UNROLL_FULL
  for (int i = 4; i >= 1; i--)
    p = mac3(acc(kP[i]), limbs(p), t);
  const acc_t y = mac3(q, limbs(p), t);

  // h = (t a 2^-9) 2^y, y = log2(e) (P(t) - z^2) + 8. The 8 keeps 2^y normal
  // for every normal h; a lane whose 2^y underflows has a subnormal h.
  const bf_t a_2m9 = aie::mul(a, bfloat16(0.001953125f)).to_vector<bfloat16>();
  const Limbs ta = limbs(aie::mac(aie::mul(t.hi, a_2m9), t.lo, a_2m9));
  const Exp2F32 ex = exp2_f32_32(y.to_vector<float>());
  const Limbs w = limbs(acc_t(ex.bits.cast_to<float>()));
  const auto h = aie::select(
      aie::mac(aie::mac(aie::mul(ta.hi, w.hi), ta.hi, w.lo), ta.lo, w.hi)
          .to_vector<float>(),
      aie::zeros<float, 32>(), ex.below);

  acc_t xa;
  xa.from_vector(x);
  // Below -127 are the negative numbers; a NaN of either sign keeps x.
  const auto x_or_0 =
      aie::select(xa.to_vector<float>(), aie::zeros<float, 32>(),
                  aie::lt(xi, aie::broadcast<int16_t, 32>(-127)));
  // Below 2^-16, x Phi(x) rounds to x / 2, which x - h cannot reach near
  // 2^-126 and a 2^-9 does not survive below 2^-117. x / 2 also flushes a
  // subnormal result to 0, except that |x| = 0x00ff halves to a tie that
  // rounds up to the smallest normal, 0x0080 with x's sign.
  const auto half_x = aie::select(
      aie::mul(x, bfloat16(0.5f)).to_vector<bfloat16>().cast_to<int16_t>(),
      aie::sub(xi, aie::broadcast<int16_t, 32>(0x007f)),
      aie::eq(ai, aie::broadcast<int16_t, 32>(0x00ff)));
  return aie::select(acc_t(aie::sub(x_or_0, h)).to_vector<bfloat16>(),
                     half_x.cast_to<bfloat16>(),
                     aie::lt(ai, aie::broadcast<int16_t, 32>(0x3780)));
}

extern "C" {

// vector_size must be a multiple of 32.
void gelu_erf_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                        int32_t vector_size) {
  event0();
  // Every limb and the output round to bf16 in this mode.
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  // Each vector is one long dependency chain; two a trip fill each other's
  // latency. AIE2 spills too much for it to pay.
#if AIE_ARCH_AIE2P
  AIE_LOOP_UNROLL(2)
#endif
  for (int i = 0; i < vector_size / 32; i++) {
    aie::store_v(output, gelu_erf_32(aie::load_v<32>(input)));
    input += 32;
    output += 32;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
