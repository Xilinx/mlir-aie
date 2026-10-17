//===- gelu_erf.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/exp2_bf16.h"
#include "../common/f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GELU_ERF_ELEMS
#define GELU_ERF_ELEMS n
#endif

// erfc(z) = t exp(P(t) - z^2), t = 1 / (1 + z / 2), with P Numerical Recipes'
// erfcc fit: within 1.2e-7 of erfc, relatively, for every z >= 0. Highest
// coefficient first.
static const float erfc_p[10] = {
    0.17087277f,  -0.82215223f, 1.48851587f, -1.13520398f, 0.27886807f,
    -0.18628806f, 0.09678418f,  0.37409196f, 1.00002368f,  -1.26551223f};

// x Phi(x) = 0.5 x (1 + erf(x / sqrt(2))) of bf16, rounded once to bf16. With
// a = |x| and e = 0.5 a erfc(a / sqrt(2)), it is x - e for x >= 0 and -e
// below, so neither side cancels. Multiplies as f32_split.h does.
extern "C" void gelu_erf_bf16(bfloat16 *restrict x, bfloat16 *restrict y,
                              int32_t n) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  auto splat = [](float c) {
    return f32_acc(aie::broadcast<float, f32_lanes>(c));
  };
  const bf16_vec one = bf16_splat(0x3f80);
  // 1 / (2 sqrt(2)) and log2(e), three bf16 limbs each.
  const Limbs<f32_lanes> c_limbs = bf16_limbs(splat(0.35355339059327373f));
  const Limbs<f32_lanes> log2e_limbs = bf16_limbs(splat(1.44269504088896341f));
  const auto magnitude = aie::broadcast<uint16_t, f32_lanes>(0x7fff);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < GELU_ERF_ELEMS; i += f32_lanes) {
    const bf16_vec xv = aie::load_v<f32_lanes>(x + i);
    const auto xbits = xv.cast_to<uint16_t>();
    // Past |x| = 16, x Phi(x) is x or 0 in f32 already.
    const bf16_vec a =
        aie::min(aie::bit_and(xbits, magnitude).cast_to<bfloat16>(),
                 aie::broadcast<bfloat16, f32_lanes>(16.0f));
    const bf16_vec half_a =
        aie::mul(a, aie::broadcast<bfloat16, f32_lanes>(0.5f))
            .to_vector<bfloat16>();
    // z^2 = a (a / 2), exact; d = 1 + z / 2 in [1, 6.7].
    const f32_acc zz = aie::mul(a, half_a);
    f32_acc d = aie::mac(splat(1.0f), a, c_limbs.lo);
    d = aie::mac(d, a, c_limbs.mid);
    d = aie::mac(d, a, c_limbs.hi);

    // t = 1 / d from d's mantissa m in [1, 2): 1 / d = (1 / m) 2^-k.
    const auto dbits = d.to_vector<float>().cast_to<int32_t>();
    const auto k = aie::bit_and(
        aie::sub(dbits, aie::broadcast<int32_t, f32_lanes>(0x3f800000)),
        aie::broadcast<int32_t, f32_lanes>(0x7f800000));
    const auto m = aie::sub(dbits, k).cast_to<float>();
    const auto tv =
        aie::sub(inv_1_2(f32_acc(m)).to_vector<float>().cast_to<int32_t>(), k)
            .cast_to<float>();
    const Limbs<f32_lanes> tl = bf16_limbs(f32_acc(tv));

    // P(t) by Horner's rule on two limbs of each factor.
    f32_acc p = splat(erfc_p[0]);
    AIE_LOOP_UNROLL_FULL
    for (int j = 1; j < 10; j++) {
      const bf16_vec hi = p.to_vector<bfloat16>();
      const bf16_vec lo = aie::msc(p, hi, one).to_vector<bfloat16>();
      f32_acc q = aie::mac(splat(erfc_p[j]), hi, tl.hi);
      q = aie::mac(q, hi, tl.mid);
      p = aie::mac(q, lo, tl.hi);
    }

    // exp(P(t) - z^2) as 2^((P(t) - z^2) log2(e)); P(t) - z^2 <= 0.
    const f32_acc arg = aie::sub(p, zz.to_vector<float>());
    const Exp2F32 e2 = exp2_f32_32(mac_limbs(aie::zeros<accfloat, f32_lanes>(),
                                             bf16_limbs(arg), log2e_limbs)
                                       .to_vector<float>());
    const f32_acc ex(aie::select(e2.bits.cast_to<float>(),
                                 aie::zeros<float, f32_lanes>(), e2.below));
    // e = (a t / 2) exp(...): a t / 2 is at least 1 wherever exp(...) is near
    // the bottom of the normals, so the product is not flushed on the way.
    f32_acc g = aie::mul(tl.lo, half_a);
    g = aie::mac(g, tl.mid, half_a);
    g = aie::mac(g, tl.hi, half_a);
    const f32_acc e = mac_limbs(aie::zeros<accfloat, f32_lanes>(),
                                bf16_limbs(g), bf16_limbs(ex));

    const auto ev = e.to_vector<float>();
    const auto negative =
        aie::lt(xbits.cast_to<int16_t>(), aie::zeros<int16_t, f32_lanes>());
    const auto above = aie::sub(aie::mul(xv, one), ev).to_vector<float>();
    const auto below =
        aie::sub(aie::zeros<accfloat, f32_lanes>(), ev).to_vector<float>();
    const bf16_vec out =
        f32_acc(aie::select(above, below, negative)).to_vector<bfloat16>();
    // A NaN passes through.
    const auto nan = aie::gt(aie::bit_and(xbits, magnitude),
                             aie::broadcast<uint16_t, f32_lanes>(0x7f80));
    aie::store_v(y + i, aie::select(out, xv, nan));
  }
  aie::set_rounding(saved);
  event1();
}
