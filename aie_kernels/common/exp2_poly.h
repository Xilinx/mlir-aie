// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_EXP2_POLY_H
#define AIE_KERNELS_COMMON_EXP2_POLY_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// Return p(f) approximating 2^f, with x = k + f and 0 <= f < 1, for
// |x| < 2^22. Keep the polynomial shared; callers own their domain and
// exponent scaling.
//
// Use N = 32: an f32 aie::mul is emulated with 32-lane bf16 products, so a
// 16-lane call pays for 32 lanes and discards half of them.
template <unsigned N>
static inline aie::vector<float, N> exp2_poly(aie::vector<float, N> x,
                                              aie::vector<int32_t, N> &k) {
  // Adding 1.5 * 2^23 rounds x to an integer and leaves it in the low
  // mantissa bits, so k needs no float-to-int conversion. aie::to_fixed and
  // aie::to_float are emulated on aie2p through the shift-round unit.
  const auto magic = aie::broadcast<float, N>(12582912.0f);
  const auto xm = aie::add(x, magic);
  const auto r = aie::sub(xm, magic);
  // Step back the lanes that rounded up.
  const auto up = aie::lt(x, r);
  k = aie::sub(xm.template cast_to<int32_t>(),
               aie::broadcast<int32_t, N>(0x4b400000));
  k = aie::sub(k, aie::select(aie::zeros<int32_t, N>(),
                              aie::broadcast<int32_t, N>(1), up));
  const auto f =
      aie::sub(x, aie::sub(r, aie::select(aie::zeros<float, N>(),
                                          aie::broadcast<float, N>(1.0f), up)));
#if AIE_TUNED_AIE2
  // Estrin, (c0 + c1 f) + f^2 (c2 + c3 f) + f^4 (c4 + c5 f), chains three
  // emulated f32 multiplies where Horner chains five.
  auto fma = [](aie::vector<float, N> a, aie::vector<float, N> b,
                aie::vector<float, N> c) {
    aie::accum<accfloat, N> acc;
    acc.from_vector(c);
    return aie::mac(acc, a, b).template to_vector<float>();
  };
  auto lin = [&](float c1, float c0) {
    return fma(f, aie::broadcast<float, N>(c1), aie::broadcast<float, N>(c0));
  };
  const auto f2 = aie::mul(f, f).template to_vector<float>();
  const auto lo = lin(0.6931471805f, 1.0f);
  const auto mid = lin(0.0555041087f, 0.2402265069f);
  const auto hi = lin(0.0013333558f, 0.0096181291f);
  const auto f4 = aie::mul(f2, f2).template to_vector<float>();
  return fma(f4, hi, fma(f2, mid, lo));
#else
  auto p = aie::broadcast<float, N>(0.0013333558f);
  p = aie::add(aie::mul(p, f).template to_vector<float>(),
               aie::broadcast<float, N>(0.0096181291f));
  p = aie::add(aie::mul(p, f).template to_vector<float>(),
               aie::broadcast<float, N>(0.0555041087f));
  p = aie::add(aie::mul(p, f).template to_vector<float>(),
               aie::broadcast<float, N>(0.2402265069f));
  p = aie::add(aie::mul(p, f).template to_vector<float>(),
               aie::broadcast<float, N>(0.6931471805f));
  p = aie::add(aie::mul(p, f).template to_vector<float>(),
               aie::broadcast<float, N>(1.0f));
  return p;
#endif
}

#endif
