// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_AIE2P_EXP2_POLY_H
#define AIE_KERNELS_AIE2P_EXP2_POLY_H

#include <aie_api/aie.hpp>
#include <stdint.h>

// Return p(f) approximating 2^f, with x = k + f and 0 <= f < 1.
// Keep the polynomial shared; callers own their domain and exponent scaling.
static inline aie::vector<float, 16> exp2_poly(aie::vector<float, 16> x,
                                               aie::vector<int32_t, 16> &k) {
  k = aie::to_fixed<int32_t>(x);
  const auto kf = aie::to_float<float>(k);
  const auto one = aie::broadcast<int32_t, 16>(1);
  const auto zero = aie::zeros<int32_t, 16>();
  // to_fixed rounds, so step back the lanes it rounded up.
  k = aie::sub(k, aie::select(zero, one, aie::lt(x, kf)));
  const auto f = aie::sub(x, aie::to_float<float>(k));
  auto p = aie::broadcast<float, 16>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, 16>(1.0f));
  return p;
}

#endif
