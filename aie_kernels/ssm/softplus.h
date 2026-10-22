//===- softplus.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Float32 softplus for the state space kernels' step sizes, per vector lane.

#ifndef AIE_KERNELS_SSM_SOFTPLUS_H
#define AIE_KERNELS_SSM_SOFTPLUS_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../common/exp_f32.h"
#include "../common/f32_split.h"
#include "../common/log_q.h"

// softplus per lane, with exp_f32 for the exponential and each f32 product
// summed from exact limb products. The caller sets conv_even.
template <unsigned W>
static inline __attribute__((always_inline)) aie::accum<accfloat, W>
softplus_f32(aie::accum<accfloat, W> z) {
  using acc_t = aie::accum<accfloat, W>;
  using f_t = aie::vector<float, W>;
  auto acc = [](float c) { return acc_t(aie::broadcast<float, W>(c)); };
  const f_t zv = z.template to_vector<float>();
  const auto zi = zv.template cast_to<int32_t>();
  // -|z|, the sign bit set.
  const f_t minus_a = aie::bit_or(zi, aie::broadcast<int32_t, W>(INT32_MIN))
                          .template cast_to<float>();
  const f_t u = exp_f32(acc_t(minus_a)).template to_vector<float>();
  const auto halve = aie::ge(
      u.template cast_to<int32_t>(),
      aie::broadcast<float, W>(0.41421356f).template cast_to<int32_t>());
  const f_t half_u_minus_1 = mul_limbs(bf16_limbs(acc_t(aie::sub(u, 1.0f))),
                                       aie::broadcast<bfloat16, W>(0.5f))
                                 .template to_vector<float>();
  const Limbs<W> f = bf16_limbs(acc_t(aie::select(u, half_u_minus_1, halve)));
  acc_t q = acc(log_q[9]);
  for (int k = 8; k >= 0; k--)
    q = aie::add(mul_limbs(bf16_limbs(q), f), acc(log_q[k]));
  const f_t z_or_0 = aie::select(zv, aie::zeros<float, W>(),
                                 aie::lt(zi, aie::zeros<int32_t, W>()));
  const acc_t sum = aie::add(acc_t(z_or_0), mul_limbs(f, bf16_limbs(q)));
  return aie::add(
      sum, acc_t(aie::select(aie::zeros<float, W>(),
                             aie::broadcast<float, W>(0.69314718f), halve)));
}

#endif
