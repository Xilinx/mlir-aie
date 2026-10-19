// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_EXP_F32_H
#define AIE_KERNELS_COMMON_EXP_F32_H

#include "f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// scalar_exp's algorithm per lane: x = k ln 2 + r by Cody-Waite (k ln 2's
// high part is exact), e^r by its Taylor series to r^7, then 2^k added into
// the exponent. Each f32 product is summed from exact limb products and each
// sum rounded once, as the scalar f32 operations are. Lanes below -87.33654
// give 0, above 88.72283 infinity (the ends of f32's normal range), and NaN
// itself. The caller sets conv_even.
template <unsigned W>
static inline __attribute__((always_inline)) aie::accum<accfloat, W>
exp_f32(aie::accum<accfloat, W> x) {
  using acc_t = aie::accum<accfloat, W>;
  using f_t = aie::vector<float, W>;
  auto acc = [](float c) { return acc_t(aie::broadcast<float, W>(c)); };
  auto bits = [](float v) {
    return aie::broadcast<float, W>(v).template cast_to<int32_t>();
  };
  const f_t xv = x.template to_vector<float>();
  const auto xi = xv.template cast_to<int32_t>();
  const auto xu = xi.template cast_to<uint32_t>();
  const auto is_nan =
      aie::gt(aie::bit_and(xi, aie::broadcast<int32_t, W>(INT32_MAX)),
              aie::broadcast<int32_t, W>(0x7f800000));
  const auto below = aie::gt(xu, bits(-87.33654f).template cast_to<uint32_t>());
  const auto above = aie::gt(xi, bits(88.72283f));

  // k = round(x log2 e) through the 1.5 * 2^23 rounding (exp2_poly.h).
  const f_t magic = aie::broadcast<float, W>(12582912.0f);
  const f_t t = mul_limbs(bf16_limbs(x), bf16_limbs(acc(1.44269504f)))
                    .template to_vector<float>();
  const f_t tm = aie::add(t, magic);
  const aie::vector<bfloat16, W> k =
      acc_t(aie::sub(tm, magic)).template to_vector<bfloat16>();
  const acc_t r =
      aie::sub(aie::sub(x, mul_limbs(bf16_limbs(acc(0.693145751953125f)), k)),
               mul_limbs(bf16_limbs(acc(1.42860677e-6f)), k));
  const Limbs<W> rl = bf16_limbs(r);
  acc_t p = acc(1.0f / 5040);
  for (float c :
       {1.0f / 720, 1.0f / 120, 1.0f / 24, 1.0f / 6, 0.5f, 1.0f, 1.0f})
    p = aie::add(mul_limbs(bf16_limbs(p), rl), acc(c));

  // 2^k scales p through its exponent field. k << 23 is k * 2^23, which fits
  // int32 for |k| <= 128; a shift here is a vector G_SHL Peano cannot
  // legalize once its amount is hoisted out of the caller's loop.
  const auto k_int = aie::sub(tm.template cast_to<int32_t>(),
                              aie::broadcast<int32_t, W>(0x4b400000));
  f_t e =
      aie::add(p.template to_vector<float>().template cast_to<int32_t>(),
               aie::mul(k_int, int32_t(1 << 23)).template to_vector<int32_t>())
          .template cast_to<float>();
  e = aie::select(e, aie::zeros<float, W>(), below);
  e = aie::select(e, aie::broadcast<float, W>(__builtin_inff()), above);
  return acc_t(aie::select(e, xv, is_nan));
}

#endif
