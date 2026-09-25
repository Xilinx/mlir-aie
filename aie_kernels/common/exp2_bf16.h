// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_EXP2_BF16_H
#define AIE_KERNELS_COMMON_EXP2_BF16_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// 2^x rounded to bf16: aie::exp2<bfloat16> on AIE2P. AIE2 has no exp2, so
// there x = k + f with k = round(x) and |f| <= 1/2. 2^f is a cubic in bf16
// products, within 0.2% of 2^f, and 2^k is added into its f32 exponent field.
// x is clamped to [-200, 127]. Past the bottom of the exponent field the sum
// goes negative, and is clamped to +0.
#if !AIE_HAS_NATIVE_EXP2
static inline aie::vector<bfloat16, 16> exp2_bf16_16(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-200.0f));
  x = aie::min(x, aie::broadcast<float, 16>(127.0f));
  // Adding 1.5 * 2^23 rounds x to an integer in the low mantissa bits.
  const auto magic = aie::broadcast<float, 16>(12582912.0f);
  const aie::vector<float, 16> xm = aie::add(x, magic);
  const aie::vector<int32_t, 16> k =
      aie::sub(xm.cast_to<int32_t>(), aie::broadcast<int32_t, 16>(0x4b400000));
  aie::accum<accfloat, 16> fa;
  fa.from_vector(aie::sub(x, aie::sub(xm, magic)));
  const aie::vector<bfloat16, 16> f = fa.to_vector<bfloat16>();

  auto horner = [&](aie::vector<bfloat16, 16> c1, float c0) {
    aie::accum<accfloat, 16> acc;
    acc.from_vector(aie::broadcast<float, 16>(c0));
    return aie::mac(acc, f, c1);
  };
  auto t = horner(aie::broadcast<bfloat16, 16>(0.0555041087f), 0.2402265069f);
  t = horner(t.to_vector<bfloat16>(), 0.6931471805f);
  t = horner(t.to_vector<bfloat16>(), 1.0f);

  aie::accum<accfloat, 16> r;
  r.from_vector(aie::max(aie::add(t.to_vector<float>().cast_to<int32_t>(),
                                  aie::upshift(k, 23)),
                         aie::zeros<int32_t, 16>())
                    .cast_to<float>());
  return r.to_vector<bfloat16>();
}

template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  if constexpr (N < 16) {
    return exp2_bf16_16(x.template grow<16>()).template extract<N>(0);
  } else {
    aie::vector<bfloat16, N> out;
    for (unsigned i = 0; i < N / 16; i++)
      out.insert(i, exp2_bf16_16(x.template extract<16>(i)));
    return out;
  }
}
#else
template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  return aie::exp2<bfloat16>(x);
}
#endif

#endif
