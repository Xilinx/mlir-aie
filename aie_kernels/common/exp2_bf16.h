// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_KERNELS_COMMON_EXP2_BF16_H
#define AIE_KERNELS_COMMON_EXP2_BF16_H

#include "../aie_arch.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// 2^x rounded to bf16: a polynomial under -DEXP2_BF16_ACCURATE (below) on
// either architecture, else aie::exp2<bfloat16> on AIE2P. AIE2 has no exp2, so
// there x = k + f with k = round(x) and |f| <= 1/2. 2^f is a cubic in bf16
// products, within 0.2% of 2^f, and 2^k is added into its f32 exponent field.
// x is clamped to [-200, 127]. Past the bottom of the exponent field the sum
// goes negative, and is clamped to +0.
#if !AIE_HAS_NATIVE_EXP2 && !defined(EXP2_BF16_ACCURATE)
static inline aie::vector<bfloat16, 16> exp2_bf16_16(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-200.0f));
  x = aie::min(x, aie::broadcast<float, 16>(127.0f));
  // The 1.5 * 2^23 rounding: see exp2_poly.h.
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
#elif defined(EXP2_BF16_ACCURATE)
// exp2f_vec.cc's limb Horner, rounded once to bf16: within 2.8e-6 of
// 2^x before that rounding, where aie::exp2<bfloat16> is off by 6% on [-1, 0]
// and by 49% on [-100, 0]. The caller sets conv_even. k = round(x) and
// f = x - k in [-1/2, 1/2]; k is added into p(f)'s exponent field, which
// leaves the normal floats when k = -126 and p < 1, so x < -125.5 (and -inf)
// gives +0, x >= 128 gives +inf, and NaN stays NaN.
static inline aie::vector<bfloat16, 32> exp2_bf16_32(aie::vector<float, 32> x) {
  using acc_t = aie::accum<accfloat, 32>;
  using bf_t = aie::vector<bfloat16, 32>;
  auto bf = [](int16_t bits) {
    return aie::broadcast<int16_t, 32>(bits).cast_to<bfloat16>();
  };
  auto acc = [](float c) { return acc_t(aie::broadcast<float, 32>(c)); };
  auto bits = [](float v) {
    return aie::broadcast<float, 32>(v).cast_to<int32_t>();
  };
  // The f32 compares are emulated, so the bits are compared: they order as
  // signed integers among positive values and in reverse as unsigned ones
  // among negative values. Doubling them drops the sign, leaving NaNs above
  // 0xff000000.
  const auto xi = x.cast_to<int32_t>();
  const auto xu = xi.cast_to<uint32_t>();
  const auto is_nan =
      aie::gt(aie::add(xu, xu), aie::broadcast<uint32_t, 32>(0xff000000u));
  const auto below = aie::gt(xu, bits(-125.5f).cast_to<uint32_t>());
  const auto overflow = aie::ge(xi, bits(128.0f));

  // The 1.5 * 2^23 rounding: see exp2_poly.h. x - k goes through the
  // accumulator, where k is exact in bf16.
  const auto magic = aie::broadcast<float, 32>(12582912.0f);
  const aie::vector<float, 32> xm = aie::add(x, magic);
  const bf_t one = bf(0x3f80);
  const bf_t k = acc_t(aie::sub(xm, magic)).to_vector<bfloat16>();
  const acc_t f = aie::msc(acc_t(x), k, one);
  const bf_t f_hi = f.to_vector<bfloat16>();
  const bf_t f_lo = aie::msc(f, f_hi, one).to_vector<bfloat16>();

  acc_t p = aie::mac(acc(0.0559063815f), f_hi, bf(0x3c1d));
  auto step = [&](float c) {
    const bf_t hi = p.to_vector<bfloat16>();
    const bf_t lo = aie::msc(p, hi, one).to_vector<bfloat16>();
    acc_t q = aie::mac(acc(c), hi, f_hi);
    q = aie::mac(q, hi, f_lo);
    p = aie::mac(q, lo, f_hi);
  };
  step(0.240241051f);
  step(0.693124175f);
  step(1.0f);

  // k << 23 is bits(xm) << 23.
  const auto y = aie::add(p.to_vector<float>().cast_to<int32_t>(),
                          aie::upshift(xm.cast_to<int32_t>(), 23));
  bf_t r = acc_t(y.cast_to<float>()).to_vector<bfloat16>();
  r = aie::select(r, aie::zeros<bfloat16, 32>(), below);
  r = aie::select(r, bf(0x7f80), overflow);
  return aie::select(r, bf(0x7fc0), is_nan);
}

template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  if constexpr (N < 32) {
    return exp2_bf16_32(x.template grow<32>()).template extract<N>(0);
  } else {
    aie::vector<bfloat16, N> out;
    for (unsigned i = 0; i < N / 32; i++)
      out.insert(i, exp2_bf16_32(x.template extract<32>(i)));
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
