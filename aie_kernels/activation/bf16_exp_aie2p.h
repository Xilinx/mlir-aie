//===- bf16_exp_aie2p.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#include "../aie_kernel_utils.h"

#define VEC_LEN 32

using namespace aie;

// exp(x) = 2^f * 2^k with t = x log2(e), k = round(t) and f = t - k in
// [-1/2, 1/2]. There is no f32 vector multiply: x is exact in bf16, so t is
// three bf16 products against log2(e)'s limbs, and the last two Horner steps
// multiply the [hi | lo] bf16 limbs of the running value by those of f.
static inline aie::vector<bfloat16, VEC_LEN>
exp_bf16_vec(aie::vector<bfloat16, VEC_LEN> x) {
  using acc_t = aie::accum<accfloat, VEC_LEN>;
  using bf_t = aie::vector<bfloat16, VEC_LEN>;
  auto bf = [](int16_t bits) {
    return aie::broadcast<int16_t, VEC_LEN>(bits).cast_to<bfloat16>();
  };
  auto acc = [](float c) { return acc_t(aie::broadcast<float, VEC_LEN>(c)); };

  // Doubling the bits drops the sign: NaNs are then above 0xff00.
  auto u16 = [](uint16_t v) { return aie::broadcast<uint16_t, VEC_LEN>(v); };
  const auto xu = x.cast_to<uint16_t>();
  const auto is_nan = aie::gt(aie::add(xu, xu), u16(0xff00));
  // Match the LUT-backed AIE2 kernel's domain, including infinite inputs.
  x = aie::min(x, aie::broadcast<bfloat16, VEC_LEN>(EXP_BF16_CLAMP));
  x = aie::max(x, aie::broadcast<bfloat16, VEC_LEN>(-EXP_BF16_CLAMP));

  // log2(e) = 0x3fb9 + 0xbb2c + 0x36ec to 2^-26; smallest product first.
  const bf_t one = bf(0x3f80);
  acc_t t = aie::mul(x, bf(0x36ec));
  t = aie::mac(t, x, bf(0xbb2c));
  t = aie::mac(t, x, bf(0x3fb9));

  // Adding 1.5 * 2^23 rounds t to an integer in the low mantissa bits.
  const auto magic = aie::broadcast<float, VEC_LEN>(12582912.0f);
  const aie::vector<float, VEC_LEN> tv = t.to_vector<float>();
  const aie::vector<float, VEC_LEN> tm = aie::add(tv, magic);
  const acc_t f(aie::sub(tv, aie::sub(tm, magic)));
  const bf_t f_hi = f.to_vector<bfloat16>();
  const bf_t f_lo = aie::msc(f, f_hi, one).to_vector<bfloat16>();

  // Degree-4 fit of 2^f, 2.8e-6 relative; the last coefficient is a bf16.
  // The first two steps use only the high limbs; every result still rounds
  // correctly.
  acc_t p = aie::mac(acc(0.05590642f), f_hi, bf(0x3c1d));
  auto step = [&](float c) {
    const bf_t hi = p.to_vector<bfloat16>();
    const bf_t lo = aie::msc(p, hi, one).to_vector<bfloat16>();
    acc_t q = aie::mac(acc(c), hi, f_hi);
    q = aie::mac(q, hi, f_lo);
    p = aie::mac(q, lo, f_hi);
  };
  p = aie::mac(acc(0.24024099f), p.to_vector<bfloat16>(), f_hi);
  step(0.69312419f);
  step(1.0f);

  // Scale in integer bits, where k << 23 is bits(tm) << 23, then round to
  // bf16. The AIE conversion mishandles f32 subnormals; the only clamped
  // inputs with subnormal results are -87.5 and -88, so those are constants.
  static_assert(EXP_BF16_CLAMP == 88.0f);
  const auto bits = p.to_vector<float>().cast_to<int32_t>();
  const auto scaled = aie::add(bits, aie::upshift(tm.cast_to<int32_t>(), 23));
  bf_t y = acc_t(scaled.cast_to<float>()).to_vector<bfloat16>();
  const auto xc = x.cast_to<uint16_t>();
  y = aie::select(y, bf(0x006d), aie::gt(xc, u16(0xc2ae)));
  y = aie::select(y, bf(0x0042), aie::gt(xc, u16(0xc2af)));
  return aie::select(y, bf(0x7fc0), is_nan);
}

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {
  static_assert(N % VEC_LEN == 0);
  auto it_in = aie::begin_restrict_vector<VEC_LEN>(in);
  auto it_out = aie::begin_restrict_vector<VEC_LEN>(out);
  // The last conversion rounds to bf16 in this mode.
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  // One vector's dependency chain does not fill the schedule; two do.
  AIE_LOOP_UNROLL(2)
  for (int i = 0; i < N / VEC_LEN; i++)
    *it_out++ = exp_bf16_vec(*it_in++);
  aie::set_rounding(saved_rounding);
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  event0();
  exp_bf16_func<1024>(a_in, c_out);
  event1();
}

} // extern "C"
