//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x for exp-family ops that need more accuracy than the
// hardware `aie::exp2<bfloat16>` LUT behind bf16_exp.cc. Measured on aie2p
// against a float64 reference, the LUT's max relative error runs from 6.1% on
// [-1, 0] to 49.1% on [-100, 0], where this poly holds 8.5e-5. [-100, 0] is
// softmax's range: scores are shifted by the row max before exponentiating.
// See programming_examples/basic/vector_exp2f.
//
// x = k + f with k = floor(x), f in [0, 1); poly5(f) by Horner, 2^k by packing
// k + 127 into the f32 exponent field. The clamps keep k + 127 inside that
// 8-bit field; past it the biased exponent walks into the sign bit and the
// kernel returns wrong-signed finite values (k = 129 gives -0.0, k = 257 gives
// -2.0) that no isfinite() check catches.
//
// x <= -100 returns 2^-100, already zero for any softmax weight. x >= 128
// returns +inf, the true 2^x there since 2^128 > FLT_MAX, via a select rather
// than a saturating multiply through the all-ones exponent pattern: aie::mul
// here returns NaN, not +inf, on f32 overflow, and that edge is ~1 ULP wide
// (127.99999 gives NaN, 127.99998 does not). Hence the arithmetic clamp at
// 127.999, which collapses [127.999, 128) onto one value at a 7.8e-4 relative
// error. NaN propagates.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// 512-bit vector register / 32-bit lanes.
static constexpr int EXP2F_VEC_LEN = 16;

static __attribute__((noinline)) aie::vector<float, EXP2F_VEC_LEN>
exp2f_vec(aie::vector<float, EXP2F_VEC_LEN> x) {
  x = aie::max(x, aie::broadcast<float, EXP2F_VEC_LEN>(-100.0f));
  // Tested against the true boundary, before the clamp below narrows x.
  aie::mask<EXP2F_VEC_LEN> overflow =
      aie::ge(x, aie::broadcast<float, EXP2F_VEC_LEN>(128.0f));
  x = aie::min(x, aie::broadcast<float, EXP2F_VEC_LEN>(127.999f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ki =
      aie::to_fixed<int32_t>(x); // round-to-nearest on aie2p
  aie::vector<float, EXP2F_VEC_LEN> kf = aie::to_float<float>(ki);
  // floor(x): to_fixed rounds, so step back the lanes it rounded up.
  aie::vector<int32_t, EXP2F_VEC_LEN> one =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(1);
  aie::vector<int32_t, EXP2F_VEC_LEN> zero =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, EXP2F_VEC_LEN> f =
      aie::sub(x, aie::to_float<float>(ki)); // f in [0,1)
  aie::vector<float, EXP2F_VEC_LEN> p =
      aie::broadcast<float, EXP2F_VEC_LEN>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(1.0f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ebits = aie::upshift(
      aie::add(ki, aie::broadcast<int32_t, EXP2F_VEC_LEN>(127)), 23);
  aie::vector<float, EXP2F_VEC_LEN> p2k = ebits.cast_to<float>();
  aie::vector<float, EXP2F_VEC_LEN> result =
      aie::mul(p, p2k).to_vector<float>();
  // Lane-wise copy, not arithmetic, so it cannot overflow to NaN.
  aie::vector<int32_t, EXP2F_VEC_LEN> pos_inf_bits =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0x7f800000);
  aie::vector<float, EXP2F_VEC_LEN> pos_inf = pos_inf_bits.cast_to<float>();
  return aie::select(result, pos_inf, overflow);
}

extern "C" {

// vector_size must be a multiple of EXP2F_VEC_LEN.
void exp2f_vec_f32(float *restrict input, float *restrict output,
                   int32_t vector_size) {
  event0();

  auto it_in = aie::cbegin_vector<EXP2F_VEC_LEN>((float *)input);
  auto it_out = aie::begin_vector<EXP2F_VEC_LEN>((float *)output);
  const int elem_iters = vector_size / EXP2F_VEC_LEN;

  for (int i = 0; i < elem_iters; i++) {
    *it_out++ = exp2f_vec(*it_in++);
  }

  event1();
}

} // extern "C"
