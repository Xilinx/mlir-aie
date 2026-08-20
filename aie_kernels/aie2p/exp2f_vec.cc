//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x for exp-family ops needing more accuracy than the hardware
// `aie::exp2<bfloat16>` LUT behind bf16_exp.cc. Measured on aie2p against a
// float64 reference, the LUT's max relative error runs 6.1% on [-1, 0] to 49.1%
// on [-100, 0], softmax's range, where this poly holds 8.9e-5. See
// programming_examples/basic/vector_exp2f.
//
// 2^x = p(f) * 2^k, k = floor(x), f = x - k, with 2^k written straight into the
// f32 exponent field as (k + 127) << 23. The field is 8 bits wide, and that
// sets the hard ends of the domain: outside k in [-126, 127] the biased
// exponent carries into the sign bit and the result is finite but wrong-signed
// (k = 129 reads back as -0.0, k = -129 as -1.7e38), which isfinite() cannot
// catch.
//
// The lower clamp sits above the hard end, at -111, because accuracy runs out
// first. `aie::mul(p, 2^k)` loses low-order bits as the product approaches the
// bottom of the f32 normal range, measured on aie2p over f in [0, 1):
//
//   k >= -111   8.9e-5     k = -116   4.6e-4     k = -118   3.2e-3
//   k =  -114   1.1e-4     k = -117   1.7e-3     k <= -119  6.5e-3
//
// -111 is the last exponent holding the poly's own 8.9e-5. Move the clamp with
// -DEXP2F_VEC_MIN_X=<float> for a narrower domain, or for a wider one down to
// -126 at the accuracy above.
//
// At the top end 2^128 exceeds FLT_MAX, so +inf is the right answer, but it is
// copied in under a mask rather than computed: aie::mul returns NaN on f32
// overflow and that edge is ~1 ULP wide (127.99999 gives NaN, 127.99998 does
// not). Hence the arithmetic clamp at 127.999, collapsing [127.999, 128) onto
// one value at 7.8e-4 relative error.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// 512-bit vector register / 32-bit lanes.
static constexpr int EXP2F_VEC_LEN = 16;

#ifndef EXP2F_VEC_MIN_X
#define EXP2F_VEC_MIN_X (-111.0f)
#endif
static constexpr float kMinX = EXP2F_VEC_MIN_X;
static_assert(kMinX >= -126.0f,
              "2^k is built in the f32 exponent field, which bottoms out at "
              "the smallest normal, k = -126");

// noinline: Peano -O2 miscompiles the inlined form to NaN under high register
// pressure.
static __attribute__((noinline)) aie::vector<float, EXP2F_VEC_LEN>
exp2f_vec(aie::vector<float, EXP2F_VEC_LEN> x) {
  x = aie::max(x, aie::broadcast<float, EXP2F_VEC_LEN>(kMinX));
  // Taken before the clamp below narrows x.
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
