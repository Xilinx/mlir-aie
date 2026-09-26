//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x for exp-family ops needing more accuracy than the hardware
// `aie::exp2<bfloat16>` LUT. Measured on aie2p against a
// float64 reference, the LUT's max relative error runs 6.1% on [-1, 0] to 49.1%
// on [-100, 0], softmax's range, where this kernel holds 9.2e-6. See
// programming_examples/basic/vector_exp2f.
//
// 2^x = p(f) * 2^k, f = x - k, with k added straight into the f32 exponent
// field. The field is 8 bits wide, and that sets the hard ends of the domain:
// outside k in [-126, 127] the biased exponent carries into the sign bit and
// the result is finite but wrong-signed (k = 129 reads back as -0.0, k = -129
// as -1.7e38), which isfinite() cannot catch. The add is exact, and on aie2p
// the accuracy above holds down to -126. The clamp defaults to -111; move it
// with -DEXP2F_VEC_MIN_X=<float>.
//
// At the top end 2^128 exceeds FLT_MAX, so +inf is copied in under a mask for
// x >= 128, and [127.999, 128) is clamped onto one value at 7.8e-4 relative
// error.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/exp2_poly.h"

using namespace aie;

// The width of one emulated f32 multiply (see exp2_poly.h).
static constexpr int EXP2F_VEC_LEN = 32;

#ifndef EXP2F_VEC_MIN_X
#define EXP2F_VEC_MIN_X (-111.0f)
#endif
static constexpr float kMinX = EXP2F_VEC_MIN_X;
static_assert(kMinX >= -126.0f,
              "2^k is built in the f32 exponent field, which bottoms out at "
              "the smallest normal, k = -126");

#if AIE_TUNED_AIE2P
// k = round(x) here, so f = x - k is in [-1/2, 1/2]. There is no f32 vector
// multiply: each Horner step multiplies the [hi | lo] bf16 limbs of the
// running value by those of f into an f32 accumulator.
static inline void exp2f_vec(const float *in, float *out) {
  using acc_t = aie::accum<accfloat, EXP2F_VEC_LEN>;
  using bf_t = aie::vector<bfloat16, EXP2F_VEC_LEN>;
  auto bf = [](int16_t bits) {
    return aie::broadcast<int16_t, EXP2F_VEC_LEN>(bits).cast_to<bfloat16>();
  };
  auto acc = [](float c) {
    return acc_t(aie::broadcast<float, EXP2F_VEC_LEN>(c));
  };

  auto fbits = [](float v) {
    return aie::broadcast<float, EXP2F_VEC_LEN>(v).cast_to<int32_t>();
  };
  aie::vector<float, EXP2F_VEC_LEN> x = aie::load_v<EXP2F_VEC_LEN>(in);
  // The f32 compares are emulated; compare the bits instead. They order as
  // signed integers when the bound is positive, and in reverse as unsigned
  // ones among negative values.
  const auto xi = x.cast_to<int32_t>();
  // Doubling the bits drops the sign: NaNs are then above 0xff000000.
  const auto xu = xi.cast_to<uint32_t>();
  const auto is_nan = aie::gt(
      aie::add(xu, xu), aie::broadcast<uint32_t, EXP2F_VEC_LEN>(0xff000000u));
  const auto below = kMinX < 0.0f
                         ? aie::gt(xu, fbits(kMinX).cast_to<uint32_t>())
                         : aie::lt(xi, fbits(kMinX));
  const auto overflow = aie::ge(xi, fbits(128.0f));
  const auto above = aie::gt(xi, fbits(127.999f));
  x = aie::select(x, aie::broadcast<float, EXP2F_VEC_LEN>(kMinX), below);
  x = aie::select(x, aie::broadcast<float, EXP2F_VEC_LEN>(127.999f), above);

  // Adding 1.5 * 2^23 rounds x to an integer in the low mantissa bits.
  const auto magic = aie::broadcast<float, EXP2F_VEC_LEN>(12582912.0f);
  const aie::vector<float, EXP2F_VEC_LEN> xm = aie::add(x, magic);
  // The f32 add rounds x to the grid of an operand with a larger exponent,
  // so x - k goes through the accumulator, where k is exact in bf16.
  const bf_t one = bf(0x3f80);
  const bf_t k = acc_t(aie::sub(xm, magic)).to_vector<bfloat16>();
  const acc_t f = aie::msc(acc_t(x), k, one);
  const bf_t f_hi = f.to_vector<bfloat16>();
  const bf_t f_lo = aie::msc(f, f_hi, one).to_vector<bfloat16>();

  // Degree-4 fit of 2^f, 2.8e-6 relative; the last coefficient is a bf16.
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

  // k << 23 is bits(xm) << 23. The clamps hold k to [-126, 128] and p below
  // 1 when k is 128, so the sum is finite, and exact.
  const auto bits = p.to_vector<float>().cast_to<int32_t>();
  const auto y = aie::add(bits, aie::upshift(xm.cast_to<int32_t>(), 23));
  auto i32 = [](int32_t v) {
    return aie::broadcast<int32_t, EXP2F_VEC_LEN>(v);
  };
  const auto z = aie::select(y, i32(0x7f800000), overflow);
  aie::store_v(out, aie::select(z, i32(0x7fc00000), is_nan).cast_to<float>());
}
#else
// Pointers keep the 32-lane vectors off the stack.
static void exp2f_vec(const float *in, float *out) {
  aie::vector<float, EXP2F_VEC_LEN> x = aie::load_v<EXP2F_VEC_LEN>(in);
  x = aie::max(x, aie::broadcast<float, EXP2F_VEC_LEN>(kMinX));
  // Taken before the clamp below narrows x.
  aie::mask<EXP2F_VEC_LEN> overflow =
      aie::ge(x, aie::broadcast<float, EXP2F_VEC_LEN>(128.0f));
  x = aie::min(x, aie::broadcast<float, EXP2F_VEC_LEN>(127.999f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ki;
  const auto p = exp2_poly(x, ki);
  // AIE2 emulates the f32 multiply, so add k to p's exponent instead. p is in
  // [1, 2) and the clamps hold k to [-126, 127], so the sum stays a normal
  // float and is exact.
  aie::vector<float, EXP2F_VEC_LEN> result =
      aie::add(p.cast_to<int32_t>(), aie::upshift(ki, 23)).cast_to<float>();
  aie::vector<int32_t, EXP2F_VEC_LEN> pos_inf_bits =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0x7f800000);
  aie::vector<float, EXP2F_VEC_LEN> pos_inf = pos_inf_bits.cast_to<float>();
  aie::store_v(out, aie::select(result, pos_inf, overflow));
}
#endif

extern "C" {

// vector_size must be a multiple of 16.
void exp2f_vec_f32(float *restrict input, float *restrict output,
                   int32_t vector_size) {
  event0();
#if AIE_TUNED_AIE2P
  // f's limbs round to bf16 in this mode.
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  // One vector's dependency chain does not fill the schedule; two do.
  AIE_LOOP_UNROLL(2)
#endif
  for (int i = 0; i < vector_size / EXP2F_VEC_LEN; i++) {
    exp2f_vec(input, output);
    input += EXP2F_VEC_LEN;
    output += EXP2F_VEC_LEN;
  }
  // A 16-element tail runs through a scratch vector, lanes 16-31 unused.
  if (vector_size % EXP2F_VEC_LEN) {
    alignas(aie::vector_decl_align) float tail[EXP2F_VEC_LEN];
    aie::store_v(tail, aie::load_v<EXP2F_VEC_LEN / 2>(input));
    exp2f_vec(tail, tail);
    aie::store_v(output, aie::load_v<EXP2F_VEC_LEN / 2>(tail));
  }
#if AIE_TUNED_AIE2P
  aie::set_rounding(saved_rounding);
#endif

  event1();
}

} // extern "C"
