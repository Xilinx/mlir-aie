//===- bf16_exp.cc ---------------------------*- C++-----*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#include "exp2_poly.h"

#define VEC_LEN 16
#define log2e 1.44269504089f

using namespace aie;

// Keep the polynomial out of the tile loop: the inlined exp2f_vec form has
// exhibited Peano -O2 register-pressure miscompiles.
static __attribute__((noinline)) aie::vector<bfloat16, VEC_LEN>
exp2_bf16(aie::vector<float, VEC_LEN> x) {
  aie::vector<int32_t, VEC_LEN> k;
  const auto p = exp2_poly(x, k);
  const auto bits = p.cast_to<int32_t>();
  const auto one = aie::broadcast<int32_t, VEC_LEN>(1);
  // x is in [-88*log2(e), 88*log2(e)], so k is in [-127, 126].
  // Scale and round in integer bits, avoiding AIE floating-point underflow
  // near -88. Normal results use the f32 exponent field and bf16 RNE.
  auto normal = aie::add(bits, aie::upshift(k, 23));
  normal = aie::add(normal, aie::broadcast<int32_t, VEC_LEN>(0x7fff));
  normal =
      aie::add(normal, aie::bit_and(aie::logical_downshift(bits, 16), one));
  normal = aie::logical_downshift(normal, 16);
  // k == -127: restore the implicit leading bit and round directly to the
  // subnormal bf16 significand (17 discarded bits, not 16).
  auto subnormal = aie::sub(bits, aie::broadcast<int32_t, VEC_LEN>(0x3f000000));
  subnormal = aie::add(subnormal, aie::broadcast<int32_t, VEC_LEN>(0xffff));
  subnormal =
      aie::add(subnormal, aie::bit_and(aie::logical_downshift(bits, 17), one));
  subnormal = aie::logical_downshift(subnormal, 17);
  const auto result = aie::select(
      normal, subnormal, aie::eq(k, aie::broadcast<int32_t, VEC_LEN>(-127)));
  return aie::pack(result).cast_to<bfloat16>();
}

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {

  auto it_exp_in = aie::cbegin_vector<VEC_LEN>((bfloat16 *)in);
  auto it_exp_out = aie::begin_vector<VEC_LEN>((bfloat16 *)out);

  const int elem_iters = N / VEC_LEN;

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<bfloat16, VEC_LEN> input_bf16;
  aie::accum<accfloat, VEC_LEN> exp_in;
  aie::vector<bfloat16, VEC_LEN> exp_val;
  aie::vector<float, VEC_LEN> log2e_vec = aie::broadcast<float, VEC_LEN>(log2e);
  const auto upper = aie::broadcast<bfloat16, VEC_LEN>(EXP_BF16_CLAMP);
  const auto lower = aie::broadcast<bfloat16, VEC_LEN>(-EXP_BF16_CLAMP);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    // Match the LUT-backed AIE2 kernel's domain, including infinite inputs.
    input_bf16 = aie::select(input_bf16, upper, aie::gt(input_bf16, upper));
    input_bf16 = aie::select(input_bf16, lower, aie::lt(input_bf16, lower));
    // A bf16 log2(e) introduces about 17% relative error at x = 88.
    const aie::accum<accfloat, VEC_LEN> input_acc(input_bf16);
    exp_in = aie::mul(input_acc.to_vector<float>(), log2e_vec);
    exp_val = exp2_bf16(exp_in.to_vector<float>());
    *it_exp_out++ = exp_val;
  }
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  event0();
  exp_bf16_func<1024>(a_in, c_out);
  event1();
}

} // extern "C"
