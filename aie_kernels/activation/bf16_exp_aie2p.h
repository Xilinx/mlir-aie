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
#include "../common/exp2_poly.h"

#define VEC_LEN 32
#define log2e 1.44269504089f

using namespace aie;

// Keep the polynomial out of the tile loop: the inlined exp2f_vec form has
// exhibited Peano -O2 register-pressure miscompiles. The call takes pointers
// because a 32-lane vector argument and return travel through the stack.
static __attribute__((noinline)) void exp_bf16_vec(const bfloat16 *in,
                                                   bfloat16 *out) {
  aie::vector<bfloat16, VEC_LEN> input_bf16 = aie::load_v<VEC_LEN>(in);
  // Match the LUT-backed AIE2 kernel's domain, including infinite inputs.
  const auto upper = aie::broadcast<bfloat16, VEC_LEN>(EXP_BF16_CLAMP);
  const auto lower = aie::broadcast<bfloat16, VEC_LEN>(-EXP_BF16_CLAMP);
  input_bf16 = aie::select(input_bf16, upper, aie::gt(input_bf16, upper));
  input_bf16 = aie::select(input_bf16, lower, aie::lt(input_bf16, lower));
  // A bf16 log2(e) introduces about 17% relative error at x = 88.
  const aie::accum<accfloat, VEC_LEN> input_acc(input_bf16);
  // NaN passes the clamp, and the rounding in exp2_poly turns it into a
  // meaningless finite exponent. Return a quiet NaN instead.
  const auto is_nan =
      aie::gt(aie::bit_and(input_acc.to_vector<float>().cast_to<int32_t>(),
                           aie::broadcast<int32_t, VEC_LEN>(0x7fffffff)),
              aie::broadcast<int32_t, VEC_LEN>(0x7f800000));
  const aie::accum<accfloat, VEC_LEN> exp_in = aie::mul(
      input_acc.to_vector<float>(), aie::broadcast<float, VEC_LEN>(log2e));

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<int32_t, VEC_LEN> k;
  const auto p = exp2_poly(exp_in.to_vector<float>(), k);
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
  auto result = aie::select(normal, subnormal,
                            aie::eq(k, aie::broadcast<int32_t, VEC_LEN>(-127)));
  result =
      aie::select(result, aie::broadcast<int32_t, VEC_LEN>(0x7fc0), is_nan);
  aie::store_v(out, aie::pack(result).cast_to<bfloat16>());
}

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {
  static_assert(N % VEC_LEN == 0);
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < N / VEC_LEN; i++) {
    exp_bf16_vec(in, out);
    in += VEC_LEN;
    out += VEC_LEN;
  }
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  event0();
  exp_bf16_func<1024>(a_in, c_out);
  event1();
}

} // extern "C"
