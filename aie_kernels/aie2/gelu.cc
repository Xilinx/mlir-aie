//===- gelu.cc --------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

void gelu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                           bfloat16 *restrict output_vector,
                           const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> input;

  // Constants
  const bfloat16 k0_5 = 0.5f;
  const bfloat16 k1 = 1.0f;
  const bfloat16 sqrt_2_over_pi = 0.79788456f; // ≈ sqrt(2/π)
  const bfloat16 kBeta = 0.044715f;

  auto v05 = aie::broadcast<bfloat16, 16>(k0_5);
  auto v1 = aie::broadcast<bfloat16, 16>(k1);
  auto vs2opi = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi);
  auto vBeta = aie::broadcast<bfloat16, 16>(kBeta);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int i = 0; i < vector_size; i += 16) {
    input = *it_in++;
    auto x = input;

    // Compute x^3
    aie::vector<bfloat16, 16> x2 = aie::mul(x, x);  // x^2
    aie::vector<bfloat16, 16> x3 = aie::mul(x, x2); // x^3

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    aie::vector<bfloat16, 16> x3_beta = aie::mul(x3, vBeta);
    aie::vector<bfloat16, 16> inner = aie::add(x, x3_beta);
    aie::vector<bfloat16, 16> inner1 = aie::mul(inner, vs2opi);

    // tanh_out = tanh(inner)
    aie::vector<bfloat16, 16> tanh_out = getTanhBf16(inner1);

    // result = 0.5 * x * (1 + tanh_out)
    aie::vector<bfloat16, 16> one_plus_tanh = aie::add(tanh_out, v1);
    // Multiply by x and 0.5
    aie::vector<bfloat16, 16> mul_v05 = aie::mul(v05, one_plus_tanh);
    auto result = aie::mul(x, mul_v05);

    *it_out++ = result.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

void gelu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  int32_t input_size = 1024; // Assuming input size is a multiple of 16
  gelu_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
