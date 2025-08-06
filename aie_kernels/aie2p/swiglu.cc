//===- swiglu.cc --------------------------------------------*- C++
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
#include <stdint.h>

using namespace aie;

void swiglu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                             bfloat16 *restrict weight_vector_1,
                             bfloat16 *restrict weight_vector_2,
                             bfloat16 *restrict output_vector,
                             const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_wt_1 = aie::begin_restrict_vector<16>((bfloat16 *)weight_vector_1);
  auto it_wt_2 = aie::begin_restrict_vector<16>((bfloat16 *)weight_vector_2);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> input;
  aie::vector<bfloat16, 16> weight_1;
  aie::vector<bfloat16, 16> weight_2;
  aie::vector<bfloat16, 16> output;
  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 16> register_1 = aie::broadcast<bfloat16, 16>(1.0f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < num_elems; i += 16) {
    // Load input vector
    input = *it_in++;
    // Load weights
    weight_1 = *it_wt_1++;
    weight_2 = *it_wt_2++;

    // Compute the first part of the SWiGLU: x * w1
    aie::vector<bfloat16, 16> mul_input_weight_1 = aie::mul(input, weight_1);
    // Compute the second part of the SWiGLU: x * w2
    aie::vector<bfloat16, 16> mul_input_weight_2 = aie::mul(input, weight_2);

    // Compute tanh approximation
    auto half_x = aie::mul(mul_input_weight_2, register_0_5);
    auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
    auto tanh_half_x_approx = aie::add(tanh_half_x, register_1);
    aie::vector<bfloat16, 16> sigmoid_approx =
        aie::mul(tanh_half_x_approx, register_0_5);
    // Compute output: x * tanh_approx
    aie::vector<bfloat16, 16> silu_output =
        aie::mul(mul_input_weight_2, sigmoid_approx);

    // Multiply the first part with the second part
    auto mul_output = aie::mul(mul_input_weight_1, silu_output);

    // Store output vector
    *it_out++ = mul_output.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

void swiglu_bf16(bfloat16 *restrict input, bfloat16 *restrict weights_1,
                 bfloat16 *restrict weights_2, bfloat16 *restrict output) {
  int32_t input_size = 1024; // Assuming input size is a multiple of 16
  swiglu_tanh_approx_bf16(input, weights_1, weights_2, output, input_size);
}

} // extern "C"
