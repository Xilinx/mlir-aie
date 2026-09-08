//===- leaky_relu.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Leaky ReLU: f(x) = max(x, alpha * x).  aie2 vector width 16.
void leaky_relu_vectorized_bf16(bfloat16 *restrict a, bfloat16 *restrict c,
                                const int32_t vector_size,
                                const bfloat16 alpha) {
  event0();

  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)a);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)c);

  vector<bfloat16, 16> alpha_vec = aie::broadcast<bfloat16, 16>(alpha);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int i = 0; i < vector_size; i += 16) {
    vector<bfloat16, 16> input = *it_in++;
    vector<bfloat16, 16> alpha_times_input = aie::mul(input, alpha_vec);
    vector<bfloat16, 16> output = aie::max(input, alpha_times_input);
    *it_out++ = output;
  }

  event1();

  return;
}

extern "C" {

void leaky_relu_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                     int input_size, bfloat16 alpha) {
  leaky_relu_vectorized_bf16(input, output, input_size, alpha);
}

} // extern "C"
