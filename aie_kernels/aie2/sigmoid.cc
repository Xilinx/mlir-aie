//===- sigmoid.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "lut_based_ops.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// sigmoid(x) = 0.5 * (1 + tanh(x/2)) via LUT tanh (aie2), 32 bf16 elems/iter.
void sigmoid_tanh_approx_bf16(bfloat16 *restrict input_vector,
                              bfloat16 *restrict output_vector,
                              const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 32> register_0_5 = aie::broadcast<bfloat16, 32>(0.5f);
  aie::vector<bfloat16, 32> register_1 = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(32)
  for (int i = 0; i < vector_size; i += 32) {
    auto input = *it_in++;

    aie::vector<bfloat16, 32> half_x = aie::mul(input, register_0_5);

    aie::vector<bfloat16, 16> tanh_lo = getTanhBf16(half_x.extract<16>(0));
    aie::vector<bfloat16, 16> tanh_hi = getTanhBf16(half_x.extract<16>(1));
    aie::vector<bfloat16, 32> tanh_half_x = aie::concat(tanh_lo, tanh_hi);

    auto one_plus = aie::add(tanh_half_x, register_1);
    aie::vector<bfloat16, 32> sigmoid_approx = aie::mul(one_plus, register_0_5);

    *it_out++ = sigmoid_approx;
  }

  event1();

  return;
}

extern "C" {

void sigmoid_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  int input_size) {
  sigmoid_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
