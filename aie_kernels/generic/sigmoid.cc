//===- sigmoid.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "activations.h" // tanh_bf16_v16
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SIGMOID_ELEMS
#define SIGMOID_ELEMS vector_size
#endif

// sigmoid(x) = 0.5 * (1 + tanh(x/2)), 32 bf16 elements per iteration, with
// tanh(x/2) on the two 16-lane halves both tanh paths work in. Passing the
// multiply's accumulator straight in keeps x/2 in f32 on AIE2P; AIE2's LUT
// narrows it, which is the accuracy difference between the two architectures.
void sigmoid_tanh_approx_bf16(bfloat16 *restrict input_vector,
                              bfloat16 *restrict output_vector,
                              const int32_t vector_size) {
  event0();

  const int num_elems = SIGMOID_ELEMS;
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 32> register_1 = aie::broadcast<bfloat16, 32>(1.0f);
  aie::vector<bfloat16, 32> register_0_5_wide =
      aie::broadcast<bfloat16, 32>(0.5f);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < num_elems; i += 32) {
    auto input = *it_in++;

    auto tanh_lo = tanh_bf16_v16(aie::mul(input.extract<16>(0), register_0_5));
    auto tanh_hi = tanh_bf16_v16(aie::mul(input.extract<16>(1), register_0_5));
    aie::vector<bfloat16, 32> tanh_half_x = aie::concat(tanh_lo, tanh_hi);

    auto one_plus = aie::add(tanh_half_x, register_1);
    aie::vector<bfloat16, 32> sigmoid_approx =
        aie::mul(one_plus, register_0_5_wide);

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
