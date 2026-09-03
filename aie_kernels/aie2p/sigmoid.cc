//===- sigmoid.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// sigmoid(x) = 0.5 * (1 + tanh(x/2)), 32 bf16 elements per iteration.  The
// native tanh works on 16 float lanes, so tanh(x/2) is computed on two halves.
void sigmoid_tanh_approx_bf16(bfloat16 *restrict input_vector,
                              bfloat16 *restrict output_vector,
                              const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, 32> register_1 = aie::broadcast<bfloat16, 32>(1.0f);
  aie::vector<bfloat16, 32> register_0_5_wide =
      aie::broadcast<bfloat16, 32>(0.5f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(32)
  for (int i = 0; i < num_elems; i += 32) {
    auto input = *it_in++;

    // tanh(x/2) computed on two 16-wide halves, then recombined.
    auto half_x_lo = aie::mul(input.extract<16>(0), register_0_5);
    auto half_x_hi = aie::mul(input.extract<16>(1), register_0_5);
    auto tanh_lo = aie::tanh<bfloat16>(half_x_lo.to_vector<float>());
    auto tanh_hi = aie::tanh<bfloat16>(half_x_hi.to_vector<float>());
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
