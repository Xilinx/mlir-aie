//===- silu.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Processes 32 bf16 elements per iteration.  The bf16 tanh intrinsic works on
// 16-wide vectors, so each 32-wide input is split into two 16-wide halves for
// the tanh and re-concatenated; everything else stays 32-wide.
void silu_tanh_approx_bf16(bfloat16 *restrict input_vector,
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
    auto mul_output = aie::mul(input, sigmoid_approx);

    *it_out++ = mul_output.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

void silu_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  int32_t input_size = 1024; // Assuming input size is a multiple of 32
  silu_tanh_approx_bf16(input, output, input_size);
}

} // extern "C"
