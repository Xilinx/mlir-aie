//===- tanh.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// 32 bf16 elements per iteration.  The native tanh intrinsic works on 16 float
// lanes, so each 32-wide input is split into two halves and re-concatenated.
void tanh_bf16_vectorized(bfloat16 *restrict input_vector,
                          bfloat16 *restrict output_vector,
                          const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(32)
  for (int i = 0; i < num_elems; i += 32) {
    auto input = *it_in++;

    aie::accum<accfloat, 16> acc_lo;
    aie::accum<accfloat, 16> acc_hi;
    acc_lo.from_vector(input.extract<16>(0), 0);
    acc_hi.from_vector(input.extract<16>(1), 0);
    auto tanh_lo = aie::tanh<bfloat16>(acc_lo.to_vector<float>());
    auto tanh_hi = aie::tanh<bfloat16>(acc_hi.to_vector<float>());

    *it_out++ = aie::concat(tanh_lo, tanh_hi);
  }

  event1();

  return;
}

extern "C" {

void tanh_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
               int input_size) {
  tanh_bf16_vectorized(input, output, input_size);
}

} // extern "C"
