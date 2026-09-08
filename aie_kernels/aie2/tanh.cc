//===- tanh.cc --------------------------------------------*- C++ -*-===//
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

// LUT-based tanh (aie2 has no native tanh intrinsic), 32 bf16 elems/iter.
void tanh_bf16_vectorized(bfloat16 *restrict input_vector,
                          bfloat16 *restrict output_vector,
                          const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(32)
  for (int i = 0; i < vector_size; i += 32) {
    auto input = *it_in++;

    aie::vector<bfloat16, 16> tanh_lo = getTanhBf16(input.extract<16>(0));
    aie::vector<bfloat16, 16> tanh_hi = getTanhBf16(input.extract<16>(1));

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
