//===- tanh.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h" // tanh_bf16_v16
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef TANH_ELEMS
#define TANH_ELEMS vector_size
#endif

// 32 bf16 elements per iteration, split into the two 16-lane halves both tanh
// paths work in and re-concatenated.
void tanh_bf16_vectorized(bfloat16 *restrict input_vector,
                          bfloat16 *restrict output_vector,
                          const int32_t vector_size) {
  event0();

  const int num_elems = TANH_ELEMS;
#if AIE_TUNED_AIE2
  // AIE2's tanh reads a table; lut_map_bf16 lays the loop out around the reads.
  lut_map_bf16<4, false>(input_vector, output_vector, num_elems,
                         [](aie::vector<bfloat16, 16> x) {
                           return aie::vector<bfloat16, 16>(tanh_bf16_v16(x));
                         });
#elif AIE_TUNED_AIE2P && !ACTIVATIONS_NATIVE_TANH
  tanh_lut_map(input_vector, output_vector, num_elems);
#else
  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)output_vector);

  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < num_elems; i += 32) {
    auto input = *it_in++;

    auto tanh_lo = tanh_bf16_v16(input.extract<16>(0));
    auto tanh_hi = tanh_bf16_v16(input.extract<16>(1));

    *it_out++ = aie::concat(tanh_lo, tanh_hi);
  }
#endif

  event1();

  return;
}

extern "C" {

void tanh_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
               int input_size) {
  tanh_bf16_vectorized(input, output, input_size);
}

} // extern "C"
