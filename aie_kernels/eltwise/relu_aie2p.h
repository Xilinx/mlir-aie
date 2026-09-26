//===- relu_aie2p.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef RELU_ELEMS
#define RELU_ELEMS vector_size
#endif

// ReLU: f(x) = max(x, 0).  32 bf16 elements per iteration (one 512-bit AIE2P
// vector register).
void relu_vectorized_bf16(bfloat16 *restrict a, bfloat16 *restrict c,
                          const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<32>((bfloat16 *)a);
  auto it_out = aie::begin_restrict_vector<32>((bfloat16 *)c);

  vector<bfloat16, 32> zeroes = aie::zeros<bfloat16, 32>();

  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < RELU_ELEMS; i += 32) {
    vector<bfloat16, 32> input = *it_in++;
    vector<bfloat16, 32> output = aie::max(input, zeroes);
    *it_out++ = output;
  }

  event1();

  return;
}

extern "C" {

void bf16_relu(bfloat16 *a_in, bfloat16 *c_out) {
  relu_vectorized_bf16(a_in, c_out, 1024);
}

void relu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  relu_vectorized_bf16(input, output, input_size);
}

} // extern "C"
