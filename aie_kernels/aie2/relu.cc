//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef RELU_ELEMS
#define RELU_ELEMS TILE_SIZE
#endif

void relu(bfloat16 *restrict a, bfloat16 *restrict c, const int TILE_SIZE) {
  const int v_factor = 32;
  v32bfloat16 zeroes = broadcast_zero_bfloat16();

  event0();
  AIE_PREPARE_FOR_PIPELINING
  for (size_t i = 0; i < RELU_ELEMS; i += v_factor) {
    v32bfloat16 input = *(v32bfloat16 *)(a + i);
    v32bfloat16 output = max(input, zeroes);
    *(v32bfloat16 *)(c + i) = output;
  }
  event1();
  return;
}

extern "C" {

void bf16_relu(bfloat16 *a_in, bfloat16 *c_out) { relu(a_in, c_out, 1024); }

void relu_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output,
                    int32_t input_size) {
  relu(input, output, input_size);
}

} // extern "C"
