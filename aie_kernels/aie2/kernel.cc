//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned long N>
void add(const T_in *__restrict inA, const T_in *__restrict inB,
         T_out *__restrict out) {
  for (int i = 0; i < N; i++) {
    out[i] = inA[i] + inB[i];
  }
}

template <typename T>
__attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(5)
  for (int j = 0; j < 10; j += 1) // Nx samples per loop
  {
    *outPtr++ = *inPtr++;
  }

  event1();
}

extern "C" {

void add_10_i32(const int *__restrict inA, const int *__restrict inB,
                int *__restrict out) {
  add<int, int, 10>(inA, inB, out);
}

void passthrough_10_i32(int32_t *in, int32_t *out) {
  passThrough_aie<int32_t>(in, out);
}
}
