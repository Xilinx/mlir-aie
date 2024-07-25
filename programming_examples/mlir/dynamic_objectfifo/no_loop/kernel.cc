//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned long N>
void sum(const T_in *__restrict inA, const T_in *__restrict inB, T_out *__restrict out) {
  for(int i = 0; i < N; i++) {
    out[i] = inA[i] + inB[i];
  }
}

extern "C" {

void sum_64_i32(const int *__restrict inA, const int *__restrict inB, int *__restrict out) {
  sum<int, int, 64>(inA, inB, out);
}
}
