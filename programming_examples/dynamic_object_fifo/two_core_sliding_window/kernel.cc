//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned long N>
void passthrough(const T_in *__restrict in, T_out *__restrict out) {
  for (int i = 0; i < N; i++) {
    out[i] = in[i];
  }
}

extern "C" {

void passthrough_10_i32(const int *__restrict in, int *__restrict out) {
  passthrough<int, int, 10>(in, out);
}
}

template <typename T_in, typename T_out, unsigned long N>
void add(const T_in *__restrict inA, const T_in *__restrict inB,
         T_out *__restrict out) {
  for (int i = 0; i < N; i++) {
    out[i] = inA[i] + inB[i];
  }
}

extern "C" {

void add_10_i32(const int *__restrict inA, const int *__restrict inB,
                int *__restrict out) {
  add<int, int, 10>(inA, inB, out);
}
}
