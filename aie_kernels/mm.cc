//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
//  THIS IS WIP AND HAS BEEN TAKEN AND MODIFIED FROM MLIR-AIE.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, int M, int K, int N>
void matmul_scalar(T_in *a, unsigned offsetA, T_in *b, unsigned offsetB,
                   T_out *c, unsigned offsetC) {
  event0();
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < K; i++) {
        running_sum += a[offsetA + row * K + i] * b[offsetB + i * N + col];
      }
      c[offsetC + row * N + col] += running_sum;
    }
  }
  event1();
}

extern "C" {

#define combos(X)                                                              \
  X(int16, i16, int16, i16, 4, 4, 4)                                           \
  X(int32, i32, int32, i32, 4, 4, 4)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, unsigned offsetA, ctype_in *b_in, unsigned offsetB,      \
      ctype_out *c_out, unsigned offsetC) {                                    \
    matmul_scalar<ctype_in, ctype_out, 4, 4, 4>(a_in, offsetA, b_in, offsetB,  \
                                                c_out, offsetC);               \
  }

combos(matmul_scalar_c_func)

} // extern "C"
