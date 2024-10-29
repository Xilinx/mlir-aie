//===- cascade_mm.cc --------------------------------------------*- C++ -*-===//
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

#include "zero.cc"

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_only(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = undef_v16int32();
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_get_only(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_get(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#define combos(X)                                                              \
  X(int16, i16, int16, i16, 4, 4, 4)                                           \
  X(int16, i16, int32, i32, 4, 4, 4)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_scalar_cascade_get_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_get_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_get_only<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(  \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_put_only<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(  \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_get_c_func(                                  \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_get_##mlir_type_in##_##mlir_type_out(         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar_cascade_put_get<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(   \
        a_in, b_in, c_out);                                                    \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                           \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

combos(matmul_scalar_cascade_get_only_c_func)
    combos(matmul_scalar_cascade_put_only_c_func)
        combos(matmul_scalar_cascade_put_get_c_func)
            combos(zero_vectorized_c_func) combos(zero_scalar_c_func)
} // extern "C"
