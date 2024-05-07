//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
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

template <bool put, bool get, int M_tile, int K_tile, int N_tile, int M, int K, int N>
void matmul_scalar_cascade_i32_i32(int32_t *a, int32_t *b, int32_t *c) {
  event0();  
  for (int m_t = 0; m_t < M_tile; m_t++) {
    for (int n_t = 0; n_t < N_tile; n_t++) {
      for (int k_t = 0; k_t < K_tile; k_t++) {
        int a_offset = (k_t * M_tile + m_t) * (M * K);
        int b_offset = (n_t * K_tile + k_t) * (K * N);
        int c_offset = (n_t * M_tile + m_t) * (M * N);
        for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
            int32_t running_sum = 0;
            if (get && k_t == 0) {
              v32int32 v32 = get_scd_v32int32();
              running_sum += ext_elem(v32, 0);
            }
            for (int k = 0; k < K; k++) {
              running_sum += a[a_offset + m * K + k] * b[b_offset + k * N + n];
            }
            c[c_offset + m * N + n] += running_sum;
            if (put && k_t == K_tile - 1) {
              v32int32 v32 = undef_v32int32();
              v32 = upd_elem(v32, 0, c[c_offset + m * N + n]);
              put_mcd(v32);
            }
          }
        }
      }
    }
  }
  event1();
}

extern "C" {

void matmul_scalar_put_4x1x4_4x8x4_i32_i32(int32_t *a, int32_t *b, int32_t *c) {
  matmul_scalar_cascade_i32_i32<true, false, 4, 1, 4, 4, 8, 4>(a, b, c);
}
void matmul_scalar_get_4x1x4_4x8x4_i32_i32(int32_t *a, int32_t *b, int32_t *c) {
  matmul_scalar_cascade_i32_i32<false, true, 4, 1, 4, 4, 8, 4>(a, b, c);
}

}

