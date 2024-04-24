//===- vector_scaler_mul.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

extern "C" {

void vector_scalar_mul_aie_scalar(int32_t *a, int32_t *c, int32_t *factor,
                                  int32_t N) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = *factor * a[i];
  }
  event1();
}
} // extern "C"
