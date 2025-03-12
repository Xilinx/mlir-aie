//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T, int M, int N, int r>
void zero_vectorized(T *__restrict pC, unsigned offsetC) {
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  T *__restrict pC1 = pC + offsetC;
  const T *__restrict c_end = pC1 + M * N;
  for (; pC1 + r < c_end; pC1 += r) {
    aie::store_v(pC1, zeros);
  }
  // Do a scalar write for any remainder not divisible by vector instruction
  // size r
  for (; pC1 < c_end; pC1++) {
    *pC1 = 0;
  }
}

#endif
