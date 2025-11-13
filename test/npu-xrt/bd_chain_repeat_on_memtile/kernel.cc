//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

template <typename T, int N>
__attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out,
                                               const int32_t height,
                                               const int32_t width) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
  {
    *outPtr++ = *inPtr++;
  }

  event1();
}

extern "C" {

void passThroughLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  passThrough_aie<uint8_t, 64>(in, out, 1, lineWidth);
}

} // extern "C"
