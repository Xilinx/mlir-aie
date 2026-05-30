//===- passthrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
// Trivial single-tile passthrough kernel for the dispatch-overhead
// bisector example.
//
// Pure memcpy: copy ``lineWidth`` uint8_t bytes from in to out using
// 64-byte vector loads/stores. No arithmetic, no branching, no
// per-element work. The bisector pushes compute down to the noise
// floor so any per-launch wall-time the host runner measures is
// dispatch / wait / DMA-setup overhead, not kernel arithmetic.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

namespace {

template <typename T, int N>
__attribute__((noinline)) void passthrough_aie(T *restrict in, T *restrict out,
                                               int32_t length) {
  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int j = 0; j < length; j += N) {
    *outPtr++ = *inPtr++;
  }
}

} // namespace

extern "C" {

void passThroughLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  passthrough_aie<uint8_t, 64>(in, out, lineWidth);
}

} // extern "C"
