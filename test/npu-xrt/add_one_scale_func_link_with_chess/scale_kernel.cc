//===- scale_kernel.cc -------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// External AIE kernel: multiply every element of a buffer by 2, writing to a
// separate output buffer.  The loop is manually unrolled for n=8 (the fixed
// tile buffer size) to avoid a chess compiler bug where software pipelining
// sets lc=1 for loops with n < 9, causing only 1 iteration to execute.
// Compiled to scale_kernel.o and linked via func-level link_with alongside
// add_one_kernel.o — exercises multi-.o linking through the func-level
// link_with path.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

extern "C" {

void scale_by_two(int32_t *__restrict in, int32_t *__restrict out, int32_t n) {
  // Manually unrolled for n=8: avoids chess sw-pipeline bug (lc=1 for n<9).
  (void)n;
  out[0] = in[0] + in[0];
  out[1] = in[1] + in[1];
  out[2] = in[2] + in[2];
  out[3] = in[3] + in[3];
  out[4] = in[4] + in[4];
  out[5] = in[5] + in[5];
  out[6] = in[6] + in[6];
  out[7] = in[7] + in[7];
}

} // extern "C"
