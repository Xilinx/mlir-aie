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
// separate output buffer.  Used with the same memref for both in and out to
// perform an in-place scale after add_one_kernel.  Two-pointer form with no
// __restrict allows chess to generate correct vectorized code when in==out.
// Compiled to scale_kernel.o and linked via func-level link_with alongside
// add_one_kernel.o — exercises multi-.o linking through the func-level
// link_with path.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

extern "C" {

void scale_by_two(int32_t *in, int32_t *out, int32_t n) {
  for (int32_t i = 0; i < n; i++)
    out[i] = in[i] + in[i];
}

} // extern "C"
