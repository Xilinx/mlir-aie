//===- add_one_kernel.cc -----------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// External AIE kernel: copy input to output, adding 1 to every element.
// Compiled to add_one_kernel.o and linked via func-level link_with.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

extern "C" {

void add_one(int32_t *__restrict in, int32_t *__restrict out, int32_t n) {
  for (int32_t i = 0; i < n; i++)
    out[i] = in[i] + 1;
}

} // extern "C"
