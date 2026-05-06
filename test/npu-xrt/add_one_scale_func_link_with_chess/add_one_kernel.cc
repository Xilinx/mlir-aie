//===- add_one_kernel.cc -----------------------------------------*- C++
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
