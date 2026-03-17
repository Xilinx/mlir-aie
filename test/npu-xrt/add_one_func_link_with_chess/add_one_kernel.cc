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
// External AIE kernel compiled with xchesscc_wrapper and linked via func-level
// link_with on func.func.  Increments every element of a buffer by 1.
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
