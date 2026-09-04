//===- data_region_overflow_kernel.cc ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for data_region_overflow.mlir. A `.bss` far larger than the
// space this core's buffers leave, so the link cannot place it.

#include <stdint.h>

volatile uint8_t hog[40000];

extern "C" void touch(uint8_t *out) {
  for (int i = 0; i < 64; i++)
    out[i] = hog[i];
}
