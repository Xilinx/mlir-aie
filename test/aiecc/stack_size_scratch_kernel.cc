//===- stack_size_scratch_kernel.cc -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for the stack_size_* tests. Compiled with -fstack-size-section,
// aiecc measures `touch_scratch`. Compiled without it, the object carries no
// `.stack_sizes` section, which is the unmeasurable path: aiecc warns and
// leaves stack_size unchecked.

#include <stdint.h>

volatile uint8_t scratch[512];

extern "C" void touch_scratch(uint8_t *out) {
  for (int i = 0; i < 512; i++)
    out[i] = scratch[i];
}
