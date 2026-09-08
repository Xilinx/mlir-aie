//===- stack_size_unmeasurable_kernel.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_unmeasurable_warns.mlir. The RUN line compiles
// this file without -fstack-size-section, so the object carries no
// `.stack_sizes` section and the analysis cannot size `touch_scratch`. That is
// the unmeasurable path: aiecc warns and leaves stack_size unchecked.

#include <stdint.h>

volatile uint8_t scratch[512];

extern "C" void touch_scratch(uint8_t *out) {
  for (int i = 0; i < 512; i++)
    out[i] = scratch[i];
}
