//===- stack_size_unmeasurable_kernel.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_unmeasurable_warns.mlir. Deliberately compiled
// WITHOUT -fstack-size-section, so the object carries no `.stack_sizes` and
// the call-graph analysis cannot size `touch_scratch` -- the "unmeasurable"
// path, which warns and leaves stack_size unvalidated rather than failing.

#include <stdint.h>

volatile uint8_t scratch[512];

extern "C" void touch_scratch(uint8_t *out) {
  for (int i = 0; i < 512; i++)
    out[i] = scratch[i];
}
