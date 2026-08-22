//===- reserved_data_size_measured_kernel.cc --------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for reserved_data_size_measured.mlir. A single zero-initialised
// static array gives this object a `.bss` of a known, fixed size (8192 bytes)
// with nothing else contributing to .data/.rodata/.bss, so the auto-measured
// reserved_data_size is exactly that size plus the driver's fixed margin.

#include <stdint.h>

volatile uint8_t scratch[8192];

extern "C" void touch_scratch(uint8_t *out) {
  for (int i = 0; i < 512; i++)
    out[i] = scratch[i];
}
