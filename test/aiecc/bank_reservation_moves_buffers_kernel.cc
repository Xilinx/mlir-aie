//===- bank_reservation_moves_buffers_kernel.cc ----------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for bank_reservation_moves_buffers.mlir. Two small tables, one
// per bank, of the shape an aie::lut<4> pair needs: far smaller than a bank,
// but each needing a bank of its own.

#include <stdint.h>

__attribute__((section(".aie.bank0"), aligned(64))) float table_a[128];
__attribute__((section(".aie.bank1"), aligned(64))) float table_b[128];

extern "C" void classify(uint8_t *out) {
  out[0] = (uint8_t)(table_a[out[0]] + table_b[out[0]]);
}
