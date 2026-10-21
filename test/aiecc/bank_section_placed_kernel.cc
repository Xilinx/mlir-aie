//===- bank_section_placed_kernel.cc ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for bank_section_placed.mlir and bank_region_overflow.mlir.

#include <stdint.h>

__attribute__((section(".aie.bank1"), aligned(32))) float tbl_ab[256];
__attribute__((section(".aie.bank2"), aligned(32))) float tbl_cd[256];

extern "C" void classify(uint8_t *out) {
  out[0] = (uint8_t)(tbl_ab[out[0]] + tbl_cd[out[0]]);
}
