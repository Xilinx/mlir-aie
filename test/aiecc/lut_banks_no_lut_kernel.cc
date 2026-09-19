//===- lut_banks_no_lut_kernel.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for lut_banks_no_lut.mlir. Two tables and no aie::lut, so the
// check has pairs to not find.

#include <stdint.h>

alignas(32) short tbl_ab[512];
alignas(32) short tbl_cd[512];

extern "C" void classify(uint8_t *out) {
  out[0] = (uint8_t)(tbl_ab[out[0]] + tbl_cd[out[0]]);
}
