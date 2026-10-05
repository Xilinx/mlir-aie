//===- lut_runtime_lib_banks_kernel.cc --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for lut_runtime_lib_banks.mlir. Annotates nothing itself: the
// bank request lives on the definitions, which is the point. The .cpp is
// included so one object carries both the tables and the caller.

#include "lut_based_ops.h"

#include "lut_based_ops.cpp"

#include <stdint.h>

extern "C" void classify(bfloat16 *inout) {
  v16accfloat e = getExpBf16(*(v16bfloat16 *)inout);
  *(v16bfloat16 *)inout = to_v16bfloat16(e);
}
