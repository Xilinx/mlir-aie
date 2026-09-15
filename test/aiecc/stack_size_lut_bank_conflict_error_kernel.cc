//===- stack_size_lut_bank_conflict_error_kernel.cc --------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for stack_size_lut_bank_conflict_error.mlir.
//
// The linked core uses LUT-style globals whose names match the conventions of
// the AIE runtime libraries and user code. The regression protects the case
// where stack_size reaches the second 16KB bank on npu2/aie2p.

#include <stdint.h>

alignas(32) const float activation_lut_ab[256] = {0.0f};
alignas(32) const float activation_lut_cd[256] = {1.0f};

extern "C" void classify(uint8_t *out) {
  unsigned i = out[0] & 63;
  out[0] = static_cast<uint8_t>(activation_lut_ab[i] + activation_lut_cd[i]);
}
