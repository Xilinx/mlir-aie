//===- bank_placement_match_kernel.cc ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for bank_placement_match.mlir. A paired resource, so the table
// is satisfied by either of two banks. That keeps the test about whether the
// check accepts a satisfied request, rather than about which bank this linker
// happens to pick.

#include "me_annotations.h"
#include <stdint.h>

float __aie_dm_resource_ab activation_lut[256];

extern "C" void classify(uint8_t *out) {
  out[0] = (uint8_t)activation_lut[out[0]];
}
