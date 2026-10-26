//===- bank_placement_mismatch_kernel.cc ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for bank_placement_mismatch.mlir. The two tables of an
// `aie::lut<4>` pair, each placed for its own bank. The chess linker packs them
// adjacently instead, so the second lands in the bank the first asked for.

#include "me_annotations.h"
#include <stdint.h>

float __aie_dm_resource_a activation_lut_ab[256];
float __aie_dm_resource_b activation_lut_cd[256];

extern "C" void classify(uint8_t *out) {
  out[0] = (uint8_t)(activation_lut_ab[out[0]] + activation_lut_cd[out[0]]);
}
