//===- data_size_measured_after_link_kernel.cc ------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for data_size_measured_after_link.mlir.
//
// `table` is larger than the tile's memory. The parameter of `lookup` shadows
// it, so nothing reads the global and --gc-sections drops its section. Only
// `bias` survives the link. A count taken from the object file would exceed the
// tile; a count taken from the linked ELF is 128 bytes.

#include <stdint.h>

const uint8_t table[300000] = {1};
const uint8_t bias[128] = {2};

static uint8_t lookup(const uint8_t *table, uint8_t i) { return table[i]; }

extern "C" void classify(uint8_t *out) {
  for (int i = 0; i < 64; i++)
    out[i] = lookup(bias, out[i]);
}
