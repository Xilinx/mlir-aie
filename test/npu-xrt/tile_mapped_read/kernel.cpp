//===- read_processor_bus.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" {

void read_processor_bus(uint32_t *data, uint32_t addr, uint32_t size,
                        uint32_t stride) {
  for (uint32_t i = 0; i < size; i++) {
    uint32_t offset = addr + (i * stride);
    data[i] = read_tm(offset);
  }
}

} // extern "C"
