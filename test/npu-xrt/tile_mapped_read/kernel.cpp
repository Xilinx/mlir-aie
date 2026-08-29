//===- read_processor_bus.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" {

constexpr uint32_t tm_start_addr = 0x80000;

void read_processor_bus(uint32_t *data, uint32_t addr, uint32_t size,
                        uint32_t stride) {
  volatile uint32_t *const addr_space_start =
      reinterpret_cast<volatile uint32_t *>(tm_start_addr);
  for (uint32_t i = 0; i < size; i++) {
    uint32_t offset = addr + (i * stride);
    data[i] = addr_space_start[offset / 4];
  }
}

} // extern "C"
