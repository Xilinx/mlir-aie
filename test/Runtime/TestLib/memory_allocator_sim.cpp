//===- memory_allocator_sim.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allocates through the AIESIM test_lib allocator and prints the simulated
// physical address of each buffer.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"

#include <cstdio>

// The simulator's global-memory accessors, which the allocator calls on sync.
extern "C" void ess_WriteGM(uint64_t, const void *, uint64_t) {}
extern "C" void ess_ReadGM(uint64_t, void *, uint64_t) {}

int main() {
  aie_libxaie_ctx_t ctx{};
  // 64 KiB each, then one word, then one more buffer.
  const int words[] = {16384, 16384, 1, 16};
  for (int size : words) {
    ext_mem_model_t handle{};
    if (!mlir_aie_mem_alloc(&ctx, handle, size))
      return 1;
    std::printf("physical 0x%llx\n", (unsigned long long)handle.physicalAddr);
  }
  return 0;
}
