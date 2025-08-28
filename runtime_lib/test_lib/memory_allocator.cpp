//===- memory_allocator.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
#include <assert.h>
#include <iostream>

extern "C" {

void ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
void ess_ReadGM(uint64_t addr, void *data, uint64_t size);
}

int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &handle,
                        int size) {
  int size_bytes = size * sizeof(int);
  handle.virtualAddr = std::malloc(size_bytes);
  if (handle.virtualAddr) {
    handle.size = size_bytes;
    // assign physical space in SystemC DDR memory controller
    handle.physicalAddr = nextAlignedAddr;
    // adjust nextAlignedAddr to the next 128-bit aligned address
    nextAlignedAddr = nextAlignedAddr + size_bytes;
    uint64_t gapToAligned = nextAlignedAddr % 16; // 16byte (128bit)
    if (gapToAligned > 0)
      nextAlignedAddr += (16 - gapToAligned);
  } else {
    printf("ExtMemModel: Failed to allocate %d memory.\n", size_bytes);
  }

  _xaie->allocations.push_back(handle);

  std::cout << "ExtMemModel constructor: " << _xaie << " virtual address "
            << std::hex << handle.virtualAddr << ", physical address "
            << handle.physicalAddr << ", size " << std::dec << handle.size
            << std::endl;
  return (int *)handle.virtualAddr;
}

void mlir_aie_sync_mem_cpu(ext_mem_model_t &handle) {
  ess_ReadGM(handle.physicalAddr, handle.virtualAddr, handle.size);
}

void mlir_aie_sync_mem_dev(ext_mem_model_t &handle) {
  ess_WriteGM(handle.physicalAddr, handle.virtualAddr, handle.size);
}

u64 mlir_aie_get_device_address(aie_libxaie_ctx_t *_xaie, void *VA) {
  std::cout << "get_device_address: " << _xaie << " VA " << std::hex << VA
            << std::dec << "\n";
  for (auto i : _xaie->allocations) {
    std::cout << "get_device_address: virtual address " << std::hex
              << i.virtualAddr << ", physical address " << i.physicalAddr
              << ", size " << std::dec << i.size << std::endl;

    if (i.virtualAddr == VA)
      return i.physicalAddr;
  }
  printf("ERROR: cannot get device address for allocation!\n");
  assert(false);
  return 0;
}