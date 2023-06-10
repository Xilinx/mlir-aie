//===- memory_allocator.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_MEMORY_ALLOCATOR_H
#define AIE_MEMORY_ALLOCATOR_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// #if defined(__AIESIM__)
// #include "xioutils.h"
// #include <iostream>
// #endif

extern "C" {

// static variable for tracking current DDR physical addr during AIESIM
static uint16_t nextAlignedAddr;
struct ext_mem_model_t {
  void *virtualAddr;
  uint64_t physicalAddr;
  size_t size;
  int fd; // The file descriptor used during allocation
};

/// @brief Allocate a buffer in device memory
/// @param bufIdx The index of the buffer to allocate.
/// @param size The number of 32-bit words to allocate
/// @return A host-side pointer that can write into the given buffer.
/// @todo This is at best a quick hack and should be replaced
int *mlir_aie_mem_alloc(ext_mem_model_t &handle, int size);

/// @brief Synchronize the buffer from the device to the host CPU.
/// This is expected to be called after the device writes data into
/// device memory, so that the data can be read by the CPU.  In
/// a non-cache coherent system, this implies invalidating the
/// processor cache associated with the buffer.
/// @param bufIdx The buffer index.
void mlir_aie_sync_mem_cpu(ext_mem_model_t &handle);

/// @brief Synchronize the buffer from the host CPU to the device.
/// This is expected to be called after the host writes data into
/// device memory, so that the data can be read by the device.  In
/// a non-cache coherent system, this implies flushing the
/// processor cache associated with the buffer.
/// @param bufIdx The buffer index.
void mlir_aie_sync_mem_dev(ext_mem_model_t &handle);

} // extern "C"

#endif
