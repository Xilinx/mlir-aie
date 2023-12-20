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

#include "target.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// #if defined(__AIESIM__)
// #include "xioutils.h"
// #include <iostream>
// #endif

extern "C" {

/// Depending on the model of a particular device, this API supports several
/// different scenarios for device memory allocation and reference.
/// For instance, on the VCK190 with ARM host programmed through libXAIE,
/// mem_alloc() might allocate data in device DDR and return a cacheable mapping
/// to it.   sync_mem_cpu and sync_mem_dev would flush and invalidate caches.
/// A device address would correspond to a DDR physical address.
/// Alternatively in the AIESIM environment, mem_alloc allocates a duplicate
/// buffer in the host memory and in the simulator memory for each allocation,
/// sync_mem_cpu and sync_mem_dev make an explicit copy between these two
/// buffers and device addresses are modeled in a simulator-specific way. Other
/// combinations are also possible, largely representing different tradeoffs
/// between efficiency of host data access vs. efficiency of accelerator access.

// static variable for tracking current DDR physical addr during AIESIM
static uint16_t nextAlignedAddr;

/// @brief Allocate a buffer in device memory
/// @param bufIdx The index of the buffer to allocate.
/// @param size The number of 32-bit words to allocate
/// @return A host-side pointer that can write into the given buffer.
/// @todo This is at best a quick hack and should be replaced
int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *_xaie, ext_mem_model_t &handle,
                        int size);

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

/// @brief Return a device address corresponding to the given host address.
/// @param host_address A host-side pointer returned from mlir_aie_mem_alloc
u64 mlir_aie_get_device_address(aie_libxaie_ctx_t *_xaie, void *host_address);

} // extern "C"

#endif
