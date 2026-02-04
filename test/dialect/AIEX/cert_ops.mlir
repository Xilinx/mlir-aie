//===- cert_ops.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
module {
  aie.device(npu2) {
    // Define a uC DMA chain
    memref.global "private" constant @dma_data_0 : memref<9xi32> = dense<[0xa8, 0xa7, 0xa6, 0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa0]>
    memref.global "private" constant @dma_data_1 : memref<9xi32> = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8]>
    // CHECK: aiex.cert.uc_dma_chain @dma_chain
    aiex.cert.uc_dma_chain @dma_chain {
      // CHECK: aiex.cert.uc_dma_bd @dma_data_0, 6292480, 128, true
      aiex.cert.uc_dma_bd @dma_data_0, 0x600400, 128, true
      // CHECK: aiex.cert.uc_dma_bd @dma_data_1, 6292736, 256, false
      aiex.cert.uc_dma_bd @dma_data_1, 0x600500, 256, false
    }
    
    
    // Define a cert job
    // CHECK: aiex.cert.job(42)
    aiex.cert.job(42) {
      // Write a 32-bit value to a specific address
      // CHECK: aiex.cert.write32(4096, 42)
      aiex.cert.write32(0x1000, 42)

      // Write a masked 32-bit value to a specific address
      // CHECK: aiex.cert.maskwrite32(8192, 42, 255)
      aiex.cert.maskwrite32(0x2000, 42, 0xFF)

      // CHECK: aiex.cert.apply_offset_57(@dma_data_0, 1, -1)
      aiex.cert.apply_offset_57(@dma_data_0, 1, 0xffff)
      // CHECK: aiex.cert.apply_offset_57(@dma_data_1, 1, 2)
      aiex.cert.apply_offset_57(@dma_data_1, 1, 2)

      // Enqueue a uC DMA transfer
      // CHECK: aiex.cert.uc_dma_write_des_sync(@dma_chain)
      aiex.cert.uc_dma_write_des_sync(@dma_chain)

      // Wait for 3 TCTs on shim column 1, channel 2
      // CHECK: aiex.cert.wait_tcts(1, 2, 3)
      aiex.cert.wait_tcts(1, 2, 3)

      // Local barrier with barrier_id=1, num_participants=2
      // CHECK: aiex.cert.local_barrier(1, 2)
      aiex.cert.local_barrier(1, 2)

      // Remote barrier with barrier_id=1, party_mask=0x3
      // CHECK: aiex.cert.remote_barrier(1, 3)
      aiex.cert.remote_barrier(1, 0x3)

      // No operation
      // CHECK: aiex.cert.nop
      aiex.cert.nop

    }

    // Attach jobs to a group
    // CHECK: aiex.cert.attach_to_group(5)
    aiex.cert.attach_to_group(5) {
      // CHECK: aiex.cert.job(100)
      aiex.cert.job(100) {
        // CHECK: aiex.cert.write32(12288, 123)
        aiex.cert.write32(0x3000, 123)
      }
      // CHECK: aiex.cert.job(101)
      aiex.cert.job(101) {
        // CHECK: aiex.cert.nop
        aiex.cert.nop
      }
    }
  }
}