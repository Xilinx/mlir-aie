//===- cert_to_asm.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-cert-to-asm %s | FileCheck %s

// CHECK: START_JOB 42
// CHECK:   WRITE_32               0x00001000, 0x0000002a
// CHECK:   uC_DMA_WRITE_DES_SYNC  @dma_chain
// CHECK:   WAIT_TCTS              1, 2, 3
// CHECK: END_JOB

// CHECK: EOF

// CHECK:   .align 16
// CHECK: dma_chain:
// CHECK:   UC_DMA_BD       0, 0x00600400, @dma_data_0, 128, 0, 0
// CHECK:   UC_DMA_BD       0, 0x00600500, @dma_data_1, 256, 0, 1

// CHECK: dma_data_0:
// CHECK:   .long           0x00000007
// CHECK:   .long           0x00000006
// CHECK:   .long           0x00000005
// CHECK:   .long           0x00000004
// CHECK:   .long           0x00000003
// CHECK:   .long           0x00000002
// CHECK: dma_data_1:
// CHECK:   .long           0x0000001f
// CHECK:   .long           0x0000002f
// CHECK:   .long           0x0000003f
// CHECK:   .long           0x0000004f
module {
  aie.device(npu2) {
    // Define a uC DMA chain
    memref.global "private" constant @dma_data_0 : memref<6xi32> = dense<[0x7, 0x6, 0x5, 0x4, 0x3, 0x2]>
    memref.global "private" constant @dma_data_1 : memref<4xi32> = dense<[0x1f, 0x2f, 0x3f, 0x4f]>

    aiex.cert.uc_dma_chain @dma_chain {
      aiex.cert.uc_dma_bd @dma_data_0, 0x600400, 128, false
      aiex.cert.uc_dma_bd @dma_data_1, 0x600500, 256, true
    }

    // Define a cert job
    aiex.cert.job(42) {
      // Write a 32-bit value to a specific address
      aiex.cert.write32(0x1000, 42)

      // Enqueue a uC DMA transfer
      aiex.cert.uc_dma_write_des_sync(@dma_chain)

      // Wait for 3 TCTs on shim column 1, channel 2
      aiex.cert.wait_tcts(1, 2, 3)
    }
  }
}