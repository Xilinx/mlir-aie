//===- cert_uc_dma_async.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip the asynchronous uC DMA push and its wait.

// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.uc_dma_write_des(2, @chain)
// CHECK-NEXT: aiex.cert.wait_uc_dma(2)
// CHECK: aiex.cert.uc_dma_write_des(23, @chain)
aie.device(xcve3858) {
  memref.global "private" constant @data : memref<1xi32> = dense<1>
  aiex.cert.uc_dma_chain @chain {
    aiex.cert.uc_dma_bd @data, 0x1000, 1, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des(2, @chain)
      aiex.cert.wait_uc_dma(2)
      aiex.cert.uc_dma_write_des(23, @chain)
    }
  }
}
