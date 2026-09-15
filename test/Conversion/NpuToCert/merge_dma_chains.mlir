//===- merge_dma_chains.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MergeConsecutiveCertUcDmaWriteDesSyncOps fuses adjacent
// cert.uc_dma_write_des_sync ops (and their backing chains) into a single
// enqueue, preserving BD order and clearing the "last BD" flag on every BD but
// the final one.
//
// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK:      aiex.cert.uc_dma_chain @[[SYM:[a-zA-Z0-9_]+]] {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, true
// CHECK-NEXT:   aiex.cert.uc_dma_bd @blockwrite_data1, 5432, 2, true
// CHECK-NEXT:   aiex.cert.uc_dma_bd @blockwrite_data2, 6543, 3, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@[[SYM]])
// CHECK-NEXT: }

aie.device(npu2) {
  memref.global "private" constant @blockwrite_data0 : memref<1xi32> = dense<[1]>
  memref.global "private" constant @blockwrite_data1 : memref<2xi32> = dense<[2, 3]>
  memref.global "private" constant @blockwrite_data2 : memref<3xi32> = dense<[4, 5, 6]>
  aiex.cert.uc_dma_chain @chain_0 {
    aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, false
  }
  aiex.cert.uc_dma_chain @chain_1 {
    aiex.cert.uc_dma_bd @blockwrite_data1, 5432, 2, false
  }
  aiex.cert.uc_dma_chain @chain_2 {
    aiex.cert.uc_dma_bd @blockwrite_data2, 6543, 3, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@chain_0)
    aiex.cert.uc_dma_write_des_sync(@chain_1)
    aiex.cert.uc_dma_write_des_sync(@chain_2)
  }
}
