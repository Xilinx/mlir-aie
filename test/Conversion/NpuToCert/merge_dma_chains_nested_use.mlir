//===- merge_dma_chains_nested_use.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MergeConsecutiveCertUcDmaWriteDesSyncOps counts chain uses anywhere in the
// device, including inside a cert.section. @c0 is also enqueued from section
// @s, so it is not merged; @c1 and @c2 are used once each and merge into @c2.
//
// RUN: aie-opt -cert-legalize-pages %s | FileCheck %s

// CHECK:      aiex.cert.uc_dma_chain @c0 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d0, 0, 1, false
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.uc_dma_chain @c2 {
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d1, 4, 1, true
// CHECK-NEXT:   aiex.cert.uc_dma_bd @d2, 8, 1, false
// CHECK-NEXT: }
// CHECK:      aiex.cert.job(1) {
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c0)
// CHECK-NEXT:   aiex.cert.uc_dma_write_des_sync(@c2)
// CHECK-NEXT: }
// CHECK:      aiex.cert.section @s {
// CHECK:          aiex.cert.uc_dma_write_des_sync(@c0)

aie.device(npu2) {
  memref.global "private" constant @d0 : memref<1xi32> = dense<[1]>
  memref.global "private" constant @d1 : memref<1xi32> = dense<[2]>
  memref.global "private" constant @d2 : memref<1xi32> = dense<[3]>
  aiex.cert.uc_dma_chain @c0 {
    aiex.cert.uc_dma_bd @d0, 0, 1, false
  }
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 4, 1, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 8, 1, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c0)
    aiex.cert.uc_dma_write_des_sync(@c1)
    aiex.cert.uc_dma_write_des_sync(@c2)
  }
  aiex.cert.section @s {
    aiex.cert.page {
      aiex.cert.job(2) {
        aiex.cert.uc_dma_write_des_sync(@c0)
      }
    }
  }
}
