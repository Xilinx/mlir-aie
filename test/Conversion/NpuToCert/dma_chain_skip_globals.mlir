//===- dma_chain_skip_globals.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Test that NpuBlockWriteToCertUcDma finds the correct insertion point for a
// newly-created uc_dma_chain when a memref.global precedes an existing
// uc_dma_chain in the device body. The scan must skip over the leading
// global rather than stopping at it, so the new chain is inserted after the
// last existing chain (preserving chain order) instead of before everything.

// CHECK: memref.global "private" constant @unrelated
// CHECK: aiex.cert.uc_dma_chain @chain_0 {
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, false
// CHECK: }
// CHECK: aiex.cert.uc_dma_chain @chain_1 {
// CHECK:   aiex.cert.uc_dma_bd @blockwrite_data1, 5432, 2, false
// CHECK: }

aie.device(npu2) {
  memref.global "private" constant @unrelated : memref<1xi32> = dense<[9]>
  aiex.cert.uc_dma_chain @chain_0 {
    aiex.cert.uc_dma_bd @blockwrite_data0, 4321, 1, false
  }
  memref.global "private" constant @blockwrite_data1 : memref<2xi32> = dense<[2, 3]>
  aie.runtime_sequence @configure() {
    %0 = memref.get_global @blockwrite_data1 : memref<2xi32>
    aiex.npu.blockwrite(%0) {address = 5432 : ui32} : memref<2xi32>
  }
}
