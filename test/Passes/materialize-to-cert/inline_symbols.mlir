//===- inline_symbols.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-npu-to-cert %s | FileCheck %s

// Test that symbol definitions (like shim_dma_allocation) are properly preserved
// when inlining runtime sequences from other devices

// NOTE: aiex.npu.dma_memcpy_nd is NOT yet converted to CERT format.
// It remains as aiex.npu.dma_memcpy_nd in the output.

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)

    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      aiex.configure @config_with_symbols {
        aiex.run @seq_with_dma(%arg0) : (memref<64xi32>)
      }
    }
  }

  // The referenced device is absorbed into the main device as a cert.section
  // and the original @config_with_symbols device is removed from output
  aie.device(npu2) @config_with_symbols {
    %tile20 = aie.tile(2, 0)
    aie.shim_dma_allocation @buffer_in(%tile20, S2MM, 0)

    aie.runtime_sequence @seq_with_dma(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @buffer_in, id = 0 : i64 } : memref<64xi32>
    }
  }
}

// CHECK: aie.device(npu2) {
// CHECK-NOT: aie.device(npu2) @config_with_symbols

// CHECK: aie.tile(2, 0)
// CHECK: aie.shim_dma_allocation @buffer_in

// Main control flow: a bare cert.job (a page is only formed later by
// cert-legalize-pages).
// CHECK: aiex.cert.job({{[0-9]+}}) {
// CHECK-NEXT: ^bb0(%[[ARG0:.*]]: memref<64xi32>):

// The configure op is converted to cert.load_pdi
// CHECK-NEXT: aiex.cert.load_pdi(1, @config_with_symbols)

// The inlined dma_memcpy_nd operation - NOT YET CONVERTED TO CERT
// TODO: This should eventually be converted to CERT DMA operations
// CHECK-NEXT: aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1])
// CHECK-SAME: {id = 0 : i64, metadata = @buffer_in}

// CHECK: }

// The referenced device's configure sequence becomes a cert.section
// CHECK: aiex.cert.section @config_with_symbols {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job({{[0-9]+}}) {
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
