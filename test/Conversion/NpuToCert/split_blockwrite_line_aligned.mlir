//===- split_blockwrite_line_aligned.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A split blockwrite must resume on a 128-bit program-memory line, not at a
// bare dataSize/2. 2012 words (8048 bytes) is a legal overlay -- past the
// split threshold and a multiple of 16 bytes -- but its old split point (word
// 1006) landed 8 bytes past a line boundary. split_blockwrite.mlir's 2000-word
// case never showed this, since its half is already line-aligned.
//
// RUN: aie-opt --aie-npu-to-cert %s | FileCheck %s

// Rounded down to word 1004 (byte 4016), so the second write starts at
// 0x22000 + 4016 = 0x22fb0, on a line. The halves are allowed to be uneven.
// CHECK-DAG: memref.global "private" constant @data_split_0 : memref<1004xi32>
// CHECK-DAG: memref.global "private" constant @data_split_1 : memref<1008xi32>
// CHECK-DAG: aiex.cert.uc_dma_bd @data_split_0, 139264, 1004, false
// CHECK-DAG: aiex.cert.uc_dma_bd @data_split_1, 143280, 1008, false

module {
  aie.device(npu2) {
    // 139264 is 0x22000: the program-memory host offset 0x20000 plus a slot at
    // 0x2000.
    memref.global "private" constant @data : memref<2012xi32> = dense<1>
    aie.runtime_sequence @configure() {
      %g = memref.get_global @data : memref<2012xi32>
      aiex.npu.blockwrite(%g) {address = 139264 : ui32} : memref<2012xi32>
      aie.end
    }
  }
}
