//===- split_blockwrite_line_aligned.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A split blockwrite must resume on a 128-bit program-memory line.
//
// SplitNpuBlockWriteOpPattern used to split at dataSize/2 elements, which is
// only guaranteed to be a multiple of two words. That is fine for the
// word-addressed config registers, but a blockwrite into a core's program
// memory has to start on a 16-byte line, because program memory is 128 bits
// wide -- and program-memory overlays are exactly the case that produces large
// blockwrites.
//
// 2012 words is 8048 bytes: past the 8000-byte split threshold, and a multiple
// of 16 bytes, so a perfectly legal overlay payload. The old split point was
// word 1006, or byte 4024, which is 8 past a line boundary. The companion case
// in split_blockwrite.mlir happens to be 2000 words, whose half is already a
// multiple of 4, so it never showed this.
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
