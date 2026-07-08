//===- memtiledma.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.buffer' op in Column 3 and Row 1 is accessed from an unreachable tile in Column 1 and Row 1

// memtiles can only access neighboring memtiles

aie.device(xcve2802) {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(3, 1)
  %t0 = aie.tile(0, 1)
  %buf = aie.buffer(%t1) : memref<256xi32>
  %buf2 = aie.buffer(%t2) : memref<256xi32>
  %buf0 = aie.buffer(%t0) : memref<256xi32>
  %mem = aie.memtile_dma(%t1) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    aie.dma_start("MM2S", 0, ^bd0, ^dma1)
    ^dma1:
    aie.dma_start("MM2S", 1, ^bd15, ^dma1)
    ^bd0:
      aie.dma_bd(%buf : memref<256xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf0 : memref<256xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.next_bd ^bd2
    ^bd2:
      aie.dma_bd(%buf2 : memref<256xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.next_bd ^bd15
    ^bd15:
      aie.dma_bd(%buf : memref<256xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.next_bd ^bd16
    ^bd16:
      aie.dma_bd(%buf : memref<256xi32> offset = %c0_i32 len = %c256_i32 sizes = [] strides = [])
      aie.end
  }
}
