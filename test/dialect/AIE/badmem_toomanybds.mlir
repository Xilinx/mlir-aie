//===- badmem_toomany_bds.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'aie.mem' op has more than 16 blocks

aie.device(xcvc1902) {
  %t1 = aie.tile(1, 1)
  aie.buffer @buf(%t1) : memref<256xi32>
  %mem = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^bd15)
    ^bd0:
      %buf = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd1
    ^bd1:
      %buf_1 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_1 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd2
    ^bd2:
      %buf_2 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_2 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd3
    ^bd3:
      %buf_3 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_3 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd4
    ^bd4:
      %buf_4 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_4 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd5
    ^bd5:
      %buf_5 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_5 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd6
    ^bd6:
      %buf_6 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_6 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd7
    ^bd7:
      %buf_7 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_7 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd8
    ^bd8:
      %buf_8 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_8 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd9
    ^bd9:
      %buf_9 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_9 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd10
    ^bd10:
      %buf_10 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_10 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd11
    ^bd11:
      %buf_11 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_11 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd12
    ^bd12:
      %buf_12 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_12 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd13
    ^bd13:
      %buf_13 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_13 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd14
    ^bd14:
      %buf_14 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_14 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd15
    ^bd15:
      %buf_15 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_15 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^bd16
    ^bd16:
      %buf_16 = aie.get_buffer @buf : memref<256xi32>
      aie.dma_bd(%buf_16 : memref<256xi32> offset = 0 len = 256)
      aie.end
  }
}
