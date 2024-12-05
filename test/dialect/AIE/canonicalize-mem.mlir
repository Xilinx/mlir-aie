//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// Verify that canonicalize does not remove chained aie.next_bd

// CHECK-LABEL:  module @test {
// CHECK-NEXT:     %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK-NEXT:     %[[VAL_1:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK-NEXT:       %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       aie.next_bd ^bb2
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       aie.next_bd ^bb3
// CHECK-NEXT:     ^bb3:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       aie.end
// CHECK-NEXT:     }

// CHECK:      %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:  %[[BUF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buf_0"} : memref<256xi32> 
// CHECK-DAG:  %[[BUF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buf_1"} : memref<256xi32> 
// CHECK-DAG:  %[[BUF_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buf_2"} : memref<256xi32> 
// CHECK-DAG:  %[[BUF_3:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buf_3"} : memref<256xi32> 
// CHECK-DAG:  %[[LOCK_0:.*]] = aie.lock(%{{.*}}, 0)
// CHECK:   aie.mem(%[[TILE_1_2]]) {
// CHECK-NEXT:    %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb4)
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK-NEXT:    aie.dma_bd(%[[BUF_0]] : memref<256xi32>, 0, 256)
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK-NEXT:    aie.next_bd ^bb3
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK-NEXT:    aie.dma_bd(%[[BUF_1]] : memref<256xi32>, 0, 256)
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK-NEXT:    aie.next_bd ^bb2
// CHECK-NEXT:  ^bb4:  // pred: ^bb1
// CHECK-NEXT:    aie.end
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb1, ^bb6
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK-NEXT:    aie.dma_bd(%[[BUF_2]] : memref<256xi32>, 0, 128)
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK-NEXT:    aie.next_bd ^bb6
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, 1)
// CHECK-NEXT:    aie.dma_bd(%[[BUF_2]] : memref<256xi32>, 128, 128)
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, 0)
// CHECK-NEXT:    aie.next_bd ^bb5

module @test {
  %t1 = aie.tile(1, 1)

  %mem11 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.next_bd ^bd1 // point to the next BD, or termination
    ^bd1:
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }


  %t2 = aie.tile(1, 2)

  %buf_0 = aie.buffer(%t2) { sym_name = "buf_0" } : memref<256xi32>
  %buf_1 = aie.buffer(%t2) { sym_name = "buf_1" } : memref<256xi32>
  %buf_2 = aie.buffer(%t2) { sym_name = "buf_2" } : memref<256xi32>
  %buf_3 = aie.buffer(%t2) { sym_name = "buf_3" } : memref<256xi32>

  %lock_0 = aie.lock(%t2, 0)
  %lock_1 = aie.lock(%t2, 1)
  %lock_2 = aie.lock(%t2, 0)
  %lock_3 = aie.lock(%t2, 0)

  %mem12 = aie.mem(%t2) {
      %start1 = aie.dma_start("MM2S", 0, ^bd0, ^dma0)
    ^dma0:
      %start2 = aie.dma_start("MM2S", 1, ^bd4, ^end)
    ^bd0:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd2
    ^bd2:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
    ^bd4:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_2 : memref<256xi32>, 0, 128)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_2 : memref<256xi32>, 128, 128)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd6
    ^bd6:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_2 : memref<256xi32>, 0, 128)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%lock_0, Acquire, 1)
      aie.dma_bd(%buf_2 : memref<256xi32>, 128, 128)
      aie.use_lock(%lock_0, Release, 0)
      aie.next_bd ^bd4
  }
}
