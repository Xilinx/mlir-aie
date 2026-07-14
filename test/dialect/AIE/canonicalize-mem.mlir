//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// Verify that canonicalize does not remove chained aie.next_bd

// CHECK-LABEL:  module @test {
// CHECK:          %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:          %[[VAL_1:.*]] = aie.mem(%[[VAL_0]]) {
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
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb4)
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF_0]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, %{{.*}})
// CHECK-NEXT:    aie.next_bd ^bb3
// CHECK-NEXT:  ^bb3:  // pred: ^bb2
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF_1]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, %{{.*}})
// CHECK-NEXT:    aie.next_bd ^bb2
// CHECK-NEXT:  ^bb4:  // pred: ^bb1
// CHECK-NEXT:    aie.end
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb1, ^bb6
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF_2]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, %{{.*}})
// CHECK-NEXT:    aie.next_bd ^bb6
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Acquire, %{{.*}})
// CHECK-NEXT:    aie.dma_bd(%[[BUF_2]] : memref<256xi32> offset = {{.*}} len = {{.*}})
// CHECK-NEXT:    aie.use_lock(%[[LOCK_0]], Release, %{{.*}})
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
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul1)
      aie.dma_bd(%buf_0 : memref<256xi32> offset = 0 len = 256)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul2)
      aie.next_bd ^bd1
    ^bd1:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul3)
      aie.dma_bd(%buf_1 : memref<256xi32> offset = 0 len = 256)
      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul4)
      aie.next_bd ^bd2
    ^bd2:
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul5)
      aie.dma_bd(%buf_0 : memref<256xi32> offset = 0 len = 256)
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul6)
      aie.next_bd ^bd3
    ^bd3:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul7)
      aie.dma_bd(%buf_1 : memref<256xi32> offset = 0 len = 256)
      %c0_ul8 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul8)
      aie.next_bd ^bd0
    ^end:
      aie.end
    ^bd4:
      %c1_ul9 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul9)
      aie.dma_bd(%buf_2 : memref<256xi32> offset = 0 len = 128)
      %c0_ul10 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul10)
      aie.next_bd ^bd5
    ^bd5:
      %c1_ul11 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul11)
      aie.dma_bd(%buf_2 : memref<256xi32> offset = 128 len = 128)
      %c0_ul12 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul12)
      aie.next_bd ^bd6
    ^bd6:
      %c1_ul13 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul13)
      aie.dma_bd(%buf_2 : memref<256xi32> offset = 0 len = 128)
      %c0_ul14 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul14)
      aie.next_bd ^bd7
    ^bd7:
      %c1_ul15 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul15)
      aie.dma_bd(%buf_2 : memref<256xi32> offset = 128 len = 128)
      %c0_ul16 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul16)
      aie.next_bd ^bd4
  }
}
