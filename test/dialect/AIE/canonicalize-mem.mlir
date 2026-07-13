//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// Verify that canonicalize does not remove chained aie.next_bd

// CHECK-LABEL: module @test {
// CHECK:         %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:         %[[VAL_1:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:           %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.next_bd ^bb3
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %[[VAL_3:.*]] = aie.tile(1, 2)
// CHECK:         %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf_0"} : memref<256xi32>
// CHECK:         %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf_1"} : memref<256xi32>
// CHECK:         %[[VAL_6:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf_2"} : memref<256xi32>
// CHECK:         %[[VAL_7:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "buf_3"} : memref<256xi32>
// CHECK:         %[[VAL_8:.*]] = aie.lock(%[[VAL_3]], 0)
// CHECK:         %[[VAL_9:.*]] = aie.lock(%[[VAL_3]], 1)
// CHECK:         %[[VAL_10:.*]] = aie.lock(%[[VAL_3]], 0)
// CHECK:         %[[VAL_11:.*]] = aie.lock(%[[VAL_3]], 0)
// CHECK:         %[[VAL_12:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_15:.*]] = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
// CHECK:         ^bb1:
// CHECK:           %[[VAL_16:.*]] = aie.dma_start(MM2S, 1, ^bb5, ^bb4)
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[VAL_8]], Acquire, %[[VAL_14]])
// CHECK:           aie.dma_bd(%[[VAL_4]] : memref<256xi32>, 0, 256)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, %[[VAL_13]])
// CHECK:           aie.next_bd ^bb3
// CHECK:         ^bb3:
// CHECK:           aie.use_lock(%[[VAL_8]], Acquire, %[[VAL_14]])
// CHECK:           aie.dma_bd(%[[VAL_5]] : memref<256xi32>, 0, 256)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, %[[VAL_13]])
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb4:
// CHECK:           aie.end
// CHECK:         ^bb5:
// CHECK:           aie.use_lock(%[[VAL_8]], Acquire, %[[VAL_14]])
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<256xi32>, 0, 128)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, %[[VAL_13]])
// CHECK:           aie.next_bd ^bb6
// CHECK:         ^bb6:
// CHECK:           aie.use_lock(%[[VAL_8]], Acquire, %[[VAL_14]])
// CHECK:           aie.dma_bd(%[[VAL_6]] : memref<256xi32>, 128, 128)
// CHECK:           aie.use_lock(%[[VAL_8]], Release, %[[VAL_13]])
// CHECK:           aie.next_bd ^bb5
// CHECK:         }
// CHECK:       }

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
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul0)
      aie.dma_bd(%buf_0 : memref<256xi32>, 0, 256)
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul1)
      aie.next_bd ^bd1
    ^bd1:
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul2)
      aie.dma_bd(%buf_1 : memref<256xi32>, 0, 256)
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul3)
      aie.next_bd ^bd2
    ^bd2:
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul4)
      aie.dma_bd(%buf_0 : memref<256xi32>, 0, 256)
      %c0_ul5 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul5)
      aie.next_bd ^bd3
    ^bd3:
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul6)
      aie.dma_bd(%buf_1 : memref<256xi32>, 0, 256)
      %c0_ul7 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul7)
      aie.next_bd ^bd0
    ^end:
      aie.end
    ^bd4:
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul8)
      aie.dma_bd(%buf_2 : memref<256xi32>, 0, 128)
      %c0_ul9 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul9)
      aie.next_bd ^bd5
    ^bd5:
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul10)
      aie.dma_bd(%buf_2 : memref<256xi32>, 128, 128)
      %c0_ul11 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul11)
      aie.next_bd ^bd6
    ^bd6:
      %c1_ul12 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul12)
      aie.dma_bd(%buf_2 : memref<256xi32>, 0, 128)
      %c0_ul13 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul13)
      aie.next_bd ^bd7
    ^bd7:
      %c1_ul14 = arith.constant 1 : i32
      aie.use_lock(%lock_0, Acquire, %c1_ul14)
      aie.dma_bd(%buf_2 : memref<256xi32>, 128, 128)
      %c0_ul15 = arith.constant 0 : i32
      aie.use_lock(%lock_0, Release, %c0_ul15)
      aie.next_bd ^bd4
  }
}
