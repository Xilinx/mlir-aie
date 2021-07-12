//===- lower_buffer_and_lock.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-llvm-lowering="tilecol=1 tilerow=1" %s | FileCheck --check-prefix=CHECK11 %s
// RUN: aie-opt --aie-llvm-lowering="tilecol=1 tilerow=2" %s | FileCheck --check-prefix=CHECK12 %s

// Test LLVM lowering for lock accesses and memory accesses (LockOp, UseLockOp, and BufferOp)
// Things to make sure:
//   - LockID: depending on which tile (or memory module) a lock is instantiated, create a lock ID
//             that has correct offset from a core's view (based on cardinal direction)
//   - Buffer: depending on which tile (or memory module) a buffer is instantiated, create an LLVM
//             static allocation (for now) for each core that can access to the buffer
module @test_core_llvm1 {
// CHECK11:         llvm.mlir.global external @a() : !llvm.array<256 x i32>
// CHECK11:         llvm.func @core11() {
// CHECK11:           %[[VAL_0:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK11:           %[[VAL_1:.*]] = llvm.mlir.addressof @a : !llvm.ptr<array<256 x i32>>
// CHECK11:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK11:           %[[VAL_3:.*]] = llvm.mlir.constant(56 : i32) : i32
// CHECK11:           llvm.call @llvm.aie.lock.acquire.reg(%[[VAL_3]], %[[VAL_2]]) : (i32, i32) -> ()
// CHECK11:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK11:           %[[VAL_5:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK11:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK11:           %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_5]]] : (!llvm.ptr<array<256 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK11:           llvm.store %[[VAL_4]], %[[VAL_7]] : !llvm.ptr<i32>
// CHECK11:           %[[VAL_8:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK11:           %[[VAL_9:.*]] = llvm.mlir.constant(56 : i32) : i32
// CHECK11:           llvm.call @llvm.aie.lock.release.reg(%[[VAL_9]], %[[VAL_8]]) : (i32, i32) -> ()
// CHECK11:           llvm.return
// CHECK11:         }

// CHECK12:         llvm.mlir.global external @a() : !llvm.array<256 x i32>
// CHECK12:         llvm.func @core12() {
// CHECK12:           %[[VAL_10:.*]] = llvm.mlir.constant(256 : index) : i64
// CHECK12:           %[[VAL_11:.*]] = llvm.mlir.addressof @a : !llvm.ptr<array<256 x i32>>
// CHECK12:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK12:           %[[VAL_13:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK12:           llvm.call @llvm.aie.lock.acquire.reg(%[[VAL_13]], %[[VAL_12]]) : (i32, i32) -> ()
// CHECK12:           %[[VAL_14:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK12:           %[[VAL_15:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK12:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_11]]{{\[}}%[[VAL_15]], %[[VAL_14]]] : (!llvm.ptr<array<256 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK12:           %[[VAL_17:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr<i32>
// CHECK12:           %[[VAL_18:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK12:           %[[VAL_19:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK12:           llvm.call @llvm.aie.lock.release.reg(%[[VAL_19]], %[[VAL_18]]) : (i32, i32) -> ()
// CHECK12:           llvm.return
// CHECK12:         }
  %tile11 = AIE.tile(1, 1)
  %tile12 = AIE.tile(1, 2)

  %lock11_8 = AIE.lock(%tile11, 8)
  %buf11_0  = AIE.buffer(%tile11) { sym_name = "a" } : memref<256xi32>

  %core11 = AIE.core(%tile11) {
    AIE.useLock(%lock11_8, Acquire, 0, 0)
    %0 = constant 1 : i32
    %i = constant 16 : index
    memref.store %0, %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 1, 0)
    AIE.end
  }

  %core12 = AIE.core(%tile12) {
    AIE.useLock(%lock11_8, Acquire, 1, 0)
    %i = constant 16 : index
    %0 = memref.load %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 0, 0)
    AIE.end
  }
}
