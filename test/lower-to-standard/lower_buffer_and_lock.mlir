//===- lower_buffer_and_lock.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=1" %s | FileCheck --check-prefix=CHECK11 %s
// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=2" %s | FileCheck --check-prefix=CHECK12 %s

// Test LLVM lowering for lock accesses and memory accesses (LockOp, UseLockOp, and BufferOp)
// Things to make sure:
//   - LockID: depending on which tile (or memory module) a lock is instantiated, create a lock ID
//             that has correct offset from a core's view (based on cardinal direction)
//   - Buffer: depending on which tile (or memory module) a buffer is instantiated, create an LLVM
//             static allocation (for now) for each core that can access to the buffer
module @test_core_llvm1 {
 AIE.device(xcvc1902) {
// CHECK11:  memref.global "public" @a : memref<256xi32>
// CHECK11:  func.func @core_1_1() {
// CHECK11:    %c56 = arith.constant 56 : index
// CHECK11:    %0 = arith.index_cast %c56 : index to i32
// CHECK11:    %c0_i32 = arith.constant 0 : i32
// CHECK11:    call @llvm.aie.lock.acquire.reg(%0, %c0_i32) : (i32, i32) -> ()
// CHECK11:    %c1_i32 = arith.constant 1 : i32
// CHECK11:    %c16 = arith.constant 16 : index
// CHECK11:    %1 = memref.get_global @a : memref<256xi32>
// CHECK11:    memref.assume_alignment %1, 32 : memref<256xi32>
// CHECK11:    memref.store %c1_i32, %1[%c16] : memref<256xi32>
// CHECK11:    %2 = arith.index_cast %c56 : index to i32
// CHECK11:    %c1_i32_0 = arith.constant 1 : i32
// CHECK11:    call @llvm.aie.lock.release.reg(%2, %c1_i32_0) : (i32, i32) -> ()
// CHECK11:    return
// CHECK11:  }

// CHECK12:  memref.global "public" @a : memref<256xi32>
// CHECK12:  func.func @core_1_2() {
// CHECK12:    %c8 = arith.constant 8 : index
// CHECK12:    %0 = arith.index_cast %c8 : index to i32
// CHECK12:    %c1_i32 = arith.constant 1 : i32
// CHECK12:    call @llvm.aie.lock.acquire.reg(%0, %c1_i32) : (i32, i32) -> ()
// CHECK12:    %c16 = arith.constant 16 : index
// CHECK12:    %1 = memref.get_global @a : memref<256xi32>
// CHECK12:    memref.assume_alignment %1, 32 : memref<256xi32>
// CHECK12:    %2 = memref.load %1[%c16] : memref<256xi32>
// CHECK12:    %3 = arith.index_cast %c8 : index to i32
// CHECK12:    %c0_i32 = arith.constant 0 : i32
// CHECK12:    call @llvm.aie.lock.release.reg(%3, %c0_i32) : (i32, i32) -> ()
// CHECK12:    return
// CHECK12:  }
  %tile11 = AIE.tile(1, 1)
  %tile12 = AIE.tile(1, 2)

  %lock11_8 = AIE.lock(%tile11, 8)
  %buf11_0  = AIE.buffer(%tile11) { sym_name = "a" } : memref<256xi32>

  %core11 = AIE.core(%tile11) {
    AIE.useLock(%lock11_8, Acquire, 0)
    %0 = arith.constant 1 : i32
    %i = arith.constant 16 : index
    memref.store %0, %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 1)
    AIE.end
  }

  %core12 = AIE.core(%tile12) {
    AIE.useLock(%lock11_8, Acquire, 1)
    %i = arith.constant 16 : index
    %0 = memref.load %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 0)
    AIE.end
  }
 }
}
