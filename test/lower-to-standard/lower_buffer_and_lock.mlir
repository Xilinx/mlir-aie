//===- lower_buffer_and_lock.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=1 tilerow=1" %s | FileCheck --check-prefix=CHECK11 %s
// RUN: aie-opt --aie-standard-lowering="tilecol=1 tilerow=2" %s | FileCheck --check-prefix=CHECK12 %s

// Test LLVM lowering for lock accesses and memory accesses (LockOp, UseLockOp, and BufferOp)
// Things to make sure:
//   - LockID: depending on which tile (or memory module) a lock is instantiated, create a lock ID
//             that has correct offset from a core's view (based on cardinal direction)
//   - Buffer: depending on which tile (or memory module) a buffer is instantiated, create an LLVM
//             static allocation (for now) for each core that can access to the buffer
module @test_core_llvm1 {
// CHECK11:    call @llvm.aie.lock.acquire.reg(%c56_i32, %c0_i32) : (i32, i32) -> ()
// CHECK11:    call @llvm.aie.lock.release.reg(%c56_i32_1, %c1_i32_0) : (i32, i32) -> ()
// CHECK12:    call @llvm.aie.lock.acquire.reg(%c8_i32, %c1_i32) : (i32, i32) -> ()
// CHECK12:    call @llvm.aie.lock.release.reg(%c8_i32_0, %c0_i32) : (i32, i32) -> ()
  %tile11 = AIE.tile(1, 1)
  %tile12 = AIE.tile(1, 2)

  %lock11_8 = AIE.lock(%tile11, 8)
  %buf11_0  = AIE.buffer(%tile11) { sym_name = "a" } : memref<256xi32>

  %core11 = AIE.core(%tile11) {
    AIE.useLock(%lock11_8, Acquire, 0, 0)
    %0 = arith.constant 1 : i32
    %i = arith.constant 16 : index
    memref.store %0, %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 1)
    AIE.end
  }

  %core12 = AIE.core(%tile12) {
    AIE.useLock(%lock11_8, Acquire, 1, 0)
    %i = arith.constant 16 : index
    %0 = memref.load %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 0)
    AIE.end
  }
}
