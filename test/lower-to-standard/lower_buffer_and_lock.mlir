//===- lower_buffer_and_lock.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
 aie.device(xcvc1902) {
// CHECK11:  memref.global "public" @a : memref<256xi32>
// CHECK11:  func.func @core_1_1() {
// CHECK11:    %c0_i32 = arith.constant 0 : i32
// CHECK11:    %[[ACQUIRE11:.*]] = arith.constant 56 : i32
// CHECK11:    call @llvm.aie.lock.acquire.reg(%[[ACQUIRE11]], %c0_i32) : (i32, i32) -> ()
// CHECK11:    %c1_i32 = arith.constant 1 : i32
// CHECK11:    %c16 = arith.constant 16 : index
// CHECK11:    %[[BUFFER11:.*]] = memref.get_global @a : memref<256xi32>
// CHECK11:    memref.assume_alignment %[[BUFFER11]], 32 : memref<256xi32>
// CHECK11:    memref.store %c1_i32, %[[BUFFER11]][%c16] : memref<256xi32>
// CHECK11:    %c1_i32_0 = arith.constant 1 : i32
// CHECK11:    %[[RELEASE11:.*]] = arith.constant 56 : i32
// CHECK11:    call @llvm.aie.lock.release.reg(%[[RELEASE11]], %c1_i32_0) : (i32, i32) -> ()
// CHECK11:    return
// CHECK11:  }

// CHECK12:  memref.global "public" @a : memref<256xi32>
// CHECK12:  func.func @core_1_2() {
// CHECK12:    %c1_i32 = arith.constant 1 : i32
// CHECK12:    %[[ACQUIRE12:.*]] = arith.constant 8 : i32
// CHECK12:    call @llvm.aie.lock.acquire.reg(%[[ACQUIRE12]], %c1_i32) : (i32, i32) -> ()
// CHECK12:    %c16 = arith.constant 16 : index
// CHECK12:    %[[BUFFER12:.*]] = memref.get_global @a : memref<256xi32>
// CHECK12:    memref.assume_alignment %[[BUFFER12]], 32 : memref<256xi32>
// CHECK12:    %{{.*}} = memref.load %[[BUFFER12]][%c16] : memref<256xi32>
// CHECK12:    %c0_i32 = arith.constant 0 : i32
// CHECK12:    %[[RELEASE12:.*]] = arith.constant 8 : i32
// CHECK12:    call @llvm.aie.lock.release.reg(%[[RELEASE12]], %c0_i32) : (i32, i32) -> ()
// CHECK12:    return
// CHECK12:  }
  %tile11 = aie.tile(1, 1)
  %tile12 = aie.tile(1, 2)

  aie.lock @lock11_8(%tile11, 8)
  aie.buffer @a(%tile11) : memref<256xi32>

  %core11 = aie.core(%tile11) {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(@lock11_8, Acquire, %c0_ul0)
    %0 = arith.constant 1 : i32
    %i = arith.constant 16 : index
    %buf11_0 = aie.get_buffer @a : memref<256xi32>
    memref.store %0, %buf11_0[%i] : memref<256xi32>
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@lock11_8, Release, %c1_ul1)
    aie.end
  }

  %core12 = aie.core(%tile12) {
    %c1_ul2 = arith.constant 1 : i32
    aie.use_lock(@lock11_8, Acquire, %c1_ul2)
    %i = arith.constant 16 : index
    %buf11_0_1 = aie.get_buffer @a : memref<256xi32>
    %0 = memref.load %buf11_0_1[%i] : memref<256xi32>
    %c0_ul3 = arith.constant 0 : i32
    aie.use_lock(@lock11_8, Release, %c0_ul3)
    aie.end
  }
 }
}
