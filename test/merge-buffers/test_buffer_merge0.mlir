//===- test_buffer_merge0.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-merge-buffers %s | FileCheck %s
// The idea of this pass is probably not a good one.

//CHECK-LABEL: module @test_buffer_merge0 {
//CHECK: %[[TILE33:.*]] = AIE.tile(3, 3)
//CHECK: %[[TILE34:.*]] = AIE.tile(3, 4)
//CHECK: %[[TILE32:.*]] = AIE.tile(3, 2)
//CHECK: %[[LOCK33:.*]] = AIE.lock(%[[TILE33]], 0)
//CHECK: %[[LOCK34:.*]] = AIE.lock(%[[TILE34]], 0)
//CHECK: %[[LOCK32:.*]] = AIE.lock(%[[TILE32]], 0)
//CHECK: %[[BUF33:.*]] = AIE.buffer(%[[TILE33]]) : memref<256xi32>
//CHECK: %[[BUF34:.*]] = AIE.buffer(%[[TILE34]]) : memref<256xi32>
//CHECK: %[[BUF32:.*]] = AIE.buffer(%[[TILE32]]) : memref<256xi32>
//CHECK: %11 = AIE.core(%2) {
//CHECK:   AIE.useLock(%1, Acquire, 0)
//CHECK:   %c16 = arith.constant 16 : index
//CHECK:   %c1_i32 = arith.constant 1 : i32
//CHECK:   store %c1_i32, %6[%c16] : memref<256xi32>
//CHECK:   AIE.useLock(%1, Release, 1)
//CHECK:   AIE.end
//CHECK: }
//CHECK: %12 = AIE.core(%3) {
//CHECK:   AIE.useLock(%1, Acquire, 1)
//CHECK:   %c16 = arith.constant 16 : index
//CHECK:   %c1_i32 = arith.constant 1 : i32
//CHECK:   %16 = memref.load %6[%c16] : memref<256xi32>
//CHECK:   AIE.useLock(%1, Release, 0)
//CHECK:   AIE.end
//CHECK: }
//CHECK: %13 = AIE.switchbox(%2) {
//CHECK: }
//CHECK: %14 = AIE.switchbox(%0) {
//CHECK: }
//CHECK: %15 = AIE.switchbox(%3) {
//CHECK: }
//CHECK: }

// In this simple test, we would like to merge buf34_0 and buf32_0 because:
//   - they are not used by cores other than core(3, 4) and core(3, 2), respectively (single user)
//   - core(3, 4) and core(3, 2) are distant (not abut)
//   - core(3, 4) uses DMA to copy data from buf34_0 to buf32_0 of core(3, 2)
//   - core(3, 4) and core(3, 2) has a shareable memory module: mem(3, 3)
//   - we want to avoid the overhead of DMA copy, and the routing resource that routes (3, 4) to (3, 2)
// After merging, the shared buf lives in mem(3, 3) that is accessed by core(3, 4), and then core(3, 2).
// Therefore, the functionality of the original netlist is still preserved.
// Merging Procedure:
//   1. Find bufs that have sharing opportunities
//   2. Find common (shareable) tile for the buf users (cores)
//   3. Create a BufferOp of that tile, and create a LockOp of that tile
//   4. Replace the usage of old buffers with the newly created buffer
//   5. Replace the usage of old locks (that guarded the old buffers) with the newly created lock
//   6. Remove the associated DMA operations (or Block Descriptors)
//   7. Remove the associated routing ConnectOps for the DMA operations
module @test_buffer_merge0 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t34 = AIE.tile(3, 4)
  %t32 = AIE.tile(3, 2)

  %l34_0 = AIE.lock(%t34, 0)
  %l32_0 = AIE.lock(%t32, 0)

  %buf34_0 = AIE.buffer(%t34) : memref<256xi32>
  %buf32_0 = AIE.buffer(%t32) : memref<256xi32>

  %m34 = AIE.mem(%t34) {
      %dmaSt = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l34_0, Acquire, 1)
      AIE.dmaBd(<%buf34_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l34_0, Release, 0)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %m32 = AIE.mem(%t32) {
      %dmaSt = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l32_0, Acquire, 0)
      AIE.dmaBd(<%buf32_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l32_0, Release, 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %c34 = AIE.core(%t34) {
    AIE.useLock(%l34_0, Acquire, 0)
    %i = arith.constant 16 : index
    %0 = arith.constant 1 : i32
    store %0, %buf34_0[%i] : memref<256xi32>
    AIE.useLock(%l34_0, Release, 1)
    AIE.end
  }

  %c32 = AIE.core(%t32) {
    AIE.useLock(%l32_0, Acquire, 1)
    %i = arith.constant 16 : index
    %0 = arith.constant 1 : i32
    %1 = memref.load %buf32_0[%i] : memref<256xi32>
    AIE.useLock(%l32_0, Release, 0)
    AIE.end
  }

  %s34 = AIE.switchbox(%t34) {
    AIE.connect<DMA: 0, South: 0>
  }

  %s33 = AIE.switchbox(%t33) {
    AIE.connect<North: 0, South: 0>
  }

  %s32 = AIE.switchbox(%t32) {
    AIE.connect<North: 0, DMA: 0>
  }
 }
}
