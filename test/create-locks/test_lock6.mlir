//===- test_lock6.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock6 {
// CHECK:  %0 = AIE.tile(5, 5)
// CHECK:  %1 = AIE.lock(%0, 1)
// CHECK:  %2 = AIE.lock(%0, 0)
// CHECK:  %3 = AIE.tile(4, 4)
// CHECK:  %4 = AIE.lock(%3, 0)
// CHECK:  %5 = AIE.tile(3, 3)
// CHECK:  %6 = AIE.lock(%5, 0)
// CHECK:  %7 = AIE.buffer(%5) : memref<256xi32>
// CHECK:  %8 = AIE.buffer(%3) : memref<256xi32>
// CHECK:  %9 = AIE.buffer(%0) : memref<256xi32>
// CHECK:  %10 = AIE.buffer(%0) : memref<256xi32>
// CHECK:  AIEX.token(0) {sym_name = "token0"}
// CHECK:  AIEX.token(0) {sym_name = "token1"}
// CHECK:  %11 = AIE.mem(%5) {
// CHECK:    AIE.useLock(%6, Acquire, 1)
// CHECK:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%6, Release, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %12 = AIE.mem(%3) {
// CHECK:    AIE.useLock(%4, Acquire, 1)
// CHECK:    AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%4, Release, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %13 = AIE.mem(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 0)
// CHECK:    AIE.dmaBd(<%9 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%1, Release, 1)
// CHECK:    AIE.useLock(%2, Acquire, 0)
// CHECK:    AIE.dmaBd(<%10 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%2, Release, 1)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %14 = AIE.core(%5) {
// CHECK:    AIE.useLock(%6, Acquire, 0)
// CHECK:    AIE.useLock(%6, Release, 1)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %15 = AIE.core(%3) {
// CHECK:    AIE.useLock(%4, Acquire, 0)
// CHECK:    AIE.useLock(%4, Release, 1)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %16 = AIE.core(%0) {
// CHECK:    AIE.useLock(%2, Acquire, 1)
// CHECK:    AIE.useLock(%1, Acquire, 1)
// CHECK:    AIE.useLock(%1, Release, 0)
// CHECK:    AIE.useLock(%2, Release, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  AIE.flow(%5, DMA : 0, %0, DMA : 0)
// CHECK:  AIE.flow(%3, DMA : 0, %0, DMA : 1)
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---\
//                [Core-Mem] (non-neighboring tiles)
// [Core-Mem] ---/
// multiple producers, single consumer
module @test_lock6 {
 AIE.device(xcvc1902) {
  %t55 = AIE.tile(5, 5)
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)

  %buf33   = AIE.buffer(%t33) : memref<256xi32>
  %buf44   = AIE.buffer(%t44) : memref<256xi32>
  %buf55_0 = AIE.buffer(%t55) : memref<256xi32>
  %buf55_1 = AIE.buffer(%t55) : memref<256xi32>

  AIEX.token(0) {sym_name = "token0"}
  AIEX.token(0) {sym_name = "token1"}

  %m33 = AIE.mem(%t33) {
      %dmaSt = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIEX.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIEX.useToken @token1(Acquire, 1)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token1(Release, 2)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %m55 = AIE.mem(%t55) {
      %dmaSt0 = AIE.dmaStart(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = AIE.dmaStart("S2MM", 1, ^bd1, ^end)
    ^bd0:
      AIEX.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf55_0 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
      AIE.nextBd ^end
    ^bd1:
      AIEX.useToken @token1(Acquire, 1)
      AIE.dmaBd(<%buf55_1 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token1(Release, 2)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIEX.useToken @token0(Acquire, 0)
    AIEX.useToken @token0(Release, 1)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIEX.useToken @token1(Acquire, 0)
    AIEX.useToken @token1(Release, 1)
    AIE.end
  }

  %c55 = AIE.core(%t55) {
    AIEX.useToken @token1(Acquire, 2)
    AIEX.useToken @token0(Acquire, 2)
    AIEX.useToken @token0(Release, 3)
    AIEX.useToken @token1(Release, 3)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t55, DMA : 0)
  AIE.flow(%t44, DMA : 0, %t55, DMA : 1)
 }
}
