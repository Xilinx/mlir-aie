//===- test_lock6.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock6 {
// CHECK:  %0 = AIE.tile(5, 5)
// CHECK:  %1 = AIE.lock(%0, 3)
// CHECK:  AIE.useLock(%1, Acquire, 0)
// CHECK:  %2 = AIE.lock(%0, 2)
// CHECK:  AIE.useLock(%2, Acquire, 0)
// CHECK:  %3 = AIE.lock(%0, 1)
// CHECK:  AIE.useLock(%3, Release, 1)
// CHECK:  %4 = AIE.lock(%0, 0)
// CHECK:  AIE.useLock(%4, Release, 1)
// CHECK:  %5 = AIE.tile(4, 4)
// CHECK:  %6 = AIE.lock(%5, 1)
// CHECK:  AIE.useLock(%6, Release, 0)
// CHECK:  %7 = AIE.lock(%5, 0)
// CHECK:  AIE.useLock(%7, Acquire, 0)
// CHECK:  %8 = AIE.tile(3, 3)
// CHECK:  %9 = AIE.lock(%8, 1)
// CHECK:  AIE.useLock(%9, Release, 0)
// CHECK:  %10 = AIE.lock(%8, 0)
// CHECK:  AIE.useLock(%10, Acquire, 0)
// CHECK:  %15 = AIE.mem(%8) {
// CHECK:    AIE.useLock(%10, Acquire, 1)
// CHECK:    AIE.dmaBd(<%11 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%10, Release, 0)
// CHECK:  }
// CHECK:  %16 = AIE.mem(%5) {
// CHECK:    AIE.useLock(%7, Acquire, 1)
// CHECK:    AIE.dmaBd(<%12 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%7, Release, 0)
// CHECK:  }
// CHECK:  %17 = AIE.mem(%0) {
// CHECK:    AIE.useLock(%4, Acquire, 1)
// CHECK:    AIE.dmaBd(<%13 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%4, Release, 0)
// CHECK:    AIE.useLock(%3, Acquire, 1)
// CHECK:    AIE.dmaBd(<%14 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%3, Release, 0)
// CHECK:  }
// CHECK:  %18 = AIE.core(%8) {
// CHECK:    AIE.useLock(%9, Acquire, 0)
// CHECK:    AIE.useLock(%10, Release, 1)
// CHECK:  }
// CHECK:  %19 = AIE.core(%5) {
// CHECK:    AIE.useLock(%6, Acquire, 0)
// CHECK:    AIE.useLock(%7, Release, 1)
// CHECK:  }
// CHECK:  %20 = AIE.core(%0) {
// CHECK:    AIE.useLock(%4, Acquire, 0)
// CHECK:    AIE.useLock(%3, Acquire, 0)
// CHECK:    AIE.useLock(%1, Release, 0)
// CHECK:    AIE.useLock(%2, Release, 0)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---\
//                [Core-Mem] (non-neighboring tiles)
// [Core-Mem] ---/
// multiple producers, single consumer
module @test_lock6 {
  %t55 = AIE.tile(5, 5)
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)

  %buf33   = AIE.buffer(%t33) : memref<256xi32>
  %buf44   = AIE.buffer(%t44) : memref<256xi32>
  %buf55_0 = AIE.buffer(%t55) : memref<256xi32>
  %buf55_1 = AIE.buffer(%t55) : memref<256xi32>

  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}
  AIE.token(0) {sym_name = "token2"}
  AIE.token(0) {sym_name = "token3"}

  %m33 = AIE.mem(%t33) {
      %dmaSt = AIE.dmaStart(MM2S0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      cf.br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token1(Acquire, 1)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token1(Release, 2)
      cf.br ^end
    ^end:
      AIE.end
  }

  %m55 = AIE.mem(%t55) {
      %dmaSt0 = AIE.dmaStart(S2MM0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = AIE.dmaStart(S2MM1, ^bd1, ^end)
    ^bd0:
      AIE.useToken @token2(Acquire, 0)
      AIE.dmaBd(<%buf55_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token2(Release, 1)
      cf.br ^end
    ^bd1:
      AIE.useToken @token3(Acquire, 0)
      AIE.dmaBd(<%buf55_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token3(Release, 1)
      cf.br ^end
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIE.useToken @token1(Acquire, 0)
    AIE.useToken @token1(Release, 1)
    AIE.end
  }

  %c55 = AIE.core(%t55) {
    AIE.useToken @token2(Acquire, 1)
    AIE.useToken @token3(Acquire, 1)
    AIE.useToken @token3(Release, 2)
    AIE.useToken @token2(Release, 2)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t55, DMA : 0)
  AIE.flow(%t44, DMA : 0, %t55, DMA : 1)
}
