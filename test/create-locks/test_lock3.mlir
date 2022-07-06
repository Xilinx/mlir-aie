//===- test_lock3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock3 {
// CHECK:  %0 = AIE.tile(4, 4)
// CHECK:  %1 = AIE.lock(%0, 0)
// CHECK:  %2 = AIE.tile(3, 3)
// CHECK:  %3 = AIE.lock(%2, 0)
// CHECK:  %6 = AIE.core(%2) {
// CHECK:    AIE.useLock(%3, Acquire, 0)
// CHECK:    AIE.useLock(%3, Release, 1)
// CHECK:  }
// CHECK:  %7 = AIE.mem(%2) {
// CHECK:    AIE.useLock(%3, Acquire, 1)
// CHECK:    AIE.dmaBd(<%4 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%3, Release, 0)
// CHECK:  }
// CHECK:  %8 = AIE.mem(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 1)
// CHECK:    AIE.dmaBd(<%4 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%1, Release, 0)
// CHECK:  }
// CHECK:  %9 = AIE.core(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 0)
// CHECK:    AIE.useLock(%1, Release, 1)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [Core-Mem] (non-neighboring tiles)
// single producer, single consumer
module @test_lock3 {
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)
  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf44 = AIE.buffer(%t44) : memref<256xi32>

  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.end
  }

  %m33 = AIE.mem(%t33) {
      %dmaSt = AIE.dmaStart(MM2S0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 0)
      cf.br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token1(Acquire, 0)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token1(Release, 1)
      cf.br ^end
    ^end:
      AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIE.useToken @token1(Acquire, 1)
    AIE.useToken @token1(Release, 0)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t44, DMA : 0)
}
