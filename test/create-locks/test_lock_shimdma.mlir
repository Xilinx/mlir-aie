//===- test_lock_shimdma.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABELs: module @test_lock_shimdma  {
// CHECK:  %0 = AIE.external_buffer 0 : memref<256xi32>
// CHECK:  %1 = AIE.tile(6, 0)
// CHECK:  %2 = AIE.lock(%1, 0)
// CHECK:  %3 = AIE.shimDMA(%1) {
// CHECK:    AIE.useLock(%2, Acquire, 1)
// CHECK:    AIE.dmaBd(<%0 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%2, Release, 0)
// CHECK:  }
// CHECK:  %4 = AIE.tile(3, 3)
// CHECK:  %5 = AIE.lock(%4, 0)
// CHECK:  %6 = AIE.buffer(%4) : memref<256xi32>
// CHECK:  %7 = AIE.core(%4) {
// CHECK:    AIE.useLock(%5, Acquire, 0)
// CHECK:    AIE.useLock(%5, Release, 1)
// CHECK:  }
// CHECK:  %8 = AIE.mem(%4) {
// CHECK:    AIE.useLock(%5, Acquire, 1)
// CHECK:    AIE.dmaBd(<%6 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%5, Release, 0)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [ShimDMA] (non-neighboring tiles)
// single producer, single consumer
module @test_lock_shimdma {
  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}
  %buf_ext = AIE.external_buffer 0 : memref<256xi32>

  %t60 = AIE.tile(6, 0)
  %m60 = AIE.shimDMA(%t60) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token1(Acquire, 0)
      AIE.dmaBd(<%buf_ext : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token1(Release, 1)
      cf.br ^end
    ^end:
      AIE.end
  }

  %t33 = AIE.tile(3, 3)
  %buf33 = AIE.buffer(%t33) : memref<256xi32>
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

  AIE.flow(%t33, DMA : 0, %t60, DMA : 0)
}
