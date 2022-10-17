//===- test_lock3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: andrab
// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABELs: module @test_lock_shimdma  {
// CHECK:   AIE.token(0) {sym_name = "token0"}
// CHECK:   %0 = AIE.external_buffer 0 : memref<256xi32>
// CHECK:   %1 = AIE.tile(6, 0)
// CHECK:   %2 = AIE.lock(%1, 0)
// CHECK:   %3 = AIE.shimDMA(%1)  {
// CHECK:     %9 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:   ^bb1:  // pred: ^bb0
// CHECK:     AIE.useLock(%2, Acquire, 0)
// CHECK:     AIE.dmaBd(<%0 : memref<256xi32>, 0, 256>, 0)
// CHECK:     AIE.useLock(%2, Release, 1)
// CHECK:     cf.br ^bb2
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %4 = AIE.tile(3, 3)
// CHECK:   %5 = AIE.lock(%4, 0)
// CHECK:   %6 = AIE.buffer(%4) : memref<256xi32>
// CHECK:   %7 = AIE.core(%4)  {
// CHECK:     AIE.useLock(%5, Acquire, 0)
// CHECK:     AIE.useLock(%5, Release, 1)
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %8 = AIE.mem(%4)  {
// CHECK:     %9 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:   ^bb1:  // pred: ^bb0
// CHECK:     AIE.useLock(%5, Acquire, 1)
// CHECK:     AIE.dmaBd(<%6 : memref<256xi32>, 0, 256>, 0)
// CHECK:     AIE.useLock(%5, Release, 0)
// CHECK:     cf.br ^bb2
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   AIE.flow(%4, DMA : 0, %1, DMA : 0)
// CHECK: }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [ShimDMA] (non-neighboring tiles)
// single producer, single consumer
module @test_lock_shimdma {
  AIE.token(0) {sym_name = "token0"}
  %buf_ext = AIE.external_buffer 0 : memref<256xi32>

  %t60 = AIE.tile(6, 0)
  %c60 = AIE.core(%t60) {
    // TODO: This represents the token uses on the host CPU. A representation of
    // the host CPU in MLIR might be a better place for holding this.
    AIE.useToken @token0(Acquire, 2)
    AIE.useToken @token0(Release, 3)
    AIE.end
  }
  %m60 = AIE.shimDMA(%t60) {
      %dmaSt = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf_ext : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
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
      %dmaSt = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      cf.br ^end
    ^end:
      AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t60, DMA : 0)
}
