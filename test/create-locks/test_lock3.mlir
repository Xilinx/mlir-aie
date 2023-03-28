//===- test_lock3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock3 {
// CHECK:       %0 = AIE.tile(4, 4)
// CHECK-NEXT:  %1 = AIE.lock(%0, 0)
// CHECK-NEXT:  %2 = AIE.tile(3, 3)
// CHECK-NEXT:  %3 = AIE.lock(%2, 0)
// CHECK-NEXT:  %4 = AIE.buffer(%2) : memref<256xi32>
// CHECK-NEXT:  %5 = AIE.buffer(%0) : memref<256xi32>
// CHECK-NEXT:  AIEX.token(0) {sym_name = "token0"}
// CHECK-NEXT:  %6 = AIE.mem(%2) {
// CHECK-NEXT:    %10 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    AIE.useLock(%3, Acquire, 1)
// CHECK-NEXT:    AIE.dmaBd(<%4 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%3, Release, 0)
// CHECK-NEXT:    AIE.nextBd ^bb2
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %7 = AIE.mem(%0) {
// CHECK-NEXT:    %10 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    AIE.useLock(%1, Acquire, 0)
// CHECK-NEXT:    AIE.dmaBd(<%5 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%1, Release, 1)
// CHECK-NEXT:    AIE.nextBd ^bb2
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %8 = AIE.core(%2) {
// CHECK-NEXT:    AIE.useLock(%3, Acquire, 0)
// CHECK-NEXT:    AIE.useLock(%3, Release, 1)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %9 = AIE.core(%0) {
// CHECK-NEXT:    AIE.useLock(%1, Acquire, 1)
// CHECK-NEXT:    AIE.useLock(%1, Release, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  AIE.flow(%2, DMA : 0, %0, DMA : 0)
// CHECK-NEXT:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [Core-Mem] (non-neighboring tiles)
// single producer, single consumer
module @test_lock3 {
 AIE.device(xcvc1902) {
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)
  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf44 = AIE.buffer(%t44) : memref<256xi32>

  AIEX.token(0) {sym_name = "token0"}

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
      AIEX.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
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
    AIEX.useToken @token0(Acquire, 2)
    AIEX.useToken @token0(Release, 3)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t44, DMA : 0)
 }
}
