//===- test_lock5.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(5, 5)
// CHECK:           %[[VAL_1:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_4]], 1)
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_4]], 0)
// CHECK:           %[[VAL_7:.*]] = AIE.buffer(%[[VAL_4]]) : memref<256xi32>
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_2]]) : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_7]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 0)
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             AIE.useLock(%[[VAL_3]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_1]], Acquire, 0)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useLock(%[[VAL_1]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = AIE.core(%[[VAL_4]]) {
// CHECK:             AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:             AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:             AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:             AIE.useLock(%[[VAL_3]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_3]], Release, 0)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.useLock(%[[VAL_1]], Acquire, 1)
// CHECK:             AIE.useLock(%[[VAL_1]], Release, 0)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.flow(%[[VAL_4]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_4]], DMA : 1, %[[VAL_0]], DMA : 0)
// CHECK:         }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [Core-Mem] (non-neighboring tiles)
//     |---------> [Core-Mem]
// single producer, multipler consumers
module @test_lock5 {
 AIE.device(xcvc1902) {
  %t55 = AIE.tile(5, 5)
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)

  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf44 = AIE.buffer(%t44) : memref<256xi32>
  %buf55 = AIE.buffer(%t55) : memref<256xi32>

  AIEX.token(0) {sym_name = "token0"}
  AIEX.token(0) {sym_name = "token1"}

  %m33 = AIE.mem(%t33) {
      %dmaSt0 = AIE.dmaStart(MM2S, 0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = AIE.dmaStart("MM2S", 1, ^bd1, ^end)
    ^bd0:
      AIEX.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
      AIE.nextBd ^end
    ^bd1:
      AIEX.useToken @token1(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token1(Release, 2)
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

  %m55 = AIE.mem(%t55) {
      %dmaSt = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIEX.useToken @token1(Acquire, 1)
      AIE.dmaBd(<%buf55 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token1(Release, 2)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIEX.useToken @token1(Acquire, 0)
    AIEX.useToken @token0(Acquire, 0)
    AIEX.useToken @token0(Release, 1)
    AIEX.useToken @token1(Release, 1)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIEX.useToken @token0(Acquire, 2)
    AIEX.useToken @token0(Release, 3)
    AIE.end
  }

  %c55 = AIE.core(%t55) {
    AIEX.useToken @token1(Acquire, 2)
    AIEX.useToken @token1(Release, 3)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t44, DMA : 0)
  AIE.flow(%t33, DMA : 1, %t55, DMA : 0)
 }
}
