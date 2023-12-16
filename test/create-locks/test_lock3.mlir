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

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(4, 4)
// CHECK:           %[[VAL_1:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_2]]) : memref<256xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_0]]) : memref<256xi32>
// CHECK:           AIEX.token(0) {sym_name = "token0"}
// CHECK:           %[[VAL_6:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_7:.*]] = AIE.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_3]], Acquire, 1)
// CHECK:             AIE.dma_bd(<%[[VAL_4]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_3]], Release, 0)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = AIE.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             AIE.use_lock(%[[VAL_1]], Acquire, 0)
// CHECK:             AIE.dma_bd(<%[[VAL_5]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.use_lock(%[[VAL_1]], Release, 1)
// CHECK:             AIE.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:             AIE.use_lock(%[[VAL_3]], Acquire, 0)
// CHECK:             AIE.use_lock(%[[VAL_3]], Release, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             AIE.use_lock(%[[VAL_1]], Acquire, 1)
// CHECK:             AIE.use_lock(%[[VAL_1]], Release, 0)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:         }

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
      %dmaSt = AIE.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIEX.useToken @token0(Acquire, 1)
      AIE.dma_bd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
      AIE.next_bd ^end
    ^end:
      AIE.end
  }
  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIEX.useToken @token0(Acquire, 1)
      AIE.dma_bd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIEX.useToken @token0(Release, 2)
      AIE.next_bd ^end
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
