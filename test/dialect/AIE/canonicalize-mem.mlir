//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s
// Verify that canonicalize does not remove chained AIE.next_bd

// CHECK-LABEL:  module @test {
// CHECK-NEXT:     %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK-NEXT:     %[[VAL_1:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK-NEXT:       %[[VAL_2:.*]] = AIE.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       AIE.next_bd ^bb2
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       AIE.next_bd ^bb3
// CHECK-NEXT:     ^bb3:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       AIE.end
// CHECK-NEXT:     }
// CHECK-NEXT:  }

module @test {
  %t1 = AIE.tile(1, 1)

  %mem13 = AIE.mem(%t1) {
    %dma0 = AIE.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      AIE.next_bd ^bd1 // point to the next BD, or termination
    ^bd1:
      AIE.next_bd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }
}
