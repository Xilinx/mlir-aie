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
// Verify that canonicalize does not remove chained aie.next_bd

// CHECK-LABEL:  module @test {
// CHECK-NEXT:     %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK-NEXT:     %[[VAL_1:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK-NEXT:       %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       aie.next_bd ^bb2
// CHECK-NEXT:     ^bb2:  // pred: ^bb1
// CHECK-NEXT:       aie.next_bd ^bb3
// CHECK-NEXT:     ^bb3:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:       aie.end
// CHECK-NEXT:     }
// CHECK-NEXT:  }

module @test {
  %t1 = aie.tile(1, 1)

  %mem13 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.next_bd ^bd1 // point to the next BD, or termination
    ^bd1:
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }
}
