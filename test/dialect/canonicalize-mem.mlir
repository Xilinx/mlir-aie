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
// Verify that canonicalize does not remove chained AIE.nextBd

// CHECK:      module @test {
// CHECK-NEXT:   %0 = AIE.tile(1, 1)
// CHECK-NEXT:   %1 = AIE.mem(%0) {
// CHECK-NEXT:     %2 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     AIE.nextBd ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     AIE.nextBd ^bb3
// CHECK-NEXT:   ^bb3:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     AIE.end
// CHECK-NEXT:   }
// CHECK-NEXT: }
module @test {
  %t1 = AIE.tile(1, 1)

  %mem13 = AIE.mem(%t1) {
    %dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
    ^bd0:
      AIE.nextBd ^bd1 // point to the next BD, or termination
    ^bd1:
      AIE.nextBd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }
}
