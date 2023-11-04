//===- shim.mlir -----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-find-flows %s | FileCheck %s

// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_6:.*]] = AIE.shimDMA(%[[VAL_1]])
// CHECK:           AIE.flow(%[[VAL_0]], Core : 0, %[[VAL_6]], DMA : 0)
module {
  AIE.device(xcvc1902) {
    %t21 = AIE.tile(2, 1)
    %t20 = AIE.tile(2, 0)
    %c21 = AIE.core(%t21)  {
      AIE.end
    }
    %s21 = AIE.switchbox(%t21)  {
      AIE.connect<Core : 0, South : 0>
    }
    %s20 = AIE.switchbox(%t20)  {
      AIE.connect<North : 0, South : 2>
    }
    %mux = AIE.shimmux(%t20)  {
      AIE.connect<North : 2, DMA : 0>
    }
    %dma = AIE.shimDMA(%t20)  {
      AIE.end
    }
    AIE.wire(%s21 : South, %s20 : North)
    AIE.wire(%s20 : South, %mux : North)
    AIE.wire(%mux : DMA, %dma : DMA)
    AIE.wire(%mux : South, %t20 : DMA)
    AIE.wire(%s21 : Core, %c21 : Core)
    AIE.wire(%s21 : Core, %t21 : Core)
  }
}