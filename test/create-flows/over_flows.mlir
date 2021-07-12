//===- over_flows.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T03:.*]] = AIE.tile(0, 3)
// CHECK: %[[T02:.*]] = AIE.tile(0, 2)
// CHECK: %[[T00:.*]] = AIE.tile(0, 0)
// CHECK: %[[T13:.*]] = AIE.tile(1, 3)
// CHECK: %[[T11:.*]] = AIE.tile(1, 1)
// CHECK: %[[T10:.*]] = AIE.tile(1, 0)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T30:.*]] = AIE.tile(3, 0)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: %[[T31:.*]] = AIE.tile(3, 1)
// CHECK: %[[T60:.*]] = AIE.tile(6, 0)
// CHECK: %[[T70:.*]] = AIE.tile(7, 0)
// CHECK: %[[T71:.*]] = AIE.tile(7, 1)
// CHECK: %[[T72:.*]] = AIE.tile(7, 2)
// CHECK: %[[T73:.*]] = AIE.tile(7, 3)
// CHECK: %[[T80:.*]] = AIE.tile(8, 0)
// CHECK: %[[T82:.*]] = AIE.tile(8, 2)
// CHECK: %[[T83:.*]] = AIE.tile(8, 3)
//
// CHECK: AIE.flow(%[[T71]], DMA : 0, %[[T20]], DMA : 0)
// CHECK: AIE.flow(%[[T71]], DMA : 1, %[[T20]], DMA : 1)
// CHECK: AIE.flow(%[[T72]], DMA : 0, %[[T60]], DMA : 0)
// CHECK: AIE.flow(%[[T72]], DMA : 1, %[[T60]], DMA : 1)
// CHECK: AIE.flow(%[[T73]], DMA : 0, %[[T70]], DMA : 0)
// CHECK: AIE.flow(%[[T73]], DMA : 1, %[[T70]], DMA : 1)
// CHECK: AIE.flow(%[[T83]], DMA : 0, %[[T30]], DMA : 0)
// CHECK: AIE.flow(%[[T83]], DMA : 1, %[[T30]], DMA : 1)

module {
%t03 = AIE.tile(0, 3)
%t02 = AIE.tile(0, 2)
%t00 = AIE.tile(0, 0)
%t13 = AIE.tile(1, 3)
%t11 = AIE.tile(1, 1)
%t10 = AIE.tile(1, 0)
%t20 = AIE.tile(2, 0)
%t30 = AIE.tile(3, 0)
%t22 = AIE.tile(2, 2)
%t31 = AIE.tile(3, 1)
%t60 = AIE.tile(6, 0)
%t70 = AIE.tile(7, 0)
%t71 = AIE.tile(7, 1)
%t72 = AIE.tile(7, 2)
%t73 = AIE.tile(7, 3)
%t80 = AIE.tile(8, 0)
%t82 = AIE.tile(8, 2)
%t83 = AIE.tile(8, 3)

AIE.flow(%t71, DMA : 0, %t20, DMA : 0)
AIE.flow(%t71, DMA : 1, %t20, DMA : 1)
AIE.flow(%t72, DMA : 0, %t60, DMA : 0)
AIE.flow(%t72, DMA : 1, %t60, DMA : 1)
AIE.flow(%t73, DMA : 0, %t70, DMA : 0)
AIE.flow(%t73, DMA : 1, %t70, DMA : 1)
AIE.flow(%t83, DMA : 0, %t30, DMA : 0)
AIE.flow(%t83, DMA : 1, %t30, DMA : 1)
}

