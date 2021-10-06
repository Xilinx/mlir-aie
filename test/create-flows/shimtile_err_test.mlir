//===- shimtile_err_test.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows --aie-find-flows %s |& FileCheck %s
// CHECK: error: 'AIE.flow' op attempts to route to ShimTile (4, 0), which has no NOC connection
// CHECK: error: 'AIE.flow' op attempts to route from ShimTile (5, 0), which has no NOC connection

module {
%t20 = AIE.tile(2, 0)
%t21 = AIE.tile(2, 1)
%t31 = AIE.tile(3, 1)
%t40 = AIE.tile(4, 0)
%t50 = AIE.tile(5, 0)
%t60 = AIE.tile(6, 0)
%t83 = AIE.tile(8, 3)

AIE.flow(%t20, DMA : 0, %t83, DMA : 0)
AIE.flow(%t21, DMA : 0, %t31, DMA : 0)
AIE.flow(%t20, DMA : 0, %t40, DMA : 1)
AIE.flow(%t50, DMA : 0, %t60, DMA : 1)
AIE.flow(%t20, DMA : 0, %t60, DMA : 0)

}