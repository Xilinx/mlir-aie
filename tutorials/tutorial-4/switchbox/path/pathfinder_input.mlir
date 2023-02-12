//===- pathfinder_input.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//


module @pathfinder{
%t01 = AIE.tile(0, 1)
%t02 = AIE.tile(0, 2)
%t03 = AIE.tile(0, 3)
%t11 = AIE.tile(1, 1)
%t12 = AIE.tile(1, 2)
%t13 = AIE.tile(1, 3)
%t21 = AIE.tile(2, 1)
%t22 = AIE.tile(2, 2)
%t23 = AIE.tile(2, 3)
%t31 = AIE.tile(3, 1)
%t32 = AIE.tile(3, 2)
%t33 = AIE.tile(3, 3)
%t41 = AIE.tile(4, 1)
%t42 = AIE.tile(4, 2)
%t43 = AIE.tile(4, 3)

AIE.flow(%t11, DMA : 0, %t42, DMA : 0)
AIE.flow(%t42, DMA : 0, %t11, DMA : 0)
AIE.flow(%t31, DMA : 0, %t43, DMA : 0)
AIE.flow(%t43, DMA : 0, %t31, DMA : 0)

//AIE.flow(%t03, DMA : 0, %t41, DMA : 0)
}

