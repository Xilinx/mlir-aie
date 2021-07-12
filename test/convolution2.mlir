//===- convolution2.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

// arbiter() {
// %1 = masterset(north:1, east:2);
// }
// packetrules(east:1) {
// rule(10000|getRow(), %1);
// }
// for(i: 1:8)
// for(j: 1:50) {
//   out[i][j] = AIE.tile()
//   }
//   flow(out[i][4], out[i][6])
//   }
module {


  %21 = AIE.tile(2, 1)
  %22 = AIE.tile(2, 2)
  %23 = AIE.tile(2, 3)
  %24 = AIE.tile(2, 4)
  %25 = AIE.tile(2, 5)
  %26 = AIE.tile(2, 6)
  %31 = AIE.tile(3, 1)
  %32 = AIE.tile(3, 2)
  %33 = AIE.tile(3, 3)
  %34 = AIE.tile(3, 4)
  %35 = AIE.tile(3, 5)
  %36 = AIE.tile(3, 6)
  %41 = AIE.tile(4, 1)
  %42 = AIE.tile(4, 2)
  %43 = AIE.tile(4, 3)
  %44 = AIE.tile(4, 4)
  %45 = AIE.tile(4, 5)
  %46 = AIE.tile(4, 6)
  %51 = AIE.tile(5, 1)
  %52 = AIE.tile(5, 2)
  %53 = AIE.tile(5, 3)
  %54 = AIE.tile(5, 4)
  %55 = AIE.tile(5, 5)
  %56 = AIE.tile(5, 6)
  %p0 = AIE.plio(0)
  %p1 = AIE.plio(1)
  %p2 = AIE.plio(2)
  %p3 = AIE.plio(3)

  AIE.flow(%21, DMA : 0, %23, DMA : 1)
  AIE.flow(%22, DMA : 0, %23, DMA : 0)
  AIE.flow(%22, DMA : 0, %24, DMA : 0)
  AIE.flow(%22, DMA : 0, %25, DMA : 0)
  AIE.flow(%22, DMA : 0, %26, DMA : 0)
  AIE.flow(%31, DMA : 0, %24, DMA : 1)
  AIE.flow(%32, DMA : 0, %33, DMA : 0)
  AIE.flow(%32, DMA : 0, %34, DMA : 0)
  AIE.flow(%32, DMA : 0, %35, DMA : 0)
  AIE.flow(%32, DMA : 0, %36, DMA : 0)
  AIE.flow(%41, DMA : 0, %25, DMA : 1)
  AIE.flow(%42, DMA : 0, %43, DMA : 0)
  AIE.flow(%42, DMA : 0, %44, DMA : 0)
  AIE.flow(%42, DMA : 0, %45, DMA : 0)
  AIE.flow(%42, DMA : 0, %46, DMA : 0)
  AIE.flow(%51, DMA : 0, %26, DMA : 1)
  AIE.flow(%52, DMA : 0, %53, DMA : 0)
  AIE.flow(%52, DMA : 0, %54, DMA : 0)
  AIE.flow(%52, DMA : 0, %55, DMA : 0)
  AIE.flow(%52, DMA : 0, %56, DMA : 0)
}
