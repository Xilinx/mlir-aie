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
// packet_rules(east:1) {
// rule(10000|getRow(), %1);
// }
// for(i: 1:8)
// for(j: 1:50) {
//   out[i][j] = aie.tile()
//   }
//   flow(out[i][4], out[i][6])
//   }
module {


  %21 = aie.tile(2, 1)
  %22 = aie.tile(2, 2)
  %23 = aie.tile(2, 3)
  %24 = aie.tile(2, 4)
  %25 = aie.tile(2, 5)
  %26 = aie.tile(2, 6)
  %31 = aie.tile(3, 1)
  %32 = aie.tile(3, 2)
  %33 = aie.tile(3, 3)
  %34 = aie.tile(3, 4)
  %35 = aie.tile(3, 5)
  %36 = aie.tile(3, 6)
  %41 = aie.tile(4, 1)
  %42 = aie.tile(4, 2)
  %43 = aie.tile(4, 3)
  %44 = aie.tile(4, 4)
  %45 = aie.tile(4, 5)
  %46 = aie.tile(4, 6)
  %51 = aie.tile(5, 1)
  %52 = aie.tile(5, 2)
  %53 = aie.tile(5, 3)
  %54 = aie.tile(5, 4)
  %55 = aie.tile(5, 5)
  %56 = aie.tile(5, 6)
  %p0 = aie.plio(0)
  %p1 = aie.plio(1)
  %p2 = aie.plio(2)
  %p3 = aie.plio(3)

  aie.flow(%21, DMA : 0, %23, DMA : 1)
  aie.flow(%22, DMA : 0, %23, DMA : 0)
  aie.flow(%22, DMA : 0, %24, DMA : 0)
  aie.flow(%22, DMA : 0, %25, DMA : 0)
  aie.flow(%22, DMA : 0, %26, DMA : 0)
  aie.flow(%31, DMA : 0, %24, DMA : 1)
  aie.flow(%32, DMA : 0, %33, DMA : 0)
  aie.flow(%32, DMA : 0, %34, DMA : 0)
  aie.flow(%32, DMA : 0, %35, DMA : 0)
  aie.flow(%32, DMA : 0, %36, DMA : 0)
  aie.flow(%41, DMA : 0, %25, DMA : 1)
  aie.flow(%42, DMA : 0, %43, DMA : 0)
  aie.flow(%42, DMA : 0, %44, DMA : 0)
  aie.flow(%42, DMA : 0, %45, DMA : 0)
  aie.flow(%42, DMA : 0, %46, DMA : 0)
  aie.flow(%51, DMA : 0, %26, DMA : 1)
  aie.flow(%52, DMA : 0, %53, DMA : 0)
  aie.flow(%52, DMA : 0, %54, DMA : 0)
  aie.flow(%52, DMA : 0, %55, DMA : 0)
  aie.flow(%52, DMA : 0, %56, DMA : 0)
}
