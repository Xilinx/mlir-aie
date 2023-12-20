//===- convolution.mlir ----------------------------------------*- MLIR -*-===//
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
module {
  %01 = aie.tile(0, 1)
  %02 = aie.tile(0, 2)
  %03 = aie.tile(0, 3)
  %04 = aie.tile(0, 4)
  %11 = aie.tile(1, 1)
  %12 = aie.tile(1, 2)
  %13 = aie.tile(1, 3)
  %14 = aie.tile(1, 4)
  %21 = aie.tile(2, 1)
  %22 = aie.tile(2, 2)
  %23 = aie.tile(2, 3)
  %24 = aie.tile(2, 4)
  %31 = aie.tile(3, 1)
  %32 = aie.tile(3, 2)
  %33 = aie.tile(3, 3)
  %34 = aie.tile(3, 4)
  %p0 = aie.plio(0)
  %p1 = aie.plio(1)
  %p2 = aie.plio(2)
  %p3 = aie.plio(3)
  // North flowing input activations
  aie.flow(%p0, North : 0, %01, Core : 1)
  aie.flow(%p0, North : 0, %11, Core : 1)
  aie.flow(%p0, North : 0, %21, Core : 1)
  aie.flow(%p0, North : 0, %31, Core : 1)
  aie.flow(%p1, North : 0, %02, Core : 1)
  aie.flow(%p1, North : 0, %12, Core : 1)
  aie.flow(%p1, North : 0, %22, Core : 1)
  aie.flow(%p1, North : 0, %32, Core : 1)
  aie.flow(%p2, North : 0, %03, Core : 1)
  aie.flow(%p2, North : 0, %13, Core : 1)
  aie.flow(%p2, North : 0, %23, Core : 1)
  aie.flow(%p2, North : 0, %33, Core : 1)
  aie.flow(%p3, North : 0, %04, Core : 1)
  aie.flow(%p3, North : 0, %14, Core : 1)
  aie.flow(%p3, North : 0, %24, Core : 1)
  aie.flow(%p3, North : 0, %34, Core : 1)
  // South-west flowing results
  aie.flow(%34, Core : 0, %p3, South : 0)
  aie.flow(%33, Core : 0, %p2, South : 0)
  aie.flow(%32, Core : 0, %p1, South : 0)
  aie.flow(%31, Core : 0, %p0, South : 0)
}
