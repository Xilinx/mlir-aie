//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-lock-ids %s | FileCheck %s
// CHECK:    %0 = AIE.tile(2, 2)
// CHECK:    %1 = AIE.tile(2, 3)
// CHECK:    %2 = AIE.tile(3, 3)
// CHECK:    %3 = AIE.tile(3, 4)
// CHECK:    %4 = AIE.lock(%0, 0)
// CHECK:    %5 = AIE.lock(%0, 2)
// CHECK:    %6 = AIE.lock(%0, 1)
// CHECK:    %7 = AIE.lock(%1, 0)
// CHECK:    %8 = AIE.lock(%1, 1)
// CHECK:    %9 = AIE.lock(%1, 4)
// CHECK:    %10 = AIE.lock(%1, 2)
// CHECK:    %11 = AIE.lock(%1, 3)
// CHECK:    %12 = AIE.lock(%1, 5)
// CHECK:    %13 = AIE.lock(%1, 6)
// CHECK:    %14 = AIE.lock(%1, 7)
// CHECK:    %15 = AIE.lock(%1, 10)
// CHECK:    %16 = AIE.lock(%1, 11)
// CHECK:    %17 = AIE.lock(%1, 8)
// CHECK:    %18 = AIE.lock(%1, 9)
// CHECK:    %19 = AIE.lock(%1, 12)
// CHECK:    %20 = AIE.lock(%1, 13)
// CHECK:    %21 = AIE.lock(%1, 14)
// CHECK:    %22 = AIE.lock(%1, 15)
// CHECK:    %23 = AIE.lock(%2, 0)
// CHECK:    %24 = AIE.lock(%2, 1)
// CHECK:    %25 = AIE.lock(%2, 9)
// CHECK:    %26 = AIE.lock(%2, 2)
// CHECK:    %27 = AIE.lock(%3, 0)
// CHECK:    %28 = AIE.lock(%3, 1)
// CHECK:    %29 = AIE.lock(%3, 2)
// CHECK:    %30 = AIE.lock(%3, 3)
// CHECK:    %31 = AIE.tile(6, 0)
// CHECK:    %32 = AIE.lock(%31, 0)

module @test_assign_lockIDs {
 AIE.device(xcvc1902) {
  %t22 = AIE.tile(2, 2)
  %t23 = AIE.tile(2, 3)
  %t33 = AIE.tile(3, 3)
  %t34 = AIE.tile(3, 4)

  %l22_0 = AIE.lock(%t22, 0)
  %l22_2 = AIE.lock(%t22, 2)
  %l22_1 = AIE.lock(%t22)

  %l23_0 = AIE.lock(%t23)
  %l23_1 = AIE.lock(%t23)
  %l23_4 = AIE.lock(%t23, 4)
  %l23_2 = AIE.lock(%t23)
  %l23_3 = AIE.lock(%t23)
  %l23_5 = AIE.lock(%t23)
  %l23_6 = AIE.lock(%t23)
  %l23_7 = AIE.lock(%t23)
  %l23_10 = AIE.lock(%t23)
  %l23_11 = AIE.lock(%t23)
  %l23_8 = AIE.lock(%t23, 8)
  %l23_9 = AIE.lock(%t23, 9)
  %l23_12 = AIE.lock(%t23)
  %l23_13 = AIE.lock(%t23)
  %l23_14 = AIE.lock(%t23)
  %l23_15 = AIE.lock(%t23)

  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33)
  %l33_9 = AIE.lock(%t33, 9)
  %l33_2 = AIE.lock(%t33)

  %l34_0 = AIE.lock(%t34)
  %l34_1 = AIE.lock(%t34)
  %l34_2 = AIE.lock(%t34)
  %l34_3 = AIE.lock(%t34)

  %t60 = AIE.tile(6, 0)
  %l60 = AIE.lock(%t60)
 }
}
