//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-lock-ids --split-input-file %s | FileCheck %s
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 4)
// CHECK:           %[[VAL_4:.*]] = AIE.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_5:.*]] = AIE.lock(%[[VAL_0]], 2)
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_1]], 0)
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_1]], 1)
// CHECK:           %[[VAL_9:.*]] = AIE.lock(%[[VAL_1]], 4)
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_1]], 2)
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_1]], 3)
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_1]], 5)
// CHECK:           %[[VAL_13:.*]] = AIE.lock(%[[VAL_1]], 6)
// CHECK:           %[[VAL_14:.*]] = AIE.lock(%[[VAL_1]], 7)
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_1]], 10)
// CHECK:           %[[VAL_16:.*]] = AIE.lock(%[[VAL_1]], 11)
// CHECK:           %[[VAL_17:.*]] = AIE.lock(%[[VAL_1]], 8)
// CHECK:           %[[VAL_18:.*]] = AIE.lock(%[[VAL_1]], 9)
// CHECK:           %[[VAL_19:.*]] = AIE.lock(%[[VAL_1]], 12)
// CHECK:           %[[VAL_20:.*]] = AIE.lock(%[[VAL_1]], 13)
// CHECK:           %[[VAL_21:.*]] = AIE.lock(%[[VAL_1]], 14)
// CHECK:           %[[VAL_22:.*]] = AIE.lock(%[[VAL_1]], 15)
// CHECK:           %[[VAL_23:.*]] = AIE.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_24:.*]] = AIE.lock(%[[VAL_2]], 1)
// CHECK:           %[[VAL_25:.*]] = AIE.lock(%[[VAL_2]], 9)
// CHECK:           %[[VAL_26:.*]] = AIE.lock(%[[VAL_2]], 2)
// CHECK:           %[[VAL_27:.*]] = AIE.lock(%[[VAL_3]], 0)
// CHECK:           %[[VAL_28:.*]] = AIE.lock(%[[VAL_3]], 1)
// CHECK:           %[[VAL_29:.*]] = AIE.lock(%[[VAL_3]], 2)
// CHECK:           %[[VAL_30:.*]] = AIE.lock(%[[VAL_3]], 3)
// CHECK:           %[[VAL_31:.*]] = AIE.tile(6, 0)
// CHECK:           %[[VAL_32:.*]] = AIE.lock(%[[VAL_31]], 0)

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

// -----

module @memTileTest {
  AIE.device(xcve2802) {

    // Memory tiles on xcve have 64 locks.
    %tmemtile = AIE.tile(1,1)
    %l0 = AIE.lock(%tmemtile, 1)
    %l1 = AIE.lock(%tmemtile, 0)
    %l2 = AIE.lock(%tmemtile)
    %l3 = AIE.lock(%tmemtile)
    %l4 = AIE.lock(%tmemtile)
    %l5 = AIE.lock(%tmemtile)
    %l6 = AIE.lock(%tmemtile)
    %l7 = AIE.lock(%tmemtile)
    %l8 = AIE.lock(%tmemtile)
    %l9 = AIE.lock(%tmemtile)
    %l10 = AIE.lock(%tmemtile)
    %l11 = AIE.lock(%tmemtile)
    %l12 = AIE.lock(%tmemtile)
    %l13 = AIE.lock(%tmemtile)
    %l14 = AIE.lock(%tmemtile,33)
    %l15 = AIE.lock(%tmemtile)
    %l16 = AIE.lock(%tmemtile)
    %l17 = AIE.lock(%tmemtile)
    %l18 = AIE.lock(%tmemtile)
    %l19 = AIE.lock(%tmemtile,2)
  }
}


// CHECK-LABEL: memTileTest
// CHECK-COUNT-20: AIE.lock
// CHECK-NOT: AIE.lock


