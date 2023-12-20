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
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_0]], 0)
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 2)
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 1)
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 0)
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]], 1)
// CHECK:           %[[VAL_9:.*]] = aie.lock(%[[VAL_1]], 4)
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_1]], 2)
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_1]], 3)
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_1]], 5)
// CHECK:           %[[VAL_13:.*]] = aie.lock(%[[VAL_1]], 6)
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_1]], 7)
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_1]], 10)
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_1]], 11)
// CHECK:           %[[VAL_17:.*]] = aie.lock(%[[VAL_1]], 8)
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_1]], 9)
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_1]], 12)
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_1]], 13)
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_1]], 14)
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_1]], 15)
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_2]], 0)
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_2]], 1)
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_2]], 9)
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_2]], 2)
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_3]], 0)
// CHECK:           %[[VAL_28:.*]] = aie.lock(%[[VAL_3]], 1)
// CHECK:           %[[VAL_29:.*]] = aie.lock(%[[VAL_3]], 2)
// CHECK:           %[[VAL_30:.*]] = aie.lock(%[[VAL_3]], 3)
// CHECK:           %[[VAL_31:.*]] = aie.tile(6, 0)
// CHECK:           %[[VAL_32:.*]] = aie.lock(%[[VAL_31]], 0)

module @test_assign_lockIDs {
 aie.device(xcvc1902) {
  %t22 = aie.tile(2, 2)
  %t23 = aie.tile(2, 3)
  %t33 = aie.tile(3, 3)
  %t34 = aie.tile(3, 4)

  %l22_0 = aie.lock(%t22, 0)
  %l22_2 = aie.lock(%t22, 2)
  %l22_1 = aie.lock(%t22)

  %l23_0 = aie.lock(%t23)
  %l23_1 = aie.lock(%t23)
  %l23_4 = aie.lock(%t23, 4)
  %l23_2 = aie.lock(%t23)
  %l23_3 = aie.lock(%t23)
  %l23_5 = aie.lock(%t23)
  %l23_6 = aie.lock(%t23)
  %l23_7 = aie.lock(%t23)
  %l23_10 = aie.lock(%t23)
  %l23_11 = aie.lock(%t23)
  %l23_8 = aie.lock(%t23, 8)
  %l23_9 = aie.lock(%t23, 9)
  %l23_12 = aie.lock(%t23)
  %l23_13 = aie.lock(%t23)
  %l23_14 = aie.lock(%t23)
  %l23_15 = aie.lock(%t23)

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33)
  %l33_9 = aie.lock(%t33, 9)
  %l33_2 = aie.lock(%t33)

  %l34_0 = aie.lock(%t34)
  %l34_1 = aie.lock(%t34)
  %l34_2 = aie.lock(%t34)
  %l34_3 = aie.lock(%t34)

  %t60 = aie.tile(6, 0)
  %l60 = aie.lock(%t60)
 }
}

// -----

module @memTileTest {
  aie.device(xcve2802) {

    // Memory tiles on xcve have 64 locks.
    %tmemtile = aie.tile(1,1)
    %l0 = aie.lock(%tmemtile, 1)
    %l1 = aie.lock(%tmemtile, 0)
    %l2 = aie.lock(%tmemtile)
    %l3 = aie.lock(%tmemtile)
    %l4 = aie.lock(%tmemtile)
    %l5 = aie.lock(%tmemtile)
    %l6 = aie.lock(%tmemtile)
    %l7 = aie.lock(%tmemtile)
    %l8 = aie.lock(%tmemtile)
    %l9 = aie.lock(%tmemtile)
    %l10 = aie.lock(%tmemtile)
    %l11 = aie.lock(%tmemtile)
    %l12 = aie.lock(%tmemtile)
    %l13 = aie.lock(%tmemtile)
    %l14 = aie.lock(%tmemtile,33)
    %l15 = aie.lock(%tmemtile)
    %l16 = aie.lock(%tmemtile)
    %l17 = aie.lock(%tmemtile)
    %l18 = aie.lock(%tmemtile)
    %l19 = aie.lock(%tmemtile,2)
  }
}


// CHECK-LABEL: memTileTest
// CHECK-COUNT-20: aie.lock
// CHECK-NOT: aie.lock
