//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-lock-ids --split-input-file %s | FileCheck %s
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 4)
// CHECK:           aie.lock @[[VAL_4:.*]](%[[VAL_0]], 0)
// CHECK:           aie.lock @[[VAL_5:.*]](%[[VAL_0]], 2)
// CHECK:           aie.lock @[[VAL_6:.*]](%[[VAL_0]], 1)
// CHECK:           aie.lock @[[VAL_7:.*]](%[[VAL_1]], 0)
// CHECK:           aie.lock @[[VAL_8:.*]](%[[VAL_1]], 1)
// CHECK:           aie.lock @[[VAL_9:.*]](%[[VAL_1]], 4)
// CHECK:           aie.lock @[[VAL_10:.*]](%[[VAL_1]], 2)
// CHECK:           aie.lock @[[VAL_11:.*]](%[[VAL_1]], 3)
// CHECK:           aie.lock @[[VAL_12:.*]](%[[VAL_1]], 5)
// CHECK:           aie.lock @[[VAL_13:.*]](%[[VAL_1]], 6)
// CHECK:           aie.lock @[[VAL_14:.*]](%[[VAL_1]], 7)
// CHECK:           aie.lock @[[VAL_15:.*]](%[[VAL_1]], 10)
// CHECK:           aie.lock @[[VAL_16:.*]](%[[VAL_1]], 11)
// CHECK:           aie.lock @[[VAL_17:.*]](%[[VAL_1]], 8)
// CHECK:           aie.lock @[[VAL_18:.*]](%[[VAL_1]], 9)
// CHECK:           aie.lock @[[VAL_19:.*]](%[[VAL_1]], 12)
// CHECK:           aie.lock @[[VAL_20:.*]](%[[VAL_1]], 13)
// CHECK:           aie.lock @[[VAL_21:.*]](%[[VAL_1]], 14)
// CHECK:           aie.lock @[[VAL_22:.*]](%[[VAL_1]], 15)
// CHECK:           aie.lock @[[VAL_23:.*]](%[[VAL_2]], 0)
// CHECK:           aie.lock @[[VAL_24:.*]](%[[VAL_2]], 1)
// CHECK:           aie.lock @[[VAL_25:.*]](%[[VAL_2]], 9)
// CHECK:           aie.lock @[[VAL_26:.*]](%[[VAL_2]], 2)
// CHECK:           aie.lock @[[VAL_27:.*]](%[[VAL_3]], 0)
// CHECK:           aie.lock @[[VAL_28:.*]](%[[VAL_3]], 1)
// CHECK:           aie.lock @[[VAL_29:.*]](%[[VAL_3]], 2)
// CHECK:           aie.lock @[[VAL_30:.*]](%[[VAL_3]], 3)
// CHECK:           %[[VAL_31:.*]] = aie.tile(6, 0)
// CHECK:           aie.lock @[[VAL_32:.*]](%[[VAL_31]], 0)

module @test_assign_lockIDs {
 aie.device(xcvc1902) {
  %t22 = aie.tile(2, 2)
  %t23 = aie.tile(2, 3)
  %t33 = aie.tile(3, 3)
  %t34 = aie.tile(3, 4)

  aie.lock @l22_0(%t22, 0)
  aie.lock @l22_2(%t22, 2)
  aie.lock @l22_1(%t22)

  aie.lock @l23_0(%t23)
  aie.lock @l23_1(%t23)
  aie.lock @l23_4(%t23, 4)
  aie.lock @l23_2(%t23)
  aie.lock @l23_3(%t23)
  aie.lock @l23_5(%t23)
  aie.lock @l23_6(%t23)
  aie.lock @l23_7(%t23)
  aie.lock @l23_10(%t23)
  aie.lock @l23_11(%t23)
  aie.lock @l23_8(%t23, 8)
  aie.lock @l23_9(%t23, 9)
  aie.lock @l23_12(%t23)
  aie.lock @l23_13(%t23)
  aie.lock @l23_14(%t23)
  aie.lock @l23_15(%t23)

  aie.lock @l33_0(%t33, 0)
  aie.lock @l33_1(%t33)
  aie.lock @l33_9(%t33, 9)
  aie.lock @l33_2(%t33)

  aie.lock @l34_0(%t34)
  aie.lock @l34_1(%t34)
  aie.lock @l34_2(%t34)
  aie.lock @l34_3(%t34)

  %t60 = aie.tile(6, 0)
  aie.lock @l60(%t60)
 }
}

// -----

module @memTileTest {
  aie.device(xcve2802) {

    // Memory tiles on xcve have 64 locks.
    %tmemtile = aie.tile(1,1)
    aie.lock @l0(%tmemtile, 1)
    aie.lock @l1(%tmemtile, 0)
    aie.lock @l2(%tmemtile)
    aie.lock @l3(%tmemtile)
    aie.lock @l4(%tmemtile)
    aie.lock @l5(%tmemtile)
    aie.lock @l6(%tmemtile)
    aie.lock @l7(%tmemtile)
    aie.lock @l8(%tmemtile)
    aie.lock @l9(%tmemtile)
    aie.lock @l10(%tmemtile)
    aie.lock @l11(%tmemtile)
    aie.lock @l12(%tmemtile)
    aie.lock @l13(%tmemtile)
    aie.lock @l14(%tmemtile,33)
    aie.lock @l15(%tmemtile)
    aie.lock @l16(%tmemtile)
    aie.lock @l17(%tmemtile)
    aie.lock @l18(%tmemtile)
    aie.lock @l19(%tmemtile,2)
  }
}


// CHECK-LABEL: memTileTest
// CHECK-COUNT-20: aie.lock
// CHECK-NOT: aie.lock
