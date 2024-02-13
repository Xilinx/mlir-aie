//===- cascade_flow.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-cascade-flows %s | FileCheck %s

// CHECK: module @cascade_flow {
// CHECK:   aie.device(xcve2802) {
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %tile_2_3 = aie.tile(2, 3)
// CHECK:     %tile_3_4 = aie.tile(3, 4)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %cascade_switchbox_1_3 = aie.cascade_switchbox(%tile_1_3) {
// CHECK:       aie.connect<North : 0, East : 0>
// CHECK:     }
// CHECK:     %cascade_switchbox_2_3 = aie.cascade_switchbox(%tile_2_3) {
// CHECK:       aie.connect<West : 0, South : 0>
// CHECK:     }
// CHECK:     %cascade_switchbox_3_4 = aie.cascade_switchbox(%tile_3_4) {
// CHECK:       aie.connect<North : 0, South : 0>
// CHECK:     }
// CHECK:     %cascade_switchbox_3_3 = aie.cascade_switchbox(%tile_3_3) {
// CHECK:       aie.connect<North : 0, South : 0>
// CHECK:     }
// CHECK:   }
// CHECK: }

module @cascade_flow {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    aie.cascade_flow(%t13, %t23)

    %t34 = aie.tile(3, 4)
    %t33 = aie.tile(3, 3)
    aie.cascade_flow(%t34, %t33)
  }
}
