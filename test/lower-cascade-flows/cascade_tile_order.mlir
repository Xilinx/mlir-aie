//===- cascade_tile_order.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-cascade-flows %s | FileCheck %s

// The configure_cascade ops come out in the order the tiles are declared.

// CHECK-LABEL: aie.device(npu1)
// CHECK:      aie.configure_cascade(%tile_0_2, North, East)
// CHECK-NEXT: aie.configure_cascade(%tile_1_2, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_2_2, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_3_2, West, South)
// CHECK-NEXT: aie.configure_cascade(%tile_0_3, North, East)
// CHECK-NEXT: aie.configure_cascade(%tile_1_3, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_2_3, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_3_3, West, South)
// CHECK-NEXT: aie.configure_cascade(%tile_0_4, North, East)
// CHECK-NEXT: aie.configure_cascade(%tile_1_4, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_2_4, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_3_4, West, South)
// CHECK-NEXT: aie.configure_cascade(%tile_0_5, North, East)
// CHECK-NEXT: aie.configure_cascade(%tile_1_5, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_2_5, West, East)
// CHECK-NEXT: aie.configure_cascade(%tile_3_5, West, South)

module {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)
    %t32 = aie.tile(3, 2)
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t04 = aie.tile(0, 4)
    %t14 = aie.tile(1, 4)
    %t24 = aie.tile(2, 4)
    %t34 = aie.tile(3, 4)
    %t05 = aie.tile(0, 5)
    %t15 = aie.tile(1, 5)
    %t25 = aie.tile(2, 5)
    %t35 = aie.tile(3, 5)
    aie.cascade_flow(%t02, %t12)
    aie.cascade_flow(%t12, %t22)
    aie.cascade_flow(%t22, %t32)
    aie.cascade_flow(%t03, %t13)
    aie.cascade_flow(%t13, %t23)
    aie.cascade_flow(%t23, %t33)
    aie.cascade_flow(%t04, %t14)
    aie.cascade_flow(%t14, %t24)
    aie.cascade_flow(%t24, %t34)
    aie.cascade_flow(%t05, %t15)
    aie.cascade_flow(%t15, %t25)
    aie.cascade_flow(%t25, %t35)
  }
}
