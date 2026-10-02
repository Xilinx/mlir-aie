//===- cascade_tile_order.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-cascade-flows %s | FileCheck %s

// The configure_cascade ops come out in the order the tiles are declared.

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
// CHECK-NOT:  aie.cascade_flow

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    aie.cascade_flow(%tile_0_2, %tile_1_2)
    aie.cascade_flow(%tile_1_2, %tile_2_2)
    aie.cascade_flow(%tile_2_2, %tile_3_2)
    aie.cascade_flow(%tile_0_3, %tile_1_3)
    aie.cascade_flow(%tile_1_3, %tile_2_3)
    aie.cascade_flow(%tile_2_3, %tile_3_3)
    aie.cascade_flow(%tile_0_4, %tile_1_4)
    aie.cascade_flow(%tile_1_4, %tile_2_4)
    aie.cascade_flow(%tile_2_4, %tile_3_4)
    aie.cascade_flow(%tile_0_5, %tile_1_5)
    aie.cascade_flow(%tile_1_5, %tile_2_5)
    aie.cascade_flow(%tile_2_5, %tile_3_5)
  }
}
