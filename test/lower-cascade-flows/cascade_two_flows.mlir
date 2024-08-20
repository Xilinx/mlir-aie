//===- cascade_two_flows.mlir -----------------------------------*- MLIR -*-===//
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
// CHECK-DAG:     %tile_1_4 = aie.tile(1, 4)
// CHECK-DAG:     %tile_2_4 = aie.tile(2, 4)
// CHECK-DAG:     %tile_3_4 = aie.tile(3, 4)
// CHECK-DAG:     aie.configure_cascade(%tile_1_4, North, East)
// CHECK-DAG:     aie.configure_cascade(%tile_2_4, West, East)
// CHECK-DAG:     aie.configure_cascade(%tile_3_4, West, South)
// CHECK:   }
// CHECK: }

module @cascade_flow {
  aie.device(xcve2802) {
    %t14 = aie.tile(1, 4)
    %t24 = aie.tile(2, 4)
    %t34 = aie.tile(3, 4)
    aie.cascade_flow(%t14, %t24)
    aie.cascade_flow(%t24, %t34)
  }
}
