//===- memtile_routing_constraints.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_4:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<North : 0, DMA : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<North : 1, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<DMA : 0, South : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_9:.*]] : North, %[[VAL_10:.*]] : South)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_9]] : DMA)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_11:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_11]] : DMA)
// CHECK:           aie.wire(%[[VAL_10]] : North, %[[VAL_11]] : South)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_12:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_12]] : DMA)
// CHECK:           aie.wire(%[[VAL_11]] : North, %[[VAL_12]] : South)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_13:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_13]] : DMA)
// CHECK:           aie.wire(%[[VAL_12]] : North, %[[VAL_13]] : South)
// CHECK:         }

module {
  aie.device(xcve2802) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.flow(%tile_2_2, DMA : 0, %tile_2_1, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %tile_2_0, DMA : 0)
  }
}
