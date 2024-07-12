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
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_7:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_13:.*]] = aie.switchbox(%[[VAL_12]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[VAL_10]] : Core, %[[VAL_17:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_10]] : DMA, %[[VAL_17]] : DMA)
// CHECK:           aie.wire(%[[VAL_18:.*]] : North, %[[VAL_17]] : South)
// CHECK:           aie.wire(%[[VAL_12]] : Core, %[[VAL_19:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_12]] : DMA, %[[VAL_19]] : DMA)
// CHECK:           aie.wire(%[[VAL_17]] : North, %[[VAL_19]] : South)
// CHECK:           aie.wire(%[[VAL_14]] : Core, %[[VAL_20:.*]] : Core)
// CHECK:           aie.wire(%[[VAL_14]] : DMA, %[[VAL_20]] : DMA)
// CHECK:           aie.wire(%[[VAL_19]] : North, %[[VAL_20]] : South)
// CHECK:           aie.wire(%[[VAL_18]] : East, %[[VAL_21:.*]] : West)
// CHECK:           aie.wire(%[[VAL_22:.*]] : North, %[[VAL_21]] : South)
// CHECK:           aie.wire(%[[VAL_0]] : DMA, %[[VAL_22]] : DMA)
// CHECK:           aie.wire(%[[VAL_17]] : East, %[[VAL_23:.*]] : West)
// CHECK:           aie.wire(%[[VAL_1]] : Core, %[[VAL_23]] : Core)
// CHECK:           aie.wire(%[[VAL_1]] : DMA, %[[VAL_23]] : DMA)
// CHECK:           aie.wire(%[[VAL_21]] : North, %[[VAL_23]] : South)
// CHECK:           aie.wire(%[[VAL_19]] : East, %[[VAL_24:.*]] : West)
// CHECK:           aie.wire(%[[VAL_2]] : Core, %[[VAL_24]] : Core)
// CHECK:           aie.wire(%[[VAL_2]] : DMA, %[[VAL_24]] : DMA)
// CHECK:           aie.wire(%[[VAL_23]] : North, %[[VAL_24]] : South)
// CHECK:           aie.wire(%[[VAL_20]] : East, %[[VAL_25:.*]] : West)
// CHECK:           aie.wire(%[[VAL_3]] : Core, %[[VAL_25]] : Core)
// CHECK:           aie.wire(%[[VAL_3]] : DMA, %[[VAL_25]] : DMA)
// CHECK:           aie.wire(%[[VAL_24]] : North, %[[VAL_25]] : South)
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
