//===- test_congestion1.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// REQUIRES: ryzen_ai, chess

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// CHECK:    %[[T00:.*]] = aie.tile(0, 0)
// CHECK:    %[[T01:.*]] = aie.tile(0, 1)
// CHECK:    %[[T02:.*]] = aie.tile(0, 2)
// CHECK:    %[[T03:.*]] = aie.tile(0, 3)
// CHECK:    %[[T04:.*]] = aie.tile(0, 4)
// CHECK:    %[[T05:.*]] = aie.tile(0, 5)
// CHECK:    %{{.*}} = aie.switchbox(%[[T01]]) {
// CHECK:      aie.connect<North : 0, DMA : 0>
// CHECK:      aie.connect<North : 1, DMA : 1>
// CHECK:      aie.connect<North : 2, DMA : 2>
// CHECK:      aie.connect<North : 3, DMA : 3>
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(DMA : 4, %0)
// CHECK:      aie.packet_rules(South : 0) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T02]]) {
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:      aie.connect<North : 0, South : 1>
// CHECK:      aie.connect<North : 1, South : 2>
// CHECK:      aie.connect<North : 2, South : 3>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T03]]) {
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:      aie.connect<North : 0, South : 1>
// CHECK:      aie.connect<North : 1, South : 2>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T04]]) {
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:      aie.connect<North : 0, South : 1>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T05]]) {
// CHECK:      aie.connect<DMA : 0, South : 0>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(East : 0, %0)
// CHECK:      aie.packet_rules(DMA : 1) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T00]]) {
// CHECK:      aie.connect<North : 0, South : 2>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(North : 0, %0)
// CHECK:      aie.packet_rules(East : 0) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.shim_mux(%[[T00]]) {
// CHECK:      aie.connect<North : 2, DMA : 0>
// CHECK:    }
module {
 aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
  aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 0)
  aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 1)
  aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 2)
  aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 3)
  aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
  aie.packet_flow(0x0) {
    aie.packet_source<%tile_0_5, DMA : 1>
    aie.packet_dest<%tile_0_1, DMA : 4>
  }
 }
}
