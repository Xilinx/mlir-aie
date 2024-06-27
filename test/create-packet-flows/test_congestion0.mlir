//===- trace_packet_routing.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// REQUIRES: ryzen_ai, chess

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// CHECK:    %[[T01:.*]] = aie.tile(0, 1)
// CHECK:    %{{.*}} = aie.switchbox(%[[T01]]) {
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.amsel<1> (0)
// CHECK:      %2 = aie.amsel<0> (1)
// CHECK:      %3 = aie.amsel<0> (2)
// CHECK:      %4 = aie.amsel<0> (3)
// CHECK:      %5 = aie.masterset(DMA : 0, %0)
// CHECK:      %6 = aie.masterset(DMA : 1, %2)
// CHECK:      %7 = aie.masterset(DMA : 2, %3)
// CHECK:      %8 = aie.masterset(DMA : 3, %4)
// CHECK:      %9 = aie.masterset(DMA : 4, %1)
// CHECK:      aie.packet_rules(North : 0) {
// CHECK:        aie.rule(31, 4, %1)
// CHECK:        aie.rule(31, 3, %4)
// CHECK:        aie.rule(31, 2, %3)
// CHECK:        aie.rule(31, 1, %2)
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %[[T02:.*]] = aie.tile(0, 2)
// CHECK:    %{{.*}} = aie.switchbox(%[[T02]]) {
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(South : 0, %0)
// CHECK:      aie.packet_rules(DMA : 1) {
// CHECK:        aie.rule(31, 4, %0)
// CHECK:      }
// CHECK:      aie.packet_rules(North : 0) {
// CHECK:        aie.rule(28, 0, %0)
// CHECK:      }
// CHECK:      aie.packet_rules(DMA : 0) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %[[T03:.*]] = aie.tile(0, 3)
// CHECK:    %{{.*}} = aie.switchbox(%[[T03]]) {
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(South : 0, %0)
// CHECK:      aie.packet_rules(North : 0) {
// CHECK:        aie.rule(30, 2, %0)
// CHECK:      }
// CHECK:      aie.packet_rules(DMA : 0) {
// CHECK:        aie.rule(31, 1, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %[[T04:.*]] = aie.tile(0, 4)
// CHECK:    %{{.*}} = aie.switchbox(%[[T04]]) {
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(South : 0, %0)
// CHECK:      aie.packet_rules(North : 0) {
// CHECK:        aie.rule(31, 3, %0)
// CHECK:      }
// CHECK:      aie.packet_rules(DMA : 0) {
// CHECK:        aie.rule(31, 2, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %[[T05:.*]] = aie.tile(0, 5)
// CHECK:    %{{.*}} = aie.switchbox(%[[T05]]) {
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(South : 0, %0)
// CHECK:      aie.packet_rules(DMA : 0) {
// CHECK:        aie.rule(31, 3, %0)
// CHECK:      }
// CHECK:    }
module {
 aie.device(npu1_1col) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)

  aie.packet_flow(0) { 
    aie.packet_source<%tile_0_2, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 0>
  }
  aie.packet_flow(1) { 
    aie.packet_source<%tile_0_3, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 1>
  }
  aie.packet_flow(2) { 
    aie.packet_source<%tile_0_4, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 2>
  }
  aie.packet_flow(3) { 
    aie.packet_source<%tile_0_5, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 3>
  }
  aie.packet_flow(4) { 
    aie.packet_source<%tile_0_2, DMA : 1> 
    aie.packet_dest<%tile_0_1, DMA : 4>
  }
 }
}
