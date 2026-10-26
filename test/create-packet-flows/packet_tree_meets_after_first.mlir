//===- packet_tree_meets_after_first.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Flow 23's tree from (0,2) DMA:0 reaches its own tile's Core:0 first, then
// meets flow 8's tree from the same source. Meeting it before reaching Core:0
// led the tree out by a master port of prioritized flow 26, which leaves the
// source only that flow's master ports to take, and none leads to Core:0.
// Reduced from router_properties.py npu2 seed 1948.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         aie.masterset(Core : 0, %[[CORE:[0-9]+]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 23, %[[CORE]])

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    aie.packet_flow(26) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_dest<%t_0_1, DMA : 3> } {priority_route = true}
    aie.packet_flow(8) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_4, Core : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_4, Core : 0> }
  }
}
