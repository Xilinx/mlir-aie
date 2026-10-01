//===- priority_route_holds_needed_channel.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// Packet id 0 goes from memtile (1,1) to core (1,2) and memtile (0,1). Alone,
// it reaches (0,1) over (1,2) and down from (0,2), but four circuit flows
// need all four channels down from (0,2) to (0,1). Memtiles have no East or
// West ports, so the packet could go down through the shims instead; a
// prioritized flow keeps the route it takes alone, though, so the router
// rejects the first design and routes the second, where the flow is not
// prioritized, over (1,0) and (0,0) into (0,1) from the South.

// CHECK: error: Unable to find a legal routing: packet flows from (1, 1) DMA:0 are prioritized (priority_route), so they keep the route they take alone, and the router found no routing for the other flows around the channels it holds{{.*}} from tile (0, 2) to (0, 1)

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[A:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 5, %[[A]])
// CHECK:         aie.packet_rules(South : {{[0-9]}}) {
// CHECK-NEXT:      aie.rule(31, 0, %[[A]])

module {
  aie.device(npu1_2col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    aie.flow(%t_0_1, DMA : 0, %t_1_4, DMA : 1)
    aie.flow(%t_0_2, DMA : 1, %t_0_1, DMA : 1)
    aie.flow(%t_0_2, DMA : 0, %t_0_1, DMA : 0)
    aie.flow(%t_0_4, DMA : 0, %t_1_1, DMA : 2)
    aie.flow(%t_1_2, DMA : 0, %t_0_0, DMA : 0)
    aie.flow(%t_1_5, Core : 0, %t_1_0, DMA : 1)
    aie.flow(%t_1_5, DMA : 1, %t_1_1, DMA : 3)
    aie.flow(%t_0_2, Core : 0, %t_0_0, DMA : 1)
    aie.flow(%t_0_5, Core : 0, %t_1_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t_1_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> aie.packet_dest<%t_1_2, Core : 0> } {priority_route = true}
  }
}

// -----

module {
  aie.device(npu1_2col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    aie.flow(%t_0_1, DMA : 0, %t_1_4, DMA : 1)
    aie.flow(%t_0_2, DMA : 1, %t_0_1, DMA : 1)
    aie.flow(%t_0_2, DMA : 0, %t_0_1, DMA : 0)
    aie.flow(%t_0_4, DMA : 0, %t_1_1, DMA : 2)
    aie.flow(%t_1_2, DMA : 0, %t_0_0, DMA : 0)
    aie.flow(%t_1_5, Core : 0, %t_1_0, DMA : 1)
    aie.flow(%t_1_5, DMA : 1, %t_1_1, DMA : 3)
    aie.flow(%t_0_2, Core : 0, %t_0_0, DMA : 1)
    aie.flow(%t_0_5, Core : 0, %t_1_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t_1_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> aie.packet_dest<%t_1_2, Core : 0> }
  }
}
