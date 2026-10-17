//===- priority_route_holds_needed_channel.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s | FileCheck %s

// Packet id 0 goes from memtile (1,1) to core (1,2) and memtile (0,1). Alone,
// it reaches (0,1) over (1,2) and down from (0,2), but four circuit flows
// need all four channels down from (0,2) to (0,1). Memtiles have no East or
// West ports, so the packet goes down through the shims instead, over (1,0)
// and (0,0) into (0,1) from the South. Without a control-packet reload, a
// prioritized flow that cannot keep the route it takes alone routes like the
// others, so the first design routes as the second does, where the flow is
// not prioritized.

// CHECK-LABEL: aie.device(npu1_2col)
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[A:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 5, %[[A]])
// CHECK:         aie.packet_rules(South : {{[0-9]}}) {
// CHECK-NEXT:      aie.rule(31, 0, %[[A]])

// CHECK-LABEL: aie.device(npu1_2col)
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[B:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 5, %[[B]])
// CHECK:         aie.packet_rules(South : {{[0-9]}}) {
// CHECK-NEXT:      aie.rule(31, 0, %[[B]])

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

// -----

// The control overlay goes up the column to every tile's TileControl, so it
// takes one of the six channels up from (0,1) to (0,2) the six circuit flows
// need, wherever a control-packet reload would let it go.

module {
  // expected-error@+1 {{Unable to find a legal routing: the circuit flows from (0, 1) DMA:0, (0, 1) DMA:1, (0, 1) DMA:2, (0, 1) DMA:3 and 2 more, with packet flows, cross from row 1 to row 2, which takes 7 channels across, and it has 6 channels.}}
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_1, DMA : 0, %t_0_3, DMA : 0)
    aie.flow(%t_0_1, DMA : 1, %t_0_3, DMA : 1)
    aie.flow(%t_0_1, DMA : 2, %t_0_4, DMA : 0)
    aie.flow(%t_0_1, DMA : 3, %t_0_4, DMA : 1)
    aie.flow(%t_0_1, DMA : 4, %t_0_5, DMA : 0)
    aie.flow(%t_0_1, DMA : 5, %t_0_5, DMA : 1)
    aie.packet_flow(0) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_1, TileControl : 0> aie.packet_dest<%t_0_2, TileControl : 0> aie.packet_dest<%t_0_3, TileControl : 0> aie.packet_dest<%t_0_4, TileControl : 0> aie.packet_dest<%t_0_5, TileControl : 0> } {priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}
