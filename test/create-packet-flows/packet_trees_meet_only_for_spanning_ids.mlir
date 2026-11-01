//===- packet_trees_meet_only_for_spanning_ids.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// The trees of (0,2) DMA:0 and (0,5) DMA:0 share (0,1) DMA:5 and (0,5) Core:0,
// but (0,2) sends id 14 to one and id 22 to the other. No packet of (0,2) goes
// to both, so none can hold an arbiter where the trees meet for one receiver
// and wait where they meet for the other, and the trees need not meet. Making
// them meet took id 22 through (0,1), where it can deadlock with id 1 from
// (0,1) DMA:0. Reduced from router_properties.py npu1 seed 18268.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-NOT:     aie.rule({{[0-9]+}}, 22,
// CHECK-LABEL: aie.switchbox(%tile_0_2)

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_5 = aie.tile(0, 5)
    %b_0_2_0 = aie.buffer(%t_0_2) {sym_name = "b_0_2_0"} : memref<16xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_2_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 22>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(3) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(14) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(1) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(22) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, Core : 0> }
  }
}
