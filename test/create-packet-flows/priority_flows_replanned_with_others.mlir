//===- priority_flows_replanned_with_others.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// Routed alone, prioritized flows 7 and 19 share arbiter 5 at (3,2). Flow 30
// follows flow 7 onto its arbiter there, and flow 19 can hold that arbiter
// while draining its receiver waits on the circuit into (3,3), which waits on
// flow 30. No plan keeps the overlay's own arbiters, so without a
// control-packet reload, which would keep them, the router plans the overlay
// jointly with the other flows.
// Reduced from router_properties.py npu2 seed 198.

// CHECK-LABEL: aie.switchbox(%tile_3_2)
// CHECK-DAG:     %[[A4:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[A5:.*]] = aie.amsel<5> (3)
// CHECK:         aie.packet_rules(Core : 0) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A5]]) {aie.priority_route}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 7, %[[A4]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:      aie.rule(31, 30, %[[A4]])

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2_4col) {
    %t_3_1 = aie.tile(3, 1)
    %t_3_2 = aie.tile(3, 2)
    %t_3_3 = aie.tile(3, 3)
    %t_3_5 = aie.tile(3, 5)
    %b_3_1_0 = aie.buffer(%t_3_1) {sym_name = "b_3_1_0"} : memref<16xi32>
    %dma_3_1 = aie.memtile_dma(%t_3_1) {
      %d0 = aie.dma_start(MM2S, 2, ^p0b0, ^end, repeat_count = 2)
    ^p0b0:
      aie.dma_bd(%b_3_1_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.flow(%t_3_5, DMA : 0, %t_3_3, DMA : 0)
    aie.packet_flow(30) { aie.packet_source<%t_3_1, DMA : 2> aie.packet_dest<%t_3_3, DMA : 1> }
    aie.packet_flow(7) { aie.packet_source<%t_3_1, DMA : 2> aie.packet_dest<%t_3_3, DMA : 1> } {priority_route = true}
    aie.packet_flow(19) { aie.packet_source<%t_3_2, Core : 0> aie.packet_dest<%t_3_5, Core : 0> } {priority_route = true}
  }
}
