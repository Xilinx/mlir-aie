//===- arbiter_pinned_tree_passes_tile_twice.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// Prioritized flow 16 from (9,8) Core:1 leaves (9,8) with prioritized flow 12
// on arbiter 5 and comes back into (9,8) Core:0 with flow 2 on arbiter 4. Each
// visit takes its own arbiter, so flows 2 and 12 share none, and the design
// routes. Taking the two visits as one put them on one arbiter and rejected
// it. Reduced from router_properties.py xcvc1902 seed 48.
//
// The trees of flow 2 find no switchbox to meet at that keeps their arbiters
// apart from flow 16's, so the pass warns once of the hold cycle through the
// receivers they share.

// CHECK-LABEL: %switchbox_9_8 = aie.switchbox(%tile_9_8) {
// CHECK-DAG:     %[[A5_2:.*]] = aie.amsel<5> (2)
// CHECK-DAG:     %[[A4:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[A5_3:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(Core : 0, %[[A4]])
// CHECK-DAG:     aie.masterset(South : {{[0-9]}}, %[[A5_2]], %[[A5_3]])
// CHECK:           aie.rule(31, 16, %[[A4]])
// CHECK:         aie.packet_rules(Core : 1) {
// CHECK-NEXT:      aie.rule(31, 16, %[[A5_2]]) {priority_route}
// CHECK-NEXT:      aie.rule(31, 12, %[[A5_3]])
// CHECK:           aie.rule(31, 2, %[[A4]])

// WARN:     warning: Packet flows into receivers they share can deadlock
// WARN-SAME: (id 2)
// WARN-NOT: warning

module {
  aie.device(xcvc1902) {
    %t_8_4 = aie.tile(8, 4)
    %t_8_5 = aie.tile(8, 5)
    %t_8_6 = aie.tile(8, 6)
    %t_8_7 = aie.tile(8, 7)
    %t_8_8 = aie.tile(8, 8)
    %t_9_3 = aie.tile(9, 3)
    %t_9_6 = aie.tile(9, 6)
    %t_9_8 = aie.tile(9, 8)
    %sb_8_4 = aie.switchbox(%t_8_4) {
      aie.connect<North : 3, Core : 0>
    }
    %sb_8_5 = aie.switchbox(%t_8_5) {
      aie.connect<North : 2, South : 3>
    }
    %sb_8_6 = aie.switchbox(%t_8_6) {
      aie.connect<Core : 1, South : 2>
    }
    aie.flow(%t_8_7, DMA : 1, %t_9_3, Core : 0)
    aie.packet_flow(2) { aie.packet_source<%t_8_5, DMA : 0> aie.packet_source<%t_8_4, Core : 0> aie.packet_dest<%t_9_3, Core : 1> aie.packet_dest<%t_9_6, Core : 1> aie.packet_dest<%t_9_8, Core : 0> } {priority_route = true}
    aie.packet_flow(12) { aie.packet_source<%t_9_8, Core : 1> aie.packet_dest<%t_8_7, Core : 0> } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(16) { aie.packet_source<%t_9_8, Core : 1> aie.packet_source<%t_9_3, Core : 1> aie.packet_dest<%t_8_7, Core : 0> aie.packet_dest<%t_8_8, Core : 0> aie.packet_dest<%t_9_8, Core : 0> } {priority_route = true}
  }
}
