//===- priority_route_shared_id.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A switchbox routes on the slave port and id alone, so a flow sharing both
// with a priority_route flow is carried as a control packet there too.

// Same id and source, one priority and one not. Both id 2 flows take the
// priority flow's route alone, and id 31, which shares the source, routes on
// its own. A control-packet reload keeps the overlay's rules, so the source's
// rules tag the control packet's rule, and id 31's rule follows it.

// CHECK-LABEL: aie.switchbox(%mem_tile_2_1) {
// CHECK-NEXT:    %[[PLAIN:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:    %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK-NEXT:    aie.masterset(North : {{[0-9]+}}, %[[PLAIN]])
// CHECK-NEXT:    aie.masterset(North : {{[0-9]+}}, %[[CTRL]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:    aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 2, %[[CTRL]]) {aie.is_ctrl_pkt_overlay, aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 31, %[[PLAIN]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-LABEL: aie.switchbox(%tile_2_2) {
// CHECK:         aie.masterset(DMA : 1, %[[CTRL:.*]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.masterset(North : {{[0-9]+}}, %[[CTRL]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 2, %[[CTRL]])
// CHECK-NEXT:    } {aie.is_ctrl_pkt_overlay}

module {
  aie.device(npu2_3col) {
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    aie.packet_flow(31) {
      aie.packet_source<%t_2_1, DMA : 0>
      aie.packet_dest<%t_2_4, DMA : 1>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t_2_1, DMA : 0>
      aie.packet_dest<%t_2_2, DMA : 1>
    } {priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%t_2_1, DMA : 0>
      aie.packet_dest<%t_2_5, Core : 0>
    }
  }
}

// -----

// Same id and destination from different sources, one priority and one not.
// The other flow joins the priority flow's master set where they meet, by a
// rule of its own outside the overlay.

// CHECK-LABEL: aie.switchbox(%tile_3_2) {
// CHECK-NEXT:    %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK-NEXT:    aie.masterset(North : 1, %[[CTRL]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:    aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 15, %[[CTRL]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 15, %[[CTRL]])
// CHECK-NEXT:    } {aie.is_ctrl_pkt_overlay}
// CHECK-LABEL: aie.switchbox(%tile_3_5) {
// CHECK-NEXT:    %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK-NEXT:    aie.masterset(DMA : 0, %[[CTRL]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:    aie.packet_rules(South : 4) {
// CHECK-NEXT:      aie.rule(31, 15, %[[CTRL]])
// CHECK-NEXT:    } {aie.is_ctrl_pkt_overlay}

module {
  aie.device(npu2_4col) {
    %t_3_0 = aie.tile(3, 0)
    %t_3_2 = aie.tile(3, 2)
    %t_3_5 = aie.tile(3, 5)
    aie.packet_flow(15) {
      aie.packet_source<%t_3_0, DMA : 1>
      aie.packet_dest<%t_3_5, DMA : 0>
    } {priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%t_3_2, DMA : 1>
      aie.packet_dest<%t_3_5, DMA : 0>
    }
  }
}
