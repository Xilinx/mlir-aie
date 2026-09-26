//===- priority_route_shared_id.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A switchbox routes on the slave port and id alone, so a flow sharing both
// with a priority_route flow is carried as a control packet there too.

// Same id and source, one priority and one not. The shared hops carry id 2 as
// a control packet, and id 31, which shares the source, keeps its own amsel
// and exact rule, so it does not reach the priority flow's DMA : 1.

// CHECK-LABEL: aie.switchbox(%tile_2_2) {
// CHECK-NEXT:    %[[PLAIN:.*]] = aie.amsel<{{[0-9]}}> ({{[0-9]}})
// CHECK-NEXT:    %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK-NEXT:    aie.masterset(DMA : 1, %[[CTRL]]) {is_ctrl_pkt_overlay}
// CHECK-NEXT:    aie.masterset(North : {{[0-9]+}}, %[[PLAIN]], %[[CTRL]]) {is_ctrl_pkt_overlay}
// CHECK-NEXT:    aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 2, %[[CTRL]])
// CHECK-NEXT:      aie.rule(31, 31, %[[PLAIN]])
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}
// CHECK-NEXT:  }

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
// The priority flow stays a control packet up to its last hop.

// CHECK-LABEL: aie.switchbox(%tile_3_5) {
// CHECK-NEXT:    %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK-NEXT:    aie.masterset(DMA : 0, %[[CTRL]]) {is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.rule(31, 15, %[[CTRL]])
// CHECK-DAG:     aie.rule(31, 15, %[[CTRL]])
// CHECK:       }

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
