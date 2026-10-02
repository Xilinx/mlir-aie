//===- priority_source_other_id_routes_apart.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-create-pathfinder-flows %s | FileCheck %s

// Id 21 shares its source with the prioritized id 0 but reaches only one of
// its destinations. Following id 0's tree, it would leave (2, 4) by South:0
// without DMA:1, a master set a control-packet reload does not keep, so it
// routes apart from the tree and meets it where they go to the same place.

// CHECK-LABEL: aie.switchbox(%tile_0_5)
// CHECK-DAG:     aie.masterset(South : 0, %[[OWN:[0-9]+]]){{$}}
// CHECK-DAG:     aie.masterset(East : 1, %[[KEPT:[0-9]+]]) {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-DAG:       aie.rule(31, 0, %[[KEPT]]) {is_ctrl_pkt_overlay, priority_route}
// CHECK-DAG:       aie.rule(31, 21, %[[OWN]])
// CHECK:         }
// CHECK-LABEL: aie.switchbox(%tile_2_4)
// CHECK-NOT:     aie.rule(31, 21
// CHECK:       aie.switchbox(

module {
  aie.device(npu2) {
    %t_0_5 = aie.tile(0, 5)
    %t_2_0 = aie.tile(2, 0)
    %t_2_4 = aie.tile(2, 4)
    aie.packet_flow(0) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_2_0, DMA : 1> aie.packet_dest<%t_2_4, DMA : 1> } {priority_route = true}
    aie.packet_flow(21) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_2_0, DMA : 1> }
  }
}
