//===- ctrl_overlay_shared_rule.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control overlay stays materialized. Flow 9 enters the shim on the
// overlay's slave port, but the router gives it a rule of its own after the
// overlay's, so flow 9 is lifted back to a packet_flow. The shim mux connect
// both use is kept, and routing the result again routes flow 9.

// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s --check-prefixes=CHECK,FIND
// RUN: aie-opt --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --aie-create-pathfinder-flows --aie-find-flows --aie-create-pathfinder-flows %s | FileCheck %s --check-prefixes=CHECK,ROUTE

// CHECK:      aie.shim_mux
// CHECK-NEXT:   aie.connect<DMA : 0, North : 3>
// CHECK-NEXT: }
// CHECK:      aie.packet_rules(South : 3) {
// CHECK-NEXT:   aie.rule(30, 26, %{{.*}}) {is_ctrl_pkt_overlay, priority_route}
// CHECK-NEXT:   aie.rule(31, 15, %{{.*}}) {is_ctrl_pkt_overlay, priority_route}
// FIND-NEXT:  }
// ROUTE-NEXT:   aie.rule(31, 9,
// CHECK:      aie.switchbox(%{{.*}}tile_0_2)
// FIND-NOT:     aie.masterset(DMA : 1,
// ROUTE:        aie.masterset(DMA : 1,
// ROUTE:        aie.rule(31, 9,
// FIND:       aie.packet_flow(9)
// ROUTE-NOT:  aie.packet_flow

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(9) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 1>
    }
  }
}
