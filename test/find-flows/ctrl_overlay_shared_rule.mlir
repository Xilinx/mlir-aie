//===- ctrl_overlay_shared_rule.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Lifting application flow 9 leaves the imported overlay and the shared shim
// mux connection intact, so the application can be routed again.

// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s --check-prefixes=CHECK,FIND
// RUN: aie-opt --aie-generate-column-control-overlay="configuration=true adaptive=false" --aie-import-overlay --aie-annotate-shim-sharing --aie-create-pathfinder-flows --aie-find-flows --aie-create-pathfinder-flows %s | FileCheck %s --check-prefixes=CHECK,ROUTE

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK:      aie.switchbox(%{{.*}}tile_0_2)
// FIND-NOT:     aie.masterset(DMA : 1,
// ROUTE:        aie.masterset(DMA : 1,
// ROUTE:        aie.rule(31, 9,
// CHECK:      aie.shim_mux
// CHECK-NEXT:   aie.connect<DMA : 0, North : 3> {aie.is_ctrl_overlay}
// CHECK-NEXT: }
// CHECK:      aie.packet_rules(South : 3) {
// CHECK-NEXT:   aie.rule(24, 24, %{{.*}}) {aie.is_ctrl_overlay}
// CHECK-NEXT:   aie.rule(31, 15, %{{.*}}) {aie.is_ctrl_overlay}
// FIND-NEXT:  }
// ROUTE-NEXT:   aie.rule(31, 9,
// FIND:       aie.packet_flow(9)
// ROUTE-NOT:  aie.packet_flow
// CHECK-LABEL: aie.device(npu1_1col) @ctrl_pkt_overlay

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
