//===- priority_rules_keep_others_off.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s | FileCheck %s

// Prioritized flows' rules match first. Another flow may take their amsel
// only where it goes exactly where they do.

// Ids 24-26 take the rule (28, 24), which also matches id 27. Id 27 goes where
// they do, so it follows their route and their rule sends it the right way.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK:         aie.masterset(North : 1, %[[A:.+]]) {aie.priority_route}
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(28, 24, %[[A]]) {aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 27, %[[A]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         aie.masterset(DMA : 0, %[[B:.+]]) {aie.priority_route}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 24, %[[B]]) {aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 27, %[[B]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.masterset(North : 1, %[[C:.+]]) {aie.priority_route}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 24, %[[C]]) {aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 27, %[[C]])
// CHECK-NEXT:    }

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_2 = aie.tile(0, 2)
    aie.packet_flow(24) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(25) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(26) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(27) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
  }
}
