//===- ctrl_pkt_overlay_user_priority_flow.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A control-packet reload keeps only the flows @ctrl_pkt_overlay holds, those
// to and from TileControl ports. The design's own prioritized flow is not one
// of them, so the reload configures it: its rules are not marked, and the
// overlay's rules beside it stay as @ctrl_pkt_overlay sets them (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD

// CHECK:      aie.switchbox(%mem_tile_0_1) {
// CHECK:        aie.packet_rules(South : 3) {
// CHECK-NEXT:     aie.rule(31, 27, %[[M1:.+]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:     aie.rule(31, 26, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:     aie.rule(31, 5, %[[M1]]){{$}}
// CHECK:      aie.switchbox(%shim_noc_tile_0_0) {
// CHECK:        aie.packet_rules(South : 3) {
// CHECK-NEXT:     aie.rule(30, 26, %[[S0:.+]]) {aie.is_ctrl_pkt_overlay, aie.priority_route}
// CHECK-NEXT:     aie.rule(31, 15, %{{.+}}) {aie.is_ctrl_pkt_overlay, aie.priority_route}
// CHECK-NEXT:     aie.rule(31, 5, %[[S0]]) {aie.priority_route}
// CHECK:      aie.switchbox(%tile_0_2) {
// CHECK-NEXT:   %[[DMA:.+]] = aie.amsel<0> (0)
// CHECK:        aie.masterset(DMA : 0, %[[DMA]]){{$}}
// CHECK:        aie.packet_rules(South : 3) {
// CHECK-NEXT:     aie.rule(31, 27, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:     aie.rule(31, 5, %[[DMA]]){{$}}

// RELOAD: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    } {priority_route = true}
  }
}
