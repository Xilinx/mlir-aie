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

// The user's flow leaves the shim through DMA 0, so the overlay's requests
// take DMA 1 and the two share no slave port.

// CHECK:      aie.switchbox(%mem_tile_0_1) {
// CHECK:        aie.packet_rules(South : 3) {
// CHECK:        } {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:   aie.packet_rules(South : 5) {
// CHECK-NEXT:     aie.rule(31, 5, %{{.+}}){{$}}
// CHECK-NEXT:   }{{$}}
// CHECK:      aie.switchbox(%shim_noc_tile_0_0) {
// CHECK:        aie.packet_rules(South : 7) {
// CHECK:        } {aie.is_ctrl_pkt_overlay}
// CHECK:        aie.packet_rules(South : 3) {
// CHECK-NEXT:     aie.rule(31, 5, %{{.+}}) {aie.priority_route}
// CHECK-NEXT:   }{{$}}
// CHECK:      aie.switchbox(%tile_0_2) {
// CHECK-NEXT:   %[[DMA:.+]] = aie.amsel<0> (0)
// CHECK:        aie.masterset(DMA : 0, %[[DMA]]){{$}}
// CHECK:        aie.packet_rules(South : 3) {
// CHECK:        } {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:   aie.packet_rules(South : 5) {
// CHECK-NEXT:     aie.rule(31, 5, %[[DMA]]){{$}}
// CHECK-NEXT:   }{{$}}

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
