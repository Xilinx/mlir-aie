//===- ctrl_pkt_overlay_shared_source.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet flow 3 leaves shim (0,0) by DMA:1, so the column's control packets
// take DMA:0 and the two share no slave port of the shim's switchbox (#3837).

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s --check-prefixes=CHECK,DESIGN
// RUN: sed -n '/@ctrl_pkt_overlay {/,/^  }/p' %t | FileCheck %s
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_0_0) {
// CHECK:         aie.connect<DMA : 0, North : 3>
// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK-DAG:     %[[A52:.+]] = aie.amsel<5> (2)
// CHECK-DAG:     aie.masterset(North : 3, %[[A52]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(24, 24, %[[A52]]) {aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 15, %{{.+}}) {aie.priority_route}
// CHECK-NEXT:    } {aie.is_ctrl_pkt_overlay}
// DESIGN:        aie.packet_rules(South : 7) {
// DESIGN-NEXT:     aie.rule(31, 3, %{{.+}}){{$}}
// DESIGN-NEXT:   }{{$}}

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
    %t01 = aie.tile(0, 1)
    %t10 = aie.tile(1, 0)
    %t11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t10, DMA : 1, %t01, DMA : 0)
    aie.flow(%t01, DMA : 0, %t10, DMA : 0)
    aie.flow(%t01, DMA : 1, %t02, DMA : 0)
    aie.flow(%t02, DMA : 0, %t01, DMA : 1)
    aie.flow(%t01, DMA : 2, %t03, DMA : 0)
    aie.flow(%t03, DMA : 0, %t01, DMA : 2)
    aie.flow(%t01, DMA : 3, %t04, DMA : 0)
    aie.flow(%t04, DMA : 0, %t01, DMA : 3)
    aie.flow(%t01, DMA : 4, %t05, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 4)
    aie.flow(%t01, DMA : 5, %t02, DMA : 1)
    aie.packet_flow(3) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t11, DMA : 0>
    }
  }
}
