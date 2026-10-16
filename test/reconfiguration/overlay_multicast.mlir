//===- overlay_multicast.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=true multicast-span=1" | FileCheck %s
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=true multicast-span=1" --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=false multicast-span=1" | FileCheck %s --check-prefix=AGAIN
// RUN: not aie-opt %s --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true multicast-span=-1" 2>&1 | FileCheck %s --check-prefix=WIDTH

// Column 0's core tiles over both configurations are rows 2, 3 and 5, and one
// multicast from its control channel reaches all three, each stamped with its
// id. Column 1 has one core tile, so no multicast. The id, 25, is the highest
// free one: no controller id (15, 26, 27, 29, 30, 31 here) and no packet flow
// (28, in @a) uses it.

// CHECK-LABEL: aie.device(npu2) @a
// CHECK-DAG:   aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 25 : i32}
// CHECK-DAG:   aie.tile(0, 3) {aie.ctrl_pkt_multicast_id = 25 : i32}
// CHECK-DAG:   aie.tile(0, 5) {aie.ctrl_pkt_multicast_id = 25 : i32}
// CHECK:       aie.packet_flow(25) {
// CHECK-NEXT:    aie.packet_source<%{{.*}}, DMA : 0>
// CHECK-NEXT:    aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK-NEXT:    aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK-NEXT:    aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK-NEXT:  } {keep_pkt_header = true, priority_route = true}
// CHECK-NOT:   aie.packet_flow(25)
// CHECK-LABEL: aie.device(npu2) @b
// CHECK-DAG:   aie.tile(1, 2){{$}}
// CHECK-DAG:   aie.packet_flow(25)
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK-DAG:   aie.tile(1, 2){{$}}
// CHECK-DAG:   aie.packet_flow(25)

// A second run keeps the id and adds no second multicast.
// AGAIN-LABEL: aie.device(npu2) @a
// AGAIN-COUNT-1: aie.packet_flow(25)
// AGAIN-NOT:   aie.packet_flow(25)
// AGAIN-LABEL: aie.device(npu2) @b

// WIDTH: multicast-span -1 is negative


module {
  aie.device(npu2) @a {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %c2 = aie.core(%t02) { aie.end }
    %c3 = aie.core(%t03) { aie.end }
    aie.packet_flow(28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
  }
  aie.device(npu2) @b {
    %t03 = aie.tile(0, 3)
    %t05 = aie.tile(0, 5)
    %t12 = aie.tile(1, 2)
    %c3 = aie.core(%t03) { aie.end }
    %c5 = aie.core(%t05) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
}

// -----

// Two columns with two core tiles each, so each gets its own id and the two
// trees share none.

// CHECK-LABEL: aie.device(npu2) @e
// CHECK-DAG:   aie.tile(0, 2) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK-DAG:   aie.tile(1, 3) {aie.ctrl_pkt_multicast_id = 25 : i32}
// CHECK-DAG:   aie.packet_flow(28)
// CHECK-DAG:   aie.packet_flow(25)
module {
  aie.device(npu2) @e {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %c02 = aie.core(%t02) { aie.end }
    %c03 = aie.core(%t03) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
    %c13 = aie.core(%t13) { aie.end }
  }
}
