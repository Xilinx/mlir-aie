//===- overlay_multicast_span.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true emit-standalone-overlay=true multicast-span=2" -verify-diagnostics | FileCheck %s

// With width 2, from the lowest core column on, each pair of columns with two
// or more core tiles gets one multicast, on its own id, entering by its lowest
// core tile's column; a pair with one core tile gets a remark.

// Columns 0-1 hold all four core tiles of @a and @b, reached from column 0.
// CHECK-LABEL: aie.device(npu2) @a
// CHECK-DAG:   aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK-DAG:   aie.tile(0, 5) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK:       aie.packet_flow(28) {
// CHECK-NEXT:    aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
// CHECK-COUNT-4: aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK-NEXT:  } {keep_pkt_header = true, priority_route = true}
module {
  aie.device(npu2) @a {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %c2 = aie.core(%t02) { aie.end }
    %c3 = aie.core(%t03) { aie.end }
  }
  aie.device(npu2) @b {
    %t05 = aie.tile(0, 5)
    %t12 = aie.tile(1, 2)
    %c5 = aie.core(%t05) { aie.end }
    %c12 = aie.core(%t12) { aie.end }
  }
}

// -----

// Width 2 from column 1. Columns 1-2 have two core tiles and one id; columns
// 3-4 have one, which gets a remark and no multicast.

// CHECK-LABEL: aie.device(npu2) @c
// CHECK-DAG:   aie.tile(1, 2) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK-DAG:   aie.tile(2, 3) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK-DAG:   aie.tile(3, 2){{$}}
// CHECK:       aie.packet_flow(28) {
// CHECK-NEXT:    aie.packet_source<%shim_noc_tile_1_0, DMA : 0>
// expected-remark @below {{control multicast: columns 3-4 have one core tile, which gets its own control packets}}
module {
  aie.device(npu2) @c {
    %t12 = aie.tile(1, 2)
    %t23 = aie.tile(2, 3)
    %t32 = aie.tile(3, 2)
    %c12 = aie.core(%t12) { aie.end }
    %c23 = aie.core(%t23) { aie.end }
    %c32 = aie.core(%t32) { aie.end }
  }
}

// -----

// From column 0, columns 0-1 have one core tile (a remark), and columns 2-3
// have two, both in column 3, so their multicast enters by column 3's channel.

// CHECK-LABEL: aie.device(npu2) @d
// CHECK-DAG:   aie.tile(0, 2){{$}}
// CHECK-DAG:   aie.tile(3, 2) {aie.ctrl_pkt_multicast_id = 28 : i32}
// CHECK:       aie.packet_flow(28) {
// CHECK-NEXT:    aie.packet_source<%shim_noc_tile_3_0, DMA : 0>
// expected-remark @below {{control multicast: columns 0-1 have one core tile, which gets its own control packets}}
module {
  aie.device(npu2) @d {
    %t02 = aie.tile(0, 2)
    %t32 = aie.tile(3, 2)
    %t33 = aie.tile(3, 3)
    %c02 = aie.core(%t02) { aie.end }
    %c32 = aie.core(%t32) { aie.end }
    %c33 = aie.core(%t33) { aie.end }
  }
}
