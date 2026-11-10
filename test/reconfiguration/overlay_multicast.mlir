//===- overlay_multicast.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=1" | FileCheck %s
// RUN: aie-opt %s --split-input-file --aie-generate-column-control-overlay="configuration=true adaptive=false multicast-span=2" | FileCheck %s --check-prefix=SPAN2

// Column 0's core tiles over both configurations are rows 2, 3 and 5, and one
// multicast from its control channel reaches all three. Column 1 has one core
// tile, so no multicast. The id, 25, is the highest free one: no controller id
// (15, 26, 27, 29, 30, 31 here) and no packet flow (28, in @a) uses it. With
// span 2, columns 0 and 1 are one multicast.

// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:       aie.ctrl_pkt_multicast = [{{.}}column = 0 : i32, id = 25 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>, #aie.tile<0, 5>]
// SPAN2-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// SPAN2:       aie.ctrl_pkt_multicast = [{{.}}column = 0 : i32, id = 25 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>, #aie.tile<0, 5>, #aie.tile<1, 2>]

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

// Every multicast goes on id 28, the highest free one; the router keeps each
// column's tree in its column.

// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:         aie.switchbox(%tile_1_2) {
// CHECK-NOT:     aie.switchbox(
// CHECK:           aie.packet_rules(South : 0) {
// CHECK-NEXT:        aie.rule(31, 28,
// CHECK:       aie.ctrl_pkt_multicast = [
// CHECK-SAME:    column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>]
// CHECK-SAME:    column = 1 : i32, id = 28 : i32, tiles = [#aie.tile<1, 2>, #aie.tile<1, 3>]
// CHECK-SAME:    column = 2 : i32, id = 28 : i32,
// CHECK-SAME:    column = 3 : i32, id = 28 : i32,
// CHECK-SAME:    column = 4 : i32, id = 28 : i32,
// CHECK-SAME:    column = 5 : i32, id = 28 : i32,
// CHECK-SAME:    column = 6 : i32, id = 28 : i32,
// CHECK-SAME:    column = 7 : i32, id = 28 : i32, tiles = [#aie.tile<7, 2>, #aie.tile<7, 3>]
// SPAN2-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// SPAN2:       aie.ctrl_pkt_multicast = [
// SPAN2-SAME:    column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<0, 3>, #aie.tile<1, 2>, #aie.tile<1, 3>]
// SPAN2-SAME:    column = 6 : i32, id = 28 : i32, tiles = [#aie.tile<6, 2>, #aie.tile<6, 3>, #aie.tile<7, 2>, #aie.tile<7, 3>]

module {
  aie.device(npu2) @wide {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %c02 = aie.core(%t02) { aie.end }
    %c03 = aie.core(%t03) { aie.end }
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %c12 = aie.core(%t12) { aie.end }
    %c13 = aie.core(%t13) { aie.end }
    %t22 = aie.tile(2, 2)
    %t23 = aie.tile(2, 3)
    %c22 = aie.core(%t22) { aie.end }
    %c23 = aie.core(%t23) { aie.end }
    %t32 = aie.tile(3, 2)
    %t33 = aie.tile(3, 3)
    %c32 = aie.core(%t32) { aie.end }
    %c33 = aie.core(%t33) { aie.end }
    %t42 = aie.tile(4, 2)
    %t43 = aie.tile(4, 3)
    %c42 = aie.core(%t42) { aie.end }
    %c43 = aie.core(%t43) { aie.end }
    %t52 = aie.tile(5, 2)
    %t53 = aie.tile(5, 3)
    %c52 = aie.core(%t52) { aie.end }
    %c53 = aie.core(%t53) { aie.end }
    %t62 = aie.tile(6, 2)
    %t63 = aie.tile(6, 3)
    %c62 = aie.core(%t62) { aie.end }
    %c63 = aie.core(%t63) { aie.end }
    %t72 = aie.tile(7, 2)
    %t73 = aie.tile(7, 3)
    %c72 = aie.core(%t72) { aie.end }
    %c73 = aie.core(%t73) { aie.end }
  }
}
