//===- same_id_fanin_4way_core_dest.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Second geometry for the direction-alias routing bug, next to
// same_id_fanin_5way_memtile_dest.mlir. That one gathers five shim sources
// into a mem tile; this one gathers four into a core tile, which has a
// different switchbox topology (a mem tile has no East/West ports and passes
// North<->South on a matching channel). Both used to fail, so pinning only one
// would let a fix that over-fits its shape through.
//
// The router keys its graph on (tile, bundle, channel) with no direction, so a
// single node stood for both a switchbox port's input side and its output side
// and a path could take two crossbar hops in a row -- turning the stream around
// inside one switchbox. Same-id fan-in is what surfaces it: same-id flows may
// not share a channel, so each extra source is pushed off the cheap direct
// entry until the two-edge aliased detour looks cheaper. The route that came
// back then dead-ended and a source was reported unroutable.
//
// Note the threshold is congestion-dependent, not monotonic: before the fix
// four and five sources here failed but six routed. Four is pinned because it
// is the narrowest failing case at this geometry.
//
// All four sources must arrive, each on its own input port, and merge onto the
// single DMA : 0 endpoint through one amsel.

// CHECK-LABEL: aie.device(npu2)
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      aie.switchbox(%[[tile_0_2]]) {
// CHECK-NEXT:   %[[a:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(DMA : 0, %[[a]]) {keep_pkt_header = true}
// CHECK-NEXT:   aie.packet_rules(North : 3) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(East : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(South : 4) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu2) {
    %d = aie.tile(0, 2)
    %s0 = aie.tile(2, 0)
    %s1 = aie.tile(3, 0)
    %s2 = aie.tile(4, 0)
    %s3 = aie.tile(5, 0)
    aie.packet_flow(0) {
      aie.packet_source<%s0, DMA : 0>
      aie.packet_dest<%d, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%s1, DMA : 0>
      aie.packet_dest<%d, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%s2, DMA : 0>
      aie.packet_dest<%d, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%s3, DMA : 0>
      aie.packet_dest<%d, DMA : 0>
    } {keep_pkt_header = true}
  }
}
