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
// The router keyed its graph on (tile, bundle, channel) with no direction, so a
// single node stood for both a switchbox port's input side and its output side
// and a path could take two crossbar hops in a row -- turning the stream around
// inside one switchbox. Same-id fan-in is what surfaced it: same-id flows could
// not share a channel, so each extra source is pushed off the cheap direct
// entry until the two-edge aliased detour looks cheaper. The route that came
// back then dead-ended and a source was reported unroutable.
//
// Same-id flows to the same destination now merge, so this design no longer
// crowds the router at all: each shim folds its own DMA:0 into the stream
// passing West, and one stream reaches the core. The test still pins that all
// four sources arrive.

// CHECK-LABEL: aie.device(npu2)
// CHECK-LABEL: aie.switchbox(%tile_0_2) {
// CHECK-NEXT:   %[[a:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(DMA : 0, %[[a]]) {keep_pkt_header = true}
// CHECK-NEXT:   aie.packet_rules({{[A-Za-z]+}} : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_2_0) {
// CHECK-NEXT:   aie.connect<DMA : 0, North : [[N2:[0-9]+]]>
// CHECK:        aie.switchbox(%shim_noc_tile_2_0) {
// CHECK-NEXT:   %[[S2:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West : {{[0-9]+}}, %[[S2]])
// CHECK-NEXT:   aie.packet_rules(East : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S2]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(South : [[N2]]) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S2]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_3_0) {
// CHECK-NEXT:   aie.connect<DMA : 0, North : [[N3:[0-9]+]]>
// CHECK:        aie.switchbox(%shim_noc_tile_3_0) {
// CHECK-NEXT:   %[[S3:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West : {{[0-9]+}}, %[[S3]])
// CHECK-NEXT:   aie.packet_rules(East : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S3]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(South : [[N3]]) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S3]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_4_0) {
// CHECK-NEXT:   aie.connect<DMA : 0, North : [[N4:[0-9]+]]>
// CHECK:        aie.switchbox(%shim_noc_tile_4_0) {
// CHECK-NEXT:   %[[S4:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West : {{[0-9]+}}, %[[S4]])
// CHECK-NEXT:   aie.packet_rules(East : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S4]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(South : [[N4]]) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S4]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_5_0) {
// CHECK-NEXT:   aie.connect<DMA : 0, North : [[N5:[0-9]+]]>
// CHECK:        aie.switchbox(%shim_noc_tile_5_0) {
// CHECK-NEXT:   %[[S5:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(West : {{[0-9]+}}, %[[S5]])
// CHECK-NEXT:   aie.packet_rules(South : [[N5]]) {
// CHECK-NEXT:     aie.rule(31, 0, %[[S5]])
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
