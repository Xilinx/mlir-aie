//===- same_id_fanin_5way_memtile_dest.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test: the pathfinder must not drop a packet_source, silently or
// otherwise. Companion to same_id_fanin_4way_core_dest.mlir, which is the same
// bug at a core-tile destination; a mem tile has no East/West ports and
// passes North<->South only on a matching channel, so the two exercise
// different switchbox topologies.
//
// Five same-id packet sources merge into a single mem_tile S2MM DMA channel.
// Same-id flows may not share a switch channel, so each source is pushed onto
// its own crossbar entry into (0,1) DMA0. That congestion used to drive
// Dijkstra onto a detour that leaves a switchbox on some "South N" and then
// re-enters the same switchbox on that same "South N" -- expressible only
// because a graph node is (tile, bundle, channel) with no direction. The
// emitted route then dead-ended and one source was reported unroutable.
//
// This file used to be named incomplete_route_edge_merge.mlir and asserted that
// error, on the assumption that no complete assignment existed here. One does:
// the route below is found by a search that is strictly more constrained than
// the one that failed, over an unchanged graph. The pass still reports a
// genuinely unroutable design loudly -- see
// create-flows/unreachable_dest_err_test.mlir.
//
// All five sources must arrive and merge onto DMA0.

// Each source must land on its OWN input port -- same-id flows may not share a
// channel -- and all five merge onto the one DMA : 0 endpoint via one amsel.
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-NEXT:    %[[AMSEL:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:    aie.masterset(DMA : 0, %[[AMSEL]]) {keep_pkt_header = true}
// CHECK-NEXT:    aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 0, %[[AMSEL]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 0, %[[AMSEL]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 2) {
// CHECK-NEXT:      aie.rule(31, 0, %[[AMSEL]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 0, %[[AMSEL]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 0, %[[AMSEL]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }

module {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_2_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_3_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_4_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_5_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_6_0, DMA : 0>
      aie.packet_dest<%mem_tile_0_1, DMA : 0>
    } {keep_pkt_header = true}
  }
}
