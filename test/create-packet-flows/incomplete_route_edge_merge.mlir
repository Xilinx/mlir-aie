//===- incomplete_route_edge_merge.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --verify-diagnostics %s

// Regression test: the pathfinder must not SILENTLY drop a packet_source.
//
// Five same-id packet sources merge into a single mem_tile S2MM DMA channel.
// Same-id flows cannot share a switch channel, so each source needs its own path
// to the destination. The congestion-aware pathfinder does not find a complete
// assignment for this placement and drops one source's connection, producing an
// incomplete routing. The design used to compile "successfully" while silently
// delivering only four of the five sources on device. The pass must instead fail
// loudly.

module {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    // expected-error @+1 {{could not be routed to destination}}
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
