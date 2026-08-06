//===- incomplete_route_edge_merge.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --verify-diagnostics %s

// Regression test: the pathfinder must not SILENTLY drop a packet_source.
//
// This is a high-fan-in merge (5 shim DMA senders in cols 2..6 ->
// a single memtile S2MM at (0,1) DMA:0) with the mem_tile placed at the column-0
// edge and a per-column control-packet overlay competing for switch resources.
// For this one-sided-edge placement the congestion-aware router produces a
// solution whose per-tile switch settings do not form a complete source->dest
// path for the col2 and col4 senders: findPathToDest() rejects those source
// connections in AIECreatePathFindFlows, so no shim_mux is emitted for them.
// The design used to compile "successfully" while silently delivering only 3 of
// the 5 packets on device. The pass must instead fail loudly. (A central
// mem_tile placement routes all 5 and is unaffected.)

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_3_0 = aie.tile(3, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_4_0 = aie.tile(4, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_5_0 = aie.tile(5, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_6_0 = aie.tile(6, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
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
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_2_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_2_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_3_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_3_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_4_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_4_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_5_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_5_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_6_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_6_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
