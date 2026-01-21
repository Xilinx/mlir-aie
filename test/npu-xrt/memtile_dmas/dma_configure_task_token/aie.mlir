//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %in_buff = aie.buffer(%tile_0_1) {sym_name = "in_buff"} : memref<4096xi32>
    %out_buff = aie.buffer(%tile_0_1) {sym_name = "out_buff"} : memref<4096xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    // Packet flows for issue_token support
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_1, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // Configure shim DMA to send data to memtile
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      // Configure memtile DMA to receive from shim
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%in_buff : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)

      // Configure memtile DMA to send data to shim
      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%in_buff : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      // Configure shim DMA to receive data from memtile
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_await_task(%t2)
    }
  }
}
