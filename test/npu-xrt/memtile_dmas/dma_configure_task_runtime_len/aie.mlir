//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dynamic counterpart of dma_configure_task_token: the same shim <-> mem
// tile round trip, but the mem tile's buffer descriptors are built from a
// dispatch-time tile count %n rather than from constants. Both the transfer
// length and the d1 wrap are runtime, so the mem tile BD is genuinely
// reprogrammed per dispatch -- one xclbin serves every n.
//
// Ordering is by task completion token: the host awaits the fill before
// configuring the drain, so no locks are needed between the two halves.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}

    %buf = aie.buffer(%tile_0_1) {sym_name = "buf"} : memref<4096xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    // Return path for the mem tile's and shim's completion tokens.
    aie.packet_flow(0x1) {
      aie.packet_source<%tile_0_1, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    aie.packet_flow(0x2) {
      aie.packet_source<%tile_0_0, "TileControl" : 0>
      aie.packet_dest<%tile_0_0, "South" : 0>
    } {keep_pkt_header = true, priority_route = true}

    // %n = dispatch-time tile count, 1..8 over a 512-element tile.
    aie.runtime_sequence(%in: memref<4096xi32>, %out: memref<4096xi32>, %n: i64) {
      %c512_i32 = arith.constant 512 : i32
      %n_i32 = arith.trunci %n : i64 to i32
      %len = arith.muli %n_i32, %c512_i32 : i32

      // Every task below sits on channel 0, which is even, so each mem tile BD
      // id has to come from the low half of the pool (isBdChannelAccessible).
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 512] strides = [0, 0, 512, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 512] strides = [0, 0, 512, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)

      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 512] strides = [0, 0, 512, 1]) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%out : memref<4096xi32> offset = 0 len = %len sizes = [1, 1, %n, 512] strides = [0, 0, 512, 1]) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_await_task(%t2)
    }
  }
}
