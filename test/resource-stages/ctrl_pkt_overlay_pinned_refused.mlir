//===- ctrl_pkt_overlay_pinned_refused.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Designs the control overlay leaves no room for, refused by both placers and
// by allocation. The sequential placer's diagnostic says what the overlay
// takes.

// RUN: not aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s 2>&1 | FileCheck %s --check-prefix=SEQ
// RUN: not aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=1})' %s 2>&1 | FileCheck %s --check-prefix=SA

// A circuit-switched end pins MM2S 1, so the overlay takes channel 0, which
// packet end @a pins too but cannot share: @a is in flight with the control
// packet.
// SEQ: error: no ShimNOCTile on the device has 0 input/2 output DMA channel(s) free
// SEQ: note: on 1 ShimNOCTile(s), with this tile's pinned ends, the control overlay has no output channel it may take or share
// SA: warning: SA placer's resource estimate is over capacity
// SA: error: 'aie.route_endpoint' op pinned MM2S DMA channel 0 is out of range or already in use on this tile
module @overlay_pinned_in_flight {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
  }
}

// -----

// Both inputs are in flight with the control packet, so the overlay takes a
// channel of its own and they need two more.
// SEQ: error: no ShimNOCTile on the device has 0 input/2 output DMA channel(s) free: all 1 ShimNOCTile(s) are at 0/2 input, 1/2 output channels used
// SEQ: note: 1 of the output channels used are the control overlay's, which aie-generate-column-control-overlay shares only with ends never in flight with its control packets
// SA: warning: SA placer's resource estimate is over capacity
// SA: error: 'aie.tile' op number of output DMA channel exceeded! requires at least 3 MM2S channels, but capacity is 2
module @overlay_own_channel_refused {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}
