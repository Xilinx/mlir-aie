//===- ctrl_pkt_overlay_channel.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control overlay takes an MM2S channel on each column's shim tile, which
// it shares with the shim ends never in flight with its control packets. A
// control packet to column 0 between the inputs' transfers keeps it off their
// channels there, so the placer leaves column 0's shim room for one input.

// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s | FileCheck %s
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false})' %s | FileCheck %s

// CHECK-LABEL: module @overlay_interleaved
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = []}
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
module @overlay_interleaved {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
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

// -----

// Two columns for three inputs: column 1's overlay sends no control packet,
// so it shares a channel with one of the two inputs there.
// CHECK-LABEL: module @overlay_two_columns
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = []}
// CHECK: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%shim_noc_tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = [@b__shim_alloc, @d__shim_alloc]}
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%shim_noc_tile_1_0, MM2S, 0,
// CHECK-DAG: aie.shim_dma_allocation @d__shim_alloc(%shim_noc_tile_1_0, MM2S, 1)
module @overlay_two_columns {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c0}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c0}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s2, {%c1}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %d = aie.objectfifo.acquire @d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
      aiex.control_packet {address = 126976 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
    }
  }
}

// -----

// With no control packet, the overlay shares the one shim tile's channel with
// an input, so two inputs in flight together still fit its two channels.
// CHECK-LABEL: module @overlay_full_shim
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = [@in0__shim_alloc, @in1__shim_alloc]}
// CHECK-DAG: aie.shim_dma_allocation @in0__shim_alloc(%shim_noc_tile_0_0, MM2S, 0,
// CHECK-DAG: aie.shim_dma_allocation @in1__shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
module @overlay_full_shim {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%s, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%c, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @in0(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
      %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Consume, 1)
      aie.objectfifo.release @in1(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in0, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @in1, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @out, id = 2 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
