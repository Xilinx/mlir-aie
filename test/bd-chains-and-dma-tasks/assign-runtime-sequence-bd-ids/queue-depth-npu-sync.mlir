//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-to-npu='enforce-queue-depth=false' --verify-diagnostics --split-input-file %s

// aiex.npu.sync is the lowered form of npu.dma_wait and the same hardware
// event, so it drains the task queue by the same rule. A sequence that has
// already been lowered past dma_wait, or that reaches for the raw op, would
// otherwise look like it never drains a channel at all.
//
// Its six operands are the channel key: column, row, direction, channel,
// column_num, row_num, with direction 0 = S2MM and 1 = MM2S.

// Push-then-sync, four times over. Each sync retires the push ahead of it, so
// the queue is empty when the fifth push lands.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// The shape of test/npu-xrt/add_one_ctrl_packet_col_overlay, reduced. That
// design pushes control packets on shim MM2S 1 and syncs after every one; it
// was the only design in the whole in-tree corpus to trip the queue check, and
// it was a false positive. It needs hardware to run, so this stands in for it.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @ctrlin1 (%tile_0_0, MM2S, 1)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_id = 29, pkt_type = 1>) {id = 0 : i64, issue_token = true, metadata = @ctrlin1} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 4][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_id = 30, pkt_type = 1>) {id = 1 : i64, issue_token = true, metadata = @ctrlin1} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 8][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_id = 31, pkt_type = 1>) {id = 2 : i64, issue_token = true, metadata = @ctrlin1} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 12][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_id = 29, pkt_type = 1>) {id = 3 : i64, issue_token = true, metadata = @ctrlin1} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16][1, 1, 1, 2][0, 0, 0, 1], packet = <pkt_id = 30, pkt_type = 1>) {id = 4 : i64, issue_token = true, metadata = @ctrlin1} : memref<1280xi32>
  }
}

// -----

// A runtime-valued channel cannot be matched against a queue, so it retires
// nothing. Over-reporting is the safe direction: a spurious warning costs a
// reader's attention, while crediting a drain that may not be on this channel
// would hide the hang the check exists to find.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>, %chan: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %chan, %c1, %c1) : i32, i32, i32, i32, i32, i32
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// A sync spanning a range of tiles retires nothing either. Whether the
// firmware waits for one token per covered tile or one for the range is not
// settled anywhere in tree, and crediting nothing is the only reading that is
// sound under both.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c0, %c2, %c1) : i32, i32, i32, i32, i32, i32
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// A sync on a different channel of the same tile drains a different queue.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// Direction is part of the key too: an S2MM sync says nothing about the MM2S
// queue these pushes went onto. This is what pins the 0 = S2MM, 1 = MM2S
// numbering the key relies on.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}
