//===- ctrl_pkt_to_dma_parallel_columns.mlir ----------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma="parallel-columns=true" --split-input-file | FileCheck %s

// Overlay-gated parallel-columns delivery, chained form (opt-in
// `parallel-columns=true`, requires `has_ctrl_pkt_overlay`). Each column's
// per-tile control BDs are delivered as ONE `next_bd`-chained transfer: one
// `aie.bd_chain` (one linear `aie.dma_bd` per controlled TILE -- a shim
// control BD carries one packet-id/tile -- `aie.next_bd`-linked), one
// `aiex.dma_start_bd_chain_for @chain(%payload) for @alloc` carrying
// `issue_token`, and one deferred `aiex.dma_await_task`. Config chains are
// pushed for ALL columns before ANY await (independent per-column shim MM2S
// channels overlap), then a join barrier of the config awaits, then the enable
// chains the same way (enable-last). Trailing teardown control packets are
// delivered the same chained, column-parallel way -- a single phase (no
// enable partition), emitted IN PLACE at their original position so any op
// that follows the teardown (e.g. a dma_free_task) keeps its position. No
// column sort and no cross-tile merge: the payload order (offsets) is
// unchanged from the serial path.
//
// Addresses are `col<<25 | row<<20 | offset` (npu2: columnShift=25,
// rowShift=20):
//   col0, row2, off=0x000   -> 2097152   (config, data=100)
//   col0, row3, off=0x000   -> 3145728   (config, data=101)
//   col1, row2, off=0x000   -> 35651584  (config, data=200)
//   col1, row3, off=0x000   -> 36700160  (config, data=201)
//   col0, row2, off=0x32000 -> 2301952   (enable, data bit0=1)
//   col1, row2, off=0x32000 -> 35856384  (enable, data bit0=1)
//   col0, row2, off=0x200   -> 2097664   (teardown, data=0, NOT a core-enable)
//   col1, row2, off=0x100   -> 35651840  (teardown, data=0, NOT a core-enable)
//
// ddrOffset (program order, each packet = 1 data word + 2 = 3), one BD per tile:
//   col0/row2 config: start=0  size=3    col0/row3 config: start=3  size=3
//   col1/row2 config: start=6  size=3    col1/row3 config: start=9  size=3
//   col0/row2 enable: start=12 size=3    col1/row2 enable: start=15 size=3
//   col0/row2 teardown: start=18 size=3  col1/row2 teardown: start=21 size=3
//   -- chained + column-parallel, emitted in place after the app op.

// Config chain defs: col0 (rows 2,3) chained; col1 (rows 2,3) chained.
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 0 len = 3)
// CHECK:   aie.next_bd ^{{.*}}
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 3 len = 3)
// CHECK:   aie.end
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 6 len = 3)
// CHECK:   aie.next_bd ^{{.*}}
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 9 len = 3)
// CHECK:   aie.end
// Enable chain defs: col0 (row2), col1 (row2).
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 12 len = 3)
// CHECK:   aie.end
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 15 len = 3)
// CHECK:   aie.end
// Teardown chain defs: col0 (row2), col1 (row2) -- same per-column chaining as
// config/enable (bd_chain defs live at device scope).
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 18 len = 3)
// CHECK:   aie.end
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 21 len = 3)
// CHECK:   aie.end
// CHECK: aie.runtime_sequence @m
// The interleaved `CHECK-NOT: npu.dma_memcpy_nd` scopes each gap between the
// positive matches through the config/enable region, so a per-tile memcpy
// spliced there is caught.
// CHECK-NOT: npu.dma_memcpy_nd
// Config phase: both columns' config chains pushed before ANY await (overlap).
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col0_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col1_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: npu.dma_memcpy_nd
// Config join barrier: the deferred config awaits.
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// Enable phase (enable-last): both enable chains pushed, then awaited.
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col0_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col1_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// Trailing non-control op survives, cloned.
// CHECK: arith.constant 42
// Teardown is now chained + column-parallel (mirrors config): one bd_chain per
// column (defs above, at device scope), both pushed with issue_token before
// either await, no serial memcpy.
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col0_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: aiex.npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col1_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK: aiex.dma_await_task
// CHECK: aiex.dma_await_task
// Emit-in-place: the op AFTER the teardown packets is cloned AFTER the
// teardown chains, not reordered before them.
// CHECK: arith.constant 43

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_2 = aie.tile(1, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_1_3 = aie.tile(1, 3) {ctrl_pkt_shim_chan = 0 : i32}
  aie.runtime_sequence @m() {
    aiex.control_packet {address = 2097152 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3145728 : ui32, data = array<i32: 101>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 35651584 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 36700160 : ui32, data = array<i32: 201>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    %c42_i32 = arith.constant 42 : i32
    aiex.control_packet {address = 2097664 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 35651840 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    %c43_i32 = arith.constant 43 : i32
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
  aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0)
} {has_ctrl_pkt_overlay = true}

// -----

// Single active column: col0 has two config tiles (rows 2,3) chained into one
// config transfer, and one enable tile (row 2) as the enable chain. Nothing to
// overlap with one column, but the chained schedule is uniform: one push + one
// await per phase.
//
// Addresses (`col<<25 | row<<20 | offset`, npu2):
//   col0, row2, off=0x000   -> 2097152   (config, data=100)
//   col0, row3, off=0x000   -> 3145728   (config, data=101)
//   col0, row2, off=0x32000 -> 2301952   (enable, data bit0=1)
// ddrOffset: col0/row2 config start=0 size=3; col0/row3 config start=3 size=3;
//            col0/row2 enable start=6 size=3.

// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 0 len = 3)
// CHECK:   aie.next_bd ^{{.*}}
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 3 len = 3)
// CHECK:   aie.end
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 6 len = 3)
// CHECK:   aie.end
// CHECK: aie.runtime_sequence @single
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col0_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_start_bd_chain_for @{{.*}}(%{{.*}}) : (memref<?xi32>) for {{.*}}@ctrlpkt_col0_mm2s_chan0 {{{.*}}issue_token = true{{.*}}}
// CHECK-NOT: npu.dma_memcpy_nd
// CHECK: aiex.dma_await_task
// CHECK-NOT: npu.dma_memcpy_nd

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
  aie.runtime_sequence @single() {
    aiex.control_packet {address = 2097152 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3145728 : ui32, data = array<i32: 101>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
} {has_ctrl_pkt_overlay = true}

// -----

// npu1 (AIE2): contiguous SAME-tile control packets must NOT coalesce -- the
// TLAST_Suppress packing the AIE2P path uses is not available on AIE2, so each
// packet keeps its own per-tile BD (matching the serial path's AIE2p-gated
// batching and the `ctrl_pkt_to_dma.mlir` "not combined on npu1" check). Two
// col0/row2 packets (each size 3) therefore stay two len=3 BDs, not one len=6.
// col0/row2 off=0x000 -> 2097152 ; col0/row2 off=0x004 -> 2097156
// col0/row3 off=0x000 -> 3145728 (a distinct tile -> its own BD either way).

// CHECK-LABEL: aie.device(npu1) {
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 0 len = 3)
// CHECK:   aie.next_bd ^{{.*}}
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 3 len = 3)
// CHECK:   aie.next_bd ^{{.*}}
// CHECK:   aie.dma_bd(%{{.*}} : memref<?xi32> offset = 6 len = 3)
// CHECK:   aie.end
// CHECK-NOT: len = 6

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
  aie.runtime_sequence @npu1_no_combine() {
    aiex.control_packet {address = 2097152 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2097156 : ui32, data = array<i32: 101>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3145728 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
} {has_ctrl_pkt_overlay = true}
