//===- ctrl_packet_to_dma_parallel_chain.mlir ---------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-ctrl-packet-to-dma="parallel-columns=true" %s | FileCheck %s

// Overlay-gated parallel-columns delivery, chained form. Each column's
// per-tile control BDs are delivered as ONE `next_bd`-chained transfer: one
// `aie.bd_chain` (one `aie.dma_bd` per controlled tile, `aie.next_bd` between
// them), one `aiex.dma_start_bd_chain_for @chain(%payload) for @alloc`
// carrying `issue_token`, and one deferred `aiex.dma_await_task`. The config
// chain (shim + memtile + 2 cores) and the enable chain (one core-enable)
// are separate chains, config-then-enable.
//
// Addresses are `col<<25 | row<<20 | offset` (npu2: columnShift=25,
// rowShift=20):
//   col0, row0, off=0x000   -> 0         (config, shim)
//   col0, row1, off=0x000   -> 1048576   (config, memtile)
//   col0, row2, off=0x000   -> 2097152   (config, core)
//   col0, row3, off=0x000   -> 3145728   (config, core)
//   col0, row2, off=0x32000 -> 2301952   (enable, data bit0=1)
//
// ddrOffset (program order, each packet = 1 data word + 2 = 3), one BD/tile:
//   config: row0 start=0 size=3; row1 start=3 size=3; row2 start=6 size=3;
//           row3 start=9 size=3.   enable: row2 start=12 size=3.

// The config chain def (device scope): one dma_bd per tile, next_bd-linked,
// offsets/lens drawn straight from the per-tile payload slices, no memcpy.
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK: aie.dma_bd(%{{.*}} : memref<?xi32> offset = 0 len = 3)
// CHECK: aie.next_bd ^{{.*}}
// CHECK: aie.dma_bd(%{{.*}} : memref<?xi32> offset = 3 len = 3)
// CHECK: aie.next_bd ^{{.*}}
// CHECK: aie.dma_bd(%{{.*}} : memref<?xi32> offset = 6 len = 3)
// CHECK: aie.next_bd ^{{.*}}
// CHECK: aie.dma_bd(%{{.*}} : memref<?xi32> offset = 9 len = 3)
// CHECK: aie.end
// The enable chain def (one tile), enable-last.
// CHECK: aie.bd_chain @{{.*}}(%{{.*}}: memref<?xi32>) {
// CHECK: aie.dma_bd(%{{.*}} : memref<?xi32> offset = 12 len = 3)
// CHECK: aie.end
// The runtime sequence: one push + one deferred await per phase (config then
// enable), each carrying issue_token, all on the same column alloc. The
// interleaved `CHECK-NOT: npu.dma_memcpy_nd` scopes each gap between the
// positive matches, so a per-tile memcpy spliced anywhere in the engaged
// region is caught (a trailing CHECK-NOT alone would only cover last-match..EOF
// and verify nothing).
// CHECK: aie.runtime_sequence @chain
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
  %tile_0_0 = aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_1 = aie.tile(0, 1) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_2 = aie.tile(0, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %tile_0_3 = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
  aie.runtime_sequence @chain() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 10>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 1048576 : ui32, data = array<i32: 11>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2097152 : ui32, data = array<i32: 12>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3145728 : ui32, data = array<i32: 13>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
} {has_ctrl_pkt_overlay = true}
