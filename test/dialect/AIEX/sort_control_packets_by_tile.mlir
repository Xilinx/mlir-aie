//===- sort_control_packets_by_tile.mlir --------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-sort-control-packets-by-tile --split-input-file | FileCheck %s

// The pass stable-sorts BOTH the leading config run AND the trailing teardown
// run by destination tile (col, then row). An app op (arith.constant) separates
// the two runs. Addresses: col<<25 | row<<20 | offset (npu2).
//   col0/row2 = 2097152   col0/row3 = 3145728
//   col1/row2 = 35651584  col1/row3 = 36700160
// teardown offsets are per-tile, not uniform (col/row extraction ignores the
// offset bits, so this doesn't affect behavior): col0/row2 = 2097664 is
// +0x200 from its config address; col1/row2 = 35651840 is +0x100 from its
// config address.

// Leading config run sorted col0<col1, row2<row3:
// CHECK: aie.runtime_sequence
// CHECK: aiex.control_packet {address = 2097152
// CHECK: aiex.control_packet {address = 3145728
// CHECK: aiex.control_packet {address = 35651584
// CHECK: arith.constant
// Trailing teardown run sorted col0 before col1:
// CHECK: aiex.control_packet {address = 2097664
// CHECK: aiex.control_packet {address = 35651840

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t02 = aie.tile(0, 2) {ctrl_pkt_shim_chan = 0 : i32}
  %t03 = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
  %t10 = aie.tile(1, 0)
  %t12 = aie.tile(1, 2) {ctrl_pkt_shim_chan = 0 : i32}
  aie.runtime_sequence @m() {
    // leading config, deliberately UNSORTED by tile:
    aiex.control_packet {address = 35651584 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 3145728 : ui32, data = array<i32: 101>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2097152 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    %c42 = arith.constant 42 : i32
    // trailing teardown, deliberately UNSORTED by tile:
    aiex.control_packet {address = 35651840 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 2097664 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%t00, MM2S, 0)
  aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%t10, MM2S, 0)
} {has_ctrl_pkt_overlay = true}

// -----

// Regression: `aie.runtime_sequence @empty() { }` is valid IR (RuntimeSequenceOp
// is NoTerminator), and a body written this way parses to a region with NO
// blocks at all -- so the pass's per-sequence scan, which reached into the
// region/block unconditionally (`Region::front()`/`Block::front()` on an
// empty region/block is UB), crashed. The pass loops over EVERY runtime
// sequence on the device, not just ctrlpkt ones, so it must pass through an
// empty one without crashing.

// CHECK: aie.runtime_sequence @empty

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  aie.runtime_sequence @empty() {
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%t00, MM2S, 0)
} {has_ctrl_pkt_overlay = true}
