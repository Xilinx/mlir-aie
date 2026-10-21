//===- ctrl_pkt_to_dma_chain_bds.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma=chain-bds=true --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-ctrl-packet-to-dma=chain-bds=true --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids -aie-dma-tasks-to-npu | FileCheck %s --check-prefix=NPU

// One run of batches on one channel is one task, one BD per batch (a batch
// is a run of packets to one tile), in stream order. Another op ends the run.

// CHECK-LABEL: @one_run
// CHECK:       %[[T:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd(%[[ARG:.*]] : memref<?xi32> offset = 0 len = 6)
// CHECK-NEXT:    aie.next_bd ^[[B1:.*]]
// CHECK-NEXT:  ^[[B1]]:
// CHECK-NEXT:    aie.dma_bd(%[[ARG]] : memref<?xi32> offset = 6 len = 3)
// CHECK-NEXT:    aie.next_bd ^[[B2:.*]]
// CHECK-NEXT:  ^[[B2]]:
// CHECK-NEXT:    aie.dma_bd(%[[ARG]] : memref<?xi32> offset = 9 len = 3)
// CHECK-NEXT:    aie.end
// CHECK-NEXT:  } {issue_token = true}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]])
// CHECK-NEXT:  aiex.dma_await_task(%[[T]])
// CHECK-NEXT:  aiex.npu.write32
// CHECK-NEXT:  %[[U:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd(%[[ARG]] : memref<?xi32> offset = 12 len = 3)
// CHECK-NEXT:    aie.end
// CHECK-NEXT:  } {issue_token = true}
// CHECK-NEXT:  aiex.dma_start_task(%[[U]])
// CHECK-NEXT:  aiex.dma_await_task(%[[U]])
// CHECK-NOT:   aiex.npu.dma_memcpy_nd

// After lowering, each task gets one push and one sync, with its BDs linked.
// NPU-LABEL: @one_run
// NPU:         aiex.npu.writebd {{.*}}bd_id = 0 {{.*}}next_bd = 1 {{.*}}use_next_bd = 1
// NPU:         aiex.npu.writebd {{.*}}bd_id = 1 {{.*}}next_bd = 2 {{.*}}use_next_bd = 1
// NPU:         aiex.npu.writebd {{.*}}bd_id = 2 {{.*}}use_next_bd = 0
// NPU:         aiex.npu.push_queue
// NPU:         aiex.npu.sync
// NPU:         aiex.npu.write32
// NPU:         aiex.npu.writebd
// NPU:         aiex.npu.push_queue
// NPU:         aiex.npu.sync
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @one_run() {
      %zero = arith.constant 0 : i32
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301960 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet {address = 2301964 : ui32, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A change of channel ends a chain, and each channel's batches stay in stream
// order relative to the other channel's.

// CHECK-LABEL: @two_columns
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK-NEXT:    aie.end
// CHECK:       aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task_for @ctrlpkt_col1_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 3 len = 3)
// CHECK-NEXT:    aie.end
// CHECK:       aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK-NEXT:    aie.end
module {
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0)
    aie.runtime_sequence @two_columns() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A chain holds at most the shim tile's 16 BDs, so 17 batches are two tasks.

// CHECK-LABEL: @long_run
// CHECK:       aiex.dma_configure_task_for
// CHECK-COUNT-15: aie.next_bd
// CHECK:       aie.dma_bd({{.*}} offset = 45 len = 3)
// CHECK-NEXT:  aie.end
// CHECK:       aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task_for
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 48 len = 3)
// CHECK-NEXT:    aie.end
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @long_run() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
