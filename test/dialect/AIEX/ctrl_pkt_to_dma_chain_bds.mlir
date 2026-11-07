//===- ctrl_pkt_to_dma_chain_bds.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma=chain-bds=true --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-ctrl-packet-to-dma=chain-bds=true --split-input-file -verify-diagnostics -o /dev/null
// RUN: aie-opt %s "-aie-ctrl-packet-to-dma=chain-bds=true parallel-shims=true" --split-input-file -verify-diagnostics -o /dev/null
// RUN: aie-opt %s -aie-ctrl-packet-to-dma=chain-bds=true --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids -aie-dma-tasks-to-npu | FileCheck %s --check-prefix=NPU

// One run of generated batches on one channel is one task, one BD per batch (a
// batch is a run of packets to one tile), in stream order. Another op ends the
// run.

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
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @one_run() {
      %zero = arith.constant 0 : i32
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301956 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1048576 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301960 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301964 : ui32, data = array<i32: 5>, opcode = 0 : i32, stream_id = 0 : i32}
      }
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
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @two_columns() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 35856384 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301956 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      }
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
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @long_run() {
      aiex.control_packet_group @cfg(Config) {
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
}

// -----

// A read ends its batch and sends only its two header words.

// CHECK-LABEL: @read
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0 {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK:         aie.dma_bd({{.*}} offset = 3 len = 2)
// CHECK:         aie.dma_bd({{.*}} offset = 5 len = 3)
// CHECK-NEXT:    aie.end
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @read() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301956 : ui32, length = 1 : i32, opcode = 1 : i32, stream_id = 2 : i32}
        aiex.control_packet {address = 2301960 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// The shim tile's static DMA holds 14 of its 16 BDs, so a chain holds 2.

// CHECK-LABEL: @static_bds
// CHECK:       aiex.dma_configure_task_for
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK-NEXT:    aie.next_bd
// CHECK:         aie.dma_bd({{.*}} offset = 3 len = 3)
// CHECK-NEXT:    aie.end
// CHECK:       aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task_for
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK-NEXT:    aie.end
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %buf = aie.external_buffer {sym_name = "buf"} : memref<16xi32>
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    %dma = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 2 : i32}
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32, next_bd_id = 4 : i32}
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 5 : i32}
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32, next_bd_id = 6 : i32}
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 6 : i32, next_bd_id = 7 : i32}
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 7 : i32, next_bd_id = 8 : i32}
        aie.next_bd ^bd8
      ^bd8:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 8 : i32, next_bd_id = 9 : i32}
        aie.next_bd ^bd9
      ^bd9:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 9 : i32, next_bd_id = 10 : i32}
        aie.next_bd ^bd10
      ^bd10:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 10 : i32, next_bd_id = 11 : i32}
        aie.next_bd ^bd11
      ^bd11:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 11 : i32, next_bd_id = 12 : i32}
        aie.next_bd ^bd12
      ^bd12:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 12 : i32, next_bd_id = 13 : i32}
        aie.next_bd ^bd13
      ^bd13:
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 13 : i32, next_bd_id = 0 : i32}
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
    aie.runtime_sequence @static_bds() {
      aiex.control_packet_group @cfg(Config) {
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
    }
  }
}

// -----

// A runtime transfer in flight holds 10 of the shim tile's 16 BDs, so a chain
// holds 6.

// CHECK-LABEL: @in_flight
// CHECK:       aiex.dma_configure_task_for @data
// CHECK:       aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-COUNT-5: aie.next_bd
// CHECK:       aie.dma_bd({{.*}} offset = 15 len = 3)
// CHECK-NEXT:  aie.end
// CHECK:       aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 18 len = 3)
// CHECK-NEXT:    aie.end

// The BD ids then fit beside the transfer's.
// NPU-LABEL: @in_flight
// NPU:         aiex.npu.push_queue
// NPU:         aiex.npu.writebd {{.*}}bd_id = 15
// NPU:         aiex.npu.push_queue
module {
  aie.device(npu2_1col) @cfg {
  }
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @data (%tile_0_0, S2MM, 0)
    aie.runtime_sequence @in_flight(%arg0: memref<16xi32>) {
      %t = aiex.dma_configure_task_for @data {
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b1
      ^b1:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b2
      ^b2:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b3
      ^b3:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b4
      ^b4:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b5
      ^b5:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b6
      ^b6:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b7
      ^b7:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b8
      ^b8:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.next_bd ^b9
      ^b9:
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.control_packet_group @cfg(Config) {
        // expected-remark @+1 {{control packet chains on shim tile (0, 0) hold at most 6 buffer descriptors: runtime transfers there not yet awaited or freed hold 10}}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
        aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      }
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Control packets outside groups keep a transfer and a sync each.

// CHECK-LABEL: @user
// CHECK-NOT:   aiex.dma_configure_task_for
// CHECK:       aiex.npu.dma_memcpy_nd
// CHECK:       aiex.npu.sync
// CHECK:       aiex.npu.dma_memcpy_nd
// CHECK:       aiex.npu.sync
// CHECK-NOT:   aiex.dma_configure_task_for
module {
  aie.device(npu2_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @user() {
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
