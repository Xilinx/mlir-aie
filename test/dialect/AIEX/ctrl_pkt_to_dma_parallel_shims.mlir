//===- ctrl_pkt_to_dma_parallel_shims.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma=parallel-shims=true --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids=chain-tasks=false | FileCheck %s --implicit-check-not=aiex.ctrl_pkt_transfer --implicit-check-not=aiex.set_device --implicit-check-not=defer_completion
// RUN: aie-opt %s -aie-ctrl-packet-to-dma="parallel-shims=true chain-bds=true" --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids=chain-tasks=true | FileCheck %s --check-prefix=CHAIN --implicit-check-not=aiex.ctrl_pkt_transfer --implicit-check-not=aiex.set_device --implicit-check-not=defer_completion
// RUN: aie-opt %s -aie-ctrl-packet-to-dma=parallel-shims=true --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids=chain-tasks=false -aie-dma-tasks-to-npu 2>&1 | FileCheck %s --check-prefix=NPU --implicit-check-not=warning --implicit-check-not=error --implicit-check-not=maskpoll32 --implicit-check-not=aiex.set_device
// RUN: aie-opt %s -aie-ctrl-packet-to-dma="parallel-shims=true chain-bds=true" --split-input-file -aie-materialize-bd-chains -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids=chain-tasks=true -aie-dma-tasks-to-npu 2>&1 | FileCheck %s --check-prefix=NPU --implicit-check-not=warning --implicit-check-not=error --implicit-check-not=maskpoll32 --implicit-check-not=aiex.set_device

// The lowered sequences need no queue guard and no BD the shim does not have.
// NPU: aiex.npu.push_queue

// Every channel's transfers are started, then all awaited; the packets after
// the barrier follow as a second phase. Without chain-bds each transfer is its
// own task; with it, a channel's transfers in a phase are one chain.

// CHECK-LABEL: @two_columns
// CHECK:       %[[A0:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK:       aiex.dma_start_task(%[[A0]])
// CHECK-NEXT:  %[[B0:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 3 len = 3)
// CHECK:       aiex.dma_start_task(%[[B0]])
// CHECK-NEXT:  %[[A1:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK:       aiex.dma_start_task(%[[A1]])
// CHECK-NEXT:  aiex.dma_await_task(%[[A0]])
// CHECK-NEXT:  aiex.dma_await_task(%[[A1]])
// CHECK-NEXT:  aiex.dma_await_task(%[[B0]])
// CHECK-NEXT:  %[[A2:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 9 len = 6)
// CHECK:       aiex.dma_start_task(%[[A2]])
// CHECK-NEXT:  %[[B1:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 15 len = 3)
// CHECK:       aiex.dma_start_task(%[[B1]])
// CHECK-NEXT:  aiex.dma_await_task(%[[A2]])
// CHECK-NEXT:  aiex.dma_await_task(%[[B1]])
// CHECK-NEXT:  aiex.npu.write32
// CHECK-NEXT:  %[[A3:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 18 len = 3)
// CHECK:       aiex.dma_start_task(%[[A3]])
// CHECK-NEXT:  aiex.dma_await_task(%[[A3]])
// CHECK-NOT:   aiex.npu.dma_memcpy_nd

// CHAIN-LABEL: @two_columns
// CHAIN:       %[[A:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHAIN-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHAIN-NEXT:    aie.next_bd
// CHAIN-NEXT:  ^bb{{[0-9]+}}:
// CHAIN-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHAIN-NEXT:    aie.end
// CHAIN:       aiex.dma_start_task(%[[A]])
// CHAIN-NEXT:  %[[B:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHAIN-NEXT:    aie.dma_bd({{.*}} offset = 3 len = 3)
// CHAIN-NEXT:    aie.end
// CHAIN:       aiex.dma_start_task(%[[B]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[A]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[B]])
// CHAIN-NEXT:  %[[C:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHAIN-NEXT:    aie.dma_bd({{.*}} offset = 9 len = 6)
// CHAIN-NEXT:    aie.end
// CHAIN:       aiex.dma_start_task(%[[C]])
// CHAIN-NEXT:  %[[D:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHAIN:       aiex.dma_start_task(%[[D]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[C]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[D]])
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @two_columns() {
      %zero = arith.constant 0 : i32
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856388 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2219540 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301952 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856384 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.write32(%zero, %zero) : i32, i32
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 4>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// Without chains a channel keeps as many tasks outstanding as its queue holds
// (4), so the fifth transfer waits for the first.

// CHECK-LABEL: @queue_window
// CHECK-COUNT-4: aiex.dma_start_task
// CHECK-NEXT:  aiex.dma_await_task
// CHECK-NEXT:  aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 12 len = 3)
// CHECK:       aiex.dma_start_task
// CHECK-COUNT-4: aiex.dma_await_task
// CHECK-NOT:   aiex.dma_start_task

// CHAIN-LABEL: @queue_window
// CHAIN-COUNT-1: aiex.dma_start_task
// CHAIN-NEXT:  aiex.dma_await_task
// CHAIN-NOT:   aiex.dma_start_task
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @queue_window() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A chain holds at most the shim tile's 16 BDs, and a channel keeps one chain
// outstanding, so the second chain on column 0 waits for the first while column
// 1's chain runs beside both.

// CHECK-LABEL: @long_phase

// CHAIN-LABEL: @long_phase
// CHAIN:       %[[A:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHAIN-COUNT-15: aie.next_bd
// CHAIN:       aiex.dma_start_task(%[[A]])
// CHAIN-NEXT:  %[[B:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHAIN:       aiex.dma_start_task(%[[B]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[A]])
// CHAIN-NEXT:  %[[C:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHAIN-NEXT:    aie.dma_bd({{.*}} offset = 51 len = 3)
// CHAIN-NEXT:    aie.end
// CHAIN:       aiex.dma_start_task(%[[C]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[C]])
// CHAIN-NEXT:  aiex.dma_await_task(%[[B]])
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @long_phase() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 1048576 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35856388 : ui32, data = array<i32: 3>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 2301956 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// A configuration's teardown lands in every column before the next
// configuration's packets start.

// CHECK-LABEL: @drain
// CHECK-NEXT:  %[[T0:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK:       %[[T1:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 3 len = 3)
// CHECK:       aiex.dma_await_task(%[[T0]])
// CHECK-NEXT:  aiex.dma_await_task(%[[T1]])
// CHECK-NEXT:  %[[C0:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK:       %[[C1:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0) {
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 9 len = 3)
// CHECK:       aiex.dma_await_task(%[[C0]])
// CHECK-NEXT:  aiex.dma_await_task(%[[C1]])
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) @next {
  }
  aie.device(npu2_2col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_1_2 = aie.tile(1, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0 (%tile_1_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @drain() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35909636 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.set_device @next
      aiex.control_packet_barrier {independent}
      aiex.control_packet {address = 2355204 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35909636 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

module {
  aie.device(npu2) @overlay {
  } {aie.ctrl_pkt_multicast = [{column = 0 : i32, id = 28 : i32, tiles = [#aie.tile<0, 2>, #aie.tile<1, 2>]}], aie.is_ctrl_pkt_overlay}
  aie.device(npu2) @cfg {
  } {overlay = @overlay}
  aie.device(npu2) @main {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    aie.shim_dma_allocation @ctrl0(%shim0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.shim_dma_allocation @ctrl1(%shim1, MM2S, 0) {aie.is_ctrl_pkt_overlay}

    // CHECK-LABEL: @multicast_interval
    // CHECK: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHECK: aiex.dma_start_task(%[[MC]])
    // CHECK: %[[OTHER:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHECK: aiex.dma_start_task(%[[OTHER]])
    // CHECK-NEXT: aiex.dma_await_task(%[[MC]])
    // CHECK-NEXT: aiex.dma_await_task(%[[OTHER]])
    // CHECK-NEXT: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHECK: aiex.dma_start_task(%[[WRITE]])
    // CHECK-NEXT: aiex.dma_await_task(%[[WRITE]])
    // CHAIN-LABEL: @multicast_interval
    // CHAIN: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHAIN: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHAIN: aiex.dma_start_task(%[[MC]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[MC]])
    // CHAIN-NEXT: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHAIN: aiex.dma_start_task(%[[WRITE]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[WRITE]])
    aie.runtime_sequence @multicast_interval() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent, unobserved}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 3276800 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, unobserved}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
    }

    // CHECK-LABEL: @multicast_batch_interval
    // CHECK: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHECK: aiex.dma_start_task(%[[MC]])
    // CHECK-NEXT: aiex.dma_await_task(%[[MC]])
    // CHECK-NEXT: %[[LATER:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHECK: aiex.dma_start_task(%[[LATER]])
    // CHECK-NEXT: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHECK: aiex.dma_start_task(%[[WRITE]])
    // CHECK-NEXT: aiex.dma_await_task(%[[LATER]])
    // CHECK-NEXT: aiex.dma_await_task(%[[WRITE]])
    // CHAIN-LABEL: @multicast_batch_interval
    // CHAIN: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHAIN: aiex.dma_start_task(%[[MC]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[MC]])
    // CHAIN-NEXT: %[[LATER:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHAIN: aiex.dma_start_task(%[[LATER]])
    // CHAIN-NEXT: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHAIN: aiex.dma_start_task(%[[WRITE]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[LATER]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[WRITE]])
    aie.runtime_sequence @multicast_batch_interval() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent, unobserved}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, unobserved}
      aiex.control_packet {address = 2228228 : ui32, data = array<i32: 3>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
    }

    // CHECK-LABEL: @multicast_unicast_batch_interval
    // CHECK: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHECK: aiex.dma_start_task(%[[MC]])
    // CHECK: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHECK: aiex.dma_await_task(%[[MC]])
    // CHECK: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHECK: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHECK: aiex.dma_start_task(%[[WRITE]])
    // CHECK-NEXT: aiex.dma_await_task(%[[WRITE]])
    // CHAIN-LABEL: @multicast_unicast_batch_interval
    // CHAIN: %[[MC:.*]] = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 0 len = 3)
    // CHAIN: aiex.dma_start_task(%[[MC]])
    // CHAIN: aie.dma_bd({{.*}} offset = 3 len = 3)
    // CHAIN: aiex.dma_await_task(%[[MC]])
    // CHAIN: %[[WRITE:.*]] = aiex.dma_configure_task(%shim_noc_tile_1_0, MM2S, 0)
    // CHAIN: aie.dma_bd({{.*}} offset = 6 len = 3)
    // CHAIN: aiex.dma_start_task(%[[WRITE]])
    // CHAIN-NEXT: aiex.dma_await_task(%[[WRITE]])
    aie.runtime_sequence @multicast_unicast_batch_interval() {
      aiex.set_device @cfg
      aiex.control_packet_barrier {independent, unobserved}
      aiex.control_packet {address = 2228224 : ui32, data = array<i32: 1>, multicast_id = 28 : i32, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35782660 : ui32, data = array<i32: 9>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier {compiler_only, unobserved}
      aiex.control_packet {address = 35782656 : ui32, data = array<i32: 2>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
    }
  }
}
