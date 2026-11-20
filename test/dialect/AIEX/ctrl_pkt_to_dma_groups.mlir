//===- ctrl_pkt_to_dma_groups.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma --split-input-file | FileCheck %s --implicit-check-not=aiex.ctrl_pkt_transfer

// The barrier separates two batches under the same device context.

// CHECK-LABEL: aie.device(npu2) {
// CHECK:       aiex.set_device @cfg
// CHECK-NEXT:  %[[A:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd(%arg0 : memref<?xi32> offset = 0 len = 9)
// CHECK:       } {issue_token = true}
// CHECK-NEXT:  aiex.dma_start_task(%[[A]])
// CHECK-NEXT:  aiex.dma_await_task(%[[A]])
// CHECK-NEXT:  %[[B:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd(%arg0 : memref<?xi32> offset = 9 len = 9)
// CHECK:       } {issue_token = true}
// CHECK-NEXT:  aiex.dma_start_task(%[[B]])
// CHECK-NEXT:  aiex.dma_await_task(%[[B]])
// CHECK-NOT:   aiex.npu.dma_memcpy_nd

module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence() {
      aiex.set_device @cfg
      aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      aiex.control_packet {address = 12 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 20 : ui32, data = array<i32: 600>, opcode = 0 : i32, stream_id = 0 : i32}
    }
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
  }
}

// -----

// CHECK-LABEL: @marker_boundary
// CHECK-NEXT:  aiex.set_device @cfg
// CHECK-NEXT:  %[[FIRST:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK:       aiex.dma_start_task(%[[FIRST]])
// CHECK-NEXT:  aiex.dma_await_task(%[[FIRST]])
// CHECK-NEXT:  aiex.set_device @cfg
// CHECK-NEXT:  %[[SECOND:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 3 len = 3)
// CHECK:       aiex.dma_start_task(%[[SECOND]])
// CHECK-NEXT:  aiex.dma_await_task(%[[SECOND]])
// CHECK-NEXT:  aiex.set_device @other
// CHECK-NEXT:  %[[THIRD:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK-NEXT:    aie.dma_bd({{.*}} offset = 6 len = 3)
// CHECK:       aiex.dma_start_task(%[[THIRD]])
// CHECK-NEXT:  aiex.dma_await_task(%[[THIRD]])
// CHECK-NEXT:  aiex.npu.load_pdi {device_ref = @other}
// CHECK-NOT:   aiex.dma_configure_task
// CHECK:       aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 9][1, 1, 1, 3][0, 0, 0, 1])
// CHECK:       aiex.npu.sync
module {
  aie.device(npu2) @cfg {
  }
  aie.device(npu2) @other {
  }
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%shim, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @marker_boundary() {
      aiex.set_device @cfg
      aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @cfg
      aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.set_device @other
      aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.npu.load_pdi {device_ref = @other}
      aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}

// -----

// CHECK-LABEL: @raw_packets
// CHECK-NOT:   aiex.set_device
// CHECK-NOT:   aiex.dma_configure_task
// CHECK:       aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 6][0, 0, 0, 1])
// CHECK:       aiex.npu.sync
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%shim, MM2S, 0) {aie.is_ctrl_pkt_overlay}
    aie.runtime_sequence @raw_packets() {
      aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 20 : ui32, data = array<i32: 600>, opcode = 0 : i32, stream_id = 0 : i32}
    }
  }
}
