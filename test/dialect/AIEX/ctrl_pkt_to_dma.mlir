//===- ctrl_pkt_to_dma.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma --split-input-file | FileCheck %s

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK: aie.dma_bd(%[[ARG0]] : memref<?xi32> offset = 0 len = 3)
// CHECK: aiex.dma_await_task

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
  aie.runtime_sequence() {
    aiex.control_packet {address = 126976 : ui32, data = array<i32: 1024>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
}

// -----

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan0
// CHECK: aie.dma_bd(%[[ARG0]] : memref<?xi32> offset = 0 len = 3)
// CHECK: aiex.dma_await_task

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
  aie.runtime_sequence() {
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
}

// -----

// Check that control packets writes are not combined on npu1

// CHECK-LABEL: aie.device(npu1) {
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 0 len = 3)
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 3 len = 3)
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 6 len = 3)
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 9 len = 3)
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 12 len = 3)
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
}

// -----

// Check that control packets writes are combined on npu2

// CHECK-LABEL: aie.device(npu2) {
// CHECK: aie.dma_bd(%arg0 : memref<?xi32> offset = 0 len = 15)
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
}

// -----

// Check that control packets writes are not combined across other operations

// CHECK-LABEL: aie.device(npu2) {
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 6)
// CHECK: aiex.dma_await_task
// CHECK: aiex.npu.maskwrite32
// CHECK: aie.dma_bd({{.*}} offset = 6 len = 9)
// CHECK: aiex.dma_await_task
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    %cst_npu_0 = arith.constant 1024 : i32
    %cst_npu_1 = arith.constant 42 : i32
    %cst_npu_2 = arith.constant 255 : i32
    aiex.npu.maskwrite32(%cst_npu_0, %cst_npu_1, %cst_npu_2) : i32, i32, i32
    aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
}
// -----

// Control packets take the channel of their column's control packet shim DMA
// allocation, whatever the row they address.

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[TASK:.*]] = aiex.dma_configure_task_for @ctrlpkt_col0_mm2s_chan1
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 3)
// CHECK: aiex.dma_start_task(%[[TASK]])
// CHECK-NEXT: aiex.dma_await_task(%[[TASK]])
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
  aie.runtime_sequence() {
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1 (%tile_0_0, MM2S, 1) {aie.is_ctrl_pkt_overlay}
}

// -----

// Read requests send only headers and require separate DMA transfers.
// CHECK-LABEL: aie.device(npu2_1col) {
// CHECK: aie.dma_bd({{.*}} offset = 0 len = 2)
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} offset = 2 len = 2)
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} offset = 4 len = 3)
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} offset = 7 len = 2)
// CHECK: aiex.dma_await_task
aie.device(npu2_1col) {
  %shim = aie.tile(0, 0)
  %core = aie.tile(0, 2)
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(%shim, MM2S, 1) {aie.is_ctrl_pkt_overlay}
  aie.runtime_sequence() {
    aiex.control_packet {address = 0x200440 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
    aiex.control_packet {address = 0x200450 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
    aiex.control_packet {address = 0x21f000 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 0x200440 : ui32, length = 4 : i32, opcode = 1 : i32, stream_id = 2 : i32}
  }
}
