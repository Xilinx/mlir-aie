//===- ctrl_pkt_to_dma.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma --split-input-file | FileCheck %s

// transforms control packet ops to dma memcpy ops and sync ops.

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<?xi32>
// CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
  aie.runtime_sequence() {
    aiex.control_packet {address = 126976 : ui32, data = array<i32: 1024>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
}


// -----

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<?xi32>
// CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
  aie.runtime_sequence() {
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0)
}


// -----

// Check that control packets writes are not combined on npu1

// CHECK-LABEL: aie.device(npu1) {
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 3][1, 1, 1, 3][0, 0, 0, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 6][1, 1, 1, 3][0, 0, 0, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 9][1, 1, 1, 3][0, 0, 0, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 12][1, 1, 1, 3][0, 0, 0, 1])
aie.device(npu1) {
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}


// -----

// Check that control packets writes are combined on npu2

// CHECK-LABEL: aie.device(npu2) {
// CHECK: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 15][0, 0, 0, 1])
aie.device(npu2) {
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 8 : ui32, data = array<i32: 300>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 12 : ui32, data = array<i32: 400>, opcode = 0 : i32, stream_id = 0 : i32}
    aiex.control_packet {address = 16 : ui32, data = array<i32: 500>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}


// -----

// Check that control packets writes are not combined across other operations

// CHECK-LABEL: aie.device(npu2) {
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[0, 0, 0, 0][1, 1, 1, 6][0, 0, 0, 1])
// CHECK: aiex.npu.sync
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[0, 0, 0, 6][1, 1, 1, 9][0, 0, 0, 1])
// CHECK: aiex.npu.sync
aie.device(npu2) {
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
}