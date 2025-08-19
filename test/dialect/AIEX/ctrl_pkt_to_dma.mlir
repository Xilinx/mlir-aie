//===- ctrl_pkt_to_dma.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-to-dma --split-input-file | FileCheck %s

// transforms control packet ops to dma memcpy ops and sync ops.

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aiex.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<?xi32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
  aiex.runtime_sequence() {
    aiex.control_packet {address = 126976 : ui32, data = array<i32: 1024>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
  memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
}

// -----

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aiex.runtime_sequence(%[[ARG0:.*]]: memref<?xi32>) {
// CHECK: aiex.npu.dma_memcpy_nd(%[[ARG0]][0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrlpkt_col0_mm2s_chan0} : memref<?xi32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 1 : i32, row = 0 : i32, row_num = 1 : i32}

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
  aiex.runtime_sequence() {
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
  memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
}
