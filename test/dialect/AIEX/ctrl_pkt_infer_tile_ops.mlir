//===- ctrl_pkt_infer_tile_ops.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-ctrl-packet-infer-tiles --split-input-file | FileCheck %s

// infer aie.tile ops based on control packet op's address

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}

aie.device(npu1_1col) {
  aiex.runtime_sequence(%arg0: memref<2048xi32>) {
    aiex.control_packet {address = 126976 : ui32, data = array<i32: 1024>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}

// -----

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}

aie.device(npu1_1col) {
  aiex.runtime_sequence(%arg0: memref<2048xi32>) {
    aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>, opcode = 0 : i32, stream_id = 0 : i32}
  }
}
