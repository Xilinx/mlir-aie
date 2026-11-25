//===- ctrl_pkt_to_dma_copy_loc.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-ctrl-packet-to-dma --mlir-print-debuginfo %s | FileCheck %s

// CHECK: aiex.dma_configure_task_for
// CHECK: } {issue_token = true} loc(#[[LOC1:loc[0-9]*]])
// CHECK: aiex.dma_configure_task_for
// CHECK: } {issue_token = true} loc(#[[LOC2:loc[0-9]*]])
// CHECK-DAG: #[[LOC1]] = loc("first_batch")
// CHECK-DAG: #[[LOC2]] = loc("second_batch")
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence() {
    aiex.control_packet {address = 0 : ui32, data = array<i32: 100>, opcode = 0 : i32, stream_id = 0 : i32} loc("first_batch")
    aiex.control_packet {address = 4 : ui32, data = array<i32: 200>, opcode = 0 : i32, stream_id = 0 : i32} loc("second_batch")
  }
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0 (%tile_0_0, MM2S, 0) {aie.is_ctrl_pkt_overlay}
} loc("device_line")
