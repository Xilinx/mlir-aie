//===- dma_configure_task_out_of_order_roundtrip.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trips the channel-level out_of_order attribute on dma_configure_task.

// RUN: aie-opt --split-input-file %s | FileCheck %s

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 1)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // CHECK: aiex.dma_configure_task({{.*}}, S2MM, 0) {
      %t = aiex.dma_configure_task(%tile, S2MM, 0) {
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      // CHECK: } {out_of_order}
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// A task-level packet (rather than a per-BD aie.dma_bd_packet) works.
// CHECK: aiex.dma_configure_task({{.*}}, S2MM, 0, <pkt_type = 0, pkt_id = 1>) {
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 1)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      %t = aiex.dma_configure_task(%tile, S2MM, 0, <pkt_type = 0, pkt_id = 1>) {
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      // CHECK: } {out_of_order}
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}
