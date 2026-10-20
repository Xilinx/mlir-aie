//===- test_insert_trace_flows_held_ids.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// An auto-assigned trace id skips ids the rest of the device already holds:
// a hand-written packet flow at 1 and an objectfifo whose transport pinned 2.
// The trace lands on 3 rather than aliasing either.

// CHECK-LABEL: module @held_ids
module @held_ids {
  aie.device(npu1_1col) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile04 = aie.tile(0, 4)
    aie.packet_flow(1) {
      aie.packet_source<%tile03, Core : 0>
      aie.packet_dest<%tile04, Core : 0>
    }
    aie.objectfifo @of(%tile00, {%tile04}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 2>>} : !aie.objectfifo<memref<16xi32>>
    aie.trace @core_trace(%tile02) {
      // CHECK: aie.trace.packet id = 3
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 65536 : i32}
      aie.trace.start_config @core_trace
    }
    // CHECK: aie.packet_flow(3)
  }
}
