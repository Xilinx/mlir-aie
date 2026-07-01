//===- test_insert_trace_flows_simple.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// Test: End-to-end trace setup with Event-Time mode
// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
    // CHECK-DAG: %[[TILE:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // CHECK: aie.trace @core_trace
    aie.trace @core_trace(%tile02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // Core trace control (row=2)
      // CHECK: aiex.npu.write32 {{{.*}}row = 2{{.*}}}
      // Shim BD setup (1048576 bytes = 262144 words)
      // CHECK: aiex.npu.writebd {{{.*}}buffer_length = 262144{{.*}}}
      // Address patch for trace buffer: appended after the 1 data arg, so idx 1
      // CHECK: aiex.npu.address_patch {{{.*}}arg_idx = 1{{.*}}}
      // Shim DMA channel setup
      // CHECK: aiex.npu.maskwrite32
      // CHECK: aiex.npu.write32
      // Shim broadcast configuration (start/stop events)
      // CHECK: aiex.npu.write32 {{{.*}}row = 0{{.*}}}
      aie.trace.host_config buffer_size = 1048576
      aie.trace.start_config @core_trace
    }

    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%[[TILE]], Trace : 0>
    // CHECK:   aie.packet_dest<%[[SHIM]], DMA : 1>
    // CHECK: keep_pkt_header = true
  }
}
