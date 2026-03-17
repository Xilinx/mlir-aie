//===- test_insert_trace_flows_simple.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
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

    // CHECK: aie.runtime_sequence
    // Timer control write for core tile
    // CHECK: aiex.npu.write32 {{{.*}}column = 0{{.*}}row = 2{{.*}}}
    // Buffer descriptor setup
    // CHECK: aiex.npu.writebd {{{.*}}bd_id = 15{{.*}}buffer_length = 1048576{{.*}}}
    // Address patch for host buffer
    // CHECK: aiex.npu.address_patch {{{.*}}arg_idx = 4{{.*}}}
    // DMA channel configuration
    // CHECK: aiex.npu.maskwrite32
    // Task queue push
    // CHECK: aiex.npu.write32
    // Shim timer control
    // CHECK: aiex.npu.write32
    // Broadcast 15 configuration
    // CHECK: aiex.npu.write32
    // Event generate (start)
    // CHECK: aiex.npu.write32
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }

    // CHECK: aie.packet_flow(1)
    // CHECK: aie.packet_source<%tile_0_2, Trace : 0>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: keep_pkt_header = true
  }
}
