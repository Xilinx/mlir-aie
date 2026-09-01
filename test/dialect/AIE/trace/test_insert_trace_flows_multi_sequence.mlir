//===- test_insert_trace_flows_multi_sequence.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// A device may declare several runtime sequences, and any of them may drive the
// device's trace units. Each one gets its own trace buffer argument and its own
// copy of the shim DMA program that drains the trace stream.

module {
  aie.device(npu2_1col) @dev {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.trace @core_trace(%core) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // CHECK: aie.runtime_sequence @first(%{{.*}}: memref<64xi32>, %[[BUF0:.*]]: memref<4096xi8>)
    // CHECK-SAME: trace_buffer = #aie.trace_buffer<arg_index = 1, offset = 0, size = 4096, dedicated = true>
    // CHECK: aiex.npu.writebd {bd_id = 15
    // CHECK: aiex.npu.address_patch(%{{.*}} : i32) buffer %[[BUF0]]
    aie.runtime_sequence @first(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32}
      aie.trace.start_config @core_trace
    }

    // CHECK: aie.runtime_sequence @second(%{{.*}}: memref<32xi32>, %[[BUF1:.*]]: memref<4096xi8>)
    // CHECK-SAME: trace_buffer = #aie.trace_buffer<arg_index = 1, offset = 0, size = 4096, dedicated = true>
    // CHECK: aiex.npu.writebd {bd_id = 15
    // CHECK: aiex.npu.address_patch(%{{.*}} : i32) buffer %[[BUF1]]
    aie.runtime_sequence @second(%arg0: memref<32xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32}
      aie.trace.start_config @core_trace
    }
  }
}
