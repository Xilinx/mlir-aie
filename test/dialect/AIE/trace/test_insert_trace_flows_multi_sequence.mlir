//===- test_insert_trace_flows_multi_sequence.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// A device may declare several runtime sequences, and any of them may drive the
// device's trace units. Each one gets its own trace buffer argument and its own
// copy of the shim DMA program that drains the trace stream. The buffer belongs
// to the sequence, so the sequences may size it differently and may differ on
// reusing the output buffer.

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
    // CHECK: aiex.npu.writebd {bd_id = 15{{.*}}buffer_length = 1024
    // CHECK: aiex.npu.address_patch(%{{.*}} : i32) buffer %[[BUF0]]
    aie.runtime_sequence @first(%arg0: memref<64xi32>) {
      aie.trace.host_config {buffer_size = 4096 : i32}
      aie.trace.start_config @core_trace
    }

    // A larger buffer on the same trace units and the same route.
    // CHECK: aie.runtime_sequence @second(%{{.*}}: memref<32xi32>, %[[BUF1:.*]]: memref<8192xi8>)
    // CHECK-SAME: trace_buffer = #aie.trace_buffer<arg_index = 1, offset = 0, size = 8192, dedicated = true>
    // CHECK: aiex.npu.writebd {bd_id = 15{{.*}}buffer_length = 2048
    // CHECK: aiex.npu.address_patch(%{{.*}} : i32) buffer %[[BUF1]]
    aie.runtime_sequence @second(%arg0: memref<32xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @core_trace
    }

    // Reuse of the output buffer stays per sequence: no argument is appended,
    // and the trace data starts past the 128 bytes of %arg0.
    // CHECK: aie.runtime_sequence @third(%{{.*}}: memref<32xi32>)
    // CHECK-SAME: trace_buffer = #aie.trace_buffer<arg_index = 0, offset = 128, size = 2048, dedicated = false>
    aie.runtime_sequence @third(%arg0: memref<32xi32>) {
      aie.trace.host_config {buffer_size = 2048 : i32, reuse_output_buffer = true}
      aie.trace.start_config @core_trace
    }
  }
}
