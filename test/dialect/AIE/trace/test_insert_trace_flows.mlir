//===- test_insert_trace_flows.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s

// -----

// Test: Shim tile is created when not defined
// CHECK-LABEL: module @shim_create
module @shim_create {
  aie.device(npu1_1col) {
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%[[TILE]], Trace : 0>
    // CHECK:   aie.packet_dest<%[[SHIM]], DMA : 1>
  }
}

// -----

// Test: Multiple traces in same column route to same shim
// CHECK-LABEL: module @multiple_traces
module @multiple_traces {
  aie.device(npu1_1col) {
    // CHECK-DAG: %[[TILE2:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[TILE3:.*]] = aie.tile(0, 3)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)
    aie.trace @trace1(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace2(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @trace1
    }
    // CHECK-DAG: aie.packet_source<%[[TILE2]], Trace : 0>
    // CHECK-DAG: aie.packet_source<%[[TILE3]], Trace : 0>
    // CHECK-DAG: aie.packet_dest<%[[SHIM]], DMA : 1>
  }
}

// -----

// Test: Auto packet ID allocation when not specified
// CHECK-LABEL: module @auto_packet_id
module @auto_packet_id {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    aie.trace @trace1(%tile02) {
      // CHECK: aie.trace.packet id = 1
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace2(%tile03) {
      // CHECK: aie.trace.packet id = 2
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @trace1
    }
  }
}

// -----

// Test: Auto packet type detection from tile type
// CHECK-LABEL: module @auto_packet_type
module @auto_packet_type {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile01 = aie.tile(0, 1)
    %tile00 = aie.tile(0, 0)
    aie.trace @core_trace(%tile02) {
      // CHECK: aie.trace.packet id = 1 type = core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @memtile_trace(%tile01) {
      // CHECK: aie.trace.packet id = 2 type = memtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: Core trace and memory trace on same tile use different trace ports
// CHECK-LABEL: module @core_and_mem
module @core_and_mem {
  aie.device(npu1_1col) {
    // CHECK-DAG: %[[TILE:.*]] = aie.tile(0, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @mem_trace(%tile02) {
      aie.trace.packet id=2 type=mem
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
    // CHECK-DAG: aie.packet_source<%[[TILE]], Trace : 0>
    // CHECK-DAG: aie.packet_source<%[[TILE]], Trace : 1>
    // CHECK-DAG: aie.packet_dest<%[[SHIM]], DMA : 1>
  }
}

// -----

// Test: MemTile trace routes from memtile to shim
// CHECK-LABEL: module @memtile
module @memtile {
  aie.device(npu1_1col) {
    // CHECK-DAG: %[[MEMTILE:.*]] = aie.tile(0, 1)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile00 = aie.tile(0, 0)
    aie.trace @memtile_trace(%tile01) {
      aie.trace.packet id=1 type=memtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @memtile_trace
    }
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%[[MEMTILE]], Trace : 0>
    // CHECK:   aie.packet_dest<%[[SHIM]], DMA : 1>
  }
}

// -----

// Test: Shim tile trace routes from shim trace port back to same shim DMA
// CHECK-LABEL: module @shimtile
module @shimtile {
  aie.device(npu1_1col) {
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile00 = aie.tile(0, 0)
    aie.trace @shim_trace(%tile00) {
      aie.trace.packet id=1 type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start event=<"TRUE">
      aie.trace.stop event=<"NONE">
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @shim_trace
    }
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%[[SHIM]], Trace : 0>
    // CHECK:   aie.packet_dest<%[[SHIM]], DMA : 1>
  }
}

// -----

// Test: Shim tile trace with broadcast rewrites to USER_EVENT
// CHECK-LABEL: module @shimtile_broadcast
module @shimtile_broadcast {
  aie.device(npu1_1col) {
    // CHECK: %[[SHIM:.*]] = aie.tile(0, 0)
    %tile00 = aie.tile(0, 0)
    aie.trace @shim_trace(%tile00) {
      aie.trace.packet id=1 type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      // CHECK: aie.trace.start event = <"USER_EVENT_1">
      aie.trace.start broadcast=15
      // CHECK: aie.trace.stop event = <"USER_EVENT_0">
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @shim_trace
    }
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%[[SHIM]], Trace : 0>
  }
}

// -----

// Test: buffer_size in host_config sets BD buffer_length (8192 bytes = 2048 words)
// CHECK-LABEL: module @buffer_size_config
module @buffer_size_config {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // CHECK: aiex.npu.writebd {{{.*}}buffer_length = 2048{{.*}}}
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: arg_idx=-1 uses last arg index with size offset (1024*4=4096)
// CHECK-LABEL: module @trace_after_last_tensor
module @trace_after_last_tensor {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<1024xi32>) {
      // CHECK: aiex.npu.address_patch {{{.*}}arg_idx = 1{{.*}}arg_plus = 4096{{.*}}}
      aie.trace.host_config buffer_size = 8192 arg_idx = -1
      aie.trace.start_config @core_trace
    }
  }
}
