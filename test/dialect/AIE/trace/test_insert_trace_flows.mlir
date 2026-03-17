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

//===----------------------------------------------------------------------===//
// Test: Shim tile creation when not defined
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @shim_create
// CHECK: aie.tile(0, 0)
// CHECK: aie.packet_flow(1)
// CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
module @shim_create {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    // No shim tile defined - pass should create one
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: Multiple traces same column
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @multiple_traces
// CHECK-DAG: aie.packet_flow(1)
// CHECK-DAG: aie.packet_flow(2)
module @multiple_traces {
  aie.device(npu1_1col) {
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
      aie.trace.start_config @trace1
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: Auto packet ID allocation and create TracePacketOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @auto_packet_id
// Verify TracePacketOp is materialized when missing
// CHECK-DAG: aie.trace.packet id = 1
// CHECK-DAG: aie.trace.packet id = 2
// CHECK-DAG: aie.packet_flow(1)
// CHECK-DAG: aie.packet_flow(2)
module @auto_packet_id {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    // No packet id specified - auto-allocate
    aie.trace @trace1(%tile02) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace2(%tile03) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @trace1
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: Auto packet type detection from tile type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @auto_packet_type
// Core tile auto-detects type=core
// CHECK: aie.trace @core_trace
// CHECK: aie.trace.packet id = 1 type = core
// Mem tile auto-detects type=memtile
// CHECK: aie.trace @memtile_trace
// CHECK: aie.trace.packet id = 2 type = memtile
module @auto_packet_type {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)  // type=core
    %tile01 = aie.tile(0, 1)  // type=memtile
    %tile00 = aie.tile(0, 0)

    aie.trace @core_trace(%tile02) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @memtile_trace(%tile01) {
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: Core trace and memory trace on same tile 
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @core_and_mem
// Two timer control writes for same tile (core module and mem module)
// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.write32 {{{.*}}row = 2{{.*}}}
// CHECK: aiex.npu.write32 {{{.*}}row = 2{{.*}}}
// Core trace uses Trace:0, mem trace uses Trace:1
// CHECK-DAG: aie.packet_flow(1)
// CHECK-DAG: aie.packet_source<%tile_0_2, Trace : 0>
// CHECK-DAG: aie.packet_flow(2)
// CHECK-DAG: aie.packet_source<%tile_0_2, Trace : 1>
module @core_and_mem {
  aie.device(npu1_1col) {
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
      aie.trace.start_config @core_trace
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: MemTile trace
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @memtile
// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.writebd
// CHECK: aiex.npu.address_patch
// CHECK: aiex.npu.maskwrite32
// CHECK: aiex.npu.write32
// CHECK: aie.packet_flow(1)
// CHECK: aie.packet_source<%mem_tile_0_1, Trace : 0>
// CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
module @memtile {
  aie.device(npu1_1col) {
    %tile01 = aie.tile(0, 1)
    %tile00 = aie.tile(0, 0)

    aie.trace @memtile_trace(%tile01) {
      aie.trace.packet id=1 type=memtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @memtile_trace
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: Shim tile trace
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @shimtile
// CHECK: aie.packet_flow(1)
// CHECK: aie.packet_source<%shim_noc_tile_0_0, Trace : 0>
// CHECK: aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
module @shimtile {
  aie.device(npu1_1col) {
    %tile00 = aie.tile(0, 0)

    aie.trace @shim_trace(%tile00) {
      aie.trace.packet id=1 type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start event=<"TRUE">
      aie.trace.stop event=<"NONE">
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @shim_trace
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test: buffer_size attribute on trace op
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @buffer_size_attr
// CHECK: aiex.npu.writebd {{{.*}}buffer_length = 8192{{.*}}}
module @buffer_size_attr {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @core_trace(%tile02) buffer_size = 8192 {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}
