//===- test_insert_trace_flows_verify.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics -aie-insert-trace-flows

// Test: Missing host_config in runtime_sequence with traces
module @missing_host_config {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{runtime_sequence with traces requires aie.trace.host_config}}
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: Trace missing aie.trace.start
module @missing_start {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    // expected-error@+1 {{trace is missing 'aie.trace.start'}}
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: Trace missing aie.trace.stop
module @missing_stop {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    // expected-error@+1 {{trace is missing 'aie.trace.stop'}}
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 65536
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// Test: LogicalTileOp must be resolved before insert-trace-flows
module @unresolved_logical_tile {
  aie.device(npu1_1col) {
    // expected-error@+1 {{LogicalTileOp must be resolved to TileOp before running -aie-insert-trace-flows (run -aie-place-tiles first)}}
    %tile = aie.logical_tile<CoreTile>(?, ?)
    aie.trace @core_trace(%tile) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
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

// Test: Trace ops without runtime_sequence
module @no_runtime_seq {
  // expected-error@+1 {{aie.trace ops found but no runtime_sequence defined}}
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
  }
}
