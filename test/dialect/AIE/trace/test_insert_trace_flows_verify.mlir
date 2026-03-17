//===- test_insert_trace_flows_verify.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --split-input-file -aie-insert-trace-flows 2>&1 | FileCheck %s

// CHECK: error: trace is missing 'aie.trace.start'
module @missing_start {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      // Missing aie.trace.start - should error
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// CHECK: error: trace is missing 'aie.trace.stop'
module @missing_stop {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      // Missing aie.trace.stop - should error
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.start_config @core_trace
    }
  }
}

// -----

// CHECK: error: aie.trace ops found but no runtime_sequence
module @no_runtime_seq {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // Missing runtime_sequence - should error
  }
}
