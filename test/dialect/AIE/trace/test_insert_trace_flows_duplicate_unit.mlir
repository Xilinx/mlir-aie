//===- test_insert_trace_flows_duplicate_unit.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// A hardware trace unit emits a single packet stream, so two aie.trace ops
// targeting the same (tile, unit) would race on one packet id and corrupt
// routing. The pass must reject that.
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows -verify-diagnostics

// -----

// Two core-unit traces on the same compute tile -> error.
module @duplicate_core_unit {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{is traced more than once on the same trace unit}}
    aie.trace @trace_b(%tile02) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }
  }
}

// -----

// The two DIFFERENT units of one compute tile (core + mem) are allowed: they
// are distinct hardware units, so this must NOT error.
module @core_and_mem_units_ok {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_core(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_mem(%tile02) {
      aie.trace.packet id=2 type=mem
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_core
      aie.trace.start_config @trace_mem
    }
  }
}
