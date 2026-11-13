//===- test_insert_trace_flows_duplicate_unit.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A hardware trace unit emits a single packet stream, so two aie.trace ops
// targeting the same (tile, unit) would race on one packet id and corrupt
// routing. Identical ones are merged (test_insert_trace_flows_merge.mlir);
// the pass must reject ones that differ.
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows -verify-diagnostics

// -----

// Two core-unit traces on the same compute tile -> error.
module @duplicate_core_unit {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // expected-note@+1 {{other trace of this unit}}
    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{is traced more than once on the same trace unit with different configurations}}
    aie.trace @trace_b(%tile02) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_1">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }
  }
}

// -----

// Traces that differ in a single field still differ: here only the stop
// broadcast. Neither may be dropped for the other.
module @duplicate_unit_one_field_differs {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // expected-note@+1 {{other trace of this unit}}
    aie.trace @trace_a(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{tile (0, 2) is traced more than once on the same trace unit with different configurations}}
    aie.trace @trace_b(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=13
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }
  }
}

// -----

// An explicit packet id is part of the configuration: traces that differ only
// in it are not merged.
module @duplicate_unit_only_packet_id_differs {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // expected-note@+1 {{other trace of this unit}}
    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    // expected-error@+1 {{tile (0, 2) is traced more than once on the same trace unit with different configurations}}
    aie.trace @trace_b(%tile02) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
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
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_core
      aie.trace.start_config @trace_mem
    }
  }
}
