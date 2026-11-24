//===- test_insert_trace_flows_no_regdb.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// xcvc1902 has no register database, so the Timer_Control register a trace's
// start broadcast resets cannot be found: the pass reports it on the trace
// and fails, where it aborted the process.

// RUN: aie-opt --verify-diagnostics -aie-insert-trace-flows %s

module @trace_without_regdb {
  aie.device(xcvc1902) {
    %tile22 = aie.tile(2, 2)
    %tile20 = aie.tile(2, 0)

    // expected-error@+1 {{Failed to lookup Timer_Control register}}
    aie.trace @trace_a(%tile22) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32, egress_shim_col = 2 : i32}
      aie.trace.start_config @trace_a
    }
  }
}
