//===- test_trace_to_config_verify.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-trace-to-config -split-input-file -verify-diagnostics

module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)

    aie.trace @unknown_trace_event(%tile02) {
      // expected-error@+1 {{unknown trace event 'NOT_A_REAL_EVENT'}}
      aie.trace.event<"NOT_A_REAL_EVENT">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)

    aie.trace @unknown_start_event(%tile02) {
      aie.trace.event<"INSTR_EVENT_0">
      // expected-error@+1 {{unknown trace event 'ALSO_NOT_REAL'}}
      aie.trace.start event=<"ALSO_NOT_REAL">
      aie.trace.stop broadcast=14
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)

    aie.trace @unknown_stop_event(%tile02) {
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      // expected-error@+1 {{unknown trace event 'STILL_NOT_REAL'}}
      aie.trace.stop event=<"STILL_NOT_REAL">
    }
  }
}
