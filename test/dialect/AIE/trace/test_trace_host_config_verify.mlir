//===- test_trace_host_config_verify.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

// Test: buffer_size must be positive
module @invalid_buffer_size {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // expected-error@+1 {{'aie.trace.host_config' op buffer_size must be positive}}
      aie.trace.host_config buffer_size = 0
    }
  }
}

// -----

// Test: unknown routing strategy parse error
module @invalid_routing {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      aie.trace.host_config buffer_size = 8192 routing = invalid_routing
    // expected-error@+1 {{custom op 'aie.trace.host_config' unknown routing strategy: invalid_routing}}
    }
  }
}

// -----

// Test: invalid trace_after_last_tensor value parse error
module @invalid_trace_after_value {
  aie.device(npu1_1col) {
    aie.runtime_sequence() {
      aie.trace.host_config buffer_size = 8192 trace_after_last_tensor = maybe
    // expected-error@+1 {{custom op 'aie.trace.host_config' expected 'true' or 'false'}}
    }
  }
}
