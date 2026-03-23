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

// Test: trace_after_last_tensor requires routing = single
module @trace_after_with_per_column {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      // expected-error@+1 {{'aie.trace.host_config' op appending trace data to the last tensor argument only works with single shim destination strategy (routing=single)}}
      aie.trace.host_config buffer_size = 65536 routing = per_column trace_after_last_tensor = true
    }
  }
}

// -----

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
