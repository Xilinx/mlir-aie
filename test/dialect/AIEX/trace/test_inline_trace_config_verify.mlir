//===- test_inline_trace_config_verify.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-trace-to-config -aie-inline-trace-config -verify-diagnostics

module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)

    // expected-error@+1 {{aie.trace.reg still has field attribute - run -aie-trace-pack-reg-writes pass first}}
    aie.trace @test_trace(%tile02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @seq(%arg0: memref<32xi32>) {
      aie.trace.start_config @test_trace
    }
  }
}
