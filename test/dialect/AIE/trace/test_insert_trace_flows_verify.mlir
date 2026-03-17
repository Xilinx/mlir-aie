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

// CHECK: error: aie.trace ops found but no runtime_sequence
module @no_runtime_seq {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
    }
    // Missing runtime_sequence - should error
  }
}
