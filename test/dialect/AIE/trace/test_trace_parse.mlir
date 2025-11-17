//===- test_trace_parse.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: aie.device(npu1_1col)
  aie.device(npu1_1col) {
    // CHECK: %[[TILE:.*]] = aie.tile(0, 2)
    %tile02 = aie.tile(0, 2)
    
    // CHECK: aie.trace @test_trace(%[[TILE]]) {
    aie.trace @test_trace(%tile02) {
      // CHECK: aie.trace.mode "Event-Time"
      aie.trace.mode "Event-Time"
      
      // CHECK: aie.trace.packet id = 1 type = core
      aie.trace.packet id=1 type="core"
      
      // CHECK: aie.trace.event <"INSTR_EVENT_0">
      aie.trace.event<"INSTR_EVENT_0">
      
      // CHECK: aie.trace.event <"INSTR_VECTOR"> label = "vector_op"
      aie.trace.event<"INSTR_VECTOR"> label="vector_op"
      
      // CHECK: aie.trace.event <"LOCK_STALL">
      aie.trace.event<"LOCK_STALL">
      
      // CHECK: aie.trace.start broadcast = 15
      aie.trace.start broadcast=15
      
      // CHECK: aie.trace.stop broadcast = 14
      aie.trace.stop broadcast=14
    }
  }
}
