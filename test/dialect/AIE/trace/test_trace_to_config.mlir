//===- test_trace_to_config.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)

    aie.trace @test_trace(%tile02) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.event<"LOCK_STALL">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Mode" value = 0
    // CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "ID" value = 1
    // CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "Packet_Type" value = 0
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event" value = 15
    // CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event" value = 14
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event0" value = "INSTR_EVENT_0"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event1" value = "INSTR_VECTOR"
    // CHECK-DAG: aie.trace.reg register = "Trace_Event0" field = "Trace_Event2" value = "LOCK_STALL"
  }
}
