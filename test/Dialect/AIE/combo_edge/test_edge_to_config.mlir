//===- test_edge_to_config.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

aie.device(npu1_1col) {
  %tile_0_2 = aie.tile(0, 2)
  
  // CHECK-LABEL: @test_edge_lowering
  aie.trace @test_edge_lowering(%tile_0_2) {
    aie.trace.edge_event<0> event=<"LOCK_STALL"> trigger=RISING
    aie.trace.event<"EDGE_DETECTION_EVENT_0">
  }
  
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_Event_0"
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Rising" value = 1
  // CHECK: aie.trace.reg register = "Edge_Detection_event_control" field = "Edge_Detection_0_Trigger_Falling" value = 0
  // CHECK: aie.trace.reg register = "Trace_Event0" field = "Trace_Event0"
}
