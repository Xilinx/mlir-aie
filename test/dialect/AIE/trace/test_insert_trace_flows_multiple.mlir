//===- test_insert_trace_flows_multiple.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-insert-trace-flows | FileCheck %s

// Test multiple traces from different tiles routed to one shim

// CHECK-LABEL: module {
module {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    // Core trace on tile (0,2)
    aie.trace @core_trace_02(%tile02) {
      aie.trace.packet id=1 type="core"
      aie.trace.event<"INSTR_VECTOR">
      aie.trace.start broadcast=15
    }

    // Mem trace on tile (0,2)
    aie.trace @mem_trace_02(%tile02) {
      aie.trace.packet id=2 type="mem"
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.start broadcast=15
    }

    // Core trace on tile (0,3)
    aie.trace @core_trace_03(%tile03) {
      aie.trace.packet id=3 type="core"
      aie.trace.event<"LOCK_STALL">
      aie.trace.start broadcast=15
    }

    // CHECK: aie.packet_flow(1)
    // CHECK: aie.packet_source<%tile_0_2, Trace : 0>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>

    // CHECK: aie.packet_flow(2)
    // CHECK: aie.packet_source<%tile_0_2, Trace : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>

    // CHECK: aie.packet_flow(3)
    // CHECK: aie.packet_source<%tile_0_3, Trace : 0>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}
