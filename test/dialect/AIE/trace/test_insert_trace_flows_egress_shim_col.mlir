//===- test_insert_trace_flows_egress_shim_col.mlir -----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s

// -----

// Test: omitting egress_shim_col preserves the default (col 0) destination.
// CHECK-LABEL: module @default_egress
// CHECK: aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
module @default_egress {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)
    aie.trace @trace0(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace0
    }
  }
}

// -----

// Test: egress_shim_col = 1 redirects the trace to col 1 shim.
// CHECK-LABEL: module @forced_egress
// CHECK: aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
module @forced_egress {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile10 = aie.tile(1, 0)
    aie.trace @trace0(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192 egress_shim_col = 1
      aie.trace.start_config @trace0
    }
  }
}
