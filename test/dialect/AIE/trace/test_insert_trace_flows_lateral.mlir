//===- test_insert_trace_flows_lateral.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true" | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s --check-prefix=NOLATRL

// -----

// Test: Lateral routing sends trace to spare column (no active core).
// Column 0 has a core (active), column 1 has no core (spare).
// CHECK-LABEL: module @lateral_basic
module @lateral_basic {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    %core = aie.core(%tile02) {
      aie.end
    }

    aie.trace @core_trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @core_trace
    }

    // Trace routes to column 1 (spare), not column 0 (active)
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%{{.*}}, Trace : 0>
    // CHECK:   aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
    // CHECK:   keep_pkt_header = true

    // Without lateral routing, trace stays on column 0
    // NOLATRL: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
  }
}

// -----

// Test: No spare column available -- falls back to column 0 shim.
// Both columns have cores.
// CHECK-LABEL: module @lateral_no_spare
module @lateral_no_spare {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile00 = aie.tile(0, 0)

    %core0 = aie.core(%tile02) { aie.end }
    %core1 = aie.core(%tile12) { aie.end }

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace
    }

    // No spare column, falls back to column 0
    // CHECK: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
  }
}
