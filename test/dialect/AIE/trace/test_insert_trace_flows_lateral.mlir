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
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true lateral-target-col=1" | FileCheck %s --check-prefix=FORCED
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="lateral-routing=true distribute-channels=true" | FileCheck %s --check-prefix=COMBO

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

    // Shim DMA config targets the redirected column 1
    // CHECK: aiex.npu.writebd {{{.*}}column = 1

    // Trace routes to column 1 (spare), not column 0 (active)
    // CHECK: aie.packet_flow(1)
    // CHECK:   aie.packet_source<%{{.*}}, Trace : 0>
    // CHECK:   aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
    // CHECK:   keep_pkt_header = true

    // With forced target column 1, routes to column 1
    // FORCED: aiex.npu.writebd {{{.*}}column = 1
    // FORCED: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
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

    // With forced target, overrides even when target column has active cores
    // FORCED: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
  }
}

// -----

// Test: Lateral routing + channel distribution compose together.
// Two traces in column 0 (active) should route laterally to column 1 (spare)
// AND distribute across two DMA channels.
// COMBO-LABEL: module @lateral_and_distribute
module @lateral_and_distribute {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    %core0 = aie.core(%tile02) { aie.end }

    aie.trace @trace_a(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_b(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // Both DMA channels configured on the redirected column 1
    // COMBO-DAG: aiex.npu.writebd {bd_id = 15{{.*}}column = 1
    // COMBO-DAG: aiex.npu.writebd {bd_id = 14{{.*}}column = 1

    // Both traces route to column 1 (spare) with different DMA channels
    // COMBO-DAG: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
    // COMBO-DAG: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 0>
  }
}

// -----

// Test: Both S2MM channels on target shim are used -- lateral redirects to
// a spare column where channels are free.
// CHECK-LABEL: module @lateral_fallback_full_shim
module @lateral_fallback_full_shim {
  aie.device(npu1_2col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    %core = aie.core(%tile02) { aie.end }

    // Both S2MM channels claimed on column 0 shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 0)
    aie.flow(%tile02, DMA : 1, %tile00, DMA : 1)

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

    // Trace redirects to column 1 (spare, both channels free)
    // CHECK: aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    // CHECK: aiex.npu.writebd {{{.*}}column = 1
    // CHECK: aie.packet_dest<%{{.*}}1_0{{.*}}, DMA : 1>
  }
}
