//===- test_insert_trace_flows_distribute.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="distribute-channels=true" | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s --check-prefix=NODIST

// -----

// Test: Two traces are distributed across DMA channels 0 and 1.
// CHECK-LABEL: module @distribute_two_traces
module @distribute_two_traces {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

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

    // First trace -> channel 1 (default shim-channel), second -> channel 0
    // CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK-DAG: aie.packet_dest<%{{.*}}, DMA : 0>
  }
}

// -----

// Test: Single trace -- no distribution even when enabled (only 1 trace).
// CHECK-LABEL: module @distribute_single_trace
module @distribute_single_trace {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

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

    // Single trace always uses default channel
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}

// -----

// Test: Without distribute-channels, both traces use same DMA channel.
// NODIST-LABEL: module @distribute_two_traces
// NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
// NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
