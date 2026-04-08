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

    // Phase 4 emits per-channel shim DMA configuration:
    // Two buffer descriptors (one per channel, distinct bd_ids).
    // CHECK-DAG: aiex.npu.writebd {bd_id = 15
    // CHECK-DAG: aiex.npu.writebd {bd_id = 14
    // Two address patches with distinct arg_idx values.
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 4
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 5

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

// -----

// Test: distribute with arg_idx=-1 falls back to single channel
// (arg_idx=-1 resolves to last arg; +1 would be out of bounds)
// CHECK-LABEL: module @distribute_auto_argidx_fallback
module @distribute_auto_argidx_fallback {
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
      aie.trace.host_config buffer_size = 8192 arg_idx = -1
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // With arg_idx=-1, distribute falls back to single channel.
    // Only one writebd and one address_patch (single channel).
    // CHECK-COUNT-1: aiex.npu.writebd
    // CHECK-COUNT-1: aiex.npu.address_patch
    // Both traces use the same DMA channel (no distribute).
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}

// -----

// Test: 4 traces across 3 columns, distributed across 2 channels.
// Round-robin: traces 0,2 -> channel 1 (primary), traces 1,3 -> channel 0.
// CHECK-LABEL: module @distribute_four_traces_three_cols
module @distribute_four_traces_three_cols {
  aie.device(npu1_3col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile12 = aie.tile(1, 2)
    %tile22 = aie.tile(2, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_0_2(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_0_3(%tile03) {
      aie.trace.packet id=2 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_1_2(%tile12) {
      aie.trace.packet id=3 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.trace @trace_2_2(%tile22) {
      aie.trace.packet id=4 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aie.trace.host_config buffer_size = 8192
      aie.trace.start_config @trace_0_2
      aie.trace.start_config @trace_0_3
      aie.trace.start_config @trace_1_2
      aie.trace.start_config @trace_2_2
    }

    // All 4 traces route to column 0 shim (Single routing strategy)
    // Distributed across 2 channels: traces alternate between DMA 1 and DMA 0
    // CHECK-DAG: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 1>
    // CHECK-DAG: aie.packet_dest<%{{.*}}0_0{{.*}}, DMA : 0>
    // Two BDs configured (one per channel)
    // CHECK-DAG: aiex.npu.writebd {bd_id = 15
    // CHECK-DAG: aiex.npu.writebd {bd_id = 14
    // Two address patches
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 4
    // CHECK-DAG: aiex.npu.address_patch {{{.*}}arg_idx = 5

    // Without distribute, all 4 traces use same channel
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
    // NODIST: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}
