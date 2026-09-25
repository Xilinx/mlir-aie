//===- test_insert_trace_flows_conflicts.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Both RUN lines share one CHECK prefix -- conflict detection behavior is
// identical with and without distribute-channels, so this implicitly verifies
// both modes produce the same result.
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows="distribute-channels=true" | FileCheck %s

// -----

// Test: Default trace channel (DMA:1) is already claimed by an existing flow.
// The pass must detect the conflict and switch the primary to DMA:0.
// This is the critical path -- it tests that the PRIMARY channel gets
// reassigned, not just that distribute collapses.
// CHECK-LABEL: module @default_channel_conflict
module @default_channel_conflict {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    // Existing flow claims S2MM channel 1 (the trace default) on shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 1)

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace
    }

    // Trace switches to DMA:0 (channel 1 is occupied)
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 0>
  }
}

// -----

// Test: S2MM channel 0 is claimed. With distribute, the secondary channel
// (DMA:0) is unavailable, so distribute collapses to single-channel on
// DMA:1. Without distribute, DMA:1 is already the default -- same output.
// CHECK-LABEL: module @secondary_channel_used
module @secondary_channel_used {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    // Existing flow claims S2MM channel 0 (distribute secondary) on shim
    aie.flow(%tile02, DMA : 0, %tile00, DMA : 0)

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
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // Single channel means single BD (appears in runtime_sequence before packet flows)
    // CHECK: aiex.npu.writebd
    // CHECK-NOT: aiex.npu.writebd
    // Both traces use channel 1 (channel 0 occupied, no distribute possible)
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}

// -----

// Test: An objectFifo that ends at the shim has no flow yet when this pass
// runs, but its lowering will take S2MM channel 0. With distribute, that
// leaves the traces a single channel, DMA:1, as in @secondary_channel_used.
// CHECK-LABEL: module @objectfifo_claims_channel
module @objectfifo_claims_channel {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    aie.objectfifo @out(%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

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
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }

    // CHECK: aiex.npu.writebd
    // CHECK-NOT: aiex.npu.writebd
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
    // CHECK: aie.packet_dest<%{{.*}}, DMA : 1>
  }
}

// -----

// Test: An objectFifo pinned to S2MM channel 1 by cons_dma_channels takes
// channel 1, not the lowest free one, so the trace gets channel 0.
// CHECK-LABEL: module @objectfifo_pinned_channel
module @objectfifo_pinned_channel {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.objectfifo @out(%tile02, {%tile00}, 2 : i32) {cons_dma_channels = array<i32: 1>} : !aie.objectfifo<memref<16xi32>>

    aie.trace @trace(%tile02) {
      aie.trace.packet id=1 type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace
    }

    // CHECK: aie.packet_dest<%{{.*}}, DMA : 0>
  }
}
