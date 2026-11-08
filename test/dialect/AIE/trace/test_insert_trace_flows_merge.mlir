//===- test_insert_trace_flows_merge.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Placement can put several aie.trace ops on one physical tile and trace unit,
// e.g. IRON traces each logical shim endpoint and the placer lands them all on
// the same shim tile. Traces whose configurations are identical program the
// same registers, so the pass keeps the first one and points every reference
// to the others at it.
// RUN: aie-opt %s --split-input-file -aie-insert-trace-flows | FileCheck %s

// -----

// Three identical shim traces on shim (1, 0) and a core trace on (1, 2): the
// shim traces merge into the first, and its start_config is not repeated.
// CHECK-LABEL: module @identical_shim_traces
// CHECK: %[[SHIM:.*]] = aie.tile(1, 0)
// CHECK: %[[CORE:.*]] = aie.tile(1, 2)
// CHECK: aie.trace @trace_shim_1(%[[SHIM]])
// CHECK-NEXT: aie.trace.packet id = 1 type = shimtile
// CHECK-NOT: aie.trace @trace_shim_2
// CHECK-NOT: aie.trace @trace_shim_3
// CHECK: aie.trace @trace_core_4(%[[CORE]])
// CHECK-NEXT: aie.trace.packet id = 2 type = core
// CHECK: aie.runtime_sequence
// CHECK: aie.trace.start_config @trace_shim_1
// CHECK-NOT: aie.trace.start_config @trace_shim
// CHECK: aie.trace.start_config @trace_core_4
// CHECK-NOT: aie.trace.start_config
// CHECK-COUNT-2: aie.packet_flow(
// CHECK-NOT: aie.packet_flow(
module @identical_shim_traces {
  aie.device(npu1) {
    %shim10 = aie.tile(1, 0)
    %tile12 = aie.tile(1, 2)

    aie.trace @trace_shim_1(%shim10) {
      aie.trace.packet type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_shim_2(%shim10) {
      aie.trace.packet type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_shim_3(%shim10) {
      aie.trace.packet type=shimtile
      aie.trace.event<"DMA_S2MM_0_START_TASK">
      aie.trace.event<"DMA_MM2S_0_START_TASK">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_core_4(%tile12) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_shim_1
      aie.trace.start_config @trace_shim_2
      aie.trace.start_config @trace_shim_3
      aie.trace.start_config @trace_core_4
    }
  }
}

// -----

// A reference to a merged trace is rewritten, not dropped: here only the
// second trace is started, so its start_config now names the first. A
// sequence without host_config is rewritten too.
// CHECK-LABEL: module @reference_rewritten
// CHECK: aie.trace @trace_a
// CHECK-NOT: aie.trace @trace_b
// CHECK: aie.runtime_sequence @with_buffer
// CHECK: aie.trace.start_config @trace_a
// CHECK: aie.runtime_sequence @restart
// CHECK-NEXT: aie.trace.start_config @trace_a
// CHECK-NOT: @trace_b
module @reference_rewritten {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_b(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence @with_buffer(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_b
    }
    aie.runtime_sequence @restart(%arg0: memref<16xi32>) {
      aie.trace.start_config @trace_b
    }
  }
}

// -----

// Identical configurations on DIFFERENT physical tiles are separate trace
// units: both stay, each with its own packet flow.
// CHECK-LABEL: module @different_tiles_untouched
// CHECK: aie.trace @trace_a
// CHECK: aie.trace @trace_b
// CHECK: aie.trace.start_config @trace_a
// CHECK: aie.trace.start_config @trace_b
// CHECK-COUNT-2: aie.packet_flow(
module @different_tiles_untouched {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_b(%tile03) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_b
    }
  }
}

// -----

// Only the start_configs the merge redirects are dropped as repeats, wherever
// they sit in the block; a repeat already in the input is kept.
// CHECK-LABEL: module @input_repeat_kept
// CHECK: aie.runtime_sequence
// CHECK: aie.trace.start_config @trace_a
// CHECK-NEXT: aie.trace.start_config @trace_a
// CHECK-NOT: aie.trace.start_config
module @input_repeat_kept {
  aie.device(npu1_1col) {
    %tile02 = aie.tile(0, 2)
    %tile00 = aie.tile(0, 0)

    aie.trace @trace_a(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    aie.trace @trace_b(%tile02) {
      aie.trace.packet type=core
      aie.trace.event<"INSTR_EVENT_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aie.trace.host_config {buffer_size = 8192 : i32}
      aie.trace.start_config @trace_b
      aie.trace.start_config @trace_a
      aie.trace.start_config @trace_a
    }
  }
}
