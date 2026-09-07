//===- decompose_large_dma_task.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for aie-decompose-large-dma-bd on task-path aie.dma_bd ops inside
// aiex.dma_configure_task_for regions (IRON rt.fill/drain tap lowering).
//
//===----------------------------------------------------------------------===//


// -----

// Test 1: FACTOR — oversized non-contiguous shim BD is rewritten in place to a
// single hardware-legal aie.dma_bd (no next_bd chain).
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=FACTOR

// FACTOR-LABEL: @factor_task_bd
// FACTOR:         aiex.dma_configure_task_for @a
// FACTOR:           aie.dma_bd
// -- The oversized 1920 wrap is gone; the NOT is bounded by the aie.end below
// -- so it only inspects this single-BD task (not later chained tests).
// FACTOR-NOT:       4, 1920]
// FACTOR-NOT:       aie.next_bd
// FACTOR:           aie.end
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @factor_task_bd(%in: memref<7684xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<7684xi32> offset = 0 len = 7680 sizes = [1, 1, 4, 1920] strides = [0, 0, 1921, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 2: UNCHANGED — a small already-legal task BD is left as-is.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=UNCHANGED

// UNCHANGED-LABEL: @small_unchanged_task
// UNCHANGED:         aie.dma_bd
// UNCHANGED-SAME:        sizes = [1, 1, 1, 8]
// UNCHANGED-SAME:        strides = [0, 0, 0, 1]
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @small_unchanged_task(%in: memref<8xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<8xi32> offset = 0 len = 8 sizes = [1, 1, 1, 8] strides = [0, 0, 0, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 3: LOWER — end-to-end through BD-ID assignment and tasks-to-npu.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// LOWER-LABEL: @lower_task_bd
// LOWER-NOT:     exceeds the [0:1023] range
// LOWER:         aiex.npu.writebd
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @lower_task_bd(%in: memref<7684xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<7684xi32> offset = 0 len = 7680 sizes = [1, 1, 4, 1920] strides = [0, 0, 1921, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 4: SLICE — a prime outer dimension (1031 > 1023) cannot be factored, so
// it is split into an aie.next_bd chain of hardware-legal BDs. Each chain
// member covers a contiguous slice of the oversized dimension in order.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=SLICE

// SLICE-LABEL: @slice_task_bd
// SLICE:         aie.dma_bd
// SLICE-SAME:        sizes = [1, 1, 1023, 2]
// SLICE:         aie.next_bd
// SLICE:         aie.dma_bd
// SLICE-SAME:        sizes = [1, 1, 8, 2]
// SLICE:         aie.end
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @slice_task_bd(%in: memref<4096xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 2062 sizes = [1, 1, 1031, 2] strides = [0, 0, 3, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 5: AXCACHE — every member of a sliced chain must inherit the template's
// shim-only attributes. The first chunk is rewritten in place (so it keeps them
// for free); the rest are freshly built and have to copy them explicitly, or a
// single logical transfer ends up issuing AXI bursts under two different cache
// attributes.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=AXCACHE

// AXCACHE-LABEL: @axcache_slice_task_bd
// AXCACHE:         aie.dma_bd
// AXCACHE-SAME:        sizes = [1, 1, 1023, 2]
// AXCACHE-SAME:        axcache = 15 : i32
// AXCACHE:         aie.next_bd
// AXCACHE:         aie.dma_bd
// AXCACHE-SAME:        sizes = [1, 1, 8, 2]
// AXCACHE-SAME:        axcache = 15 : i32
// AXCACHE:         aie.end
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @axcache_slice_task_bd(%in: memref<4096xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 2062 sizes = [1, 1, 1031, 2] strides = [0, 0, 3, 1])
          {burst_length = 0 : i32, axcache = 15 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 6: OOO_FACTOR — oversized out-of-order task BD whose extent factors
// into hardware-legal dimensions (2046 = 2 x 1023) is rewritten to a single BD
// that keeps out_of_order_id and packet header.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=OOO-FACTOR

// OOO-FACTOR-LABEL: @factor_ooo_task
// OOO-FACTOR:         aie.dma_bd
// OOO-FACTOR-SAME:        sizes = [1, 2, 1023, 2]
// OOO-FACTOR-SAME:        out_of_order_id = 5
// OOO-FACTOR-SAME:        packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>
// OOO-FACTOR-NOT:     aie.next_bd
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @factor_ooo_task(%in: memref<8192xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 4092 sizes = [1, 1, 2046, 2] strides = [0, 0, 3, 1])
          {burst_length = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, out_of_order_id = 5 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}
