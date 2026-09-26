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


// -----

// Test 7: REPEAT_LEN — factoring an overlong innermost run adds a dimension,
// which pushes the outermost one into the fourth (repeat) slot. Two things then
// have to follow the shape:
//   * len describes a single BD invocation, so it must shrink to the innermost
//     three dimensions; at the full extent aie-dma-tasks-to-npu rejects the BD.
//   * the queue-push repeat count must grow by the same factor. The BD's
//     iteration state only advances once per execution, so a BD left at one
//     execution moves 1/8th of the data here and the consumer hangs waiting for
//     the rest -- a silent, hardware-only failure, hence the check below.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=REPEAT-LEN
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=REPEAT-LOWER

// The 4096-element innermost run needs two dimensions, so the outermost 8 moves
// into the repeat slot and len drops from 262144 to one invocation's 32768.
// REPEAT-LEN-LABEL: @repeat_len_task_bd
// REPEAT-LEN:         aie.dma_bd
// REPEAT-LEN-SAME:        len = 32768

// The same 32768 elements, expressed as 8192 four-byte words, and 8 executions
// of it -- iteration_size and the queue-push repeat agree, both 0-based.
// REPEAT-LOWER-LABEL: @repeat_len_task_bd
// REPEAT-LOWER-NOT:     Buffer descriptor length does not match
// REPEAT-LOWER:         aiex.npu.writebd
// REPEAT-LOWER-SAME:        buffer_length = 8192
// REPEAT-LOWER-SAME:        iteration_size = 7
// REPEAT-LOWER:         %[[REPEAT:.*]] = arith.constant 7 : i32
// REPEAT-LOWER:         aiex.npu.push_queue
// REPEAT-LOWER-SAME:        repeat %[[REPEAT]]
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @repeat_len_task_bd(%in: memref<16x16x4096xi8>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Scaling 32 executions by 8 reaches the target's maximum repeat count.
// REPEAT-LEN-LABEL: @max_repeat_task_bd
// REPEAT-LEN:         aie.dma_bd
// REPEAT-LEN-SAME:        len = 32768
// REPEAT-LEN:         repeat_count = 255 : i32
// REPEAT-LOWER-LABEL: @max_repeat_task_bd
// REPEAT-LOWER:         %[[MAX_REPEAT:.*]] = arith.constant 255 : i32
// REPEAT-LOWER:         aiex.npu.push_queue
// REPEAT-LOWER-SAME:        repeat %[[MAX_REPEAT]]
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @max_repeat_task_bd(%in: memref<16x16x4096xi8>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 31 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Scaling 201 executions by 8 gives 1608 runs, past one push's 256. That is
// no longer an error: the BD-ID pass issues it as six full pushes and a
// 72-run remainder, and only the last push carries the token. A start that
// overrides the count repeats the same BD, so decomposition scales it too:
// 2 runs become 16.
// REPEAT-LEN-LABEL: @split_repeat_task_bd
// REPEAT-LEN:         aie.dma_bd
// REPEAT-LEN-SAME:        len = 32768
// REPEAT-LEN:         repeat_count = 1607 : i32
// REPEAT-LEN:         aiex.dma_start_task(%{{.*}}) {repeat_count = 15 : i32}
// REPEAT-LOWER-LABEL: @split_repeat_task_bd
// REPEAT-LOWER:         %[[FULL:.*]] = arith.constant 255 : i32
// REPEAT-LOWER:         aiex.npu.push_queue
// REPEAT-LOWER-SAME:        repeat %[[FULL]] {issue_token = false}
// REPEAT-LOWER-COUNT-5: repeat %c255_i32_{{[0-9]+}} {issue_token = false}
// REPEAT-LOWER:         %[[REST:.*]] = arith.constant 71 : i32
// REPEAT-LOWER:         aiex.npu.push_queue
// REPEAT-LOWER-SAME:        repeat %[[REST]] {issue_token = true}
// REPEAT-LOWER:         %[[OVERRIDE:.*]] = arith.constant 15 : i32
// REPEAT-LOWER:         aiex.npu.push_queue
// REPEAT-LOWER-SAME:        repeat %[[OVERRIDE]] {issue_token = true}
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @split_repeat_task_bd(%in: memref<16x16x4096xi8>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 200 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
      aiex.dma_start_task(%tk) {repeat_count = 1 : i32}
      aiex.dma_await_task(%tk)
    }
  }
}
