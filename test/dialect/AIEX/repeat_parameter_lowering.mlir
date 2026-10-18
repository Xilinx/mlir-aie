//===- repeat_parameter_lowering.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dma_start_task's repeat_parameter is the push's repeat count, set per run
// from the scratchpad. The parameter is `repeat` kind, bounded by the task's
// own repeat count, and the push lowers to an update_from_scratchpad<incr> of
// the channel's task queue register adding the host's (n << 16) to
// bd_id | issue_token << 31. The update clears the queue word's low two bits,
// so the task's head BD takes an id that is a multiple of 4: the S2MM task
// holds BD 0, and the MM2S task gets BD 4, not 1. BD 4 of tile (0, 0) is at
// 0x1D080 = 118912; MM2S 0's task queue is at 0x1D214 = 119316.

// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters %s | FileCheck %s --check-prefix=MARK
// RUN: aie-opt --split-input-file --aie-lower-scratchpad-parameters=output-params-file=%t.params %s > /dev/null
// RUN: FileCheck %s --check-prefix=PARAMS < %t.params

// The params file is the last case's.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @runs : i32 {kind = 2 : i32, max_value = 15 : i32, min_value = 0 : i32, state_table_idx = 0 : ui8}
// CHECK: @blockwrite_data_1 : memref<8xi32> = dense<[64, 0, 0, 0, -1073741824, 33554432, 15728703, 33554432]>
// CHECK: aie.runtime_sequence @stepped
// CHECK: aiex.npu.create_scratchpad {size = 4 : ui32}
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118912 : ui32}
// CHECK: aiex.npu.address_patch
// CHECK-NOT: aiex.npu.write32(%c119316
// CHECK: aiex.npu.update_from_scratchpad {address = 119316 : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
// CHECK: aiex.npu.sync
// MARK-LABEL: aie.runtime_sequence @stepped
// MARK: aiex.dma_start_task(%{{.*}}) {repeat_state_table_idx = 0 : ui8}
aiex.scratchpad_parameter @runs : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @stepped(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 1024)
      aie.end
    } {issue_token = true}
    %in = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
      aie.end
    } {repeat_count = 15 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) {repeat_parameter = @runs}
    aiex.dma_await_task(%out)
  }
}

// -----

// The xclbin form of the first task: the repeat count is a dispatch-time
// scalar (the start's repeat_count_val), the push a write32 of a word computed
// per dispatch, range-checked on the host. The head BD needs no alignment.

// CHECK-LABEL: aie.runtime_sequence @stepped_xclbin
// CHECK-NOT: aiex.npu.create_scratchpad
// CHECK: aiex.npu.blockwrite(%{{.*}}) {address = 118816 : ui32}
// CHECK: cf.assert %{{.*}}, "a runtime DMA repeat count exceeds the task queue's [0:255] range (at most 256 executions)"
// CHECK: %[[MASKED:.*]] = arith.andi %arg2, %{{.*}} : i32
// CHECK: %[[FIELD:.*]] = arith.shli %[[MASKED]], %{{.*}} : i32
// CHECK: %[[WORD:.*]] = arith.ori %[[FIELD]], %{{.*}} : i32
// CHECK: %[[QUEUE:.*]] = arith.constant 119316 : i32
// CHECK: aiex.npu.write32(%[[QUEUE]], %[[WORD]])
// CHECK-NOT: aiex.npu.update_from_scratchpad
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @stepped_xclbin(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>, %reps : i32) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 1024)
      aie.end
    } {issue_token = true}
    %in = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
      aie.end
    } {repeat_count = 15 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) repeat %reps : i32
    aiex.dma_await_task(%out)
  }
}

// -----

// On MM2S 1 of tile (0, 0) the update is at its task queue, 0x1D21C = 119324.
// It writes the pair of registers from there: the queue and 0x1D220,
// DMA_S2MM_Status_0, written back with the value read, which clears its
// sticky write-1-to-clear error bits (16 to 18). This pins that address.

// CHECK-LABEL: aie.runtime_sequence @mm2s1
// CHECK: aiex.npu.update_from_scratchpad {address = 119324 : ui32, func_arg = 4 : ui32, state_table_idx = 0 : ui8}
aiex.scratchpad_parameter @runs : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence @mm2s1(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 1024)
      aie.end
    } {issue_token = true}
    %in = aiex.dma_configure_task(%t, MM2S, 1) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64)
      aie.end
    } {repeat_count = 15 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) {repeat_parameter = @runs}
    aiex.dma_await_task(%out)
  }
}

// -----

// A token-issuing task through a shim allocation, repeating one tile (no
// iteration): the token bit joins the update's func_arg, after the controller
// id the token needs is set as for any push. The S2MM task holds BD 0, so the
// head is BD 4: func_arg = 1 << 31 | 4. The repeat parameter is the second
// parameter, so it reads state table entry 1. MM2S 1's task queue of tile
// (1, 0) is at 0x201D21C = 33673756.

// CHECK-LABEL: module
// CHECK: aiex.scratchpad_parameter @rows : i32 {kind = 1 : i32
// CHECK: aiex.scratchpad_parameter @reps : i32 {kind = 2 : i32, max_value = 7 : i32, min_value = 0 : i32, state_table_idx = 1 : ui8}
// CHECK: aie.runtime_sequence @token
// CHECK: aiex.npu.create_scratchpad {size = 8 : ui32}
// CHECK: aiex.npu.update_from_scratchpad<mul>
// CHECK: aiex.npu.maskwrite32
// PARAMS: rows 0 i32 addr 0 8
// PARAMS-NEXT: reps 1 i32 repeat 0 7
// CHECK: aiex.npu.update_from_scratchpad {address = 33673756 : ui32, func_arg = 2147483652 : ui32, state_table_idx = 1 : ui8}
aiex.scratchpad_parameter @rows : i32
aiex.scratchpad_parameter @reps : i32
aie.device(npu2) {
  %t = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
  aie.shim_dma_allocation @in(%t, MM2S, 1)
  aie.runtime_sequence @token(%arg0 : memref<64xi32>, %arg1 : memref<512xi32>) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<512xi32> offset = 0 len = 0) {length_parameter = @rows, length_unit = 64 : i32}
      aie.end
    }
    %in = aiex.dma_configure_task_for @in {
      aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {issue_token = true, repeat_count = 7 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) {repeat_parameter = @reps}
    aiex.dma_await_task(%in)
  }
}
