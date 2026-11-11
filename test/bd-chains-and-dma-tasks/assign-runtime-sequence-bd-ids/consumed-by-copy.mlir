//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:         --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true warn-unreleased-transfers=false' \
// RUN:         --verify-diagnostics --split-input-file %s

// An aiex.npu.dma_memcpy_nd push on an input with a #aie.consumed_by proof
// (consumed-by.mlir's ProjectionGEMM round, with the input a copy instead of
// a task) retires the same way a task's push does: once the output's await
// proves its bytes taken, a later BD on the same id -- a task naming it, or
// another copy -- is not in flight anymore.

// A named task reuses the copy's BD id once the output proves it consumed.
module @task_over_proven_copy {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<512xi32>
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      %in1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%a : memref<512xi32> offset = 64 len = 64) {bd_id = 1 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)
    }
  }
}

// -----

// A second copy reuses the first's id the same way, and the round repeats
// cleanly.
module @copy_over_proven_copy {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<512xi32>
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 64][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<512xi32>
      %out1 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 64 len = 64)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
    }
  }
}

// -----

// The same, but the second copy runs before the output await retires the
// first: the proof has not yet fired, so this is a genuine overlap and still
// warns (the mechanism stays exact, it does not just silence every copy on a
// proven channel).
module @copy_over_copy_not_yet_proven {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%tile_0_0, MM2S, 0) {consumed_by = [#aie.consumed_by<output = @out, input_bytes = 1, output_bytes = 1>]}
    aie.shim_dma_allocation @out(%tile_0_0, S2MM, 0)
    aie.runtime_sequence @rounds(%a: memref<512xi32>, %c: memref<512xi32>) {
      // expected-note@+1 {{the earlier copy is here}}
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<512xi32>
      // expected-warning@+1 {{reuses buffer descriptor ID 1 on tile (0,0) while an earlier aiex.npu.dma_memcpy_nd of that ID may still be in flight}}
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 64][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @in} : memref<512xi32>
      %out0 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%c : memref<512xi32> offset = 0 len = 128)
        aie.end
      } {alloc = @out, issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
    }
  }
}
