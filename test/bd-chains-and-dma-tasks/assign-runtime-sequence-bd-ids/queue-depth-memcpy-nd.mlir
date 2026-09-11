//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-to-npu --verify-diagnostics --split-input-file %s

// The npu.dma_memcpy_nd path shares the task queue with dma_start_task, so it
// gets the same bound. aie-dma-to-npu is pattern-driven and visits ops in
// worklist order, so the count runs as a program-order pre-walk; the rule
// itself is shared (DmaQueueModel.h) so the two paths cannot disagree about the
// same hardware.

// Five pushes on one channel: the fifth is the first that can overflow. Only
// the first occurrence per channel is reported.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @alloc0} : memref<1280xi32>
    // expected-warning@+1 {{whose task queue is only 4 deep, with 4 push(es) not yet known to have completed}}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// Four pushes fit exactly, so this must stay quiet.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// An npu.dma_wait retires the outstanding issue_token push and everything
// queued ahead of it, so the fifth push lands in a drained queue.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, issue_token = true, metadata = @alloc0} : memref<1280xi32>
    aiex.npu.dma_wait {symbol = @alloc0}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1024][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @alloc0} : memref<1280xi32>
  }
}

// -----

// Independent channels have independent queues, so four on each is fine.
aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @in0 (%tile_0_0, MM2S, 0)
  aie.shim_dma_allocation @out0 (%tile_0_0, S2MM, 0)
  aie.runtime_sequence (%arg0: memref<1280xi32>, %arg1: memref<1280xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @out0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @in0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 256][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @out0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 4 : i64, metadata = @in0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 512][1, 1, 1, 256][0, 0, 0, 1]) {id = 5 : i64, metadata = @out0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 6 : i64, metadata = @in0} : memref<1280xi32>
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 768][1, 1, 1, 256][0, 0, 0, 1]) {id = 7 : i64, metadata = @out0} : memref<1280xi32>
  }
}
