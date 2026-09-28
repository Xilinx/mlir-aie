// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-dma-tasks-to-npu --aie-lower-dma-channel-reset --aie-dma-to-npu --split-input-file %s | FileCheck %s

// Task starts, explicit queue pushes and memcpy transfers use the same channel.
// Neither high-level path sees five starts by itself.
// CHECK-LABEL: @mixed
// CHECK-COUNT-4: aiex.npu.write32
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @mixed(%arg0: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256) {bd_id = 7 : i32}
      aie.end
    }
    aiex.dma_start_task(%task)
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc} : memref<256xi32>
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 2 : i64, metadata = @alloc} : memref<256xi32>
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 3 : i64, metadata = @alloc} : memref<256xi32>
  }
}

// -----

// A lowered wait must retire the token from a previously lowered task start.
// CHECK-LABEL: @mixed_wait
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @mixed_wait(%arg0: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256) {bd_id = 7 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%task)
    aiex.npu.dma_wait {symbol = @alloc}
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 1 : i64, metadata = @alloc} : memref<256xi32>
  }
}

// -----

// Compiler-generated resident-channel rearm pushes count too. Four later raw
// starts fill the channel; the last one must be guarded.
// CHECK-LABEL: @rearm
// CHECK-COUNT-4: aiex.npu.write32
// CHECK: aiex.npu.maskpoll
// CHECK: aiex.npu.write32
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  %tile = aie.tile(0, 3)
  %pl = aie.lock(%tile, 0) {init = 1 : i32}
  %cl = aie.lock(%tile, 1) {init = 0 : i32}
  aie.objectfifo_rearm_binding @binding channels(%tile : index) locks(%pl, %cl : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1, 0>, head_bd_ids = array<i32: 5>, repeat_counts = array<i32: 0>}
  aie.runtime_sequence @rearm() {
    %c0 = arith.constant 0 : i32
    aiex.dma_channel_reset_for(@binding)
    aiex.npu.push_queue (0, 3, S2MM:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 3, S2MM:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 3, S2MM:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 3, S2MM:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}
