// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics %s

// Enforcement cannot prove progress when the target has no pollable status
// register. The oldest task stays potentially live after queue truncation.
aie.device(xcve3858) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @unpollable(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+2 {{whose task queue is only 4 deep}}
      // expected-note@+1 {{this target reports no pollable task-queue occupancy register}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}
