// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: sed '/expected-warning/d' %s | aie-opt --aie-lower-dynamic-bd-pool \
// RUN:   --verify-diagnostics --split-input-file

// Unknown region semantics must not hide a nested release of a running task.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @nested_free(%buf: memref<256xi32>, %cond: i1) {
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      scf.execute_region {
        // expected-warning@+1 {{with no completion guarantee}}
        aiex.dma_free_task(%task)
        scf.yield
      }
    }
  }
}

// -----

// An unstructured conditional await cannot guarantee completion at the free.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @conditional_nested_await(%buf: memref<256xi32>, %cond: i1, %wait: i1) {
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      scf.execute_region {
        cf.cond_br %wait, ^await, ^free
      ^await:
        aiex.dma_await_task(%task)
        cf.br ^free
      ^free:
        // expected-warning@+1 {{with no completion guarantee}}
        aiex.dma_free_task(%task)
        scf.yield
      }
    }
  }
}
