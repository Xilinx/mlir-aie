//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids))' %s | FileCheck %s

// @main calls @p0 and then @p1, which both send on shim (0, 0) MM2S 0. @p0
// frees its task without awaiting it, so when @p1 reprograms the same BD id
// in the inlined sequence, the task may still be running on it.
// CHECK-LABEL: module @free_then_reuse
// CHECK:       aie.runtime_sequence @main_seq
// CHECK-COUNT-2: {bd_id = 0 : i32}
module @free_then_reuse {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @main_seq(%a: memref<4096xi32>, %b: memref<4096xi32>) {
      aiex.configure @dev {
        aiex.run @p0(%a) : (memref<4096xi32>)
        aiex.run @p1(%b) : (memref<4096xi32>)
      }
    }
  }
  aie.device(npu2_1col) @dev {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%t00, MM2S, 0)
    aie.shim_dma_allocation @in1(%t00, MM2S, 0)
    aie.runtime_sequence @p0(%a: memref<4096xi32>) {
      %t = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 4096)
        aie.end
      }
      aiex.dma_start_task(%t)
      // expected-note @below {{released here}}
      aiex.dma_free_task(%t)
    }
    aie.runtime_sequence @p1(%b: memref<4096xi32>) {
      %t = aiex.dma_configure_task_for @in1 {
        // expected-warning@+2 {{reuses buffer descriptor ID 0 on tile (0,0) after it was released by an aiex.dma_free_task that had no completion guarantee}}
        // expected-note@+1 {{not proven finished: aie-objectfifo-verify found no output whose progress proves @in0 consumed}}
        aie.dma_bd(%b : memref<4096xi32> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// @p0 awaits its task before freeing it, so @p1 reuses BD id 0 silently.
// CHECK-LABEL: module @await_then_reuse
// CHECK:       aie.runtime_sequence @main_seq
// CHECK-COUNT-2: {bd_id = 0 : i32}
module @await_then_reuse {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @main_seq(%a: memref<4096xi32>, %b: memref<4096xi32>) {
      aiex.configure @dev {
        aiex.run @p0(%a) : (memref<4096xi32>)
        aiex.run @p1(%b) : (memref<4096xi32>)
      }
    }
  }
  aie.device(npu2_1col) @dev {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0(%t00, MM2S, 0)
    aie.shim_dma_allocation @in1(%t00, MM2S, 0)
    aie.runtime_sequence @p0(%a: memref<4096xi32>) {
      %t = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a : memref<4096xi32> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
    aie.runtime_sequence @p1(%b: memref<4096xi32>) {
      %t = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%b : memref<4096xi32> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
