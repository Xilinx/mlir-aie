//===- decompose_large_dma_task_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// aie-decompose-large-dma-bd rejects an out-of-order BD that needs splitting.

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @ooo_too_large(%in: memref<4096xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{splitting an out-of-order buffer descriptor into multiple descriptors is not implemented}}
        aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 2062 sizes = [1, 1, 1031, 2] strides = [0, 0, 3, 1])
          {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>, out_of_order_id = 5 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Decomposition that moves extent into the iteration dimension has to scale the
// task's repeat count to match. A runtime repeat count cannot be scaled at
// compile time, so the BD is rejected rather than left under-running.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @runtime_repeat(%in: memref<16x16x4096xi8>, %r: i32) {
      %tk = aiex.dma_configure_task_for @a repeat %r : i32 {
        // expected-error@+1 {{cannot decompose a buffer descriptor whose repeat count is a runtime value: decomposition needs to scale it by 8}}
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
    }
  }
}

// -----

// The scaled repeat count has to fit the queue's 8-bit field. Saying so here
// names the factor that got us there, instead of failing later at the push.

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @repeat_overflows(%in: memref<16x16x4096xi8>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error@+1 {{decomposition scales the repeat count by 8 to 1607, beyond the [0:255] a queue push can carry}}
        aie.dma_bd(%in : memref<16x16x4096xi8> offset = 4096 len = 262144 sizes = [1, 8, 8, 4096] strides = [0, 131072, 8192, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 200 : i32}
      aiex.dma_start_task(%tk)
    }
  }
}
