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
