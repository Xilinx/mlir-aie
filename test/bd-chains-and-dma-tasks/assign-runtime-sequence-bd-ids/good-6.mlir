//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// RUN: aie-opt --aie-substitute-shim-dma-allocations %s | FileCheck %s

// This test ensures that packet info in the shim_dma_alloc op is lowered to dma_configure_task op.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.shim_dma_allocation @alloc0 (MM2S, 0, 0, <pkt_type = 0, pkt_id = 2>)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%{{.*}}tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>)
      %t1 = aiex.dma_configure_task_for @alloc0 {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
}

