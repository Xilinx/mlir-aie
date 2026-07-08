//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-substitute-shim-dma-allocations %s | FileCheck %s

// This test ensures that packet info in the shim_dma_alloc op is lowered to dma_configure_task op.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.shim_dma_allocation @alloc0 (%tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>)

    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%{{.*}}tile_0_0, MM2S, 0, <pkt_type = 0, pkt_id = 2>)
      %t1 = aiex.dma_configure_task_for @alloc0 {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = [])
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
}

