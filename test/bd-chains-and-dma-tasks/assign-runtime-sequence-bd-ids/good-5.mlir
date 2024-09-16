//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// This test ensures that all available 16 buffer descriptors are used.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.shim_dma_allocation @alloc0 (MM2S, 0, 0)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      // Allocate all available BD IDs
      %t1 = aiex.dma_configure_task_for @alloc0 {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }
}

