//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// This test ensures that the proper error is issued if the user tries to reuse buffer descriptor IDs
// withou explicit ops `aiex.dma_free_task` or `aiex.dma_await_task` between them.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.end
      }
      // Reuse BD ID without explicit free
      // expected-error@+1 {{Specified buffer descriptor ID 7 is already in use}}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.end
      }
    }
  }
}

