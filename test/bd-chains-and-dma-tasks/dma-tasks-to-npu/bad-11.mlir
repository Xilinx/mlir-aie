//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// This test ensures the proper error is emitted if a user tries to lower a 
// BD in a aiex.dma_configure_task operation in the runtime sequence before
// the address of all referenced buffers is known.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_1) : memref<32xi8> 

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // expected-error@+1 {{without associated address}}
          aie.dma_bd(%buf : memref<32xi8>, 4, 32) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

