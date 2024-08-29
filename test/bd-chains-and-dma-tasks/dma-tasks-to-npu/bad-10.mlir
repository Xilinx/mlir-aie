//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// This test ensures the proper error is emitted if a user attempts to configure a task 
// that accesses a buffer that is inaccessible from the tile it is configured to run on.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{accessed from an unreachable tile}}
    %buf = aie.buffer(%tile_0_2) {addr = 0xBEEF : i32} : memref<32xi8> 

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-note@+1 {{user}}
          aie.dma_bd(%buf : memref<32xi8>, 4, 32) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

