//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{unreachable}}
    %buf = aie.buffer(%tile_0_2) : memref<8xi16>

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>) {
            aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
            aie.next_bd ^bd1
        ^bd1:
            // expected-note@+1{{user}}
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.end
    }

    aiex.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @simple_chain(%arg0) : (memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }
  }
}

