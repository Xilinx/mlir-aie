//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s
// XFAIL:*

// This test ensures that the correct error gets emitted when a BD "chain" is not
// actually a proper chain, i.e. some blocks are not connected.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.bd_chain @simple_chain(%buf: memref<8xi16>) {
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.end
        ^bd2:
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            // expected-error@+1 {{Block ending in this terminator does not form a chain with entry block}}
            aie.end
    }

    aiex.runtime_sequence(%buf: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }
  }
}

