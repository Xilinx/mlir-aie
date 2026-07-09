//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s
// XFAIL:*

// This test ensures that the correct error gets emitted when a BD "chain" is not
// actually a proper chain, i.e. some blocks are not connected.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.bd_chain @simple_chain(%buf: memref<8xi16>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
            aie.dma_bd(%buf : memref<8xi16> offset = %c0_i32 len = %c8_i32)
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%buf : memref<8xi16> offset = %c0_i32 len = %c8_i32)
            aie.end
        ^bd2:
            aie.dma_bd(%buf : memref<8xi16> offset = %c0_i32 len = %c8_i32)
            // expected-error@+1 {{Block ending in this terminator does not form a chain with entry block}}
            aie.end
    }

    aie.runtime_sequence(%buf: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }
  }
}

