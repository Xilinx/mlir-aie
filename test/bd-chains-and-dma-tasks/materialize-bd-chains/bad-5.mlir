//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{unreachable}}
    %buf = aie.buffer(%tile_0_2) : memref<8xi16>

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
            aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32)
            aie.end
    }

    aie.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
      // expected-note@+1{{user}}
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }
  }
}

