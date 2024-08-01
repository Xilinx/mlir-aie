//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-materialize-bd-chains %s
// XFAIL: *
// Referencing locks inside sequence function not yet implemented

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.bd_chain @simple_chain(%buf : memref<9xi16>) {
      aie.end
    }

    aiex.runtime_sequence(%buf: memref<8xi16>) {
      // expected-error@+1 {{Argument 1 types mismatch}}
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf) : (memref<8xi16>) 
                                    on (%tile_0_0, MM2S, 0) 
      aiex.dma_await_task(%t1)
    }

  }
}

