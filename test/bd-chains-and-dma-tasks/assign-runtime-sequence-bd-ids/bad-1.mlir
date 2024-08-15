//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// This test ensures that the proper error is emitted if a user tries to reference not-yet-lowered BD chains in aiex.dma_await_task.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
      aie.end
    }

    aiex.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
      // expected-note@+1 {{Lower this operation first using the --aie-materialize-bd-chains pass}}
      %t1 = aiex.dma_start_bd_chain @simple_chain(%arg0, %arg1, %arg2) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      // expected-error@+1 {{op does not reference a valid configure_task operation}}
      aiex.dma_await_task(%t1)
    }
  }
}

