//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures that the proper error is emitted if the user attempts to sepcify more than
// the architecturally possible number of data layout transformation dimensions in a `aie.dma_bd` 
// operation inside the runtime sequence.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{At most four data layout transformation dimensions may be provided}}
          aie.dma_bd(%arg0 : memref<32xi8>, 4, 32,
                     [<size=1, stride=4>, <size=1, stride=4>, <size=2, stride=4>, <size=2, stride=8>, <size=4, stride=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

