//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures the proper error is emitted if a single block inside a `aiex.dma_configure_task` op
// contains multiple `aie.dma_bd` operations -- only one such operation is allowed per basic block.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8>, 3, 4)
          // expected-note@+1 {{Extra}}
          aie.dma_bd(%arg0 : memref<32xi8>, 3, 4)
          // expected-error@+1 {{This block contains multiple}}
          aie.end
      }
    }
  }
}

