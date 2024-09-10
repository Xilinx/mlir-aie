//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures the proper error is emitted if a task with no BDs is issued in the runtime sequence.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      // expected-note@+1 {{Error encountered}}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        ^bb0:
          aie.next_bd ^bb1
        ^bb1:
          // expected-error@+1 {{Block ending in this terminator does not contain a required}}
          aie.end
      }
    }
  }
}

