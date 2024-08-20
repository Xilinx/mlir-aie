//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      // expected-note@+1 {{Error encountered}}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+2 {{Cannot lower buffer descriptor without assigned ID}}
          // expected-note@+1 {{Run the `--aie-assign-runtime-sequence-bd-ids` pass first or manually assign an ID to this buffer descriptor}}
          aie.dma_bd(%arg0 : memref<32xi8>, 3, 4)
          aie.end
      }
    }
  }
}

