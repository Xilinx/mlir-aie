//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 
       
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0xBEEF : i32 } : memref<32xi8> 

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // expected-error@+1 {{Padding requires n-d data layouts expressed as wrap(s) and stride(s).}} 
          aie.dma_bd(%buf : memref<32xi8>, 4, 16,
                     [], [<const_pad_before=2, const_pad_after=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

