//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s
       
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0xBEEF : i32 } : memref<32xi8> 

    aiex.runtime_sequence(%arg0: memref<32xi8>) {
      // expected-error@+1 {{Mismatch number of dimensions between padding(s) and wrap(s) and stride(s).}} 

      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf : memref<32xi8>, 4, 16,
                     [<size=2, stride=4>], [<const_pad_before=2, const_pad_after=1>, <const_pad_before=1, const_pad_after=1>]) 
                     {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

