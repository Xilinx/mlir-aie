//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 
       
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) { address = 0xBEEF : i32 } : memref<32xi8> 

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c4_i32 = arith.constant 4 : i32
      %c16_i32 = arith.constant 16 : i32
      %t1 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
      // expected-error@+1 {{Padding is supported only on MemTiles.}} 
          aie.dma_bd(%buf : memref<32xi8> offset = %c4_i32 len = %c16_i32 pad [<const_pad_before=2, const_pad_after=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

