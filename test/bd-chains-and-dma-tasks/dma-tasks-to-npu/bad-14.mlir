//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 
       
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0xBEEF : i32 } : memref<32xi8> 

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          // expected-error@+1 {{Padding requires n-d data layouts expressed as wrap(s) and stride(s).}} 
          aie.dma_bd(%buf : memref<32xi8> offset = 4 len = 16 pad [<const_pad_before=2, const_pad_after=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

