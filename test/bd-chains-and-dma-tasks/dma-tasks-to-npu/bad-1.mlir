//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures that the proper error is emitted if the transfer length specified in a aie.dma_bd
// op's dimensions and its overall transfer length do not match up.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c4_i32 = arith.constant 4 : i32
      %c32_i32 = arith.constant 32 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+2 {{Buffer descriptor length does not match length of transfer}}
          // expected-note@+1 {{}}
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c4_i32 len = %c32_i32 sizes = [2, 2, 4] strides = [4, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

