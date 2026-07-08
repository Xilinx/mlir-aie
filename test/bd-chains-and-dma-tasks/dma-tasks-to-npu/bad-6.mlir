//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures that the proper error is emitted if the user specifies an illegal offset in a dma_bd operation inside the runtime sequence.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{Offset must be aligned to 4 byte boundary}}
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c3_i32 len = %c4_i32 sizes = [] strides = []) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

