//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures that the correct error is emitted if an illegal data layout transformation is specified
// in a `aie.dma_bd` operation inside the runtime sequence.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c4_i32 = arith.constant 4 : i32
      %c8_i32 = arith.constant 8 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{Stride 0 is 2 elements * 1 bytes = 2 bytes, which is not divisible by 4}}
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c4_i32 len = %c8_i32 sizes = [8] strides = [2]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

