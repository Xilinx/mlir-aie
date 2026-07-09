//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s 

// This test ensures the proper error is emitted if a single block inside a `aiex.dma_configure_task` op
// contains multiple `aie.dma_bd` operations -- only one such operation is allowed per basic block.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c3_i32 len = %c4_i32)
          // expected-note@+1 {{Extra}}
          aie.dma_bd(%arg0 : memref<32xi8> offset = %c3_i32 len = %c4_i32)
          // expected-error@+1 {{This block contains multiple}}
          aie.end
      }
    }
  }
}

