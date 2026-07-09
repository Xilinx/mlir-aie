//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// This test ensures the proper error is emitted if a user attempts to configure a task 
// that accesses a buffer that is inaccessible from the tile it is configured to run on.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{accessed from an unreachable tile}}
    %buf = aie.buffer(%tile_0_2) {addr = 0xBEEF : i32} : memref<32xi8> 

    aie.runtime_sequence(%arg0: memref<32xi8>) {
      %c4_i32 = arith.constant 4 : i32
      %c32_i32 = arith.constant 32 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-note@+1 {{user}}
          aie.dma_bd(%buf : memref<32xi8> offset = %c4_i32 len = %c32_i32) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

