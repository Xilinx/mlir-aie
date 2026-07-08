//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// This test ensures that the proper error is issued if the user tries to reuse buffer descriptor IDs
// withou explicit ops `aiex.dma_free_task` or `aiex.dma_await_task` between them.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) {bd_id = 7 : i32}
        aie.end
      }
      // Reuse BD ID without explicit free
      // expected-error@+1 {{Specified buffer descriptor ID 7 is already in use}}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = %c0_i32 len = %c8_i32 sizes = [] strides = []) {bd_id = 7 : i32}
        aie.end
      }
    }
  }
}

