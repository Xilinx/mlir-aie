//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// A MemTile BD has pad fields for d0..d2 only.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0 : i32 } : memref<64xi32>

    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        // expected-error@+1 {{D3 zero padding is not supported on this tile.}}
        aie.dma_bd(%buf : memref<64xi32> offset = 0 len = 64 sizes = [2, 2, 2, 8] strides = [32, 16, 8, 1] pad [<const_pad_before = 1, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
