//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// Rejected cases for the dynamic (runtime SSA size/stride/len) dma_task path.
// The dynamic BD-word encoder only covers the shim-NOC layout with an inner
// contiguous scan and no padding; anything else stays a clean diagnostic.

// A runtime size/stride/len on a non-shim (memtile) tile is not encodable.
module {
  aie.device(npu2) {
    %tile = aie.tile(1, 1)
    %buf = aie.buffer(%tile) {sym_name = "b"} : memref<4096xi32>
    aie.runtime_sequence(%len: i32) {
      %t = aiex.dma_configure_task(%tile, MM2S, 0) {
          // expected-error@+1 {{runtime-valued BD size/stride/len is only supported on shim NOC tiles}}
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// The innermost stride must be a compile-time constant 1.
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>, %s: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{innermost stride must be a compile-time constant 1}}
          aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 4096 sizes = [1, 8, 16, 32] strides = [4096, 512, 32, %s]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
