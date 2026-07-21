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
          // expected-error@+1 {{runtime-valued BD size/stride/len/bd_id is only supported on shim NOC tiles}}
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// A constant innermost stride whose byte extent isn't a whole granule is not
// realizable (int8, stride 2 = 16 bits vs the 32-bit granule). A runtime len
// forces the dynamic path. (A unit or granule-aligned innermost stride, runtime
// or constant, is fine -- see runtime-len.mlir.)
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi8>, %len: i32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{stride 0 is 2 elements at 1 bytes each, not a multiple of the 4-byte address-gen granule}}
          aie.dma_bd(%arg0 : memref<4096xi8> offset = 0 len = %len sizes = [1, 8, 16, 4] strides = [4096, 512, 4, 2]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
