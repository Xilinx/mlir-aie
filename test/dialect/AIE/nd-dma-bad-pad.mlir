//===- nd-dma-bad-pad.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

// CHECK-LABEL: module {
// CHECK:       }

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xi8>
    %mem = aie.memtile_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Inner-most padding-before count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<256xi8> offset = %c0_i32 len = %c8_i32 sizes = [4] strides = [1] pad [<const_pad_before = 2, const_pad_after = 2>] pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
