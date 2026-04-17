//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// This test verifies that when a large-length dma_bd has mismatched dimensions,
// the error message correctly reports the byte count without integer overflow.
//
// memref<536870912xi64>: 536870912 elements * 8 bytes = 4294967296 bytes.
// BD length in 32-bit words = 4294967296 / 4 = 1073741824.
// The lowest three dims product: sizes[0] = 2*64/32 = 4 words, sizes[1] = sizes[2] = 1.
// Total = 4 words = 16 bytes, which does not match 4294967296 bytes.
// The error message must report "4294967296 bytes" (not a truncated 32-bit value
// like "0 bytes" that the overflow bug would produce) and "16 bytes" for the dims.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<536870912xi64>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+2 {{Buffer descriptor length does not match length of transfer expressed by lowest three dimensions of data layout transformation strides/wraps. BD length is 4294967296 bytes. Lowest three dimensions of data layout transformation would result in transfer of 16 bytes.}}
          // expected-note@+1 {{}}
          aie.dma_bd(%arg0 : memref<536870912xi64>, 0, 536870912,
                     [<size=1, stride=1>, <size=1, stride=1>, <size=2, stride=1>]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
