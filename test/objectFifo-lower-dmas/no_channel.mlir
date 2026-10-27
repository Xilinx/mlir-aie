// RUN: aie-opt --aie-objectfifo-lower-dmas --verify-diagnostics %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An endpoint reaches this pass without a channel only if allocation was
// skipped.

module @no_channel {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    aie.buffer @b0(%tile12) : memref<16xi32>
    aie.lock @free(%tile12) {init = 1 : i32}
    aie.lock @full(%tile12) {init = 0 : i32}
    aie.objectfifo.pool @pool(%tile12) {
      depth = 1 : i32, buffers = [@b0]
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {consumeLock = @full, offset = 0 : i32, produceLock = @free, size = 16 : i32}
    }
    // expected-error @+1 {{has no channel; run --aie-objectfifo-allocate}}
    aie.objectfifo.dma_endpoint @dma(%tile12) drains @pool
  }
}
