//===- external_pool.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-verify %s

// The host fills an input external buffer; the shim DMA drains it.
module @input {
  aie.device(xcvc1902) {
    %shim = aie.tile(2, 0)
    %tile = aie.tile(2, 2)
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<32xi32>
    aie.objectfifo.pool @pool(%shim) {depth = 1 : i32, buffers = [@buffer]}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.dma_endpoint @dma(%shim) drains @pool
    aie.route_endpoint @sink(%tile) Core {channelIndex = 0 : i32}
    aie.route from @dma to [@sink]
  }
}

// -----

// The shim DMA fills an output external buffer; the host drains it.
module @output {
  aie.device(xcvc1902) {
    %shim = aie.tile(2, 0)
    %tile = aie.tile(2, 2)
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<32xi32>
    aie.objectfifo.pool @pool(%shim) {depth = 1 : i32, buffers = [@buffer]}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.dma_endpoint @dma(%shim) fills @pool
    aie.route_endpoint @source(%tile) Core {channelIndex = 0 : i32}
    aie.route from @source to [@dma]
  }
}

// -----

// An external buffer without a device-side actor is incomplete.
module @no_device_actor {
  aie.device(xcvc1902) {
    %shim = aie.tile(2, 0)
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<32xi32>
    // expected-error@+1 {{segment 0 has no drainer}}
    aie.objectfifo.pool @pool(%shim) {depth = 1 : i32, buffers = [@buffer]}
      : memref<32xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
    }
  }
}
