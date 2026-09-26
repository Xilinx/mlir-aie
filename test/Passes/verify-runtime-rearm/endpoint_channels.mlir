// RUN: not aie-opt --split-input-file --aie-verify-runtime-rearm %s 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-verify-runtime-rearm %s | FileCheck %s --check-prefix=ALLOCATED

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Endpoint-backed starts require channel allocation, not an asserting accessor.
// The same core and mem tile programs pass verification after allocation.

// ERROR: error: 'aie.dma_start' op requires an allocated DMA channel; run --aie-objectfifo-allocate before --aie-verify-runtime-rearm
// ALLOCATED-LABEL: module @core_endpoint
// ALLOCATED: aie.dma_start(S2MM, 0,
// ALLOCATED: aiex.dma_channel_reset
// ALLOCATED: aiex.set_lock
module @core_endpoint {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %lock = aie.lock(%t, 0) {init = 1 : i32}
    aie.route_endpoint @source(%t) Core {channelIndex = 0 : i32}
    aie.route_endpoint @input(%t) DMA
    aie.route from @source to [@input]
    aie.mem(%t) {
      aie.dma_start(S2MM, @input, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, S2MM, 0)
      aiex.set_lock(%lock, 1)
    }
  }
}

// -----

// ERROR: error: 'aie.dma_start' op requires an allocated DMA channel; run --aie-objectfifo-allocate before --aie-verify-runtime-rearm
// ALLOCATED-LABEL: module @mem_endpoint
// ALLOCATED: aie.dma_start(MM2S, 0,
// ALLOCATED: aiex.dma_channel_reset
// ALLOCATED: aiex.set_lock
module @mem_endpoint {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<16xi32>
    %lock = aie.lock(%t, 0) {init = 1 : i32}
    aie.route_endpoint @output(%t) DMA
    aie.route_endpoint @sink(%core) Core {channelIndex = 0 : i32}
    aie.route from @output to [@sink]
    aie.memtile_dma(%t) {
      aie.dma_start(MM2S, @output, ^bd0, ^end)
    ^bd0:
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }
    aie.runtime_sequence() {
      aiex.dma_channel_reset(%t, MM2S, 0)
      aiex.set_lock(%lock, 1)
    }
  }
}
