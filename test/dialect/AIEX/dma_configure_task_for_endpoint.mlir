// RUN: aie-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A runtime task may name a route endpoint's DMA channel on any tile; the
// buffers it moves are checked once allocation turns it into a
// dma_configure_task on that tile. Unlike a shim allocation's, its BDs can
// take the tile's locks.

// CHECK-LABEL: @memtile_task
// CHECK: aiex.dma_configure_task_for @b_in {
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
module @memtile_task {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    %b = aie.buffer(%mt) {sym_name = "b"} : memref<64xi32>
    %free = aie.lock(%mt) {init = 1 : i32}
    %full = aie.lock(%mt) {init = 0 : i32}
    aie.route_endpoint @b_in(%mt) DMA
    aie.runtime_sequence() {
      %t = aiex.dma_configure_task_for @b_in {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%free, AcquireGreaterEqual, %c1)
        aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
        aie.use_lock(%full, Release, %c1)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}

// -----

aie.device(npu2) {
  %core = aie.tile(0, 2)
  %b = aie.buffer(%core) {sym_name = "b"} : memref<64xi32>
  aie.route_endpoint @port(%core) Core {channelIndex = 0 : i32}
  aie.runtime_sequence() {
    // expected-error @+1 {{'@port' names a Core port, not a DMA channel}}
    %t = aiex.dma_configure_task_for @port {
      aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64)
      aie.end
    }
    aiex.dma_start_task(%t)
  }
}
