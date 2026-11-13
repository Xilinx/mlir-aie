//===- bad_memtile_task_placement.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -o /dev/null
// RUN: aie-opt --aie-place-tiles --verify-diagnostics %s -o /dev/null

// Unknown coordinates defer the locality check; placement resolves these
// resources to the east neighbour and must then reject both accesses.
aie.device(npu1_2col) {
  %tile = aie.tile(0, 1)
  %logical = aie.logical_tile<MemTile>(1, ?)
  // expected-note@+1 {{buffer}}
  %buf = aie.buffer(%logical) : memref<16xi32>
  // expected-note@+1 {{lock}}
  %lock = aie.lock(%logical, 0) {init = 1 : i32}
  aie.runtime_sequence() {
    %task = aiex.dma_configure_task(%tile, MM2S, 4) {
      %c1 = arith.constant 1 : i32
      // expected-error@+1 {{'aie.use_lock' op is in a task on memtile DMA channel 4 and attempts to access a non-local lock; only channels below 4 reach neighbouring tiles}}
      aie.use_lock(%lock, AcquireGreaterEqual, %c1)
      // expected-error@+1 {{'aie.dma_bd' op is in a task on memtile DMA channel 4 and attempts to access a non-local buffer; only channels below 4 reach neighbouring tiles}}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
  }
}
