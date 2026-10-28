//===- bad_dma_bd_pool_partition.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  %tile_0_2 = aie.tile(0, 2)
  aie.runtime_sequence() {
    // expected-error@+1 {{partition [8, 24) is not a nonempty range of the 16 buffer descriptors on tile (0, 2)}}
    %bd = aiex.dma_bd_pool_pop(0, 2) partition [8, 24) : i32
  }
}

// -----

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  aie.runtime_sequence(%bd: i32) {
    // expected-error@+1 {{partition [24, 24) is not a nonempty range of the 48 buffer descriptors on tile (0, 1)}}
    aiex.dma_bd_pool_push(0, 1) partition [24, 24) bd_id %bd : i32
  }
}
