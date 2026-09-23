//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-translate --split-input-file %s --aie-npu-to-cpp | FileCheck %s

// A mem tile partitions its 48 buffer descriptors by channel PARITY: an even
// channel can only submit ids 0-23, an odd channel only 24-47
// (AIETargetModel::isBdChannelAccessible). The runtime free-list therefore
// cannot be one flat pool per tile -- an odd channel drawing from it would be
// handed an id it can never submit, and the transfer would simply never run.
//
// Pools are keyed by the id range rather than by channel, so channels sharing
// a partition still share one pool and cannot be handed the same id twice.

// CHECK: BdPool bd_pool_0_1_0 = aie_runtime::bd_pool_init_range(0, 24);
// CHECK: BdPool bd_pool_0_1_24 = aie_runtime::bd_pool_init_range(24, 48);
// CHECK: bd_pool_pop(bd_pool_0_1_0,
// CHECK: bd_pool_pop(bd_pool_0_1_24,
// CHECK: bd_pool_push(bd_pool_0_1_0,
// CHECK: bd_pool_push(bd_pool_0_1_24,

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<1024xi32>
  aie.runtime_sequence @memtile_parity_pools() {
    %even = aiex.dma_bd_pool_pop(0, 1, 0) : i32
    %odd = aiex.dma_bd_pool_pop(0, 1, 1) : i32
    aiex.dma_bd_pool_push(0, 1, 0) bd_id %even : i32
    aiex.dma_bd_pool_push(0, 1, 1) bd_id %odd : i32
  }
}

// -----

// A shim tile reports every id accessible from every channel, so both channels
// share one pool over the whole table -- splitting it by parity there would
// halve the ids available to each.

// CHECK: BdPool bd_pool_0_0_0 = aie_runtime::bd_pool_init_range(0, 16);
// CHECK-NOT: BdPool bd_pool_0_0_
// CHECK: bd_pool_pop(bd_pool_0_0_0,
// CHECK: bd_pool_pop(bd_pool_0_0_0,

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @shim_shares_one_pool() {
    %a = aiex.dma_bd_pool_pop(0, 0, 0) : i32
    %b = aiex.dma_bd_pool_pop(0, 0, 1) : i32
  }
}
