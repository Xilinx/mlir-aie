//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// bd_pool_init seeds the free list with every id on the tile, and pop hands
// out id 0 first. But the BD table is one per tile, shared with whatever the
// static allocator already placed there -- so on a tile carrying both a
// static shim BD and this pool, an unreserved pop would hand out the static
// BD's id and overwrite its slot.
//
// Here tile (0, 0) carries a static shim BD at id 0 (MM2S channel 0) and a
// runtime sequence that draws from the pool for a second stream (MM2S
// channel 1). The pool must reserve id 0 right after init, so its first pop
// returns something other than 0.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s \
// RUN: | aie-translate --aie-npu-to-cpp | FileCheck %s

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_
// CHECK: aie_runtime::BdPool bd_pool_0_0 = aie_runtime::bd_pool_init(16);
// The static shim BD's id (0) is withheld right after the pool is seeded.
// CHECK-NEXT: aie_runtime::bd_pool_reserve(bd_pool_0_0, 0);
// CHECK: uint32_t bd_{{[0-9]+}}; if (!aie_runtime::bd_pool_pop(bd_pool_0_0, bd_{{[0-9]+}})) return std::nullopt;

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %sb = aie.external_buffer {sym_name = "sb"} : memref<8xi32>

  // Static shim BD, statically assigned id 0.
  aie.shim_dma(%tile_0_0) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.dma_bd(%sb : memref<8xi32> offset = 0 len = 4) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 1)
  aie.runtime_sequence @pool(%in: memref<8192xi32>) {
    %bd = aiex.dma_bd_pool_pop(0, 0) : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
      aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1]) bd_id_val %bd : i32
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
  }
}
