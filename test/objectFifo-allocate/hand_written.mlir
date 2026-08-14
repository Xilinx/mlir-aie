// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whatever a design writes down for itself is kept: hand-placed buffers, locks
// already attached to a segment, and a pinned channel.

module @hand_written {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    %mine_0 = aie.buffer(%tile12) {sym_name = "mine_0", address = 4096 : i32} : memref<16xi32>
    %mine_1 = aie.buffer(%tile12) {sym_name = "mine_1", address = 8192 : i32} : memref<16xi32>
    %free = aie.lock(%tile12) {init = 2 : i32, sym_name = "free"}
    %full = aie.lock(%tile12) {init = 0 : i32, sym_name = "full"}

    aie.objectfifo.pool @prod_pool(%tile12) {
      depth = 2 : i32,
      buffers = [@mine_0, @mine_1],
      segments = [#aie.objectfifo_segment<offset = 0, size = 16,
                                          produceLock = @free, consumeLock = @full>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool {pinnedChannel = 1 : i32}

    aie.objectfifo.pool @cons_pool(%tile33) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons_pool

    aie.objectfifo.flow from @prod_dma to [@cons_dma]
  }
}

// CHECK-LABEL: @hand_written
// CHECK-NOT: prod_buff_0
// CHECK-NOT: prod_prod_lock_0
// CHECK: aie.objectfifo.pool @prod_pool({{.*}}) {buffers = [@mine_0, @mine_1], depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16, produceLock = @free, consumeLock = @full>]}
// CHECK: aie.objectfifo.dma_endpoint @prod_dma({{.*}}) drains @prod_pool {channel = #aie.objectfifo_channel<MM2S : 1>, pinnedChannel = 1 : i32}
// CHECK: aie.flow({{.*}}, DMA : 1, {{.*}}, DMA : 0)
