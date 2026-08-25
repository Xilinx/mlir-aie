//===- bd_chain_on_core/iter_count.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-allocate --aie-objectfifo-lower-dmas %s | FileCheck %s

// `iterCount` on a compute tile's DMA endpoint ends the chain after its final
// iteration and carries the count on the channel's start queue.
//
// It counts the passes of one chain, and the two ends of a fifo run chains of
// their own: a `repeat_count` drainer replays each object, so it makes fewer
// passes than the filler opposite it.

module @iter_count_on_core {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %shim = aie.tile(0, 0)

    aie.objectfifo.pool @p(%tile02) {depth = 2 : i32,
        segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
    aie.objectfifo.core_endpoint @fill(%tile02) fills @p
    aie.objectfifo.dma_endpoint @drain(%tile02) drains @p {iterCount = 4 : i32}
    aie.route_endpoint @sink(%shim) DMA
    aie.route from @drain to [@sink]
  }
}

// CHECK: %mem_0_2 = aie.mem(%tile_0_2) {
// CHECK:   %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4, repeat_count = 3)
// CHECK: ^bb1:  // pred: ^bb0
// CHECK:   aie.dma_bd(%p_buff_0 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:   aie.next_bd ^bb2
// CHECK: ^bb2:  // pred: ^bb1
// CHECK:   aie.dma_bd(%p_buff_1 : memref<16xi32> offset = {{.*}} len = {{.*}})
// CHECK:   aie.next_bd ^bb3
// CHECK: ^bb3:  // pred: ^bb2
// CHECK:   aie.end
// CHECK: ^bb4:  // pred: ^bb0
// CHECK:   aie.end
