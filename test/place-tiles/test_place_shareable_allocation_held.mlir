//===- test_place_shareable_allocation_held.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim allocation with aie.shareable_with holds its channel from every end
// its list does not name, as allocation does. This one names none, and MM2S 1
// is reserved outright, so the logical shim pinned to (0, 0) has no MM2S
// channel left: the sequential placer refuses and the SA charges it. Before,
// both counted the shareable allocation nowhere, and allocation then failed
// ("requires at least 3 MM2S channels").

// RUN: not aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s --check-prefix=SEQ
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA

// SEQ: error: tile (0, 0) requires 0 input/1 output DMA channels, but only 2 input/0 output available
// SA: warning: SA placer's resource estimate is over capacity

module @shareable_alloc_uncounted {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %ls = aie.logical_tile<ShimNOCTile>(0, 0)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.shim_dma_allocation @shared(%shim, MM2S, 0) {aie.shareable_with = []}
    aie.shim_dma_allocation @reserved(%shim, MM2S, 1)
    aie.objectfifo @a(%ls, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
  }
}
