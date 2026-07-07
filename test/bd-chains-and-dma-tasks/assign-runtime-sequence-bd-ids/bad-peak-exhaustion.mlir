//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics --split-input-file %s

// Peak simultaneous BD liveness exceeds the tile's BD pool. An npu2 shim tile
// has 16 BDs; configuring 17 tasks that are all held at once (never freed)
// cannot be allocated. This is a genuine hardware limit and is rejected with a
// clear diagnostic rather than silently overflowing the pool.

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  // expected-error@+1 {{Too many simultaneously active buffer descriptors}}
  aie.runtime_sequence(%arg0: memref<8xi16>) {
    %t0  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t1  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t2  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t3  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t4  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t5  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t6  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t7  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t8  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t9  = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
    %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) { aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) aie.end }
  }
}
