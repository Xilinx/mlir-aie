//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      // Allocate more than there are available BD IDs
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t6 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t7 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t8 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t9 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      // expected-error@+1 {{Allocator exhausted available }}
      %t17 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
    }
  }
}

