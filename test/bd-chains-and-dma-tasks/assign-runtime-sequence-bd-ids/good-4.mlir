//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// This test ensures that all available 16 buffer descriptors are used.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      // Allocate all available BD IDs
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 2 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 3 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 4 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t6 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 5 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t7 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 6 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t8 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t9 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 8 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t10 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 9 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t11 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 10 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t12 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 11 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t13 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 12 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t14 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 13 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t15 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 14 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t16 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 15 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
    }
  }
}

