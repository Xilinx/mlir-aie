//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// This tests ensures that buffer descriptor IDs assigned to `aie.dma_bd` ops are reused after
// calls to aiex.dma_free_task and aiex.dma_await_task, but are unique otherwise.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 2 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 3 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 1) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 4 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 5 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }

      // The following is submitted to a different tile, so BD IDs should start from 0.
      %t4 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }

      aiex.dma_start_task(%t2)
      // After freeing task 2, BD IDs 2 and 3 should become available again on tile 0 0
      aiex.dma_free_task(%t2)

      %t6 = aiex.dma_configure_task(%tile_0_0, S2MM, 1) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 2 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 3 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }

      aiex.dma_start_task(%t3)
      // Awaiting a task should free its BD IDs too
      aiex.dma_await_task(%t3)

      %t7 = aiex.dma_configure_task(%tile_0_0, S2MM, 1) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 4 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 5 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
    }
  }
}

