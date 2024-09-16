//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --verify-diagnostics --aie-assign-runtime-sequence-bd-ids %s

// This test ensures that automatic buffer descriptor allocation does not collide 
// when there are user-specified hard-coded BD IDs in the input.

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aiex.runtime_sequence(%arg0: memref<8xi16>) {
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 7 : i32}
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 1) {
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.next_bd ^bb1
      ^bb1:
      // CHECK:  aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 2 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
    }
  }
}

