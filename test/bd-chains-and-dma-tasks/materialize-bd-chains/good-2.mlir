//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s
// XFAIL: *
// Referencing locks inside sequence function not yet implemented

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %lock_0 = aie.lock(%tile_0_0, 0)
    %lock_1 = aie.lock(%tile_0_0, 1)
    %lock_2 = aie.lock(%tile_0_0, 2)

    aie.bd_chain @simple_chain(%buf: memref<8xi16>, %l0: index, %l1: index, %l2: index) {
            aie.use_lock(%l0, "Acquire", 1)
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.use_lock(%l1, "Release", 1)
            aie.next_bd ^bd1
        ^bd1:
            aie.use_lock(%l1, "Acquire", 1)
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.use_lock(%l2, "Release", 1)
            aie.next_bd ^bd2
        ^bd2:
            aie.use_lock(%l2, "Acquire", 1)
            aie.dma_bd(%buf : memref<8xi16>, 0, 8)
            aie.use_lock(%l0, "Release", 1)
            aie.end
    }

    aiex.runtime_sequence(%buf: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain @simple_chain(%buf, %lock_0, %lock_1, %lock_2) : (memref<8xi16>, index, index, index)  
                                    on (%tile_0_0, MM2S, 0) 
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(%lock_0, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_1, "Release", 1)
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(%lock_1, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<12xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_2, "Release", 1)
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(%lock_2, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_1, "Release", 1)
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      %t2 = aiex.dma_start_bd_chain @simple_chain(%buf, %lock_0, %lock_0, %lock_0) : (memref<8xi16>, index, index, index)  
                                    on (%tile_0_0, MM2S, 1) 
      // CHECK: %[[task2:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(%lock_0, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_0, "Release", 1)
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(%lock_0, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<12xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_0, "Release", 1)
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(%lock_0, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_0, "Release", 1)
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task2]])
      %t3 = aiex.dma_start_bd_chain @simple_chain(%buf, %lock_2, %lock_1, %lock_0) : (memref<8xi16>, index, index, index)  
                                    on (%tile_0_0, S2MM, 0) 
      // CHECK: %[[task3:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.use_lock(%lock_2, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_1, "Release", 1)
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.use_lock(%lock_1, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<12xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_0, "Release", 1)
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.use_lock(%lock_0, "Acquire", 1)
      // CHECK:   aie.dma_bd(%buf : memref<8xi16>, 0, 8)
      // CHECK:   aie.use_lock(%lock_2, "Release", 1)
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task3]])
      aiex.dma_await_task(%t1)
    }
  }
}

