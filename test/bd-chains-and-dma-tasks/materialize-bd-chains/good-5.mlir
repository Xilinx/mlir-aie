//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-materialize-bd-chains %s | FileCheck %s

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)

    aie.shim_dma_allocation @alloc0 (MM2S, 0, 0)
    aie.shim_dma_allocation @alloc1 (MM2S, 1, 2)

    aie.bd_chain @simple_chain(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
            aie.dma_bd(%arg0 : memref<8xi16>, 0, 8, [<size=1, stride=0>, <size=2, stride=2>, <size=2, stride=4>, <size=2, stride=1>])
            aie.next_bd ^bd1
        ^bd1:
            aie.dma_bd(%arg1 : memref<12xi16>, 0, 12)
            aie.next_bd ^bd2
        ^bd2:
            aie.dma_bd(%arg2 : memref<8xi16>, 0, 8)
            aie.end
    }

    aiex.runtime_sequence(%arg0: memref<8xi16>, %arg1: memref<12xi16>, %arg2: memref<8xi16>) {
      %t1 = aiex.dma_start_bd_chain_for @simple_chain(%arg0, %arg1, %arg2) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                        for @alloc0
      // CHECK: %[[task1:.+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8, [<size = 1, stride = 0>, <size = 2, stride = 2>, <size = 2, stride = 4>, <size = 2, stride = 1>])
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12)
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.dma_bd(%arg2 : memref<8xi16>, 0, 8)
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      %t2 = aiex.dma_start_bd_chain_for @simple_chain(%arg2, %arg1, %arg0) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                        for @alloc1
      // CHECK: %[[task2:.+]] = aiex.dma_configure_task(%tile_2_0, MM2S, 1) {
      // CHECK:   aie.dma_bd(%arg2 : memref<8xi16>, 0, 8, [<size = 1, stride = 0>, <size = 2, stride = 2>, <size = 2, stride = 4>, <size = 2, stride = 1>])
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12)
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task2]])
      aiex.dma_await_task(%t1)
    }
  }
}

