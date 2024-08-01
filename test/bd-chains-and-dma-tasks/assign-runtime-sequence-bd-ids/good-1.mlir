//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// REQUIRES: ryzen_ai
//
// RUN: aie-opt --aie-materialize-bd-chains --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

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
      %t1 = aiex.dma_start_bd_chain @simple_chain(%arg0, %arg1, %arg2) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 0) 
      // CHECK: %[[task1:[0-9]+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8, [<size = 1, stride = 0>, <size = 2, stride = 2>, <size = 2, stride = 4>, <size = 2, stride = 1>]) {bd_id = 0 : i32}
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12) {bd_id = 1 : i32}
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.dma_bd(%arg2 : memref<8xi16>, 0, 8) {bd_id = 2 : i32}
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task1]])
      %t2 = aiex.dma_start_bd_chain @simple_chain(%arg2, %arg1, %arg0) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                    on (%tile_0_0, MM2S, 1) 
      // CHECK: %[[task2:[0-9]+]] = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
      // CHECK:   aie.dma_bd(%arg2 : memref<8xi16>, 0, 8, [<size = 1, stride = 0>, <size = 2, stride = 2>, <size = 2, stride = 4>, <size = 2, stride = 1>]) {bd_id = 3 : i32}
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12) {bd_id = 4 : i32}
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 5 : i32}
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task2]])
      %t3 = aiex.dma_start_bd_chain @simple_chain(%arg0, %arg1, %arg0) : (memref<8xi16>, memref<12xi16>, memref<8xi16>)  
                                    on (%tile_0_0, S2MM, 0) 
      // CHECK: %[[task3:[0-9]+]] = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8, [<size = 1, stride = 0>, <size = 2, stride = 2>, <size = 2, stride = 4>, <size = 2, stride = 1>]) {bd_id = 6 : i32}
      // CHECK:   aie.next_bd ^bb1
      // CHECK: ^bb1:
      // CHECK:   aie.dma_bd(%arg1 : memref<12xi16>, 0, 12) {bd_id = 7 : i32}
      // CHECK:   aie.next_bd ^bb2
      // CHECK: ^bb2:
      // CHECK:   aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 8 : i32}
      // CHECK:   aie.end
      // CHECK: }
      // CHECK: aiex.dma_start_task(%[[task3]])
      aiex.dma_await_task(%t1)
    }
  }
}

