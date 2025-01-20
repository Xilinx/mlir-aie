//===- init_values_join_output_test.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init_join_output {
module @init_join_output {
  aie.device(xcve2302) {
    memref.global "public" @of2_cons : memref<4xi32>
    memref.global "public" @of2 : memref<4xi32>
    memref.global "public" @of1_cons : memref<2xi32>
    memref.global "public" @of1 : memref<2xi32>
    memref.global "public" @of0_cons : memref<2xi32>
    memref.global "public" @of0 : memref<2xi32>
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_3 = aie.tile(2, 3)
    %of2_cons_prod_lock = aie.lock(%tile_1_0, 0) {init = 1 : i32, sym_name = "of2_cons_prod_lock"}
    %of2_cons_cons_lock = aie.lock(%tile_1_0, 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock"}
    %of2_buff_0 = aie.buffer(%tile_1_1) {sym_name = "of2_buff_0"} : memref<4xi32> = dense<[0, 1, 2, 3]>
    %of2_buff_1 = aie.buffer(%tile_1_1) {sym_name = "of2_buff_1"} : memref<4xi32> = dense<[4, 5, 6, 7]>
    %of2_prod_lock = aie.lock(%tile_1_1, 0) {init = 2 : i32, sym_name = "of2_prod_lock"}
    %of2_cons_lock = aie.lock(%tile_1_1, 1) {init = 2 : i32, sym_name = "of2_cons_lock"}
    %of1_buff_0 = aie.buffer(%tile_2_3) {sym_name = "of1_buff_0"} : memref<2xi32> 
    %of1_buff_1 = aie.buffer(%tile_2_3) {sym_name = "of1_buff_1"} : memref<2xi32> 
    %of1_prod_lock = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "of1_prod_lock"}
    %of1_cons_lock = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "of1_cons_lock"}
    %of0_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_0"} : memref<2xi32> 
    %of0_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_1"} : memref<2xi32> 
    %of0_prod_lock = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of0_prod_lock"}
    %of0_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
    aie.flow(%tile_1_2, DMA : 0, %tile_1_1, DMA : 0)
    aie.flow(%tile_2_3, DMA : 0, %tile_1_1, DMA : 1)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_0, DMA : 0)
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of0_buff_0 : memref<2xi32>, 0, 2)
      aie.use_lock(%of0_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of0_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of0_buff_1 : memref<2xi32>, 0, 2)
      aie.use_lock(%of0_prod_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of2_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of2_buff_0 : memref<4xi32>, 0, 2)
      aie.use_lock(%of2_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of2_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of2_buff_1 : memref<4xi32>, 0, 2)
      aie.use_lock(%of2_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%of2_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of2_buff_0 : memref<4xi32>, 2, 2)
      aie.use_lock(%of2_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%of2_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of2_buff_1 : memref<4xi32>, 2, 2)
      aie.use_lock(%of2_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%of2_cons_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%of2_buff_0 : memref<4xi32>, 0, 4)
      aie.use_lock(%of2_prod_lock, Release, 2)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%of2_cons_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%of2_buff_1 : memref<4xi32>, 0, 4)
      aie.use_lock(%of2_prod_lock, Release, 2)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of1_buff_0 : memref<2xi32>, 0, 2)
      aie.use_lock(%of1_prod_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%of1_buff_1 : memref<2xi32>, 0, 2)
      aie.use_lock(%of1_prod_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    aie.shim_dma_allocation @of2(S2MM, 0, 1)
  }
}

module @init_join_output {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2xi32>> 
    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>, 
                                                                                          dense<[4, 5, 6, 7]> : memref<4xi32>]

    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 2] [])
 }
}
