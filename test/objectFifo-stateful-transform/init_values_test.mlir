//===- init_values_test.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @of0_cons : memref<2x2xi32>
// CHECK:     memref.global "public" @of0 : memref<2x2xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_2_3 = aie.tile(2, 3)
// CHECK:     %of0_cons_buff_0 = aie.buffer(%tile_2_3) {sym_name = "of0_cons_buff_0"} : memref<2x2xi32> 
// CHECK:     %of0_cons_buff_1 = aie.buffer(%tile_2_3) {sym_name = "of0_cons_buff_1"} : memref<2x2xi32> 
// CHECK:     %of0_cons_buff_2 = aie.buffer(%tile_2_3) {sym_name = "of0_cons_buff_2"} : memref<2x2xi32> 
// CHECK:     %of0_cons_prod_lock = aie.lock(%tile_2_3, 0) {init = 3 : i32, sym_name = "of0_cons_prod_lock"}
// CHECK:     %of0_cons_cons_lock = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock"}
// CHECK:     %of0_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK:     %of0_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// CHECK:     %of0_buff_2 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_2"} : memref<2x2xi32> = dense<{{\[}}[8, 9], [10, 11]]>
// CHECK:     %of0_prod_lock = aie.lock(%tile_1_2, 0) {init = 0 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %of0_cons_lock = aie.lock(%tile_1_2, 1) {init = 3 : i32, sym_name = "of0_cons_lock"}
// CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_2_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%of0_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_buff_0 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of0_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_buff_1 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%of0_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_buff_2 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%tile_2_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%of0_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_cons_buff_0 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of0_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_cons_buff_1 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%of0_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of0_cons_buff_2 : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%of0_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>,
                                                                                            dense<[[8, 9], [10, 11]]> : memref<2x2xi32>]
 }
}
