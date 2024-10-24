//===- repeat_count_test.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @repeatCount {
// CHECK:   aie.device(npu1) {
// CHECK:     memref.global "public" @of2_cons : memref<16xi32>
// CHECK:     memref.global "public" @of2 : memref<16xi32>
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of2_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "of2_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of2_cons_buff_1 = aie.buffer(%tile_3_3) {sym_name = "of2_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of2_cons_prod_lock = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock"}
// CHECK:     %of2_cons_cons_lock = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock"}
// CHECK:     %of2_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_0"} : memref<16xi32> 
// CHECK:     %of2_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of2_buff_1"} : memref<16xi32> 
// CHECK:     %of2_prod_lock = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "of2_prod_lock"}
// CHECK:     %of2_cons_lock = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "of2_cons_lock"}
// CHECK:     %of1_cons_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_buff_1 = aie.buffer(%tile_1_3) {sym_name = "of1_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of1_cons_prod_lock = aie.lock(%tile_1_3, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %of1_cons_cons_lock = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %of1_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %of1_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32> 
// CHECK:     %of1_prod_lock = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:     %of1_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_1_3, DMA : 0)
// CHECK:     aie.flow(%tile_1_2, DMA : 1, %tile_3_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3, repeat_count = 2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6, repeat_count = 2)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       aie.use_lock(%of2_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of2_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%of2_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of2_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%of2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of2_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @repeatCount {
 aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile13}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile12, {%tile33}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
