//===- lock_analysis_test.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @lockAnalysis {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of1_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_buff_1 = aie.buffer(%tile_3_3) {sym_name = "of1_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of1_cons_prod_lock = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:     %of1_cons_cons_lock = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:     %of1_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %of1_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32> 
// CHECK:     %of1_prod_lock = aie.lock(%tile_1_2, 2) {init = 2 : i32, sym_name = "of1_prod_lock"}
// CHECK:     %of1_cons_lock = aie.lock(%tile_1_2, 3) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:     %test_buff = aie.buffer(%tile_1_2) {sym_name = "test_buff"} : memref<16xi32> 
// CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_3_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:       %test_prod_lock = aie.lock(%tile_1_2, 0) {init = 1 : i32, sym_name = "test_prod_lock"}
// CHECK:       %test_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "test_cons_lock"}
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%test_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%test_buff : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%test_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb5)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb4
// CHECK:       aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%of1_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of1_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of1_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb5:  // pred: ^bb2
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%tile_3_3) {
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
// CHECK:   }
// CHECK: }

module @lockAnalysis {
   aie.device(xcve2302) {
      %tile12 = aie.tile(1, 2)
      %tile33 = aie.tile(3, 3)

      %test_buff = aie.buffer(%tile12) {sym_name = "test_buff"} : memref<16xi32>

      aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

      %mem_1_2 = aie.mem(%tile12) {
         %test_prod_lock = aie.lock(%tile12, 0) {init = 1 : i32, sym_name = "test_prod_lock"}
         %test_cons_lock = aie.lock(%tile12, 1) {init = 0 : i32, sym_name = "test_cons_lock"}
         %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
         ^bb1:
            aie.use_lock(%test_prod_lock, AcquireGreaterEqual, 1)
            aie.dma_bd(%test_buff : memref<16xi32>, 0, 16)
            aie.use_lock(%test_cons_lock, Release, 1)
            aie.next_bd ^bb1
         ^bb2:
            aie.end
      }
   }
}
