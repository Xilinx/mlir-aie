//===- link_test_output_sizes.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @link_distribute_output_sizes {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @link3_cons : memref<16xi32>
// CHECK:     memref.global "public" @link3 : memref<16xi32>
// CHECK:     memref.global "public" @link2_cons : memref<16xi32>
// CHECK:     memref.global "public" @link2 : memref<16xi32>
// CHECK:     memref.global "public" @link1_cons : memref<64xi32>
// CHECK:     memref.global "public" @link1 : memref<64xi32>
// CHECK:     %tile_2_0 = aie.tile(2, 0)
// CHECK:     %tile_2_1 = aie.tile(2, 1)
// CHECK:     %tile_2_2 = aie.tile(2, 2)
// CHECK:     %tile_2_3 = aie.tile(2, 3)
// CHECK:     %link3_cons_buff_0 = aie.buffer(%tile_2_3) {sym_name = "link3_cons_buff_0"} : memref<16xi32> 
// CHECK:     %link3_cons_buff_1 = aie.buffer(%tile_2_3) {sym_name = "link3_cons_buff_1"} : memref<16xi32> 
// CHECK:     %link3_cons_prod_lock = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
// CHECK:     %link3_cons_cons_lock = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
// CHECK:     %link2_cons_buff_0 = aie.buffer(%tile_2_2) {sym_name = "link2_cons_buff_0"} : memref<16xi32> 
// CHECK:     %link2_cons_buff_1 = aie.buffer(%tile_2_2) {sym_name = "link2_cons_buff_1"} : memref<16xi32> 
// CHECK:     %link2_cons_prod_lock = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
// CHECK:     %link2_cons_cons_lock = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
// CHECK:     %link1_cons_buff_0 = aie.buffer(%tile_2_1) {sym_name = "link1_cons_buff_0"} : memref<64xi32> 
// CHECK:     %link1_cons_prod_lock = aie.lock(%tile_2_1, 0) {init = 2 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:     %link1_cons_cons_lock = aie.lock(%tile_2_1, 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:     %link1_prod_lock = aie.lock(%tile_2_0, 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
// CHECK:     %link1_cons_lock = aie.lock(%tile_2_0, 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:     aie.flow(%tile_2_0, DMA : 0, %tile_2_1, DMA : 0)
// CHECK:     aie.flow(%tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
// CHECK:     aie.flow(%tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
// CHECK:     aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%link1_cons_prod_lock, AcquireGreaterEqual, 2)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<64xi32>, 0, 64)
// CHECK:       aie.use_lock(%link1_cons_cons_lock, Release, 2)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<64xi32>, 0, 32)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
// CHECK:     ^bb5:  // 2 preds: ^bb4, ^bb5
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<64xi32>, 32, 32)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb6:  // pred: ^bb4
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%tile_2_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link2_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%link2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link2_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%link2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%tile_2_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link3_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link3_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%link3_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link3_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link3_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%link3_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @link_distribute_output_sizes {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@link1] -> [@link2, @link3] ([][0, 32])
    }
}
