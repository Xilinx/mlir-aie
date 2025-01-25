//===- link_test_distribute_offsets.mlir -----------------------*- MLIR -*-===//
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

// CHECK: module @link_distribute_offsets {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @link4_cons : memref<12xi32>
// CHECK:     memref.global "public" @link4 : memref<12xi32>
// CHECK:     memref.global "public" @link3_cons : memref<20xi32>
// CHECK:     memref.global "public" @link3 : memref<20xi32>
// CHECK:     memref.global "public" @link2_cons : memref<4x4xi32>
// CHECK:     memref.global "public" @link2 : memref<4x4xi32>
// CHECK:     memref.global "public" @link1_cons : memref<48xi32>
// CHECK:     memref.global "public" @link1 : memref<48xi32>
// CHECK:     %tile_2_0 = aie.tile(2, 0)
// CHECK:     %tile_2_1 = aie.tile(2, 1)
// CHECK:     %tile_2_2 = aie.tile(2, 2)
// CHECK:     %tile_2_3 = aie.tile(2, 3)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %link4_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "link4_cons_buff_0"} : memref<12xi32> 
// CHECK:     %link4_cons_buff_1 = aie.buffer(%tile_3_3) {sym_name = "link4_cons_buff_1"} : memref<12xi32> 
// CHECK:     %link4_cons_prod_lock = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "link4_cons_prod_lock"}
// CHECK:     %link4_cons_cons_lock = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "link4_cons_cons_lock"}
// CHECK:     %link3_cons_buff_0 = aie.buffer(%tile_2_3) {sym_name = "link3_cons_buff_0"} : memref<20xi32> 
// CHECK:     %link3_cons_buff_1 = aie.buffer(%tile_2_3) {sym_name = "link3_cons_buff_1"} : memref<20xi32> 
// CHECK:     %link3_cons_prod_lock = aie.lock(%tile_2_3, 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
// CHECK:     %link3_cons_cons_lock = aie.lock(%tile_2_3, 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
// CHECK:     %link2_cons_buff_0 = aie.buffer(%tile_2_2) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32> 
// CHECK:     %link2_cons_buff_1 = aie.buffer(%tile_2_2) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32> 
// CHECK:     %link2_cons_prod_lock = aie.lock(%tile_2_2, 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
// CHECK:     %link2_cons_cons_lock = aie.lock(%tile_2_2, 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
// CHECK:     %link1_cons_buff_0 = aie.buffer(%tile_2_1) {sym_name = "link1_cons_buff_0"} : memref<48xi32> 
// CHECK:     %link1_cons_buff_1 = aie.buffer(%tile_2_1) {sym_name = "link1_cons_buff_1"} : memref<48xi32> 
// CHECK:     %link1_cons_prod_lock = aie.lock(%tile_2_1, 0) {init = 6 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:     %link1_cons_cons_lock = aie.lock(%tile_2_1, 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:     %link1_prod_lock = aie.lock(%tile_2_0, 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
// CHECK:     %link1_cons_lock = aie.lock(%tile_2_0, 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:     aie.flow(%tile_2_0, DMA : 0, %tile_2_1, DMA : 0)
// CHECK:     aie.flow(%tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
// CHECK:     aie.flow(%tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
// CHECK:     aie.flow(%tile_2_1, DMA : 2, %tile_3_3, DMA : 0)
// CHECK:     aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link1_cons_prod_lock, AcquireGreaterEqual, 3)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<48xi32>, 0, 48)
// CHECK:       aie.use_lock(%link1_cons_cons_lock, Release, 3)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link1_cons_prod_lock, AcquireGreaterEqual, 3)
// CHECK:       aie.dma_bd(%link1_cons_buff_1 : memref<48xi32>, 0, 48)
// CHECK:       aie.use_lock(%link1_cons_cons_lock, Release, 3)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_1 : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:     ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:  // pred: ^bb7
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_1 : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb9:  // pred: ^bb6
// CHECK:       %3 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
// CHECK:     ^bb10:  // 2 preds: ^bb9, ^bb11
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_0 : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:  // pred: ^bb10
// CHECK:       aie.use_lock(%link1_cons_cons_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link1_cons_buff_1 : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%link1_cons_prod_lock, Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb12:  // pred: ^bb9
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%tile_2_2) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link2_cons_buff_0 : memref<4x4xi32>, 0, 16)
// CHECK:       aie.use_lock(%link2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link2_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link2_cons_buff_1 : memref<4x4xi32>, 0, 16)
// CHECK:       aie.use_lock(%link2_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%tile_2_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link3_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link3_cons_buff_0 : memref<20xi32>, 0, 20)
// CHECK:       aie.use_lock(%link3_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link3_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link3_cons_buff_1 : memref<20xi32>, 0, 20)
// CHECK:       aie.use_lock(%link3_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%link4_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link4_cons_buff_0 : memref<12xi32>, 0, 12)
// CHECK:       aie.use_lock(%link4_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%link4_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%link4_cons_buff_1 : memref<12xi32>, 0, 12)
// CHECK:       aie.use_lock(%link4_cons_cons_lock, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @link_distribute_offsets {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
    }
}
