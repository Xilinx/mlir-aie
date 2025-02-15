//===- disable_synchronization_test.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @disable_sync {
// CHECK:   aie.device(xcve2302) {
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     memref.global "public" @of0 : memref<16xi32>
// CHECK:     %tile_1_2 = aie.tile(1, 2)
// CHECK:     %tile_1_3 = aie.tile(1, 3)
// CHECK:     %tile_3_3 = aie.tile(3, 3)
// CHECK:     %of1_cons_buff_0 = aie.buffer(%tile_3_3) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of1_cons_buff_1 = aie.buffer(%tile_3_3) {sym_name = "of1_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of1_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %of1_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_buff_1"} : memref<16xi32> 
// CHECK:     %of0_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:     aie.flow(%tile_1_2, DMA : 0, %tile_3_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.dma_bd(%of1_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.dma_bd(%of1_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%tile_3_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.dma_bd(%of1_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @disable_sync {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<16xi32>>
 }
}
