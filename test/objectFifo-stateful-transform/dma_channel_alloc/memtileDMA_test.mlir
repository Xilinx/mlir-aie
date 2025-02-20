//===- memtileDMA_test.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:           memref.global "public" @objfifo : memref<16xi32>
// CHECK:           %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK:           %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK:           %objfifo_cons_buff_0 = aie.buffer(%{{.*}}tile_3_3) {sym_name = "objfifo_cons_buff_0"} : memref<16xi32> 
// CHECK:           %objfifo_cons_buff_1 = aie.buffer(%{{.*}}tile_3_3) {sym_name = "objfifo_cons_buff_1"} : memref<16xi32> 
// CHECK:           %objfifo_cons_prod_lock = aie.lock(%{{.*}}tile_3_3, 0) {init = 2 : i32, sym_name = "objfifo_cons_prod_lock"}
// CHECK:           %objfifo_cons_cons_lock = aie.lock(%{{.*}}tile_3_3, 1) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock"}
// CHECK:           %objfifo_buff_0 = aie.buffer(%{{.*}}tile_1_1) {sym_name = "objfifo_buff_0"} : memref<16xi32> 
// CHECK:           %objfifo_buff_1 = aie.buffer(%{{.*}}tile_1_1) {sym_name = "objfifo_buff_1"} : memref<16xi32> 
// CHECK:           %objfifo_prod_lock = aie.lock(%{{.*}}tile_1_1, 3) {init = 2 : i32, sym_name = "objfifo_prod_lock"}
// CHECK:           %objfifo_cons_lock = aie.lock(%{{.*}}tile_1_1, 4) {init = 0 : i32, sym_name = "objfifo_cons_lock"}
// CHECK:           %buffer_1_1 = aie.buffer(%{{.*}}tile_1_1) : memref<16xi32> 
// CHECK:           %lock_1_1 = aie.lock(%{{.*}}tile_1_1, 0)
// CHECK:           %buffer_1_1_0 = aie.buffer(%{{.*}}tile_1_1) : memref<16xi32> 
// CHECK:           %lock_1_1_1 = aie.lock(%{{.*}}tile_1_1, 1)
// CHECK:           %buffer_1_1_2 = aie.buffer(%{{.*}}tile_1_1) : memref<16xi32> 
// CHECK:           %lock_1_1_3 = aie.lock(%{{.*}}tile_1_1, 2)
// CHECK:           aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_3_3, DMA : 0)
// CHECK:           %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:             %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%lock_1_1, Acquire, 1)
// CHECK:             aie.dma_bd(%buffer_1_1 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_1_1, Release, 0)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%lock_1_1_1, Acquire, 1)
// CHECK:             aie.dma_bd(%buffer_1_1_0 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_1_1_1, Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:             aie.use_lock(%lock_1_1_3, Acquire, 0)
// CHECK:             aie.dma_bd(%buffer_1_1_2 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%lock_1_1_3, Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:  // pred: ^bb3
// CHECK:             %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:           ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:             aie.use_lock(%objfifo_cons_lock, AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%objfifo_buff_0 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%objfifo_prod_lock, Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:  // pred: ^bb6
// CHECK:             aie.use_lock(%objfifo_cons_lock, AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%objfifo_buff_1 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%objfifo_prod_lock, Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb8:  // pred: ^bb5
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:             %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%objfifo_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%objfifo_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%objfifo_cons_cons_lock, Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%objfifo_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%objfifo_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%objfifo_cons_cons_lock, Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtileDMA_channels {
    aie.device(xcve2302) {
        %tile11 = aie.tile(1, 1)
        %tile33 = aie.tile(3, 3)

        %buff0 = aie.buffer(%tile11) : memref<16xi32>
        %lock0 = aie.lock(%tile11, 0)
        %buff1 = aie.buffer(%tile11) : memref<16xi32>
        %lock1 = aie.lock(%tile11, 1)
        %buff2 = aie.buffer(%tile11) : memref<16xi32>
        %lock2 = aie.lock(%tile11, 2)

        aie.objectfifo @objfifo (%tile11, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        %mem12 = aie.memtile_dma(%tile11) {
            %dma1 = aie.dma_start(MM2S, 1, ^bb1, ^bb3)
        ^bb1:
            aie.use_lock(%lock0, Acquire, 1)
            aie.dma_bd(%buff0 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock0, Release, 0)
            aie.next_bd ^bb2
        ^bb2:
            aie.use_lock(%lock1, Acquire, 1)
            aie.dma_bd(%buff1 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock1, Release, 0)
            aie.next_bd ^bb1
        ^bb3:
            %dma2 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
        ^bb4:
            aie.use_lock(%lock2, Acquire, 0)
            aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
            aie.use_lock(%lock2, Release, 1)
            aie.next_bd ^bb4
        ^bb5:
            aie.end
        }
    }
}
