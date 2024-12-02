//===- link_via_shared_mem3.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
// Date: October 1st 2024
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// CHECK:  aie.device(xcve2302) {
// CHECK:    memref.global "public" @of2 : memref<16xi32>
// CHECK:    memref.global "public" @of1_cons : memref<16xi32>
// CHECK:    memref.global "public" @of1 : memref<16xi32>
// CHECK:    %tile_2_0 = aie.tile(2, 0)
// CHECK:    %tile_1_2 = aie.tile(1, 2)
// CHECK:    %tile_2_2 = aie.tile(2, 2)
// CHECK:    %of1_cons_buff_0 = aie.buffer(%tile_1_2) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:    %of1_cons_buff_1 = aie.buffer(%tile_1_2) {sym_name = "of1_cons_buff_1"} : memref<16xi32> 
// CHECK:    %of1_cons_prod_lock = aie.lock(%tile_1_2, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock"}
// CHECK:    %of1_cons_cons_lock = aie.lock(%tile_1_2, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock"}
// CHECK:    %of1_prod_lock = aie.lock(%tile_2_0, 0) {init = 1 : i32, sym_name = "of1_prod_lock"}
// CHECK:    %of1_cons_lock = aie.lock(%tile_2_0, 1) {init = 0 : i32, sym_name = "of1_cons_lock"}
// CHECK:    aie.flow(%tile_2_0, DMA : 0, %tile_1_2, DMA : 0)
// CHECK:    %mem_1_2 = aie.mem(%tile_1_2) {
// CHECK:      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:  
// CHECK:      aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%of1_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:      aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:  
// CHECK:      aie.use_lock(%of1_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%of1_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:      aie.use_lock(%of1_cons_cons_lock, Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:  
// CHECK:      aie.end
// CHECK:    }
// CHECK:  }

module @link_AIE2 {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) {via_shared_mem = 1 : i32} : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])
    }
}