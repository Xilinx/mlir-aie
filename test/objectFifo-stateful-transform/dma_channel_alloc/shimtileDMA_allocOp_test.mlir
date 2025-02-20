//===- shimtileDMA_allocOp_test.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:     aie.device(xcve2302) {
// CHECK:             memref.global "public" @objfifo_cons : memref<16xi32>
// CHECK:             memref.global "public" @objfifo : memref<16xi32>
// CHECK:             %shim_noc_tile_2_0 = aie.tile(2, 0)
// CHECK:             %tile_3_3 = aie.tile(3, 3)
// CHECK:             %objfifo_cons_prod_lock = aie.lock(%shim_noc_tile_2_0, 0) {init = 1 : i32, sym_name = "objfifo_cons_prod_lock"}
// CHECK:             %objfifo_cons_cons_lock = aie.lock(%shim_noc_tile_2_0, 1) {init = 0 : i32, sym_name = "objfifo_cons_cons_lock"}
// CHECK:             %objfifo_buff_0 = aie.buffer(%tile_3_3) {sym_name = "objfifo_buff_0"} : memref<16xi32> 
// CHECK:             %objfifo_buff_1 = aie.buffer(%tile_3_3) {sym_name = "objfifo_buff_1"} : memref<16xi32> 
// CHECK:             %objfifo_prod_lock = aie.lock(%tile_3_3, 0) {init = 2 : i32, sym_name = "objfifo_prod_lock"}
// CHECK:             %objfifo_cons_lock = aie.lock(%tile_3_3, 1) {init = 0 : i32, sym_name = "objfifo_cons_lock"}
// CHECK:             aie.flow(%tile_3_3, DMA : 0, %shim_noc_tile_2_0, DMA : 1)
// CHECK:             aie.shim_dma_allocation @objfifo(S2MM, 1, 2)
// CHECK:             %ext_buffer_out = aie.external_buffer {sym_name = "ext_buffer_out"} : memref<16xi32>
// CHECK:             %mem_3_3 = aie.mem(%tile_3_3) {
// CHECK:               %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:             ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:               aie.use_lock(%objfifo_cons_lock, AcquireGreaterEqual, 1)
// CHECK:               aie.dma_bd(%objfifo_buff_0 : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%objfifo_prod_lock, Release, 1)
// CHECK:               aie.next_bd ^bb2
// CHECK:             ^bb2:  // pred: ^bb1
// CHECK:               aie.use_lock(%objfifo_cons_lock, AcquireGreaterEqual, 1)
// CHECK:               aie.dma_bd(%objfifo_buff_1 : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%objfifo_prod_lock, Release, 1)
// CHECK:               aie.next_bd ^bb1
// CHECK:             ^bb3:  // pred: ^bb0
// CHECK:               aie.end
// CHECK:             }
// CHECK:             %shim_dma_2_0 = aie.shim_dma(%shim_noc_tile_2_0) {
// CHECK:               %0 = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
// CHECK:             ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:               aie.use_lock(%objfifo_cons_prod_lock, AcquireGreaterEqual, 1)
// CHECK:               aie.dma_bd(%ext_buffer_out : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%objfifo_cons_cons_lock, Release, 1)
// CHECK:               aie.next_bd ^bb1
// CHECK:             ^bb2:  // pred: ^bb0
// CHECK:               aie.end
// CHECK:             }
// CHECK:           }

module @shimtileDMA_alloc {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @objfifo (%tile33, {%tile20}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.shim_dma_allocation @objfifo(S2MM, 1, 2)

        %ext_buffer_out = aie.external_buffer {sym_name = "ext_buffer_out"}: memref<16xi32>
        aie.objectfifo.register_external_buffers @objfifo (%tile20, {%ext_buffer_out}) : (memref<16xi32>)
    }
}
