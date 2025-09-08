//===- no_register_external_buffers_test.mlir -------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK:           aie.device(xcvc1902) {
// CHECK:             memref.global "public" @ext_of_cons : memref<16xi32>
// CHECK:             memref.global "public" @ext_of : memref<16xi32>
// CHECK:             %{{.*}}tile_7_2 = aie.tile(7, 2)
// CHECK:             %{{.*}}tile_7_0 = aie.tile(7, 0)
// CHECK:             %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_7_2) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32> 
// CHECK:             %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_7_2) {sym_name = "ext_of_cons_buff_1"} : memref<16xi32> 
// CHECK:             %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_7_2) {sym_name = "ext_of_cons_buff_2"} : memref<16xi32> 
// CHECK:             %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_7_2, 0) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:             %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_7_2, 1) {init = 0 : i32, sym_name = "ext_of_cons_lock_1"}
// CHECK:             %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_7_2, 2) {init = 0 : i32, sym_name = "ext_of_cons_lock_2"}
// CHECK:             aie.flow(%{{.*}}tile_7_0, DMA : 0, %{{.*}}tile_7_2, DMA : 0)
// CHECK:             aie.shim_dma_allocation @ext_of(MM2S, 0, 7)
// CHECK:             %mem_7_2 = aie.mem(%{{.*}}tile_7_2) {
// CHECK:               %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:             ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:               aie.use_lock(%[[VAL_3]], Acquire, 0)
// CHECK:               aie.dma_bd(%[[VAL_0]] : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:               aie.next_bd ^bb2
// CHECK:             ^bb2:  // pred: ^bb1
// CHECK:               aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:               aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:               aie.next_bd ^bb3
// CHECK:             ^bb3:  // pred: ^bb2
// CHECK:               aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:               aie.dma_bd(%[[VAL_2]] : memref<16xi32>, 0, 16)
// CHECK:               aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:               aie.next_bd ^bb1
// CHECK:             ^bb4:  // pred: ^bb0
// CHECK:               aie.end
// CHECK:             }
// CHECK:           }

module @no_register_external_buffers {
 aie.device(xcvc1902) {
    %tile72 = aie.tile(7, 2)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile72}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
