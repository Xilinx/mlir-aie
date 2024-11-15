//===- link_test_AIE1.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @of2 : memref<16xi32>
// CHECK:           memref.global "public" @of1_cons : memref<16xi32>
// CHECK:           memref.global "public" @of1 : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_8:.*]] = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
// CHECK:           aie.shim_dma_allocation @of1(MM2S, 0, 2)
// CHECK:           %[[VAL_9:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32>, 0, 16)
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_AIE1 {
    aie.device(xcvc1902) {
        %tile20 = aie.tile(2, 0)
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @of1 (%tile20, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of2 (%tile12, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@of1] -> [@of2] ([] [])

        %ext_buff_in = aie.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
        aie.objectfifo.register_external_buffers @of1 (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
