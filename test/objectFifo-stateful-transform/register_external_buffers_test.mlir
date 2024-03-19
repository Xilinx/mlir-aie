//===- register_external_buffers_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: January 27th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           memref.global "public" @ext_of_cons : memref<16xi32>
// CHECK:           memref.global "public" @ext_of : memref<16xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(7, 1)
// CHECK:           %[[VAL_1:.*]] = aie.tile(7, 0)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ext_of_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "ext_of_cons_lock_1"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "ext_of_cons_lock_2"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "ext_of_lock_0"}
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_9:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:           func.func @some_work(%[[VAL_10:.*]]: memref<16xi32>, %[[VAL_11:.*]]: memref<16xi32>) {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_15:.*]] = arith.constant 12 : index
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 1)
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 1)
// CHECK:             func.call @some_work(%[[VAL_2]], %[[VAL_3]]) : (memref<16xi32>, memref<16xi32>) -> ()
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 0)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @ext_of(MM2S, 0, 7)
// CHECK:           %[[VAL_16:.*]] = aie.shim_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(%[[VAL_8]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<64xi32>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 0)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_6]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_3]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @register_external_buffers {
 aie.device(xcvc1902) {
    %tile71 = aie.tile(7, 1)
    %tile70 = aie.tile(7, 0)

    aie.objectfifo @ext_of (%tile70, {%tile71}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    aie.objectfifo.register_external_buffers @ext_of (%tile70, {%ext_buffer_in}) : (memref<64xi32>)

    func.func @some_work(%a : memref<16xi32>, %b : memref<16xi32>) -> () {
        return
    }

    %core71 = aie.core(%tile71) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        %subview = aie.objectfifo.acquire @ext_of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0, %elem1) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.objectfifo.release @ext_of (Consume, 1)
        
        aie.end
    }
 }
}
